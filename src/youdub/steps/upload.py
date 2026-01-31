from __future__ import annotations

import json
import os
import queue
import selectors
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from loguru import logger

from ..interrupts import CancelledByUser, check_cancelled, sleep_with_cancel
from .generate_info import resize_thumbnail

load_dotenv()

_RESULT_PREFIX = "__YOUDUB_RESULT__"


# --- Async Bilibili upload worker (non-blocking) ---
_BILI_UPLOAD_QUEUE: "queue.Queue[str]" = queue.Queue()
_BILI_UPLOAD_WORKER: threading.Thread | None = None
_BILI_UPLOAD_LOCK = threading.Lock()
_BILI_UPLOAD_ENQUEUED: set[str] = set()


def _is_uploaded(submission_result: object) -> bool:
    if not isinstance(submission_result, dict):
        return False
    results = submission_result.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        return results[0].get("code") == 0
    return submission_result.get("code") == 0


def _success_marker(payload: dict[str, object] | None = None) -> dict[str, object]:
    # Keep compatibility with pipeline._already_uploaded() which checks results[0].code == 0.
    out: dict[str, object] = {"results": [{"code": 0}]}
    if payload:
        out.update(payload)
    return out


def _repo_root() -> Path:
    # upload.py -> steps -> youdub -> src -> <repo root>
    return Path(__file__).resolve().parents[3]


def _biliapi_dir() -> Path:
    return _repo_root() / "scripts" / "biliapi"


def _biliapi_upload_script() -> Path:
    return _biliapi_dir() / "upload.mjs"


def _biliapi_node_modules_ready() -> bool:
    pkg = _biliapi_dir() / "node_modules" / "@renmu" / "bili-api" / "package.json"
    return pkg.exists()


def _biliapi_availability_error() -> str | None:
    node = shutil.which("node")
    if not node:
        return "错误：未找到 node（biliAPI 需要 node>=18）。请先安装 Node.js。"

    script = _biliapi_upload_script()
    if not script.exists():
        return f"错误：缺少 biliAPI 上传脚本: {script}"

    if not _biliapi_node_modules_ready():
        return "错误：biliAPI Node 依赖未安装。请先执行：cd scripts/biliapi && npm install"

    return None


def _ensure_cookie_file(cookie_path: Path) -> bool:
    try:
        cookie_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"创建cookie目录失败 {cookie_path.parent}: {exc}")
        return False

    try:
        if cookie_path.exists() and cookie_path.stat().st_size > 0:
            logger.info(f"使用现有B站cookie文件: {cookie_path}")
            return True
    except Exception:
        pass

    # Backward-compat: if user generated `cookies.json` in the same dir, adopt it.
    candidate = cookie_path.parent / "cookies.json"
    try:
        if candidate.exists() and candidate.stat().st_size > 0:
            if candidate.resolve() != cookie_path.resolve():
                shutil.copy2(candidate, cookie_path)
                logger.info(f"已采用现有cookies.json -> {cookie_path}")
            else:
                logger.info(f"使用现有B站cookie文件: {cookie_path}")
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"采用现有cookies.json失败: {exc}")

    return False


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _submit_order_from_env() -> list[str]:
    raw = (os.getenv("BILI_SUBMIT_APIS") or "").strip()
    if raw:
        return [x.strip() for x in raw.replace("|", ",").split(",") if x.strip()]
    return ["web", "b-cut", "client"]


def _bili_tid() -> int:
    try:
        return int(os.getenv("BILI_TID") or "201")
    except Exception:
        return 201


def _bili_access_token() -> str | None:
    v = (os.getenv("BILI_ACCESS_TOKEN") or "").strip()
    return v or None


def _bili_bcut_preupload_enabled() -> bool | None:
    raw = (os.getenv("BILI_BCUT_PREUPLOAD") or "").strip().lower()
    if not raw:
        return None
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return None


def _run_biliapi_upload(payload: dict[str, Any]) -> dict[str, Any]:
    err = _biliapi_availability_error()
    if err:
        return {"ok": False, "error": err}

    node = shutil.which("node")
    assert node is not None  # for type checker
    script = _biliapi_upload_script()

    proc: subprocess.Popen[str] | None = None
    sel: selectors.BaseSelector | None = None
    result: dict[str, Any] | None = None

    try:
        proc = subprocess.Popen(
            [node, str(script)],
            cwd=str(_repo_root()),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        proc.stdin.write(json.dumps(payload, ensure_ascii=False))
        proc.stdin.close()

        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)

        while True:
            check_cancelled()
            if proc.poll() is not None and not sel.get_map():
                break

            events = sel.select(timeout=0.2)
            if not events:
                if proc.poll() is not None:
                    break
                continue

            for key, _mask in events:
                line = key.fileobj.readline()
                if not line:
                    try:
                        sel.unregister(key.fileobj)
                    except Exception:
                        pass
                    continue

                s = line.rstrip("\n")
                if s.startswith(_RESULT_PREFIX):
                    raw = s[len(_RESULT_PREFIX) :].strip()
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        logger.error(f"biliAPI 返回结果解析失败: {raw[:200]}")
                        continue
                    if isinstance(obj, dict):
                        result = obj
                    else:
                        result = {"ok": False, "error": f"biliAPI 返回结果不是对象: {type(obj)}"}
                else:
                    # Forward node-side logs to Python logger so Gradio can stream them.
                    if s.strip():
                        logger.info(s)

        rc = proc.wait(timeout=1)
        if result is None:
            if rc == 0:
                return {"ok": False, "error": "biliAPI 子进程未输出结果"}
            return {"ok": False, "error": f"biliAPI 子进程退出码 {rc}，且未输出结果"}

        # Normalize
        if result.get("ok") is True:
            return result
        if rc == 0:
            # script says not ok but exit 0, still treat as failure
            return result
        return result
    except CancelledByUser:
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        raise
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"运行 biliAPI 子进程失败: {exc}"}
    finally:
        if sel is not None:
            try:
                sel.close()
            except Exception:
                pass


def _upload_video_with_biliapi(
    folder: str,
    proxy: Optional[str],
    upload_cdn: Optional[str],
    cookie_path: Path,
) -> tuple[bool, bool]:
    """Upload video to Bilibili.

    Returns:
        (success, actually_uploaded): success 表示操作成功（包括跳过已上传的），
        actually_uploaded 表示是否真正执行了上传（用于决定是否需要等待间隔）。
    """
    check_cancelled()

    folder_path = Path(folder).expanduser()
    try:
        folder_path = folder_path.resolve()
    except Exception:
        pass

    submission_result_path = folder_path / "bilibili.json"
    if submission_result_path.exists():
        try:
            submission_result = _read_json(submission_result_path)
            if _is_uploaded(submission_result):
                logger.info("视频已上传，跳过")
                return (True, False)  # success but not actually uploaded
        except Exception:
            pass

    video_path = folder_path / "video.mp4"
    # 封面优先级：video.png > download.webp（video.png 尺寸更符合 B 站要求）
    # 如果 video.png 不存在，尝试从 download.* 生成
    cover_path = folder_path / "video.png"
    if not cover_path.exists():
        try:
            generated = resize_thumbnail(str(folder_path))
            if generated:
                cover_path = Path(generated)
                logger.info(f"已生成封面: {cover_path}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"生成封面失败: {exc}")
    # 如果还是没有 video.png，回退到 download.webp
    if not cover_path.exists():
        fallback = folder_path / "download.webp"
        if fallback.exists():
            cover_path = fallback
    if not video_path.exists():
        logger.warning(f"未找到视频文件: {video_path}")
        return (False, False)

    summary_path = folder_path / "summary.json"
    if not summary_path.exists():
        logger.warning(f"未找到摘要文件: {summary_path}")
        return (False, False)
    summary = _read_json(summary_path)

    info_path = folder_path / "download.info.json"
    if not info_path.exists():
        logger.warning(f"未找到信息文件: {info_path}")
        return (False, False)
    data = _read_json(info_path)

    summary_title = str(summary.get("title", "Untitled")).replace("视频标题：", "").strip()
    summary_text = (
        str(summary.get("summary", ""))
        .replace("视频摘要：", "")
        .replace("视频简介：", "")
        .strip()
    )

    tags = summary.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    author = summary.get("author", "Unknown")
    title = f"【中配】{summary_title} - {author}"[:80]

    title_english = str(data.get("title", "") or "")
    webpage_url = str(data.get("webpage_url", "") or "")

    # Read model information for description
    translation_model = str(summary.get("translation_model", "") or "")
    tts_method = ""
    tts_done_path = folder_path / "wavs" / ".tts_done.json"
    if tts_done_path.exists():
        try:
            tts_done = _read_json(tts_done_path)
            tts_method = str(tts_done.get("tts_method", "") or "")
        except Exception:
            pass

    model_info_parts: list[str] = []
    if translation_model:
        model_info_parts.append(f"翻译模型: {translation_model}")
    if tts_method:
        tts_display = {"bytedance": "ByteDance TTS", "gemini": "Gemini TTS", "qwen": "Qwen3-TTS"}.get(
            tts_method, tts_method
        )
        model_info_parts.append(f"配音模型: {tts_display}")
    model_info_line = " | ".join(model_info_parts) if model_info_parts else ""

    # Note: webpage_url is already shown by Bilibili as "source" for reprinted videos (copyright=2),
    # so we don't include it in description to avoid duplication.
    description = (
        f"{title_english}\n{summary_text}\n\n"
        + (f"{model_info_line}\n" if model_info_line else "")
        + "项目地址：https://github.com/StarDuster/YAYD\n"
        "YAYD 是 YouDub-webui 的一个 fork，旨在将 YouTube 和其他平台上的高质量视频翻译和配音成中文版本。"
        "该工具结合了最新的 AI 技术，包括语音识别、大型语言模型翻译，以及 AI 声音克隆技术，"
        "提供与原视频相似的中文配音，为中文用户提供卓越的观看体验。"
    )

    base_tags = ["YouDub", str(author), "AI", "ChatGPT", "中文配音", "科学", "科普"]
    all_tags = base_tags + tags
    seen_tags: set[str] = set()
    final_tags: list[str] = []
    for t in all_tags:
        if not isinstance(t, str):
            continue
        if t not in seen_tags and len(t) <= 20:
            final_tags.append(t)
            seen_tags.add(t)
        if len(final_tags) >= 12:
            break

    try:
        cookie_path = cookie_path.expanduser()
        if not cookie_path.exists() or cookie_path.stat().st_size <= 0:
            logger.error(
                f"B站cookie文件未找到或为空: {cookie_path}. "
                "请先安装 Node 依赖（cd scripts/biliapi && npm install），"
                "再运行 `node scripts/biliapi/login.mjs` 登录生成 cookies.json，然后设置 BILI_COOKIE_PATH。"
            )
            return (False, False)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"访问cookie文件失败 {cookie_path}: {exc}")
        return (False, False)

    tid = _bili_tid()
    source = webpage_url.strip()
    copyright_ = 2 if source else 1

    options: dict[str, Any] = {
        "title": title,
        "tid": tid,
        "tag": ",".join(final_tags),
        "desc": description,
        "copyright": copyright_,
    }
    if copyright_ == 2:
        options["source"] = source[:200]

    if cover_path.exists():
        options["cover"] = str(cover_path)

    upload_payload: dict[str, Any] = {
        "cookieFile": str(cookie_path),
        "accessToken": _bili_access_token(),
        "proxy": proxy,
        "videoPaths": [str(video_path)],
        "options": options,
        "submitOrder": _submit_order_from_env(),
        "upload": {
            "line": upload_cdn,
        },
    }
    bcut_pre = _bili_bcut_preupload_enabled()
    if bcut_pre is not None:
        upload_payload["upload"]["bcutPreUpload"] = bcut_pre

    logger.info("通过 biliAPI 上传中…")
    result = _run_biliapi_upload(upload_payload)
    if result.get("ok") is not True:
        logger.error(f"biliAPI 上传失败: {result.get('error')}")
        return (False, False)

    # write a success marker compatible with pipeline._already_uploaded()
    marker_payload: dict[str, object] = {
        "tool": "biliapi",
        "title": title,
        "tid": tid,
        "tag": final_tags,
        "source": source,
        "submit": result.get("submit") or "unknown",
    }
    if result.get("aid") is not None:
        marker_payload["aid"] = result.get("aid")
    if result.get("bvid") is not None:
        marker_payload["bvid"] = result.get("bvid")

    with submission_result_path.open("w", encoding="utf-8") as f:
        json.dump(_success_marker(marker_payload), f, ensure_ascii=False, indent=4)

    logger.info("上传成功")
    return (True, True)


def upload_video(folder: str) -> bool:
    """Upload a single video folder (expects video.mp4 etc)."""
    err = _biliapi_availability_error()
    if err:
        logger.error(err)
        return False

    proxy = (os.getenv("BILI_PROXY") or "").strip() or None
    upload_cdn = (os.getenv("BILI_UPLOAD_CDN") or "").strip() or None
    cookie_path = Path((os.getenv("BILI_COOKIE_PATH") or "bili_cookies.json").strip() or "bili_cookies.json")
    if not _ensure_cookie_file(cookie_path):
        logger.error(
            f"B站cookie文件未就绪: {cookie_path}。"
            "请先运行 `node scripts/biliapi/login.mjs` 登录生成 cookies.json，再设置 BILI_COOKIE_PATH。"
        )
        return False
    success, _ = _upload_video_with_biliapi(folder, proxy, upload_cdn, cookie_path)
    return success


def upload_all_videos_under_folder(folder: str) -> str:
    err = _biliapi_availability_error()
    if err:
        return err

    proxy = (os.getenv("BILI_PROXY") or "").strip() or None
    upload_cdn = (os.getenv("BILI_UPLOAD_CDN") or "").strip() or None
    cookie_path = Path((os.getenv("BILI_COOKIE_PATH") or "bili_cookies.json").strip() or "bili_cookies.json")
    try:
        upload_interval = int(os.getenv("BILI_UPLOAD_INTERVAL") or "60")
    except ValueError:
        upload_interval = 60

    if proxy:
        logger.info(f"B站上传代理已配置: {proxy}")
    if upload_cdn:
        logger.info(f"B站首选上传线路: {upload_cdn}")
    logger.info(f"B站cookie路径: {cookie_path}")
    logger.info(f"B站上传间隔: {upload_interval}秒")

    if not _ensure_cookie_file(cookie_path):
        return (
            "错误：B站 cookie 未就绪。"
            "请先运行 `node scripts/biliapi/login.mjs` 登录生成 cookies.json，再设置 BILI_COOKIE_PATH（或直接放在默认位置）。"
        )

    count = 0
    need_wait = False  # 只有真正执行了上传才需要等待
    for root, _, files in os.walk(folder):
        check_cancelled()
        if "video.mp4" in files:
            # 已上传的直接跳过：不计数，也不需要等待间隔
            if _folder_uploaded(root):
                logger.info(f"跳过（已上传）: {root}")
                continue

            if need_wait and upload_interval > 0:
                logger.info(f"等待 {upload_interval} 秒后上传下一个视频...")
                sleep_with_cancel(upload_interval)

            success, actually_uploaded = _upload_video_with_biliapi(root, proxy, upload_cdn, cookie_path)
            if success:
                count += 1
            # 只有真正执行了上传才需要等待间隔
            need_wait = actually_uploaded
    return f"上传完成: {folder}（成功 {count} 个）"


def _folder_uploaded(folder: str) -> bool:
    marker = os.path.join(folder, "bilibili.json")
    if not os.path.exists(marker):
        return False
    try:
        with open(marker, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return _is_uploaded(obj)
    except Exception:
        return False


def _bili_upload_interval_seconds() -> int:
    try:
        return max(0, int(os.getenv("BILI_UPLOAD_INTERVAL") or "60"))
    except Exception:
        return 60


def _clear_pending_bili_uploads() -> None:
    while True:
        try:
            folder = _BILI_UPLOAD_QUEUE.get_nowait()
        except queue.Empty:
            break
        try:
            with _BILI_UPLOAD_LOCK:
                _BILI_UPLOAD_ENQUEUED.discard(folder)
        finally:
            try:
                _BILI_UPLOAD_QUEUE.task_done()
            except Exception:
                pass


def _bili_upload_worker() -> None:
    logger.info("B站后台上传线程已启动")
    first_upload = True
    while True:
        folder = _BILI_UPLOAD_QUEUE.get()
        try:
            if _folder_uploaded(folder):
                logger.info(f"B站后台上传跳过（已上传）: {folder}")
                continue

            if not first_upload:
                interval = _bili_upload_interval_seconds()
                if interval > 0:
                    logger.info(f"等待 {interval} 秒后进行下一个 B 站上传…")
                    sleep_with_cancel(interval)

            ok = upload_video(folder)
            first_upload = False
            if ok:
                logger.info(f"B站后台上传成功: {folder}")
            else:
                logger.error(f"B站后台上传失败: {folder}")
        except CancelledByUser as exc:
            logger.warning(f"B站后台上传被取消: {exc}")
            _clear_pending_bili_uploads()
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"B站后台上传异常: {exc}")
        finally:
            with _BILI_UPLOAD_LOCK:
                _BILI_UPLOAD_ENQUEUED.discard(folder)
            try:
                _BILI_UPLOAD_QUEUE.task_done()
            except Exception:
                pass


def upload_video_async(folder: str) -> str:
    err = _biliapi_availability_error()
    if err:
        return err

    folder = str(folder or "").strip()
    if not folder:
        return "错误：目录不能为空"
    if not os.path.exists(folder):
        return f"错误：目录不存在：{folder}"

    if _folder_uploaded(folder):
        return f"已上传，跳过：{folder}"

    with _BILI_UPLOAD_LOCK:
        if folder in _BILI_UPLOAD_ENQUEUED:
            return f"已在后台上传队列中：{folder}"
        _BILI_UPLOAD_ENQUEUED.add(folder)
        _BILI_UPLOAD_QUEUE.put(folder)

        global _BILI_UPLOAD_WORKER  # noqa: PLW0603
        if _BILI_UPLOAD_WORKER is None or not _BILI_UPLOAD_WORKER.is_alive():
            _BILI_UPLOAD_WORKER = threading.Thread(
                target=_bili_upload_worker,
                name="youdub-bili-upload-worker",
                daemon=True,
            )
            _BILI_UPLOAD_WORKER.start()

    logger.info(f"已加入B站上传队列: {folder}")
    return f"已加入B站后台上传队列: {folder}"

