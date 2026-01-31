from __future__ import annotations

import json
import os
import queue
import re
import shutil
import threading
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from loguru import logger

from ..interrupts import CancelledByUser, check_cancelled, sleep_with_cancel

load_dotenv()

try:
    import stream_gears  # type: ignore
except Exception as exc:  # noqa: BLE001 - keep best-effort import to avoid breaking non-upload features
    stream_gears = None  # type: ignore[assignment]
    _STREAM_GEARS_IMPORT_ERROR: Exception | None = exc
else:
    _STREAM_GEARS_IMPORT_ERROR = None


_STDIO_CAPTURE_LOCK = threading.Lock()


# --- Async Bilibili upload worker (non-blocking) ---
_BILI_UPLOAD_QUEUE: "queue.Queue[str]" = queue.Queue()
_BILI_UPLOAD_WORKER: threading.Thread | None = None
_BILI_UPLOAD_LOCK = threading.Lock()
_BILI_UPLOAD_ENQUEUED: set[str] = set()


class _StdioTeeCapture:
    """
    Capture C/Rust side logs by tee-ing process stdio fds.

    Why:
    - `stream_gears` prints Rust logs like `biliup::...` directly to stdout/stderr,
      which does NOT go through Python's `logging` module.
    - This capture keeps output visible (tee to original fd) while collecting it
      for error code extraction.

    Safety:
    - Uses a process-wide lock to avoid concurrent fd redirections.
    - Disabled automatically under pytest to avoid breaking test capture.
    """

    def __init__(self, *, max_bytes: int = 512_000, fds: tuple[int, ...] = (1, 2)):
        self._max_bytes = max_bytes
        self._fds = fds
        self._enabled = os.getenv("PYTEST_CURRENT_TEST") is None
        self._buffers: dict[int, bytearray] = {}
        self._saved_fds: dict[int, int] = {}
        self._read_fds: dict[int, int] = {}
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> "_StdioTeeCapture":
        if not self._enabled:
            return self
        _STDIO_CAPTURE_LOCK.acquire()
        try:
            for fd in self._fds:
                saved = os.dup(fd)
                r, w = os.pipe()
                os.dup2(w, fd)
                os.close(w)

                self._saved_fds[fd] = saved
                self._read_fds[fd] = r
                self._buffers[fd] = bytearray()

                t = threading.Thread(
                    target=self._reader,
                    args=(fd,),
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
            return self
        except Exception:
            # Best-effort restore if capture setup fails.
            self._restore()
            _STDIO_CAPTURE_LOCK.release()
            raise

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        if not self._enabled:
            return
        try:
            self._restore()
            # Give reader threads a moment to drain remaining output.
            for t in self._threads:
                t.join(timeout=0.5)
        finally:
            for saved in self._saved_fds.values():
                try:
                    os.close(saved)
                except Exception:
                    pass
            for r in self._read_fds.values():
                try:
                    os.close(r)
                except Exception:
                    pass
            _STDIO_CAPTURE_LOCK.release()

    def _restore(self) -> None:
        for fd, saved in self._saved_fds.items():
            try:
                os.dup2(saved, fd)
            except Exception:
                # If restoration fails, we can't do much.
                pass

    def _reader(self, fd: int) -> None:
        r = self._read_fds.get(fd)
        saved = self._saved_fds.get(fd)
        buf = self._buffers.get(fd)
        if r is None or saved is None or buf is None:
            return

        try:
            while True:
                chunk = os.read(r, 4096)
                if not chunk:
                    break
                # Tee to original fd so user still sees logs.
                try:
                    os.write(saved, chunk)
                except Exception:
                    pass
                # Store (bounded).
                if self._max_bytes > 0:
                    buf.extend(chunk)
                    if len(buf) > self._max_bytes:
                        del buf[: len(buf) - self._max_bytes]
        except Exception:
            return

    def lines(self) -> list[str]:
        out: list[str] = []
        for fd in self._fds:
            b = self._buffers.get(fd)
            if not b:
                continue
            try:
                text = b.decode("utf-8", errors="replace")
            except Exception:
                continue
            out.extend([ln for ln in text.splitlines() if ln.strip()])
        return out


def _extract_biliup_error(lines: list[str]) -> tuple[int | None, str | None, str | None]:
    """
    Best-effort extract a Bilibili error (code/message) from biliup logs.

    Returns (code, message, raw_line).
    """
    if not lines:
        return None, None, None

    text = "\n".join(lines)

    # Rust debug formatting e.g.:
    # ResponseData { code: 21566, data: None, message: "...", ttl: Some(1) }
    m_all = list(
        re.finditer(
            r'ResponseData\s*\{\s*code:\s*(\d+),.*?message:\s*"([^"]*)"',
            text,
            flags=re.DOTALL,
        )
    )
    if m_all:
        m = m_all[-1]
        try:
            return int(m.group(1)), m.group(2), m.group(0)
        except Exception:
            pass

    # JSON error line e.g.:
    # Failed to pre_upload from {"code":601,"message":"...","info":"..."}
    for line in reversed(lines):
        if '"code"' not in line or "{" not in line or "}" not in line:
            continue
        candidate = line[line.find("{") : line.rfind("}") + 1]
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        code = obj.get("code")
        if isinstance(code, int):
            msg = obj.get("message") or obj.get("info")
            return code, msg if isinstance(msg, str) else None, line

    # Fallback: any "code: NNN" / "(code: NNN)" style.
    for line in reversed(lines):
        m = re.search(r"\bcode[:=]\s*(\d+)\b", line)
        if not m:
            continue
        try:
            return int(m.group(1)), None, line
        except Exception:
            continue

    return None, None, None


def _is_uploaded(submission_result: object) -> bool:
    if not isinstance(submission_result, dict):
        return False
    results = submission_result.get("results")
    if isinstance(results, list) and results and isinstance(results[0], dict):
        return results[0].get("code") == 0
    # Backward-compat for any custom markers.
    return submission_result.get("code") == 0


def _success_marker(payload: dict[str, object] | None = None) -> dict[str, object]:
    # Keep compatibility with pipeline._already_uploaded() which checks results[0].code == 0.
    out: dict[str, object] = {"results": [{"code": 0}]}
    if payload:
        out.update(payload)
    return out


def _cdn_to_upload_line(cdn: Optional[str]):
    if stream_gears is None:
        return None
    # Dynamically build mapping from available UploadLine members (PyPI version may differ from source).
    mapping: dict[str, object] = {}
    for name in ("Bda", "Bda2", "Tx", "Txa", "Bldsa", "Alia", "Qn"):
        if hasattr(stream_gears.UploadLine, name):
            mapping[name.lower()] = getattr(stream_gears.UploadLine, name)
    if not cdn:
        return None
    return mapping.get(cdn.strip().lower())


def _iter_upload_lines(preferred_cdn: Optional[str]) -> list[object | None]:
    # order: preferred -> auto(None) -> common fallbacks
    # keep small & predictable; avoid over-design
    fallbacks = ["bda2", "bda", "tx", "txa", "bldsa"]
    out: list[object | None] = []

    preferred_line = _cdn_to_upload_line(preferred_cdn)
    if preferred_line is not None:
        out.append(preferred_line)

    # auto selection
    out.append(None)

    # explicit fallbacks
    for cdn in fallbacks:
        line = _cdn_to_upload_line(cdn)
        if line is None:
            continue
        if line not in out:
            out.append(line)
    return out


def _iter_submit_apis() -> list[object | None]:
    """
    Build a small list of submit APIs to rotate through.

    Notes:
    - `submit=None` keeps backward-compatible default behavior (usually app).
    - Newer biliup versions may support `submit="web"`.
    - `BILI_SUBMIT_APIS` can override the order, e.g. "app,web" or "b-cut-android".
    """
    raw = (os.getenv("BILI_SUBMIT_APIS") or "").strip()
    if raw:
        candidates = [x.strip() for x in raw.replace("|", ",").split(",") if x.strip()]
    else:
        # Default: prefer web submit, then fall back to app and other backends.
        candidates = ["web", "app", "b-cut-android"]

    out: list[object | None] = []
    for item in candidates:
        key = item.strip().lower()
        if key in ("app", "default", "auto", "none"):
            val: object | None = None
        else:
            val = key
        if val not in out:
            out.append(val)
    return out or [None]


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
        # If stat fails, just attempt re-login.
        pass

    # Prefer using an existing cookies.json generated by biliup login (or any other means).
    # We intentionally do NOT auto-login here (web cookies flow breaks frequently).
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


def _upload_video_with_biliup(
    folder: str,
    proxy: Optional[str],
    upload_cdn: Optional[str],
    cookie_path: Path,
) -> bool:
    check_cancelled()

    submission_result_path = os.path.join(folder, "bilibili.json")
    if os.path.exists(submission_result_path):
        try:
            with open(submission_result_path, "r", encoding="utf-8") as f:
                submission_result = json.load(f)
            if _is_uploaded(submission_result):
                logger.info("视频已上传")
                return True
        except Exception:
            # Corrupted marker -> proceed to upload.
            pass

    video_path = os.path.join(folder, "video.mp4")
    cover_path = os.path.join(folder, "video.png")
    if not os.path.exists(video_path):
        logger.warning(f"未找到视频文件: {video_path}")
        return False

    summary_path = os.path.join(folder, "summary.json")
    if not os.path.exists(summary_path):
        logger.warning(f"未找到摘要文件: {summary_path}")
        return False
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    info_path = os.path.join(folder, "download.info.json")
    if not os.path.exists(info_path):
        logger.warning(f"未找到信息文件: {info_path}")
        return False
    with open(info_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary_title = summary.get("title", "Untitled").replace("视频标题：", "").strip()
    summary_text = summary.get("summary", "").replace("视频摘要：", "").replace("视频简介：", "").strip()

    tags = summary.get("tags", [])
    if not isinstance(tags, list):
        tags = []

    author = summary.get("author", "Unknown")
    title = f"【中配】{summary_title} - {author}"[:80]

    title_english = data.get("title", "")
    webpage_url = data.get("webpage_url", "")

    # Read model information for description
    translation_model = summary.get("translation_model", "")
    tts_method = ""
    tts_done_path = os.path.join(folder, "wavs", ".tts_done.json")
    if os.path.exists(tts_done_path):
        try:
            with open(tts_done_path, "r", encoding="utf-8") as f:
                tts_done = json.load(f)
            tts_method = tts_done.get("tts_method", "")
        except Exception:
            pass

    # Build model info line
    model_info_parts = []
    if translation_model:
        model_info_parts.append(f"翻译模型: {translation_model}")
    if tts_method:
        tts_display = {"bytedance": "ByteDance TTS", "gemini": "Gemini TTS", "qwen": "Qwen3-TTS"}.get(tts_method, tts_method)
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

    base_tags = ["YouDub", author, "AI", "ChatGPT", "中文配音", "科学", "科普"]
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
        if not cookie_path.exists() or cookie_path.stat().st_size <= 0:
            logger.error(
                f"B站cookie文件未找到或为空: {cookie_path}. "
                "请通过 `biliup login` 生成（会创建cookies.json），然后设置 BILI_COOKIE_PATH。"
            )
            return False
    except Exception as exc:  # noqa: BLE001
        logger.error(f"访问cookie文件失败 {cookie_path}: {exc}")
        return False

    # Retry logic: try different lines, with exponential backoff.
    # Total attempts capped at MAX_ATTEMPTS to avoid endless retries.
    MAX_ATTEMPTS = 5
    lines = _iter_upload_lines(upload_cdn)
    submit_apis = _iter_submit_apis()
    strategies: list[tuple[object | None, object | None]] = [
        (line, submit) for line in lines for submit in submit_apis
    ]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        line, submit = strategies[(attempt - 1) % len(strategies)]
        check_cancelled()
        line_name = getattr(line, "name", None) or "auto"
        submit_name = "app" if submit is None else str(submit)
        try:
            logger.info(
                f"通过biliup上传中 (尝试 {attempt}/{MAX_ATTEMPTS}, 线路={line_name}, 投稿接口={submit_name})"
            )
            with _StdioTeeCapture() as cap:
                stream_gears.upload(  # type: ignore[union-attr]
                    video_path=[video_path],
                    cookie_file=str(cookie_path),
                    title=title,
                    tid=201,
                    tag=",".join(final_tags),
                    copyright=2,
                    source=str(webpage_url or ""),
                    desc=description,
                    cover=cover_path if os.path.exists(cover_path) else "",
                    line=line,
                    submit=submit,
                    proxy=proxy,
                )

            # write a success marker compatible with pipeline._already_uploaded()
            with open(submission_result_path, "w", encoding="utf-8") as f:
                json.dump(
                    _success_marker(
                        {
                            "tool": "biliup",
                            "title": title,
                            "tid": 201,
                            "tag": final_tags,
                            "source": webpage_url,
                        }
                    ),
                    f,
                    ensure_ascii=False,
                    indent=4,
                )
            logger.info("上传成功")
            return True
        except Exception as exc:  # noqa: BLE001
            # Try to recover the real code/message from rust-side logs.
            code, code_msg, raw = _extract_biliup_error(cap.lines() if "cap" in locals() else [])
            if code is not None:
                msg_part = f": {code_msg}" if code_msg else ""
                logger.error(f"biliup上传失败 (code={code}){msg_part}")
                if raw:
                    logger.debug(f"biliup错误详情: {raw}")
                # Avoid printing huge tracebacks for common rate-limit responses.
                logger.debug(f"biliup异常: {exc}")
            else:
                logger.exception(f"biliup上传失败: {exc}")
            msg = str(exc).lower()
            if "cookie" in msg or "csrf" in msg or "登录" in msg or "unauthorized" in msg:
                logger.error(
                    "检测到可能的认证/cookie错误。 "
                    "请重新运行 `biliup login` 刷新cookies.json，然后重试。"
                )
                return False
            # Backoff strategy:
            # - For known rate-limit codes (e.g. 601/21566), wait minutes (like biliup-cli does).
            # - Otherwise keep the short exponential backoff for transient errors.
            rate_limit_codes = {601, 21566, 406}
            if code in rate_limit_codes:
                wait = 60 * attempt  # 60s, 120s, 180s, 240s...
            else:
                # Exponential backoff: 5s, 10s, 20s, 40s
                wait = 5 * (2 ** (attempt - 1))
            if attempt < MAX_ATTEMPTS:
                logger.info(f"等待 {wait} 秒后重试...")
                sleep_with_cancel(wait)

    logger.error(f"重试 {MAX_ATTEMPTS} 次后仍上传失败")
    return False


def upload_video(folder: str) -> bool:
    """Upload a single video folder (expects video.mp4 etc)."""
    if stream_gears is None:
        logger.error(f"stream_gears不可用。请安装依赖 'biliup'。({_STREAM_GEARS_IMPORT_ERROR})")
        return False

    proxy = (os.getenv("BILI_PROXY") or "").strip() or None
    upload_cdn = (os.getenv("BILI_UPLOAD_CDN") or "").strip() or None
    cookie_path = Path((os.getenv("BILI_COOKIE_PATH") or "bili_cookies.json").strip() or "bili_cookies.json")
    if not _ensure_cookie_file(cookie_path):
        logger.error(
            f"B站cookie文件未就绪: {cookie_path}。 "
            "通过 `biliup login` 生成（会创建cookies.json），然后设置 BILI_COOKIE_PATH。"
        )
        return False
    return _upload_video_with_biliup(folder, proxy, upload_cdn, cookie_path)


def upload_all_videos_under_folder(folder: str) -> str:
    if stream_gears is None:
        return f"错误：stream_gears 不可用。请安装依赖 'biliup'。({_STREAM_GEARS_IMPORT_ERROR})"

    proxy = (os.getenv("BILI_PROXY") or "").strip() or None
    upload_cdn = (os.getenv("BILI_UPLOAD_CDN") or "").strip() or None
    cookie_path = Path((os.getenv("BILI_COOKIE_PATH") or "bili_cookies.json").strip() or "bili_cookies.json")
    # Interval between uploads to avoid rate limiting (default 60s)
    try:
        upload_interval = int(os.getenv("BILI_UPLOAD_INTERVAL") or "60")
    except ValueError:
        upload_interval = 60

    if proxy:
        logger.info(f"B站上传代理已配置: {proxy}")
    if upload_cdn:
        logger.info(f"B站首选上传CDN: {upload_cdn}")
    logger.info(f"B站cookie路径: {cookie_path}")
    logger.info(f"B站上传间隔: {upload_interval}秒")

    # Prepare cookie file once to avoid repeated attempts for each video folder.
    if not _ensure_cookie_file(cookie_path):
        return (
            "错误：B站 cookie 未就绪。"
            "请先运行 `biliup login` 生成 cookies.json，再设置 BILI_COOKIE_PATH（或直接放在默认位置）。"
        )

    count = 0
    first_upload = True
    for root, _, files in os.walk(folder):
        check_cancelled()
        if "video.mp4" in files:
            # Wait between uploads to avoid rate limiting
            if not first_upload and upload_interval > 0:
                logger.info(f"等待 {upload_interval} 秒后上传下一个视频...")
                sleep_with_cancel(upload_interval)
            if _upload_video_with_biliup(root, proxy, upload_cdn, cookie_path):
                count += 1
                first_upload = False
    return f"上传完成: {folder}（成功 {count} 个）"


def _folder_uploaded(folder: str) -> bool:
    """Return True if `bilibili.json` exists and indicates success."""
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
    """Best-effort clear queued uploads (used on cancellation)."""
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
            # De-dupe + skip if already uploaded.
            if _folder_uploaded(folder):
                logger.info(f"B站后台上传跳过（已上传）: {folder}")
                continue

            # Apply interval between *actual* uploads to reduce rate limiting.
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
    """
    Enqueue Bilibili upload in a background thread (non-blocking).

    Notes:
    - Uses a single daemon worker thread to avoid concurrent uploads (rate-limit friendly).
    - Returns immediately so upstream steps (pipeline / video synthesis) won't be blocked.
    """
    if stream_gears is None:
        return f"错误：stream_gears 不可用。请安装依赖 'biliup'。({_STREAM_GEARS_IMPORT_ERROR})"

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
