import os
import re
import shutil
import subprocess
import sys
import time
import queue
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import gradio as gr
from loguru import logger

from youdub.config import Settings
from youdub.pipeline import VideoPipeline
from youdub.interrupts import cancel_requested, ignore_signals_during_shutdown, install_signal_handlers, request_cancel
from youdub.steps import (
    download_from_url,
    generate_all_info_under_folder_stream,
    generate_all_wavs_under_folder,
    separate_all_audio_under_folder,
    synthesize_all_video_under_folder,
    transcribe_all_audio_under_folder,
    translate_all_transcript_under_folder,
    upload_all_videos_under_folder,
)
from youdub.models import ModelCheckError, ModelManager

settings = Settings()
model_manager = ModelManager(settings)
pipeline = VideoPipeline(settings=settings, model_manager=model_manager)

DEFAULT_SPEED_UP = 1.2

# --- Gradio UI（中文）---
_RESOLUTION_CHOICES = ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"]
_DEVICE_CHOICES = [("自动", "auto"), ("GPU", "cuda"), ("CPU", "cpu")]
_TARGET_LANGUAGE_CHOICES = [
    ("简体中文", "简体中文"),
    ("繁体中文", "繁体中文"),
    ("英语", "English"),
    ("德语", "Deutsch"),
    ("法语", "Français"),
    ("俄语", "русский"),
]
_TRANSLATION_STRATEGY_CHOICES = [("串行（带上下文，慢）", "history"), ("并行（先生成指南，快）", "guide_parallel")]
_TTS_METHOD_CHOICES = [("ByteDance", "bytedance"), ("Qwen", "qwen"), ("Gemini", "gemini")]
_ASR_METHOD_CHOICES = [("Whisper (本地)", "whisper"), ("Qwen3-ASR (本地)", "qwen")]

# Gradio 6.x 默认 time_limit=30s，会导致长任务“前端一直转圈但没有输出”
# （后台线程仍在跑，所以你能在终端看到日志）。这里显式关掉时间限制。
_INTERFACE_STREAM_KWARGS: dict[str, object] = {"time_limit": None, "stream_every": 0.2}

_LOGURU_LINE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} \| "
    r"(TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL)\s+\| "
)
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

_AUTO_SCROLL_JS = r"""
// NOTE: Blocks.launch(js=...) 会把这段当作“直接执行”的脚本。
// 之前写成 `() => { ... }` 实际上不会运行。
(function () {
  // 只在“用户本来就在底部附近”且“内容发生变化”时跟随到底，避免抖动/抢滚动条。
  const THRESHOLD_PX = 12;
  const STATE = new WeakMap();

  const isVisible = (el) => {
    try {
      return !!(el && el.isConnected && el.offsetParent !== null);
    } catch (e) {
      return false;
    }
  };

  const isNearBottom = (ta) =>
    (ta.scrollHeight - ta.scrollTop - ta.clientHeight) <= THRESHOLD_PX;

  const ensure = (ta) => {
    if (STATE.has(ta)) return;
    STATE.set(ta, { pinned: isNearBottom(ta), lastLen: (ta.value || "").length });
    ta.addEventListener(
      "scroll",
      () => {
        const st = STATE.get(ta);
        if (!st) return;
        st.pinned = isNearBottom(ta);
      },
      { passive: true }
    );
  };

  const tick = () => {
    const areas = document.querySelectorAll(".youdub-output textarea");
    areas.forEach((ta) => {
      if (!isVisible(ta)) return;
      ensure(ta);
      const st = STATE.get(ta);
      if (!st) return;
      const len = (ta.value || "").length;
      if (len === st.lastLen) return;
      st.lastLen = len;
      if (!st.pinned) return;
      // 在 rAF 循环里同步对齐到底部，减少“先上跳再下跳”的闪动。
      try {
        ta.scrollTop = ta.scrollHeight;
      } catch (e) {}
    });
  };

  // 初次和后续都用轻量轮询（textarea.value 变化不一定会触发 DOM mutation）。
  tick();
  const loop = () => {
    tick();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);

  // 监听“按段自适应拉伸”checkbox，禁用/灰化同一页的“加速倍率”slider。
  // TabbedInterface 会同时挂载多个页面，因此这里显式绑定两组（全自动 / 视频合成）。
  const setupAdaptiveStretchToggles = () => {
    const pairs = [
      {
        checkboxSelector: "#adaptive-stretch-checkbox input[type='checkbox']",
        sliderId: "speed-up-slider",
      },
      {
        checkboxSelector: "#adaptive-stretch-checkbox-synthesize input[type='checkbox']",
        sliderId: "speed-up-slider-synthesize",
      },
    ];

    let anyFound = false;
    let allReady = true;

    pairs.forEach(({ checkboxSelector, sliderId }) => {
      const checkbox = document.querySelector(checkboxSelector);
      const sliderContainer = document.getElementById(sliderId);
      if (!checkbox || !sliderContainer) {
        allReady = false;
        return;
      }
      anyFound = true;

      // Avoid duplicate binding across retries.
      if (checkbox.dataset.youdubAdaptiveBound === sliderId) return;

      const setDisabled = (disabled) => {
        sliderContainer.classList.toggle("youdub-disabled", !!disabled);
        sliderContainer.setAttribute("aria-disabled", disabled ? "true" : "false");

        // Gradio slider 通常包含 range + number + reset button；统一禁用。
        const controls = sliderContainer.querySelectorAll("input, button, select, textarea");
        controls.forEach((el) => {
          try {
            el.disabled = !!disabled;
          } catch (e) {}
          try {
            el.setAttribute("aria-disabled", disabled ? "true" : "false");
          } catch (e) {}
        });
      };

      const update = () => {
        const isAdaptive = !!checkbox.checked;
        setDisabled(isAdaptive);
      };

      checkbox.addEventListener("change", update);
      update();
      checkbox.dataset.youdubAdaptiveBound = sliderId;
    });

    // Wait until both pages are mounted so both bindings work reliably.
    return anyFound && allReady;
  };

  // 重试直到元素就绪（Gradio 动态加载/切 tab 时会延迟挂载）
  const trySetup = () => {
    if (!setupAdaptiveStretchToggles()) {
      setTimeout(trySetup, 500);
    }
  };
  trySetup();
})();
"""

_OUTPUT_CSS = r"""
/* 让输出框在 disabled 状态下依旧可读（部分环境 disabled textarea 会变得几乎不可见） */
.youdub-output textarea:disabled,
.youdub-output textarea[disabled] {
  opacity: 1 !important;
  color: #111 !important;
  -webkit-text-fill-color: #111 !important;
  background: #fcfcfc !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 12px;
  line-height: 1.4;
}

/* 通用禁用态：用于“启用自适应拉伸后，加速倍率无效”的灰化展示 */
.youdub-disabled {
  opacity: 0.5 !important;
  filter: grayscale(1) !important;
}
.youdub-disabled * {
  pointer-events: none !important;
  cursor: not-allowed !important;
}
"""


class _QueueWriter:
    """File-like writer which forwards lines to a queue."""

    def __init__(
        self,
        q: "queue.Queue[str]",
        prefix: str = "",
        mirror: Any | None = None,
        *,
        drop_loguru_lines: bool = False,
    ) -> None:
        self._q = q
        self._prefix = prefix
        self._mirror = mirror
        self._buf = ""
        self._overwrite = False
        self._drop_loguru_lines = drop_loguru_lines

    def write(self, s: str) -> int:  # file-like
        if not s:
            return 0
        # Mirror raw bytes to original stream (keep terminal logs).
        if self._mirror is not None:
            try:
                self._mirror.write(s)
            except Exception:
                pass

        # Strip ANSI to keep UI clean (colors, cursor movement, etc.)
        s_ui = _ANSI_ESCAPE_RE.sub("", s)

        # Handle dynamic progress updates: '\r' means overwrite current line.
        for ch in s_ui:
            if ch == "\r":
                # Carriage return: subsequent text overwrites the current line.
                self._overwrite = True
                self._buf = ""
                continue
            if ch == "\n":
                line = self._buf.rstrip()
                self._buf = ""
                self._overwrite = False
                if line and not (self._drop_loguru_lines and _LOGURU_LINE_RE.match(line)):
                    self._q.put(f"{self._prefix}{line}")
                continue
            self._buf += ch

        # If we're in overwrite mode (tqdm/yt-dlp), emit current line as a replace update.
        if self._overwrite and self._buf.strip():
            line = self._buf.rstrip()
            if not (self._drop_loguru_lines and _LOGURU_LINE_RE.match(line)):
                self._q.put("\r" + f"{self._prefix}{line}")
        return len(s)

    def flush(self) -> None:  # file-like
        if self._mirror is not None:
            try:
                self._mirror.flush()
            except Exception:
                pass
        line = self._buf.strip()
        if line and not (self._drop_loguru_lines and _LOGURU_LINE_RE.match(line)):
            if self._overwrite:
                self._q.put("\r" + f"{self._prefix}{line}")
            else:
                self._q.put(f"{self._prefix}{line}")
        self._buf = ""

    def isatty(self) -> bool:  # type: ignore[override]
        """Pretend to be a TTY so tqdm/yt-dlp will render progress updates."""
        if self._mirror is None:
            return True
        try:
            if hasattr(self._mirror, "isatty"):
                return bool(self._mirror.isatty())
        except Exception:
            pass
        return True

    def fileno(self) -> int:  # type: ignore[override]
        if self._mirror is None:
            raise OSError("No underlying file descriptor")
        if hasattr(self._mirror, "fileno"):
            return int(self._mirror.fileno())
        raise OSError("No fileno()")


@contextmanager
def _capture_output_to_queue(q: "queue.Queue[str]"):
    """Temporarily mirror loguru + stdout/stderr into a queue (per Gradio run)."""
    old_stdout, old_stderr = sys.stdout, sys.stderr

    def _sink(message: Any) -> None:
        try:
            text = str(message).rstrip("\n")
        except Exception:
            return
        if text:
            q.put(text)

    sink_id: int | None
    try:
        # IMPORTANT:
        # - enqueue=False so UI gets logs immediately (enqueue=True may delay output)
        # - keep format consistent with terminal output for readability
        sink_id = logger.add(
            _sink,
            level="INFO",
            enqueue=False,
            colorize=False,
            backtrace=False,
            diagnose=False,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        )
    except Exception:
        sink_id = None

    try:
        sys.stdout = _QueueWriter(q, mirror=old_stdout)  # type: ignore[assignment]
        # tqdm/yt-dlp progress often goes to stderr; capture it too.
        # Drop loguru-like lines here to avoid duplicates (we already capture loguru via sink).
        sys.stderr = _QueueWriter(q, mirror=old_stderr, drop_loguru_lines=True)  # type: ignore[assignment]
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        if sink_id is not None:
            try:
                logger.remove(sink_id)
            except Exception:
                pass


def _stream_run(fn: Callable[[], Any], *, max_lines: int = 400) -> Iterator[str]:
    """Run `fn` in a thread and stream collected logs to Gradio output."""

    q: "queue.Queue[str]" = queue.Queue()
    done = threading.Event()
    result_box: dict[str, Any] = {}
    exc_box: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            with _capture_output_to_queue(q):
                result_box["result"] = fn()
        except BaseException as exc:  # pylint: disable=broad-except
            exc_box["exc"] = exc
        finally:
            done.set()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()

    lines: list[str] = []
    progress_line: str | None = None

    def _render() -> str:
        if progress_line:
            return "\n".join(lines + [progress_line])
        return "\n".join(lines)

    def _append_text(text: str) -> None:
        nonlocal progress_line
        s = str(text)
        if s.startswith("\r"):
            progress_line = s[1:].rstrip()
            return
        else:
            for part in s.splitlines():
                part = part.rstrip()
                if part:
                    if progress_line is not None and part == progress_line:
                        progress_line = None
                    lines.append(part)
        if len(lines) > max_lines:
            # Drop oldest lines to keep UI payload bounded.
            del lines[: max(50, max_lines // 5)]

    # Always emit an initial line so the UI never stays blank.
    _append_text("开始执行…")
    yield _render()

    last_yield = 0.0
    while True:
        try:
            item = q.get(timeout=0.2)
        except queue.Empty:
            if done.is_set():
                break
            continue
        _append_text(item)
        now = time.monotonic()
        if (now - last_yield) >= 0.2:
            last_yield = now
            yield _render()

    # Finalize any in-place progress line as a normal line.
    if progress_line:
        lines.append(progress_line)
        progress_line = None
        if len(lines) > max_lines:
            del lines[: max(50, max_lines // 5)]

    t.join(timeout=0)

    if "exc" in exc_box:
        exc = exc_box["exc"]
        _append_text(f"异常: {type(exc).__name__}: {exc}")
        yield _render()
        return

    # Always show final result (if any).
    result = result_box.get("result")
    if result is not None:
        if isinstance(result, str):
            _append_text(result)
        else:
            _append_text(str(result))

    # Flush one last time (may include final result line).
    yield _render()


def _streamify(fn: Callable[..., Any]) -> Callable[..., Iterator[str]]:
    """Wrap a non-streaming function so its logs show up in Gradio Output."""

    def _wrapped(*args: Any, **kwargs: Any) -> Iterator[str]:
        # IMPORTANT: Gradio detects streaming by `inspect.isgeneratorfunction(fn)`.
        # Returning a generator object is not enough; this wrapper itself must be
        # a generator function.
        yield from _stream_run(lambda: fn(*args, **kwargs))

    return _wrapped


def _default_use_nvenc() -> bool:
    # Best-effort NVIDIA GPU detection for NVENC default.
    # Keep this independent of torch installation/config, since NVENC only needs GPU+ffmpeg.
    nvidia_paths = ("/dev/nvidia0", "/dev/nvidiactl", "/proc/driver/nvidia/version")
    if any(os.path.exists(p) for p in nvidia_paths):
        return True

    candidates: list[str] = []
    exe = shutil.which("nvidia-smi")
    if exe:
        candidates.append(exe)
    # WSL2: nvidia-smi is often available but not on PATH.
    if os.path.exists("/usr/lib/wsl/lib/nvidia-smi"):
        candidates.append("/usr/lib/wsl/lib/nvidia-smi")

    for exe in candidates:
        try:
            r = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=2)
        except Exception:
            continue
        if r.returncode == 0 and "GPU" in (r.stdout or ""):
            return True
    return False


DEFAULT_USE_NVENC = _default_use_nvenc()


def _safe_run(names, func, *args, **kwargs):
    try:
        model_manager.enforce_offline()
        if names:
            model_manager.ensure_ready(names)
        return func(*args, **kwargs)
    except ModelCheckError as exc:
        return str(exc)


@contextmanager
def _temp_env(updates: dict[str, str | None]):
    """Temporarily set environment variables for this run only."""
    old: dict[str, str | None] = {}
    try:
        for k, v in updates.items():
            old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_pipeline(
    root_folder,
    url,
    num_videos,
    resolution,
    demucs_model,
    device,
    shifts,
    asr_method,
    qwen_asr_model_dir,
    whisper_model,
    whisper_device,
    whisper_cpu_model,
    whisper_batch_size,
    qwen_asr_num_threads,
    qwen_asr_vad_segment_threshold,
    whisper_diarization,
    whisper_min_speakers,
    whisper_max_speakers,
    translation_target_language,
    translation_strategy,
    translation_max_concurrency,
    translation_chunk_size,
    translation_guide_max_chars,
    tts_method,
    qwen_tts_batch_size,
    subtitles,
    bilingual_subtitle,
    tts_adaptive_segment_stretch,
    speed_up,
    fps,
    use_nvenc,
    target_resolution,
    max_workers,
    max_retries,
    auto_upload_video,
):
    try:
        with _temp_env(
            {
                "TRANSLATION_STRATEGY": translation_strategy,
                "TRANSLATION_MAX_CONCURRENCY": str(int(translation_max_concurrency)),
                "TRANSLATION_CHUNK_SIZE": str(int(translation_chunk_size)),
                "TRANSLATION_GUIDE_MAX_CHARS": str(int(translation_guide_max_chars)),
            }
        ):
            return pipeline.run(
                root_folder=root_folder,
                url=url,
                num_videos=num_videos,
                resolution=resolution,
                demucs_model=demucs_model,
                device=device,
                shifts=shifts,
                asr_method=asr_method,
                qwen_asr_model_dir=qwen_asr_model_dir,
                whisper_model=whisper_model,
                whisper_device=whisper_device,
                whisper_cpu_model=whisper_cpu_model,
                whisper_batch_size=whisper_batch_size,
                qwen_asr_num_threads=qwen_asr_num_threads,
                qwen_asr_vad_segment_threshold=qwen_asr_vad_segment_threshold,
                whisper_diarization=whisper_diarization,
                whisper_min_speakers=whisper_min_speakers,
                whisper_max_speakers=whisper_max_speakers,
                translation_target_language=translation_target_language,
                tts_method=tts_method,
                qwen_tts_batch_size=qwen_tts_batch_size,
                tts_adaptive_segment_stretch=bool(tts_adaptive_segment_stretch),
                subtitles=subtitles,
                bilingual_subtitle=bool(bilingual_subtitle),
                speed_up=speed_up,
                fps=fps,
                target_resolution=target_resolution,
                use_nvenc=use_nvenc,
                max_workers=max_workers,
                max_retries=max_retries,
                auto_upload_video=auto_upload_video,
            )
    except ModelCheckError as exc:
        return str(exc)


def show_model_status():
    return model_manager.describe_status()


def download_models(hf_token: str):
    """下载所有需要的离线模型。"""
    import torch

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return "缺少 huggingface_hub：请运行 `pip install huggingface-hub`（或 `uv sync`）"

    # 模型版本配置
    MODELS_TO_DOWNLOAD = {
        "whisper": {
            "repo_id": "Systran/faster-whisper-large-v3",
            "revision": "edc79942a0352e00c3b03657b4943f293cf0f1d0",
        },
        "diarization": {
            "repo_id": "pyannote/speaker-diarization-3.1",
            "revision": "84fd25912480287da0247647c3d2b4853cb3ee5d",
        },
        "segmentation": {
            "repo_id": "pyannote/segmentation-3.0",
            "revision": "4ca4d5a8d2ab82ddfbea8aa3b29c15431671239c",
        },
    }

    # 临时清除离线模式环境变量
    env_vars_to_clear = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]
    old_env = {}
    for var in env_vars_to_clear:
        old_env[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    # 启用快速传输
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    try:
        logger.info("开始下载离线模型...")
        logger.info(f"目标目录: {settings.root_folder}")

        # 1. Demucs
        logger.info("\n=== 1. Demucs 模型 ===")
        try:
            from demucs_infer.pretrained import get_model

            demucs_dir = settings.resolve_path(settings.demucs_model_dir)
            if demucs_dir:
                demucs_dir.mkdir(parents=True, exist_ok=True)
                torch.hub.set_dir(str(demucs_dir))
                logger.info(f"设置 torch hub 目录: {demucs_dir}")

            model_name = settings.demucs_model_name
            logger.info(f"下载 Demucs 模型: {model_name} ...")
            get_model(model_name)
            logger.info("Demucs 模型下载完成。")
        except Exception as e:
            logger.error(f"Demucs 下载失败: {e}")

        # 2. Whisper
        logger.info("\n=== 2. Whisper 模型 ===")
        wx_conf = MODELS_TO_DOWNLOAD["whisper"]
        wx_path = settings.resolve_path(settings.whisper_model_path)
        logger.info(f"下载 Whisper 模型 ({wx_conf['revision'][:7]}) -> {wx_path} ...")
        try:
            snapshot_download(
                repo_id=wx_conf["repo_id"],
                revision=wx_conf["revision"],
                local_dir=wx_path,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            logger.info("Whisper 模型下载完成。")
        except Exception as e:
            logger.error(f"Whisper 下载失败: {e}")

        # 3. Pyannote 说话人分离
        logger.info("\n=== 3. Pyannote 说话人分离 ===")
        if not hf_token:
            logger.warning("未提供 HF_TOKEN，跳过 Pyannote 模型下载（需要同意协议并设置 token）")
        else:
            diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
            old_hf_home = os.environ.get("HF_HOME")
            os.environ["HF_HOME"] = str(diar_dir)

            for key in ["diarization", "segmentation"]:
                conf = MODELS_TO_DOWNLOAD[key]
                logger.info(f"下载 {conf['repo_id']} ({conf['revision'][:7]}) ...")
                try:
                    snapshot_download(
                        repo_id=conf["repo_id"],
                        revision=conf["revision"],
                        token=hf_token,
                        resume_download=True,
                    )
                    logger.info(f"{conf['repo_id']} 下载完成。")
                except Exception as e:
                    logger.error(f"下载失败 {conf['repo_id']}: {e}")

            if old_hf_home is not None:
                os.environ["HF_HOME"] = old_hf_home
            else:
                os.environ.pop("HF_HOME", None)

        logger.info("\n下载流程已结束。")
        return model_manager.describe_status()

    finally:
        # 恢复环境变量
        for var, val in old_env.items():
            if val is not None:
                os.environ[var] = val
            else:
                os.environ.pop(var, None)


do_everything_interface = gr.Interface(
    fn=_streamify(run_pipeline),
    inputs=[
        gr.Textbox(label="根目录（下载/输出）", value=str(settings.root_folder)),
        gr.Textbox(
            label="视频链接",
            placeholder="支持：视频 / 播放列表 / 频道链接",
            value="https://www.youtube.com/watch?v=4_SH2nfbQZ8",
        ),
        gr.Slider(minimum=1, maximum=500, step=1, label="下载视频数量", value=20),
        gr.Radio(
            _RESOLUTION_CHOICES,
            label="下载分辨率",
            value="1080p",
        ),
        gr.Radio(
            ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q", "SIG"],
            label="Demucs 模型",
            value=settings.demucs_model_name,
        ),
        gr.Radio(_DEVICE_CHOICES, label="Demucs 运行设备", value=settings.demucs_device),
        gr.Slider(minimum=0, maximum=10, step=1, label="随机移位次数", value=settings.demucs_shifts),
        gr.Dropdown(_ASR_METHOD_CHOICES, label="语音识别方式", value=settings.asr_method),
        gr.Textbox(label="Qwen3-ASR 模型路径", value=str(settings.qwen_asr_model_path or "")),
        gr.Textbox(label="Whisper 模型路径", value=str(settings.whisper_model_path)),
        gr.Radio(_DEVICE_CHOICES, label="Whisper 运行设备", value=settings.whisper_device),
        gr.Textbox(label="Whisper CPU 模型路径（可选）", value=str(settings.whisper_cpu_model_path or "")),
        gr.Slider(minimum=1, maximum=128, step=1, label="Whisper 批大小", value=settings.whisper_batch_size),
        gr.Slider(minimum=1, maximum=16, step=1, label="Qwen3-ASR 并发线程数", value=settings.qwen_asr_num_threads),
        gr.Slider(minimum=30, maximum=180, step=10, label="Qwen3-ASR VAD分段时长(秒)", value=settings.qwen_asr_vad_segment_threshold),
        gr.Checkbox(label="说话人分离", value=True),
        gr.Number(label="最少说话人数（可选）", value=None, step=1, precision=0),
        gr.Number(label="最多说话人数（可选）", value=None, step=1, precision=0),
        gr.Dropdown(
            _TARGET_LANGUAGE_CHOICES,
            label="翻译目标语言",
            value=settings.translation_target_language,
        ),
        gr.Dropdown(
            _TRANSLATION_STRATEGY_CHOICES,
            label="翻译策略",
            value="guide_parallel",
        ),
        gr.Slider(minimum=1, maximum=32, step=1, label="翻译最大并发数", value=4),
        gr.Slider(minimum=1, maximum=64, step=1, label="翻译分块大小", value=8),
        gr.Slider(minimum=800, maximum=5000, step=100, label="翻译引导信息最大字符数", value=2500),
        gr.Dropdown(
            _TTS_METHOD_CHOICES,
            label="配音方式",
            value=settings.tts_method,
        ),
        gr.Slider(minimum=1, maximum=64, step=1, label="Qwen TTS 批大小", value=settings.qwen_tts_batch_size),
        gr.Checkbox(label="字幕", value=True),
        gr.Checkbox(label="双语字幕", value=False),
        gr.Checkbox(
            label="按段自适应拉伸语音(减少无声)",
            value=False,
            elem_id="adaptive-stretch-checkbox",
            info="启用后下方的加速倍率无效",
        ),
        gr.Slider(
            minimum=0.5,
            maximum=2,
            step=0.05,
            label="加速倍率",
            value=DEFAULT_SPEED_UP,
            elem_id="speed-up-slider",
        ),
        gr.Slider(minimum=1, maximum=60, step=1, label="帧率（每秒帧数）", value=30),
        gr.Checkbox(label="使用 NVENC（h264_nvenc）", value=DEFAULT_USE_NVENC, info="需要 NVIDIA GPU"),
        gr.Radio(
            _RESOLUTION_CHOICES,
            label="输出分辨率",
            value="1080p",
        ),
        gr.Slider(minimum=1, maximum=100, step=1, label="最大并发任务数", value=1),
        gr.Slider(minimum=1, maximum=10, step=1, label="最大重试次数", value=3),
        gr.Checkbox(label="自动上传到 B 站", value=False),
    ],
    outputs=gr.Textbox(label="输出", lines=20, max_lines=20, autoscroll=False, elem_classes=["youdub-output"]),
    title="全自动",
    flagging_mode="never",
    submit_btn="开始全流程",
    stop_btn="停止",
    clear_btn="清空",
    **_INTERFACE_STREAM_KWARGS,
)

youtube_interface = gr.Interface(
    fn=_streamify(download_from_url),
    inputs=[
        gr.Textbox(
            label="视频链接",
            placeholder="支持：视频 / 播放列表 / 频道链接",
            value="https://www.bilibili.com/list/1263732318",
        ),
        gr.Textbox(label="输出目录", value=str(settings.root_folder)),
        gr.Radio(
            _RESOLUTION_CHOICES,
            label="下载分辨率",
            value="1080p",
        ),
        gr.Slider(minimum=1, maximum=100, step=1, label="下载视频数量", value=5),
    ],
    outputs=gr.Textbox(label="输出", lines=20, max_lines=20, autoscroll=False, elem_classes=["youdub-output"]),
    title="下载视频",
    flagging_mode="never",
    submit_btn="开始下载",
    stop_btn="停止",
    clear_btn="清空",
    **_INTERFACE_STREAM_KWARGS,
)

demucs_interface = gr.Interface(
    fn=_streamify(lambda folder, model, device, progress, shifts: _safe_run(
        [model_manager._demucs_requirement().name],  # type: ignore[attr-defined]
        separate_all_audio_under_folder,
        folder,
        model_name=model,
        device=device,
        progress=progress,
        shifts=shifts,
        settings=settings,
        model_manager=model_manager,
    )),
    inputs=[
        gr.Textbox(label="目录", value=str(settings.root_folder)),
        gr.Radio(
            ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q", "SIG"],
            label="Demucs 模型",
            value=settings.demucs_model_name,
        ),
        gr.Radio(_DEVICE_CHOICES, label="运行设备", value=settings.demucs_device),
        gr.Checkbox(label="在控制台显示进度条", value=True),
        gr.Slider(minimum=0, maximum=10, step=1, label="随机移位次数", value=settings.demucs_shifts),
    ],
    outputs=gr.Textbox(label="输出", lines=20, max_lines=20, autoscroll=False, elem_classes=["youdub-output"]),
    title="人声分离",
    flagging_mode="never",
    submit_btn="开始分离",
    stop_btn="停止",
    clear_btn="清空",
    **_INTERFACE_STREAM_KWARGS,
)

def _run_transcribe(folder, asr_method, qwen_model_dir, model, cpu_model, device, batch_size, qwen_threads, qwen_vad, diarization, min_speakers, max_speakers):
    # Determine required models based on ASR method
    names = []
    if asr_method == "qwen":
        names.append(model_manager._qwen_asr_requirement().name)  # type: ignore[attr-defined]
    if diarization:
        names.append(model_manager._whisper_diarization_requirement().name)  # type: ignore[attr-defined]
    return _safe_run(
        names,
        transcribe_all_audio_under_folder,
        folder,
        model_name=model,
        cpu_model_name=cpu_model,
        device=device,
        batch_size=batch_size,
        diarization=diarization,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        settings=settings,
        model_manager=model_manager,
        asr_method=asr_method,
        qwen_asr_model_dir=qwen_model_dir,
        qwen_asr_num_threads=qwen_threads,
        qwen_asr_vad_segment_threshold=qwen_vad,
    )

whisper_inference = gr.Interface(
    fn=_streamify(_run_transcribe),
    inputs=[
        gr.Textbox(label="目录", value=str(settings.root_folder)),
        gr.Dropdown(_ASR_METHOD_CHOICES, label="语音识别方式", value=settings.asr_method),
        gr.Textbox(label="Qwen3-ASR 模型路径", value=str(settings.qwen_asr_model_path or "")),
        gr.Textbox(label="Whisper 模型路径", value=str(settings.whisper_model_path)),
        gr.Textbox(label="Whisper CPU 模型路径（可选）", value=str(settings.whisper_cpu_model_path or "")),
        gr.Radio(_DEVICE_CHOICES, label="Whisper 运行设备", value=settings.whisper_device),
        gr.Slider(minimum=1, maximum=128, step=1, label="Whisper 批大小", value=settings.whisper_batch_size),
        gr.Slider(minimum=1, maximum=16, step=1, label="Qwen3-ASR 并发线程数", value=settings.qwen_asr_num_threads),
        gr.Slider(minimum=30, maximum=180, step=10, label="Qwen3-ASR VAD分段时长(秒)", value=settings.qwen_asr_vad_segment_threshold),
        gr.Checkbox(label="说话人分离", value=True),
        gr.Number(label="最少说话人数（可选）", value=None, step=1, precision=0),
        gr.Number(label="最多说话人数（可选）", value=None, step=1, precision=0),
    ],
    outputs=gr.Textbox(label="输出", lines=20, max_lines=20, autoscroll=False, elem_classes=["youdub-output"]),
    title="语音识别",
    flagging_mode="never",
    submit_btn="开始识别",
    stop_btn="停止",
    clear_btn="清空",
    **_INTERFACE_STREAM_KWARGS,
)


def run_translation(
    folder,
    target_language,
    translation_strategy,
    translation_max_concurrency,
    translation_chunk_size,
    translation_guide_max_chars,
):
    with _temp_env(
        {
            "TRANSLATION_STRATEGY": translation_strategy,
            "TRANSLATION_MAX_CONCURRENCY": str(int(translation_max_concurrency)),
            "TRANSLATION_CHUNK_SIZE": str(int(translation_chunk_size)),
            "TRANSLATION_GUIDE_MAX_CHARS": str(int(translation_guide_max_chars)),
        }
    ):
        return translate_all_transcript_under_folder(folder, target_language, settings=settings)


translation_interface = gr.Interface(
    fn=_streamify(run_translation),
    inputs=[
        gr.Textbox(label="目录", value=str(settings.root_folder)),
        gr.Dropdown(
            _TARGET_LANGUAGE_CHOICES,
            label="目标语言",
            value=settings.translation_target_language,
        ),
        gr.Dropdown(_TRANSLATION_STRATEGY_CHOICES, label="翻译策略", value="guide_parallel"),
        gr.Slider(minimum=1, maximum=32, step=1, label="翻译最大并发数", value=4),
        gr.Slider(minimum=1, maximum=64, step=1, label="翻译分块大小", value=8),
        gr.Slider(minimum=800, maximum=5000, step=100, label="翻译引导信息最大字符数", value=2500),
    ],
    outputs=gr.Textbox(label="输出", lines=20, max_lines=20, autoscroll=False, elem_classes=["youdub-output"]),
    title="字幕翻译",
    flagging_mode="never",
    submit_btn="开始翻译",
    stop_btn="停止",
    clear_btn="清空",
    **_INTERFACE_STREAM_KWARGS,
)


def _tts_wrapper(folder, tts_method, qwen_tts_batch_size):
    names = (
        [model_manager._bytedance_requirement().name]  # type: ignore[attr-defined]
        if tts_method == "bytedance"
        else [model_manager._gemini_tts_requirement().name]  # type: ignore[attr-defined]
        if tts_method == "gemini"
        else [
            model_manager._qwen_tts_runtime_requirement().name,  # type: ignore[attr-defined]
            model_manager._qwen_tts_weights_requirement().name,  # type: ignore[attr-defined]
        ]
        if tts_method == "qwen"
        else []
    )
    return _safe_run(
        names,
        generate_all_wavs_under_folder,
        folder,
        tts_method=tts_method,
        qwen_tts_batch_size=qwen_tts_batch_size,
    )


tts_interface = gr.Interface(
    fn=_streamify(_tts_wrapper),
    inputs=[
        gr.Textbox(label="目录", value=str(settings.root_folder)),
        gr.Dropdown(
            _TTS_METHOD_CHOICES,
            label="配音方式",
            value=settings.tts_method,
        ),
        gr.Slider(minimum=1, maximum=64, step=1, label="Qwen TTS 批大小", value=settings.qwen_tts_batch_size),
    ],
    outputs=gr.Textbox(label="输出", lines=20, max_lines=20, autoscroll=False, elem_classes=["youdub-output"]),
    title="语音合成",
    flagging_mode="never",
    submit_btn="开始配音",
    stop_btn="停止",
    clear_btn="清空",
    **_INTERFACE_STREAM_KWARGS,
)

def _synthesize_video_wrapper(folder, subtitles, bilingual_subtitle, adaptive_stretch, speed_up, fps, resolution, use_nvenc):
    # 当启用自适应拉伸时，忽略 speed_up，强制设为 1.0
    effective_speed_up = 1.0 if adaptive_stretch else speed_up
    return synthesize_all_video_under_folder(
        folder,
        subtitles=subtitles,
        bilingual_subtitle=bilingual_subtitle,
        speed_up=effective_speed_up,
        fps=fps,
        resolution=resolution,
        use_nvenc=use_nvenc,
    )


synthesize_video_interface = gr.Interface(
    fn=_streamify(_synthesize_video_wrapper),
    inputs=[
        gr.Textbox(label="目录", value=str(settings.root_folder)),
        gr.Checkbox(label="字幕", value=True),
        gr.Checkbox(label="双语字幕", value=False),
        gr.Checkbox(
            label="按段自适应拉伸语音(减少无声)",
            value=False,
            elem_id="adaptive-stretch-checkbox-synthesize",
            info="启用后下方的加速倍率无效",
        ),
        gr.Slider(
            minimum=0.5,
            maximum=2,
            step=0.05,
            label="加速倍率",
            value=DEFAULT_SPEED_UP,
            elem_id="speed-up-slider-synthesize",
        ),
        gr.Slider(minimum=1, maximum=60, step=1, label="帧率（每秒帧数）", value=30),
        gr.Radio(
            _RESOLUTION_CHOICES,
            label="输出分辨率",
            value="1080p",
        ),
        gr.Checkbox(label="使用 NVENC（h264_nvenc）", value=DEFAULT_USE_NVENC, info="需要 NVIDIA GPU"),
    ],
    outputs=gr.Textbox(label="输出", lines=20, max_lines=20, autoscroll=False, elem_classes=["youdub-output"]),
    title="视频合成",
    flagging_mode="never",
    submit_btn="开始合成",
    stop_btn="停止",
    clear_btn="清空",
    **_INTERFACE_STREAM_KWARGS,
)

generate_info_interface = gr.Interface(
    fn=generate_all_info_under_folder_stream,
    inputs=[
        gr.Textbox(label="目录", value=str(settings.root_folder)),
    ],
    outputs=gr.Textbox(label="输出", lines=20, max_lines=20, autoscroll=False, elem_classes=["youdub-output"]),
    title="生成信息",
    flagging_mode="never",
    submit_btn="开始生成",
    stop_btn="停止",
    clear_btn="清空",
    **_INTERFACE_STREAM_KWARGS,
)

upload_bilibili_interface = gr.Interface(
    fn=_streamify(upload_all_videos_under_folder),
    inputs=[
        gr.Textbox(label="目录", value=str(settings.root_folder)),
    ],
    outputs=gr.Textbox(label="输出", lines=20, max_lines=20, autoscroll=False, elem_classes=["youdub-output"]),
    title="上传 B 站",
    flagging_mode="never",
    submit_btn="开始上传",
    stop_btn="停止",
    clear_btn="清空",
    **_INTERFACE_STREAM_KWARGS,
)

with gr.Blocks(title="模型检查") as model_status_interface:
    gr.Markdown("## 模型检查")
    gr.Markdown("当前需要的语音识别/语音合成模型状态（仅本地，不会自动下载）。")

    output_box = gr.Textbox(label="输出", lines=20, max_lines=20, autoscroll=False, elem_classes=["youdub-output"])

    with gr.Row():
        refresh_btn = gr.Button("刷新状态", variant="secondary")

    gr.Markdown("---")
    gr.Markdown("### 模型下载")
    gr.Markdown("下载 Demucs、Whisper 和 Pyannote 说话人分离模型到本地。Pyannote 需要 HF_TOKEN（需先在 HuggingFace 同意协议）。")

    hf_token_input = gr.Textbox(
        label="HF_TOKEN（Hugging Face Token）",
        placeholder="hf_xxx...",
        type="password",
        value=settings.hf_token or "",
    )
    download_btn = gr.Button("下载模型", variant="primary")

    refresh_btn.click(
        fn=_streamify(show_model_status),
        inputs=[],
        outputs=output_box,
        **_INTERFACE_STREAM_KWARGS,
    )

    download_btn.click(
        fn=_streamify(download_models),
        inputs=[hf_token_input],
        outputs=output_box,
        **_INTERFACE_STREAM_KWARGS,
    )

app = gr.TabbedInterface(
    interface_list=[
        model_status_interface,
        do_everything_interface,
        youtube_interface,
        demucs_interface,
        whisper_inference,
        translation_interface,
        tts_interface,
        synthesize_video_interface,
        generate_info_interface,
        upload_bilibili_interface,
    ],
    tab_names=['模型检查', '全自动', '下载视频', '人声分离', '语音识别', '字幕翻译', '语音合成', '视频合成', '生成信息', '上传B站'],
    title='YouDub｜视频翻译与配音',
)

try:
    # Enable Gradio queue so generator outputs can stream progressively.
    if hasattr(app, "queue"):
        app = app.queue()
except Exception:
    pass


def main():
    import inspect

    install_signal_handlers()

    launch_kwargs: dict[str, object] = {}
    try:
        if "prevent_thread_lock" in inspect.signature(app.launch).parameters:
            launch_kwargs["prevent_thread_lock"] = True
    except Exception:
        # Best-effort: fall back to default launch behavior.
        pass
    # Auto-follow Output scroll (tail -f style).
    launch_kwargs.setdefault("js", _AUTO_SCROLL_JS)
    # Ensure output textbox is readable across browsers/webviews.
    launch_kwargs.setdefault("css", _OUTPUT_CSS)

    try:
        app.launch(**launch_kwargs)
        if launch_kwargs.get("prevent_thread_lock"):
            while not cancel_requested():
                time.sleep(0.2)
    except KeyboardInterrupt:
        request_cancel("SIGINT")
        logger.info("收到 Ctrl+C，正在退出…")
    finally:
        # Avoid double Ctrl+C breaking shutdown and causing hard aborts.
        from contextlib import suppress

        with ignore_signals_during_shutdown():
            with suppress(Exception, KeyboardInterrupt):
                if hasattr(app, "close"):
                    app.close()


if __name__ == "__main__":
    main()
