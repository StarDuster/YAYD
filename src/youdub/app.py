import os
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
from youdub.core.pipeline import VideoPipeline
from youdub.core.interrupts import cancel_requested, ignore_signals_during_shutdown, install_signal_handlers, request_cancel
from youdub.core.steps import (
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


class _QueueWriter:
    """File-like writer which forwards lines to a queue."""

    def __init__(self, q: "queue.Queue[str]", prefix: str = "", mirror: Any | None = None) -> None:
        self._q = q
        self._prefix = prefix
        self._mirror = mirror
        self._buf = ""

    def write(self, s: str) -> int:  # file-like
        if not s:
            return 0
        # Mirror raw bytes to original stream (keep terminal logs).
        if self._mirror is not None:
            try:
                self._mirror.write(s)
            except Exception:
                pass

        # tqdm/progress bars often use '\r' updates; translate to '\n' for UI friendliness.
        s_ui = s.replace("\r", "\n")
        self._buf += s_ui
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip()
            if line:
                self._q.put(f"{self._prefix}{line}")
        return len(s)

    def flush(self) -> None:  # file-like
        if self._mirror is not None:
            try:
                self._mirror.flush()
            except Exception:
                pass
        line = self._buf.strip()
        if line:
            self._q.put(f"{self._prefix}{line}")
        self._buf = ""


@contextmanager
def _capture_output_to_queue(q: "queue.Queue[str]"):
    """Temporarily mirror loguru + stdout/stderr into a queue (per Gradio run)."""
    old_stdout = sys.stdout

    def _sink(message: Any) -> None:
        try:
            text = str(message).rstrip("\n")
        except Exception:
            return
        if text:
            q.put(text)

    sink_id: int | None
    try:
        # Match loguru's common terminal style (module:function:line) for readability.
        sink_id = logger.add(
            _sink,
            level="INFO",
            enqueue=True,
            colorize=False,
            backtrace=False,
            diagnose=False,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        )
    except Exception:
        sink_id = None

    try:
        sys.stdout = _QueueWriter(q, mirror=old_stdout)  # type: ignore[assignment]
        yield
    finally:
        sys.stdout = old_stdout
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

    def _append_text(text: str) -> None:
        for part in str(text).replace("\r", "\n").split("\n"):
            part = part.rstrip()
            if part:
                lines.append(part)
        if len(lines) > max_lines:
            # Drop oldest lines to keep UI payload bounded.
            del lines[: max(50, max_lines // 5)]

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
            yield "\n".join(lines)

    t.join(timeout=0)

    if "exc" in exc_box:
        exc = exc_box["exc"]
        _append_text(f"异常: {type(exc).__name__}: {exc}")
        yield "\n".join(lines)
        return

    # Always show final result (if any).
    result = result_box.get("result")
    if result is not None:
        if isinstance(result, str):
            _append_text(result)
        else:
            _append_text(str(result))

    # Flush one last time (may include final result line).
    yield "\n".join(lines)


def _streamify(fn: Callable[..., Any]) -> Callable[..., Iterator[str]]:
    """Wrap a non-streaming function so its logs show up in Gradio Output."""

    def _wrapped(*args: Any, **kwargs: Any) -> Iterator[str]:
        return _stream_run(lambda: fn(*args, **kwargs))

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
    whisper_model,
    whisper_device,
    whisper_cpu_model,
    whisper_batch_size,
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
    tts_adaptive_segment_stretch,
    subtitles,
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
                whisper_model=whisper_model,
                whisper_device=whisper_device,
                whisper_cpu_model=whisper_cpu_model,
                whisper_batch_size=whisper_batch_size,
                whisper_diarization=whisper_diarization,
                whisper_min_speakers=whisper_min_speakers,
                whisper_max_speakers=whisper_max_speakers,
                translation_target_language=translation_target_language,
                tts_method=tts_method,
                qwen_tts_batch_size=qwen_tts_batch_size,
                tts_adaptive_segment_stretch=bool(tts_adaptive_segment_stretch),
                subtitles=subtitles,
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


do_everything_interface = gr.Interface(
    fn=_streamify(run_pipeline),
    inputs=[
        gr.Textbox(label="Root Folder", value=str(settings.root_folder)),
        gr.Textbox(
            label="Video URL",
            placeholder="Video or Playlist or Channel URL",
            value="https://www.youtube.com/watch?v=4_SH2nfbQZ8",
        ),
        gr.Slider(minimum=1, maximum=500, step=1, label="Number of videos to download", value=20),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Resolution",
            value="1080p",
        ),
        gr.Radio(
            ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q", "SIG"],
            label="Demucs Model",
            value=settings.demucs_model_name,
        ),
        gr.Radio(["auto", "cuda", "cpu"], label="Demucs Device", value=settings.demucs_device),
        gr.Slider(minimum=0, maximum=10, step=1, label="Number of shifts", value=settings.demucs_shifts),
        gr.Textbox(label="Whisper Model", value=str(settings.whisper_model_path)),
        gr.Radio(["auto", "cuda", "cpu"], label="Whisper Device", value=settings.whisper_device),
        gr.Textbox(label="Whisper CPU Model Path", value=str(settings.whisper_cpu_model_path or "")),
        gr.Slider(minimum=1, maximum=128, step=1, label="Whisper Batch Size", value=settings.whisper_batch_size),
        gr.Checkbox(label="Whisper Diarization", value=True),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Whisper Min Speakers", value=None),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Whisper Max Speakers", value=None),
        gr.Dropdown(
            ["简体中文", "繁体中文", "English", "Deutsch", "Français", "русский"],
            label="Translation Target Language",
            value=settings.translation_target_language,
        ),
        gr.Dropdown(
            ["history", "guide_parallel"],
            label="Translation Strategy",
            value="guide_parallel",
        ),
        gr.Slider(minimum=1, maximum=32, step=1, label="Translation Max Concurrency", value=4),
        gr.Slider(minimum=1, maximum=64, step=1, label="Translation Chunk Size", value=8),
        gr.Slider(minimum=800, maximum=5000, step=100, label="Translation Guide Max Chars", value=2500),
        gr.Dropdown(
            ["bytedance", "qwen", "gemini"],
            label="TTS Method",
            value=settings.tts_method
        ),
        gr.Slider(minimum=1, maximum=64, step=1, label="Qwen TTS Batch Size", value=settings.qwen_tts_batch_size),
        gr.Checkbox(label="按段自适应拉伸语音(减少无声)", value=False),
        gr.Checkbox(label="Subtitles", value=True),
        gr.Slider(minimum=0.5, maximum=2, step=0.05, label="Speed Up", value=DEFAULT_SPEED_UP),
        gr.Slider(minimum=1, maximum=60, step=1, label="FPS", value=30),
        gr.Checkbox(label="Use NVENC (h264_nvenc)", value=DEFAULT_USE_NVENC),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Resolution",
            value="1080p",
        ),
        gr.Slider(minimum=1, maximum=100, step=1, label="Max Workers", value=1),
        gr.Slider(minimum=1, maximum=10, step=1, label="Max Retries", value=3),
        gr.Checkbox(label="Auto Upload Video", value=False),
    ],
    outputs=gr.Textbox(label="Output", lines=20, max_lines=200, autoscroll=True),
)

youtube_interface = gr.Interface(
    fn=_streamify(download_from_url),
    inputs=[
        gr.Textbox(
            label="Video URL",
            placeholder="Video or Playlist or Channel URL",
            value="https://www.bilibili.com/list/1263732318",
        ),
        gr.Textbox(label="Output Folder", value=str(settings.root_folder)),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Resolution",
            value="1080p",
        ),
        gr.Slider(minimum=1, maximum=100, step=1, label="Number of videos to download", value=5),
    ],
    outputs=gr.Textbox(label="Output", lines=20, max_lines=200, autoscroll=True),
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
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
        gr.Radio(
            ["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q", "SIG"],
            label="Model",
            value=settings.demucs_model_name,
        ),
        gr.Radio(["auto", "cuda", "cpu"], label="Device", value=settings.demucs_device),
        gr.Checkbox(label="Progress Bar in Console", value=True),
        gr.Slider(minimum=0, maximum=10, step=1, label="Number of shifts", value=settings.demucs_shifts),
    ],
    outputs=gr.Textbox(label="Output", lines=20, max_lines=200, autoscroll=True),
)

whisper_inference = gr.Interface(
    fn=_streamify(lambda folder, model, cpu_model, device, batch_size, diarization, min_speakers, max_speakers: _safe_run(
        (
            [model_manager._whisper_diarization_requirement().name]  # type: ignore[attr-defined]
            if diarization
            else []
        ),
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
    )),
    inputs=[
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
        gr.Textbox(label="Model", value=str(settings.whisper_model_path)),
        gr.Textbox(label="CPU Model (optional)", value=str(settings.whisper_cpu_model_path or "")),
        gr.Radio(["auto", "cuda", "cpu"], label="Whisper Device", value=settings.whisper_device),
        gr.Slider(minimum=1, maximum=128, step=1, label="Batch Size", value=settings.whisper_batch_size),
        gr.Checkbox(label="Diarization", value=True),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Whisper Min Speakers", value=None),
        gr.Radio([None, 1, 2, 3, 4, 5, 6, 7, 8, 9], label="Whisper Max Speakers", value=None),
    ],
    outputs=gr.Textbox(label="Output", lines=20, max_lines=200, autoscroll=True),
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
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
        gr.Dropdown(
            ["简体中文", "繁体中文", "English", "Deutsch", "Français", "русский"],
            label="Target Language",
            value=settings.translation_target_language,
        ),
        gr.Dropdown(["history", "guide_parallel"], label="Translation Strategy", value="guide_parallel"),
        gr.Slider(minimum=1, maximum=32, step=1, label="Translation Max Concurrency", value=4),
        gr.Slider(minimum=1, maximum=64, step=1, label="Translation Chunk Size", value=8),
        gr.Slider(minimum=800, maximum=5000, step=100, label="Translation Guide Max Chars", value=2500),
    ],
    outputs=gr.Textbox(label="Output", lines=20, max_lines=200, autoscroll=True),
)


def _tts_wrapper(folder, tts_method, qwen_tts_batch_size, tts_adaptive_segment_stretch):
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
        adaptive_segment_stretch=bool(tts_adaptive_segment_stretch),
    )


tts_interface = gr.Interface(
    fn=_streamify(_tts_wrapper),
    inputs=[
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
        gr.Dropdown(
            ["bytedance", "qwen", "gemini"],
            label="TTS Method",
            value=settings.tts_method
        ),
        gr.Slider(minimum=1, maximum=64, step=1, label="Qwen TTS Batch Size", value=settings.qwen_tts_batch_size),
        gr.Checkbox(label="按段自适应拉伸语音(减少无声)", value=False),
    ],
    outputs=gr.Textbox(label="Output", lines=20, max_lines=200, autoscroll=True),
)

syntehsize_video_interface = gr.Interface(
    fn=_streamify(synthesize_all_video_under_folder),
    inputs=[
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
        gr.Checkbox(label="Subtitles", value=True),
        gr.Slider(minimum=0.5, maximum=2, step=0.05, label="Speed Up", value=DEFAULT_SPEED_UP),
        gr.Slider(minimum=1, maximum=60, step=1, label="FPS", value=30),
        gr.Radio(
            ["4320p", "2160p", "1440p", "1080p", "720p", "480p", "360p", "240p", "144p"],
            label="Resolution",
            value="1080p",
        ),
        gr.Checkbox(label="Use NVENC (h264_nvenc)", value=DEFAULT_USE_NVENC),
    ],
    outputs=gr.Textbox(label="Output", lines=20, max_lines=200, autoscroll=True),
)

genearte_info_interface = gr.Interface(
    fn=generate_all_info_under_folder_stream,
    inputs=[
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
    ],
    outputs=gr.Textbox(label="Output", lines=20, max_lines=200, autoscroll=True),
)

upload_bilibili_interface = gr.Interface(
    fn=_streamify(upload_all_videos_under_folder),
    inputs=[
        gr.Textbox(label="Folder", value=str(settings.root_folder)),
    ],
    outputs=gr.Textbox(label="Output", lines=20, max_lines=200, autoscroll=True),
)

model_status_interface = gr.Interface(
    fn=_streamify(show_model_status),
    inputs=[],
    outputs=gr.Textbox(label="Output", lines=20, max_lines=200, autoscroll=True),
    description="当前需要的 ASR/TTS 模型状态（仅本地，不会自动下载）。",
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
        syntehsize_video_interface,
        genearte_info_interface,
        upload_bilibili_interface,
    ],
    tab_names=['模型检查', '全自动', '下载视频', '人声分离', '语音识别', '字幕翻译', '语音合成', '视频合成', '生成信息', '上传B站'],
    title='YouDub',
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
