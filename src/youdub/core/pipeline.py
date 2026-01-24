from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from ..config import Settings
from ..models import ModelCheckError, ModelManager
from .interrupts import check_cancelled, sleep_with_cancel
from .steps import (
    download,
    generate_all_info_under_folder,
    separate_vocals,
    synthesize_all_video_under_folder,
    synthesize_speech,
    transcribe,
    translate,
    upload_all_videos_under_folder,
)


class VideoPipeline:
    """Coordinate the full video -> dub -> upload flow."""

    def __init__(self, settings: Settings | None = None, model_manager: ModelManager | None = None):
        self.settings = settings or Settings()
        self.model_manager = model_manager or ModelManager(self.settings)

    def _ensure_models(self, names: Iterable[str] | None = None) -> None:
        self.model_manager.enforce_offline()
        self.model_manager.ensure_ready(names=names)

    def _already_uploaded(self, folder: str) -> bool:
        bilibili_json = os.path.join(folder, "bilibili.json")
        if os.path.exists(bilibili_json):
            with open(bilibili_json, "r", encoding="utf-8") as f:
                info = json.load(f)
            return info["results"][0]["code"] == 0
        return False

    def process_single(
        self,
        info: dict[str, Any],
        root_folder: str,
        resolution: str,
        demucs_model: str,
        device: str,
        shifts: int,
        whisper_model: str,
        whisper_batch_size: int,
        whisper_diarization: bool,
        whisper_min_speakers: int | None,
        whisper_max_speakers: int | None,
        translation_target_language: str,
        tts_method: str,
        qwen_tts_batch_size: int,
        tts_adaptive_segment_stretch: bool,
        subtitles: bool,
        speed_up: float,
        fps: int,
        target_resolution: str,
        max_retries: int,
        auto_upload_video: bool,
        use_nvenc: bool = False,
        whisper_device: str | None = None,
        whisper_cpu_model: str | None = None,
    ) -> bool:
        for retry in range(max_retries):
            check_cancelled()
            try:
                def _require_file(path: str, desc: str, min_bytes: int = 1) -> None:
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"缺少{desc}: {path}")
                    try:
                        if os.path.getsize(path) < min_bytes:
                            raise FileNotFoundError(f"{desc}文件过小/疑似损坏: {path}")
                    except OSError:
                        raise FileNotFoundError(f"无法读取{desc}: {path}") from None

                check_cancelled()
                folder = download.get_target_folder(info, root_folder)
                if folder is None:
                    logger.warning(f"获取视频目录失败: {info.get('title')}")
                    return False

                if self._already_uploaded(folder):
                    logger.info(f"已上传: {folder}")
                    return True

                check_cancelled()
                folder = download.download_single_video(info, root_folder, resolution, settings=self.settings)
                if folder is None:
                    logger.warning(f"下载失败: {info.get('title')}")
                    return False

                logger.info(f"开始处理: {folder}")

                _require_file(os.path.join(folder, "download.mp4"), "下载视频(download.mp4)", min_bytes=1024)

                check_cancelled()
                try:
                    separate_vocals.separate_all_audio_under_folder(
                        folder,
                        model_name=demucs_model,
                        device=device,
                        progress=True,
                        shifts=shifts,
                        settings=self.settings,
                        model_manager=self.model_manager,
                    )
                finally:
                    # Best-effort: reduce GPU memory leaks when interrupted.
                    try:
                        separate_vocals.unload_model()
                    except Exception:
                        pass

                _require_file(os.path.join(folder, "audio_vocals.wav"), "人声轨(audio_vocals.wav)", min_bytes=44)
                _require_file(os.path.join(folder, "audio_instruments.wav"), "伴奏轨(audio_instruments.wav)", min_bytes=44)

                asr_device = whisper_device or device
                check_cancelled()
                try:
                    transcribe.transcribe_all_audio_under_folder(
                        folder,
                        model_name=whisper_model,
                        cpu_model_name=whisper_cpu_model,
                        device=asr_device,
                        batch_size=whisper_batch_size,
                        diarization=whisper_diarization,
                        min_speakers=whisper_min_speakers,
                        max_speakers=whisper_max_speakers,
                        settings=self.settings,
                        model_manager=self.model_manager,
                    )
                finally:
                    try:
                        transcribe.unload_all_models()
                    except Exception:
                        pass

                _require_file(os.path.join(folder, "transcript.json"), "转写结果(transcript.json)", min_bytes=2)

                check_cancelled()
                translate.translate_all_transcript_under_folder(
                    folder, target_language=translation_target_language, settings=self.settings
                )

                _require_file(os.path.join(folder, "translation.json"), "翻译结果(translation.json)", min_bytes=2)

                check_cancelled()
                synthesize_speech.generate_all_wavs_under_folder(
                    folder, 
                    tts_method=tts_method,
                    qwen_tts_batch_size=qwen_tts_batch_size,
                    adaptive_segment_stretch=tts_adaptive_segment_stretch,
                )

                _require_file(os.path.join(folder, "audio_combined.wav"), "配音合成(audio_combined.wav)", min_bytes=44)

                check_cancelled()
                synthesize_all_video_under_folder(
                    folder,
                    subtitles=subtitles,
                    speed_up=speed_up,
                    fps=fps,
                    resolution=target_resolution,
                    use_nvenc=use_nvenc,
                )

                _require_file(os.path.join(folder, "video.mp4"), "最终视频(video.mp4)", min_bytes=1024)

                check_cancelled()
                generate_all_info_under_folder(folder)
                if auto_upload_video:
                    sleep_with_cancel(1)
                    check_cancelled()
                    upload_all_videos_under_folder(folder)
                    if not self._already_uploaded(folder):
                        raise RuntimeError(f"自动上传失败: {folder}")
                return True
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(f"处理失败: {info.get('title')} ({exc})")
        return False

    def run(
        self,
        root_folder: str | None = None,
        url: str = "",
        num_videos: int = 5,
        resolution: str = "1080p",
        demucs_model: str | None = None,
        device: str | None = None,
        shifts: int | None = None,
        whisper_model: str | None = None,
        whisper_cpu_model: str | None = None,
        whisper_device: str | None = None,
        whisper_batch_size: int | None = None,
        whisper_diarization: bool = True,
        whisper_min_speakers: int | None = None,
        whisper_max_speakers: int | None = None,
        translation_target_language: str | None = None,
        tts_method: str | None = None,
        qwen_tts_batch_size: int | None = None,
        tts_adaptive_segment_stretch: bool = False,
        subtitles: bool = True,
        speed_up: float = 1.2,
        fps: int = 30,
        target_resolution: str = "1080p",
        use_nvenc: bool = False,
        max_workers: int = 1,
        max_retries: int = 3,
        auto_upload_video: bool = True,
    ) -> str:
        """Execute the full pipeline."""
        root_folder = str(root_folder or self.settings.root_folder)
        demucs_model = demucs_model or self.settings.demucs_model_name
        device = device or self.settings.demucs_device
        shifts = self.settings.demucs_shifts if shifts is None else shifts
        whisper_model = whisper_model or str(self.settings.whisper_model_path)
        # Backward compatible defaults: ASR device defaults to Demucs device, unless explicitly set.
        whisper_device = (whisper_device or getattr(self.settings, "whisper_device", None) or device).strip()
        whisper_device = whisper_device or device
        whisper_cpu_model = whisper_cpu_model or (str(getattr(self.settings, "whisper_cpu_model_path", "") or "") or None)
        if whisper_cpu_model is not None:
            whisper_cpu_model = str(whisper_cpu_model).strip() or None
        whisper_batch_size = whisper_batch_size or self.settings.whisper_batch_size
        translation_target_language = translation_target_language or self.settings.translation_target_language
        tts_method = tts_method or self.settings.tts_method
        qwen_tts_batch_size = qwen_tts_batch_size or getattr(self.settings, "qwen_tts_batch_size", 1)

        def _has_whisper_model_bin(model_dir: str | None) -> bool:
            if not model_dir:
                return False
            try:
                path = Path(str(model_dir)).expanduser()
            except Exception:
                return False
            return path.is_dir() and (path / "model.bin").exists()

        wd = (whisper_device or "auto").lower().strip()
        if wd not in ("auto", "cuda", "cpu"):
            wd = "auto"

        # Validate ASR model path(s) early so UI can show friendly error instead of silent failures.
        if wd == "cuda":
            if not _has_whisper_model_bin(whisper_model):
                raise ModelCheckError(
                    f"Whisper GPU 模型目录无效或缺少 model.bin：{whisper_model}\n"
                    "请在 UI 中填写正确路径，或在 .env 中设置 WHISPER_MODEL_PATH。"
                )
        elif wd == "cpu":
            chosen = whisper_cpu_model or whisper_model
            if not _has_whisper_model_bin(chosen):
                raise ModelCheckError(
                    f"Whisper CPU 模型目录无效或缺少 model.bin：{chosen}\n"
                    "请在 UI 中填写正确路径，或在 .env 中设置 WHISPER_CPU_MODEL_PATH / WHISPER_MODEL_PATH。"
                )
        else:
            if not (_has_whisper_model_bin(whisper_model) or _has_whisper_model_bin(whisper_cpu_model)):
                raise ModelCheckError(
                    "Whisper 模型目录无效或缺少 model.bin。\n"
                    f"- WHISPER_MODEL_PATH: {whisper_model}\n"
                    f"- WHISPER_CPU_MODEL_PATH: {whisper_cpu_model}\n"
                    "请在 UI 中填写正确路径，或在 .env 中设置 WHISPER_MODEL_PATH / WHISPER_CPU_MODEL_PATH。"
                )
        
        required_models = [
            self.model_manager._demucs_requirement().name,  # type: ignore[attr-defined]
        ]
        if whisper_diarization:
            required_models.append(self.model_manager._whisper_diarization_requirement().name)  # type: ignore[attr-defined]
            
        if tts_method == "gemini":
            required_models.append(self.model_manager._gemini_tts_requirement().name)  # type: ignore[attr-defined]
        elif tts_method == "qwen":
            required_models.append(self.model_manager._qwen_tts_runtime_requirement().name)  # type: ignore[attr-defined]
            required_models.append(self.model_manager._qwen_tts_weights_requirement().name)  # type: ignore[attr-defined]
        else:
            required_models.append(self.model_manager._bytedance_requirement().name)  # type: ignore[attr-defined]
        self._ensure_models(required_models)

        url = url.replace(" ", "").replace("，", "\n").replace(",", "\n")
        urls = [_ for _ in url.split("\n") if _]

        # Warm-up models best-effort. Keep it sequential so Ctrl+C can stop cleanly.
        check_cancelled()
        try:
            separate_vocals.init_demucs(self.settings, self.model_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"Demucs 预热失败（忽略）: {exc}")
        check_cancelled()
        try:
            synthesize_speech.init_TTS(self.settings, self.model_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"TTS 预热失败（忽略）: {exc}")
        check_cancelled()
        try:
            # Preload ASR model best-effort to reduce first-request latency.
            # Respect explicit Whisper device selection from UI; for "auto" keep legacy behavior.
            if wd == "cpu":
                transcribe.load_asr_model(
                    whisper_cpu_model or whisper_model,
                    device="cpu",
                    settings=self.settings,
                    model_manager=self.model_manager,
                )
            elif wd == "cuda":
                transcribe.load_asr_model(
                    whisper_model,
                    device="cuda",
                    settings=self.settings,
                    model_manager=self.model_manager,
                )
            else:
                transcribe.init_asr(self.settings, self.model_manager)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"ASR 预热失败（忽略）: {exc}")

        success_list: list[dict[str, Any]] = []
        fail_list: list[dict[str, Any]] = []

        info_list = list(download.get_info_list_from_url(urls, num_videos, settings=self.settings))

        if max_workers <= 1:
            for info in info_list:
                check_cancelled()
                success = self.process_single(
                    info,
                    root_folder,
                    resolution,
                    demucs_model,
                    device,
                    shifts,
                    whisper_model,
                    whisper_batch_size,
                    whisper_diarization,
                    whisper_min_speakers,
                    whisper_max_speakers,
                    translation_target_language,
                    tts_method,
                    int(qwen_tts_batch_size),
                    bool(tts_adaptive_segment_stretch),
                    subtitles,
                    speed_up,
                    fps,
                    target_resolution,
                    max_retries,
                    auto_upload_video,
                    use_nvenc,
                    wd,
                    whisper_cpu_model,
                )
                if success:
                    success_list.append(info)
                else:
                    fail_list.append(info)
        else:
            logger.info(f"并发处理 {len(info_list)} 个视频: max_workers={max_workers}")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_info = {
                    executor.submit(
                        self.process_single,
                        info,
                        root_folder,
                        resolution,
                        demucs_model,
                        device,
                        shifts,
                        whisper_model,
                        whisper_batch_size,
                        whisper_diarization,
                        whisper_min_speakers,
                        whisper_max_speakers,
                        translation_target_language,
                        tts_method,
                        int(qwen_tts_batch_size),
                        bool(tts_adaptive_segment_stretch),
                        subtitles,
                        speed_up,
                        fps,
                        target_resolution,
                        max_retries,
                        auto_upload_video,
                        use_nvenc,
                        wd,
                        whisper_cpu_model,
                    ): info
                    for info in info_list
                }
                for future in as_completed(future_to_info):
                    check_cancelled()
                    info = future_to_info[future]
                    try:
                        success = future.result()
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.error(f"未处理异常: {info.get('title')} ({exc})")
                        success = False
                    if success:
                        success_list.append(info)
                    else:
                        fail_list.append(info)

        return f"成功: {len(success_list)}\n失败: {len(fail_list)}"
