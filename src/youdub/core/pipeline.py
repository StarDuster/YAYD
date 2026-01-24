from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

from loguru import logger

from ..config import Settings
from ..models import ModelCheckError, ModelManager
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
        subtitles: bool,
        speed_up: float,
        fps: int,
        target_resolution: str,
        max_retries: int,
        auto_upload_video: bool,
    ) -> bool:
        for retry in range(max_retries):
            try:
                folder = download.get_target_folder(info, root_folder)
                if folder is None:
                    logger.warning(f"Failed to get target folder for video {info.get('title')}")
                    return False

                if self._already_uploaded(folder):
                    logger.info(f"Video already uploaded in {folder}")
                    return True

                folder = download.download_single_video(info, root_folder, resolution)
                if folder is None:
                    logger.warning(f"Failed to download video {info.get('title')}")
                    return True

                logger.info(f"Process video in {folder}")

                separate_vocals.separate_all_audio_under_folder(
                    folder,
                    model_name=demucs_model,
                    device=device,
                    progress=True,
                    shifts=shifts,
                    settings=self.settings,
                    model_manager=self.model_manager,
                )
                separate_vocals.unload_model()

                transcribe.transcribe_all_audio_under_folder(
                    folder,
                    model_name=whisper_model,
                    device=device,
                    batch_size=whisper_batch_size,
                    diarization=whisper_diarization,
                    min_speakers=whisper_min_speakers,
                    max_speakers=whisper_max_speakers,
                    settings=self.settings,
                    model_manager=self.model_manager,
                )
                transcribe.unload_all_models()

                translate.translate_all_transcript_under_folder(
                    folder, target_language=translation_target_language, settings=self.settings
                )
                synthesize_speech.generate_all_wavs_under_folder(
                    folder, 
                    tts_method=tts_method,
                )
                synthesize_all_video_under_folder(
                    folder, subtitles=subtitles, speed_up=speed_up, fps=fps, resolution=target_resolution
                )
                # Info + upload are optional/non-model heavy
                generate_all_info_under_folder(folder)
                if auto_upload_video:
                    time.sleep(1)
                    upload_all_videos_under_folder(folder)
                return True
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(f"Error processing video {info.get('title')}: {exc}")
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
        whisper_batch_size: int | None = None,
        whisper_diarization: bool = True,
        whisper_min_speakers: int | None = None,
        whisper_max_speakers: int | None = None,
        translation_target_language: str | None = None,
        tts_method: str | None = None,
        subtitles: bool = True,
        speed_up: float = 1.05,
        fps: int = 30,
        target_resolution: str = "1080p",
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
        whisper_batch_size = whisper_batch_size or self.settings.whisper_batch_size
        translation_target_language = translation_target_language or self.settings.translation_target_language
        tts_method = tts_method or self.settings.tts_method
        
        required_models = [
            self.model_manager._demucs_requirement().name,  # type: ignore[attr-defined]
            self.model_manager._whisper_requirement().name,  # type: ignore[attr-defined]
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

        # Warm up models asynchronously
        with ThreadPoolExecutor() as executor:
            executor.submit(separate_vocals.init_demucs, self.settings, self.model_manager)
            executor.submit(synthesize_speech.init_TTS, self.settings, self.model_manager)
            executor.submit(transcribe.init_asr, self.settings, self.model_manager)

        success_list: list[dict[str, Any]] = []
        fail_list: list[dict[str, Any]] = []

        info_list = list(download.get_info_list_from_url(urls, num_videos))

        if max_workers <= 1:
            for info in info_list:
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
                    subtitles,
                    speed_up,
                    fps,
                    target_resolution,
                    max_retries,
                    auto_upload_video,
                )
                if success:
                    success_list.append(info)
                else:
                    fail_list.append(info)
        else:
            logger.info(f"Processing {len(info_list)} videos with max_workers={max_workers}")
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
                        subtitles,
                        speed_up,
                        fps,
                        target_resolution,
                        max_retries,
                        auto_upload_video,
                    ): info
                    for info in info_list
                }
                for future in as_completed(future_to_info):
                    info = future_to_info[future]
                    try:
                        success = future.result()
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.error(f"Unhandled exception processing {info.get('title')}: {exc}")
                        success = False
                    if success:
                        success_list.append(info)
                    else:
                        fail_list.append(info)

        return f"Success: {len(success_list)}\nFail: {len(fail_list)}"
