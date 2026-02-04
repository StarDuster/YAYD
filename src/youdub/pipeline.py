from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import torch
from loguru import logger

from .config import Settings
from .models import ModelCheckError, ModelManager
from .interrupts import check_cancelled
from .utils import require_file, valid_file
from .steps import (
    download,
    generate_all_info_under_folder,
    separate_vocals,
    synthesize_all_video_under_folder,
    synthesize_speech,
    transcribe,
    translate,
    upload_video_async,
)
from .steps.separate_vocals import CorruptedVideoError


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
        fps: int,
        target_resolution: str,
        max_retries: int,
        auto_upload_video: bool,
        bilingual_subtitle: bool = False,
        use_nvenc: bool = False,
        whisper_device: str | None = None,
        whisper_cpu_model: str | None = None,
        *,
        asr_method: str = "whisper",
        qwen_asr_model_dir: str | None = None,
        qwen_asr_num_threads: int = 1,
        qwen_asr_vad_segment_threshold: int = 60,
    ) -> bool:
        for retry in range(max_retries):
            check_cancelled()
            try:
                check_cancelled()
                folder = download.download_single_video(info, root_folder, resolution, settings=self.settings)
                if folder is None:
                    logger.warning(f"下载失败: {info.get('title')}")
                    return False

                logger.info(f"开始处理: {folder}")

                require_file(os.path.join(folder, "download.mp4"), "下载视频(download.mp4)", min_bytes=1024)

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

                require_file(os.path.join(folder, "audio_vocals.wav"), "人声轨(audio_vocals.wav)", min_bytes=44)
                require_file(os.path.join(folder, "audio_instruments.wav"), "伴奏轨(audio_instruments.wav)", min_bytes=44)

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
                        asr_method=asr_method,
                        qwen_asr_model_dir=qwen_asr_model_dir,
                        qwen_asr_num_threads=qwen_asr_num_threads,
                        qwen_asr_vad_segment_threshold=qwen_asr_vad_segment_threshold,
                    )
                finally:
                    try:
                        transcribe.unload_all_models()
                    except Exception:
                        pass

                require_file(os.path.join(folder, "transcript.json"), "转写结果(transcript.json)", min_bytes=2)

                check_cancelled()
                translate.translate_all_transcript_under_folder(
                    folder, target_language=translation_target_language, settings=self.settings
                )

                require_file(os.path.join(folder, "translation.json"), "翻译结果(translation.json)", min_bytes=2)

                check_cancelled()
                synthesize_speech.generate_all_wavs_under_folder(
                    folder, 
                    tts_method=tts_method,
                    qwen_tts_batch_size=qwen_tts_batch_size,
                )

                require_file(os.path.join(folder, "wavs", ".tts_done.json"), "语音合成标记(wavs/.tts_done.json)", min_bytes=2)
                require_file(os.path.join(folder, "wavs", "0000.wav"), "TTS分段音频(wavs/0000.wav)", min_bytes=44)

                check_cancelled()
                synthesize_all_video_under_folder(
                    folder,
                    subtitles=subtitles,
                    bilingual_subtitle=bilingual_subtitle,
                    adaptive_segment_stretch=bool(tts_adaptive_segment_stretch),
                    fps=fps,
                    resolution=target_resolution,
                    use_nvenc=use_nvenc,
                )

                require_file(os.path.join(folder, "video.mp4"), "最终视频(video.mp4)", min_bytes=1024)

                check_cancelled()
                generate_all_info_under_folder(folder)
                if auto_upload_video:
                    # Run B站上传 in background so it won't block video processing.
                    if self._already_uploaded(folder):
                        logger.info(f"检测到已上传，跳过自动上传: {folder}")
                    else:
                        upload_video_async(folder)
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
        asr_method: str | None = None,
        qwen_asr_model_dir: str | None = None,
        qwen_asr_num_threads: int | None = None,
        qwen_asr_vad_segment_threshold: int | None = None,
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
        bilingual_subtitle: bool = False,
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
        asr_method = (asr_method or getattr(self.settings, "asr_method", None) or "whisper").strip().lower()
        if asr_method not in ("whisper", "qwen"):
            asr_method = "whisper"
        qwen_asr_model_dir = (
            (qwen_asr_model_dir or str(getattr(self.settings, "qwen_asr_model_path", "") or "")).strip() or None
        )
        qwen_asr_num_threads = int(qwen_asr_num_threads or getattr(self.settings, "qwen_asr_num_threads", 1) or 1)
        qwen_asr_vad_segment_threshold = int(
            qwen_asr_vad_segment_threshold or getattr(self.settings, "qwen_asr_vad_segment_threshold", 60) or 60
        )
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

        def _looks_like_qwen_asr_model_dir(model_dir: str | None) -> bool:
            if not model_dir:
                return False
            try:
                path = Path(str(model_dir)).expanduser()
            except Exception:
                return False
            if not path.is_dir():
                return False
            if (path / "config.json").exists():
                return True
            if (path / "model.safetensors.index.json").exists():
                return True
            try:
                for p in path.glob("*.safetensors"):
                    if p.is_file():
                        return True
            except Exception:
                pass
            # Fallback: non-empty dir
            try:
                return any(path.iterdir())
            except Exception:
                return False

        wd = (whisper_device or "auto").lower().strip()
        if wd not in ("auto", "cuda", "cpu"):
            wd = "auto"

        # Validate ASR model path(s) early so UI can show friendly error instead of silent failures.
        if asr_method == "qwen":
            if not _looks_like_qwen_asr_model_dir(qwen_asr_model_dir):
                raise ModelCheckError(
                    "Qwen3-ASR 模型目录无效或不存在。\n"
                    f"- QWEN_ASR_MODEL_PATH: {qwen_asr_model_dir}\n"
                    "请在 UI 中填写正确路径，或在 .env 中设置 QWEN_ASR_MODEL_PATH。"
                )
        else:
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
        if asr_method == "qwen":
            required_models.append(self.model_manager._qwen_asr_requirement().name)  # type: ignore[attr-defined]
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

        info_list = list(download.get_info_list_from_url(urls, num_videos, settings=self.settings))

        def _valid_transcript(path: str) -> bool:
            try:
                if not os.path.exists(path) or os.path.getsize(path) < 2:
                    return False
            except OSError:
                return False
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                return isinstance(obj, list)
            except Exception:
                return False

        # 若目标目录里“人声/伴奏 + transcript.json”都已存在，则不需要预热 Demucs/ASR。
        need_demucs_warmup = False
        need_asr_warmup = False
        for info in info_list:
            check_cancelled()
            folder = download.get_target_folder(info, root_folder)
            if not folder:
                need_demucs_warmup = True
                need_asr_warmup = True
                break
            if not os.path.exists(folder):
                need_demucs_warmup = True
                need_asr_warmup = True
                break

            vocals_ok = valid_file(os.path.join(folder, "audio_vocals.wav"), min_bytes=44)
            inst_ok = valid_file(os.path.join(folder, "audio_instruments.wav"), min_bytes=44)
            if not (vocals_ok and inst_ok):
                need_demucs_warmup = True

            if not _valid_transcript(os.path.join(folder, "transcript.json")):
                need_asr_warmup = True

            if need_demucs_warmup and need_asr_warmup:
                break

        # Warm-up models best-effort. Keep it sequential so Ctrl+C can stop cleanly.
        if need_demucs_warmup:
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
        if need_asr_warmup:
            check_cancelled()
            try:
                # Preload ASR model best-effort to reduce first-request latency.
                # Respect explicit ASR device selection from UI; for "auto" keep legacy behavior.
                if asr_method == "qwen":
                    resolved = wd
                    if resolved == "auto":
                        resolved = "cuda" if torch.cuda.is_available() else "cpu"
                    transcribe.load_qwen_asr_model(
                        qwen_asr_model_dir or "",
                        device=resolved,
                        settings=self.settings,
                        model_manager=self.model_manager,
                        max_inference_batch_size=1,
                    )
                else:
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

        # NOTE: max_workers 的新语义：
        # - 主流程（Demucs/ASR/TTS/翻译等）永远串行，避免各步骤的全局缓存/卸载竞态。
        # - 仅在 use_nvenc=True 且多视频时，把“视频合成(FFmpeg/NVENC)”放到后台线程池执行；
        #   max_workers 只影响该后台线程池的并发数（encode_workers）。

        def _preprocess_until_tts(info: dict[str, Any]) -> str | None:
            """Run steps up to TTS generation; return folder on success."""
            for _retry in range(int(max_retries)):
                check_cancelled()
                try:
                    folder = download.download_single_video(info, root_folder, resolution, settings=self.settings)
                    if folder is None:
                        logger.warning(f"下载失败: {info.get('title')}")
                        return None

                    logger.info(f"开始处理: {folder}")
                    require_file(os.path.join(folder, "download.mp4"), "下载视频(download.mp4)", min_bytes=1024)

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

                    require_file(os.path.join(folder, "audio_vocals.wav"), "人声轨(audio_vocals.wav)", min_bytes=44)
                    require_file(
                        os.path.join(folder, "audio_instruments.wav"),
                        "伴奏轨(audio_instruments.wav)",
                        min_bytes=44,
                    )

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
                            asr_method=asr_method,
                            qwen_asr_model_dir=qwen_asr_model_dir,
                            qwen_asr_num_threads=int(qwen_asr_num_threads),
                            qwen_asr_vad_segment_threshold=int(qwen_asr_vad_segment_threshold),
                        )
                    finally:
                        try:
                            transcribe.unload_all_models()
                        except Exception:
                            pass

                    require_file(os.path.join(folder, "transcript.json"), "转写结果(transcript.json)", min_bytes=2)

                    check_cancelled()
                    translate.translate_all_transcript_under_folder(
                        folder, target_language=translation_target_language, settings=self.settings
                    )
                    require_file(os.path.join(folder, "translation.json"), "翻译结果(translation.json)", min_bytes=2)

                    check_cancelled()
                    synthesize_speech.generate_all_wavs_under_folder(
                        folder,
                        tts_method=tts_method,
                        qwen_tts_batch_size=qwen_tts_batch_size,
                    )

                    require_file(
                        os.path.join(folder, "wavs", ".tts_done.json"),
                        "语音合成标记(wavs/.tts_done.json)",
                        min_bytes=2,
                    )
                    require_file(os.path.join(folder, "wavs", "0000.wav"), "TTS分段音频(wavs/0000.wav)", min_bytes=44)

                    return folder
                except CorruptedVideoError as exc:
                    # 视频文件损坏，已被删除，下次重试会重新下载
                    logger.warning(f"视频文件损坏已删除，将重新下载: {info.get('title')} ({exc})")
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception(f"处理失败(到TTS阶段): {info.get('title')} ({exc})")
            return None

        def _finalize_after_encode(folder: str) -> None:
            require_file(os.path.join(folder, "video.mp4"), "最终视频(video.mp4)", min_bytes=1024)

            check_cancelled()
            generate_all_info_under_folder(folder)

            if auto_upload_video:
                # Run B站上传 in background so it won't block video processing.
                if self._already_uploaded(folder):
                    logger.info(f"检测到已上传，跳过自动上传: {folder}")
                else:
                    upload_video_async(folder)

        if use_nvenc and len(info_list) > 1:
            requested = int(max_workers) if int(max_workers) > 0 else 1
            encode_workers = min(requested, 8, len(info_list))
            logger.info(
                "检测到 NVENC 且有多个视频：视频合成将后台执行（不阻塞后续视频处理）"
                f"（encode_workers={encode_workers}，max_workers仅影响该值）"
            )

            encode_jobs: list[tuple[dict[str, Any], str, object]] = []
            with ThreadPoolExecutor(max_workers=encode_workers) as encode_executor:
                for info in info_list:
                    check_cancelled()
                    folder = _preprocess_until_tts(info)
                    if not folder:
                        fail_list.append(info)
                        continue

                    fut = encode_executor.submit(
                        synthesize_all_video_under_folder,
                        folder,
                        subtitles=subtitles,
                        bilingual_subtitle=bool(bilingual_subtitle),
                        adaptive_segment_stretch=bool(tts_adaptive_segment_stretch),
                        fps=fps,
                        resolution=target_resolution,
                        use_nvenc=use_nvenc,
                        auto_upload_video=False,
                    )
                    encode_jobs.append((info, folder, fut))
                    logger.info(f"已加入后台视频合成队列: {folder}")

                for info, folder, fut in encode_jobs:
                    check_cancelled()
                    try:
                        _ = fut.result()
                        _finalize_after_encode(folder)
                        success_list.append(info)
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.exception(f"处理失败(视频合成阶段): {info.get('title')} ({exc})")
                        fail_list.append(info)
        else:
            if int(max_workers) > 1:
                logger.info("max_workers>1 将被忽略：当前不启用全流程并发，仅 NVENC 多视频时用于后台编码并发数。")
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
                    fps,
                    target_resolution,
                    max_retries,
                    auto_upload_video,
                    bool(bilingual_subtitle),
                    use_nvenc,
                    wd,
                    whisper_cpu_model,
                    asr_method=asr_method,
                    qwen_asr_model_dir=qwen_asr_model_dir,
                    qwen_asr_num_threads=int(qwen_asr_num_threads),
                    qwen_asr_vad_segment_threshold=int(qwen_asr_vad_segment_threshold),
                )
                if success:
                    success_list.append(info)
                else:
                    fail_list.append(info)

        return f"成功: {len(success_list)}\n失败: {len(fail_list)}"
