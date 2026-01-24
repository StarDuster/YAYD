from __future__ import annotations

import ctypes
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from loguru import logger

from ...config import Settings
from ...models import ModelCheckError, ModelManager
from ..utils import ensure_torchaudio_backend_compat, save_wav


def _import_faster_whisper():
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel  # type: ignore
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "缺少依赖 faster-whisper，语音识别功能不可用。\n"
            "请在当前虚拟环境中安装：`pip install -U faster-whisper`（或使用 `uv sync`）。\n"
            f"Original error: {exc}"
        ) from exc
    return WhisperModel, BatchedInferencePipeline


_DEFAULT_SETTINGS = Settings()
_DEFAULT_MODEL_MANAGER = ModelManager(_DEFAULT_SETTINGS)

_ASR_MODEL = None
_ASR_PIPELINE = None
_ASR_KEY: str | None = None

_DIARIZATION_PIPELINE = None
_DIARIZATION_KEY: str | None = None


_CUDNN_PRELOADED = False


def _read_speaker_ref_seconds(default: float = 15.0) -> float:
    """
    Speaker reference audio duration (seconds) for downstream TTS voice cloning.

    Official recommendation is usually 10-20s; we default to 15s.
    Clamp to [3, 60] seconds to avoid pathological inputs.
    """
    raw = os.getenv("TTS_SPEAKER_REF_SECONDS")
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    if not (v > 0):
        return default
    return float(max(3.0, min(v, 60.0)))


def _preload_cudnn_for_onnxruntime_gpu() -> None:
    """Ensure onnxruntime-gpu can find cuDNN shipped as pip wheels.

    In some environments (notably WSL / minimal containers), CUDA libraries are available
    system-wide, but cuDNN is installed via `nvidia-cudnn-cu12` inside the venv only.
    `onnxruntime-gpu` loads `libonnxruntime_providers_cuda.so`, which depends on `libcudnn.so.9`.
    The dynamic loader won't search the venv site-packages directory by default, leading to
    a hard crash/abort in downstream libraries.

    We preload `libcudnn.so.9` via an absolute path and RTLD_GLOBAL so that subsequent loads
    can resolve the dependency by SONAME.
    """

    global _CUDNN_PRELOADED  # noqa: PLW0603

    if _CUDNN_PRELOADED:
        return
    _CUDNN_PRELOADED = True

    if sys.platform != "linux":
        return

    try:
        ort_spec = importlib.util.find_spec("onnxruntime")
        if not ort_spec or not ort_spec.submodule_search_locations:
            return
        ort_root = Path(list(ort_spec.submodule_search_locations)[0])
        if not (ort_root / "capi" / "libonnxruntime_providers_cuda.so").exists():
            # CPU-only build; nothing to do.
            return

        cudnn_spec = importlib.util.find_spec("nvidia.cudnn")
        if not cudnn_spec or not cudnn_spec.submodule_search_locations:
            return
        cudnn_root = Path(list(cudnn_spec.submodule_search_locations)[0])
        cudnn_lib = cudnn_root / "lib" / "libcudnn.so.9"
        if not cudnn_lib.exists():
            return

        ctypes.CDLL(str(cudnn_lib), mode=ctypes.RTLD_GLOBAL)
        logger.info(f"Preloaded cuDNN for onnxruntime-gpu: {cudnn_lib}")
    except Exception as exc:  # pylint: disable=broad-except
        # Never hard-fail here; downstream may still work if the system has cuDNN installed.
        logger.warning(f"Failed to preload cuDNN for onnxruntime-gpu: {exc}")


def unload_all_models() -> None:
    global _ASR_MODEL, _ASR_PIPELINE, _DIARIZATION_PIPELINE

    cleared = False
    if _ASR_PIPELINE is not None:
        del _ASR_PIPELINE
        _ASR_PIPELINE = None
        cleared = True
    if _ASR_MODEL is not None:
        del _ASR_MODEL
        _ASR_MODEL = None
        cleared = True
    if _DIARIZATION_PIPELINE is not None:
        del _DIARIZATION_PIPELINE
        _DIARIZATION_PIPELINE = None
        cleared = True

    if cleared:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()
        logger.info("ASR/diarization models unloaded")


def _ensure_offline_mode() -> None:
    """Prevent unexpected online downloads (Hugging Face)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _ensure_assets(
    settings: Settings,
    model_manager: ModelManager,
    require_diarization: bool,
) -> None:
    model_manager.enforce_offline()
    # NOTE: Whisper ASR model path is often provided via UI arguments (not only Settings).
    # Avoid enforcing a fixed Settings-based path here; `load_asr_model()` will validate the
    # actual model_dir passed in (expects model.bin).
    if require_diarization:
        model_manager.ensure_ready(
            names=[model_manager._whisper_diarization_requirement().name]  # type: ignore[attr-defined]
        )


def _default_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def load_asr_model(
    model_dir: str,
    device: str = "auto",
    compute_type: str | None = None,
    use_batched: bool = True,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> None:
    global _ASR_MODEL, _ASR_PIPELINE, _ASR_KEY

    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER

    _ensure_assets(settings, model_manager, require_diarization=False)
    _ensure_offline_mode()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if compute_type is None:
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE") or _default_compute_type(device)

    model_path = Path(model_dir)
    if not model_path.exists():
        raise ModelCheckError(
            "Whisper model directory not found: "
            f"{model_path}. Please download faster-whisper CTranslate2 model and set "
            "WHISPER_MODEL_PATH (or WHISPER_CPU_MODEL_PATH when using CPU)."
        )
    if model_path.is_dir() and not (model_path / "model.bin").exists():
        raise ModelCheckError(f"Whisper CTranslate2 model missing model.bin: {model_path}")

    key = f"{model_path.absolute()}|device={device}|compute={compute_type}|batched={int(use_batched)}"
    if _ASR_KEY == key and _ASR_MODEL is not None:
        return

    WhisperModel, BatchedInferencePipeline = _import_faster_whisper()

    logger.info(f"Loading Whisper model from {model_path} (device={device}, compute_type={compute_type})")
    t0 = time.time()
    _ASR_MODEL = WhisperModel(str(model_path), device=device, compute_type=compute_type)
    _ASR_PIPELINE = BatchedInferencePipeline(model=_ASR_MODEL) if use_batched else None
    _ASR_KEY = key
    logger.info(f"Loaded Whisper model in {time.time() - t0:.2f}s")


def init_asr(settings: Settings | None = None, model_manager: ModelManager | None = None) -> None:
    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER
    requested_device = getattr(settings, "whisper_device", "auto")
    resolved_device = requested_device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = str(settings.whisper_model_path)
    cpu_dir = getattr(settings, "whisper_cpu_model_path", None)
    if resolved_device == "cpu" and cpu_dir:
        model_dir = str(cpu_dir)

    load_asr_model(
        model_dir=model_dir,
        device=resolved_device,
        settings=settings,
        model_manager=model_manager,
    )


def _find_pyannote_config(root_dir: Path) -> Path | None:
    if not root_dir.exists():
        return None
    for path in root_dir.rglob("config.yaml"):
        if "snapshots" in path.parts:
            return path
    return None


def load_diarize_model(
    device: str = "auto",
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> None:
    global _DIARIZATION_PIPELINE, _DIARIZATION_KEY

    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER

    _ensure_assets(settings, model_manager, require_diarization=True)
    _ensure_offline_mode()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
    diar_dir_str = str(diar_dir) if diar_dir else ""
    key = f"{device}|{diar_dir_str}"
    if _DIARIZATION_PIPELINE is not None and _DIARIZATION_KEY == key:
        return

    # pyannote.audio<=3.1 imports torchaudio.set_audio_backend at import time, but torchaudio>=2.10 removed it.
    # Provide a no-op shim so diarization can still work with the default backend.
    ensure_torchaudio_backend_compat()

    try:
        from pyannote.audio import Pipeline  # type: ignore
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "缺少依赖 pyannote.audio，说话人分离不可用。\n"
            "请安装 `pyannote.audio` 并准备好离线模型缓存（或关闭 diarization）。\n"
            f"Original error: {exc}"
        ) from exc

    if diar_dir:
        # Force HF cache into the provided directory for offline execution.
        os.environ["HF_HOME"] = diar_dir_str
        os.environ["TRANSFORMERS_CACHE"] = diar_dir_str

    cfg = _find_pyannote_config(diar_dir) if diar_dir else None

    logger.info(f"Loading diarization pipeline (device={device})")
    t0 = time.time()
    if cfg and cfg.exists():
        pipeline = Pipeline.from_pretrained(str(cfg))
    else:
        token = settings.hf_token
        if not token:
            raise ModelCheckError("Missing HF_TOKEN, cannot load pyannote/speaker-diarization-3.1.")
        # pyannote.audio v4 uses `token=...`; older versions use `use_auth_token=...`.
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)

    pipeline.to(torch.device(device))
    _DIARIZATION_PIPELINE = pipeline
    _DIARIZATION_KEY = key
    logger.info(f"Loaded diarization pipeline in {time.time() - t0:.2f}s")


def merge_segments(transcript: list[dict[str, Any]], ending: str = '!"\').:;?]}~') -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    buffer: dict[str, Any] | None = None

    for segment in transcript:
        if buffer is None:
            buffer = segment
            continue

        # Never merge across speaker boundaries.
        if buffer.get("speaker") != segment.get("speaker"):
            merged.append(buffer)
            buffer = segment
            continue

        if buffer.get("text") and str(buffer["text"])[-1] in ending:
            merged.append(buffer)
            buffer = segment
            continue

        buffer["text"] = (str(buffer.get("text", "")).strip() + " " + str(segment.get("text", "")).strip()).strip()
        buffer["end"] = segment.get("end", buffer.get("end"))

    if buffer is not None:
        merged.append(buffer)
    return merged


def generate_speaker_audio(folder: str, transcript: list[dict[str, Any]]) -> None:
    wav_path = os.path.join(folder, "audio_vocals.wav")
    if not os.path.exists(wav_path):
        logger.warning(f"Audio file not found: {wav_path}")
        return

    target_sr = 24000
    max_ref_seconds = _read_speaker_ref_seconds()
    max_ref_samples = int(max_ref_seconds * float(target_sr))
    delay = 0.05
    speakers = {str(seg.get("speaker") or "SPEAKER_00") for seg in transcript}
    speaker_dict: dict[str, np.ndarray] = {}
    for spk in speakers:
        speaker_dict[spk] = np.zeros((0,), dtype=np.float32)

    for segment in transcript:
        start_s = float(segment.get("start", 0.0))
        end_s = float(segment.get("end", 0.0))
        speaker = str(segment.get("speaker") or "SPEAKER_00")
        if max_ref_samples > 0 and speaker_dict.get(speaker, np.zeros((0,), dtype=np.float32)).shape[0] >= max_ref_samples:
            continue

        offset = max(0.0, start_s - delay)
        duration = max(0.0, (end_s - start_s) + 2.0 * delay)
        if duration <= 0:
            continue
        try:
            chunk, _sr = librosa.load(wav_path, sr=target_sr, mono=True, offset=offset, duration=duration)
        except Exception as exc:
            logger.warning(f"Failed to load speaker audio chunk (speaker={speaker}, offset={offset:.2f}s): {exc}")
            continue
        if chunk.size <= 0:
            continue

        remaining = max_ref_samples - int(speaker_dict[speaker].shape[0]) if max_ref_samples > 0 else chunk.shape[0]
        if remaining <= 0:
            continue
        if chunk.shape[0] > remaining:
            chunk = chunk[:remaining]
        speaker_dict[speaker] = np.concatenate((speaker_dict[speaker], chunk.astype(np.float32)))

        if max_ref_samples > 0 and all(v.shape[0] >= max_ref_samples for v in speaker_dict.values()):
            break

    speaker_folder = os.path.join(folder, "SPEAKER")
    os.makedirs(speaker_folder, exist_ok=True)

    # Fallback: if a speaker has 0 samples, take the first N seconds from the file.
    for speaker in speakers:
        if speaker_dict[speaker].size > 0:
            continue
        try:
            chunk, _sr = librosa.load(wav_path, sr=target_sr, mono=True, offset=0.0, duration=max_ref_seconds)
            if chunk.size > 0:
                speaker_dict[speaker] = chunk.astype(np.float32)
        except Exception:
            # Best-effort: keep empty
            continue

    for speaker, audio in speaker_dict.items():
        if audio.size <= 0:
            continue
        speaker_file_path = os.path.join(speaker_folder, f"{speaker}.wav")
        save_wav(audio, speaker_file_path, sample_rate=target_sr)
        if max_ref_samples > 0 and audio.shape[0] >= max_ref_samples:
            logger.info(f"Saved speaker reference ({max_ref_seconds:.1f}s): {speaker_file_path}")


def _assign_speakers_by_overlap(
    segments: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    default_speaker: str = "SPEAKER_00",
) -> None:
    if not segments or not turns:
        for seg in segments:
            seg["speaker"] = default_speaker
        return

    turns_sorted = sorted(turns, key=lambda x: float(x["start"]))
    idx = 0

    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        if seg_end <= seg_start:
            seg["speaker"] = default_speaker
            continue

        while idx < len(turns_sorted) and float(turns_sorted[idx]["end"]) <= seg_start:
            idx += 1

        best_speaker = None
        best_overlap = 0.0
        j = idx
        while j < len(turns_sorted) and float(turns_sorted[j]["start"]) < seg_end:
            t = turns_sorted[j]
            ov = max(0.0, min(seg_end, float(t["end"])) - max(seg_start, float(t["start"])))
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = str(t.get("speaker") or default_speaker)
            j += 1

        seg["speaker"] = best_speaker if best_speaker is not None else default_speaker


def transcribe_audio(
    folder: str,
    model_name: str | None = None,
    cpu_model_name: str | None = None,
    device: str = "auto",
    batch_size: int = 32,
    diarization: bool = True,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> bool:
    transcript_path = os.path.join(folder, "transcript.json")
    if os.path.exists(transcript_path):
        transcript: Any | None = None
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
            if not isinstance(transcript, list):
                raise ValueError("transcript.json is not a list")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"Invalid transcript file, will regenerate: {transcript_path} ({exc})")
            try:
                os.remove(transcript_path)
            except Exception:
                pass
        else:
            logger.info(f"Transcript already exists in {folder}")
            # Ensure speaker reference files exist even when transcript step is skipped.
            try:
                speakers = {str(seg.get("speaker") or "SPEAKER_00") for seg in transcript}
                speaker_dir = os.path.join(folder, "SPEAKER")
                need = False
                for spk in speakers:
                    p = os.path.join(speaker_dir, f"{spk}.wav")
                    if not os.path.exists(p) or os.path.getsize(p) < 44:
                        need = True
                        break
                if need:
                    generate_speaker_audio(folder, transcript)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(f"Failed to ensure speaker reference wavs: {exc}")
            return True

    wav_path = os.path.join(folder, "audio_vocals.wav")
    if not os.path.exists(wav_path):
        return False

    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER

    if model_name is None:
        model_name = str(settings.whisper_model_path)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        # Prefer explicit CPU model path (UI arg) -> settings default -> fallback to model_name.
        cpu_candidate = (cpu_model_name or "").strip()
        if not cpu_candidate:
            cpu_dir = getattr(settings, "whisper_cpu_model_path", None)
            cpu_candidate = str(cpu_dir).strip() if cpu_dir else ""
        if cpu_candidate:
            model_name = cpu_candidate

    load_asr_model(
        model_dir=model_name,
        device=device,
        settings=settings,
        model_manager=model_manager,
        use_batched=True,
    )

    assert _ASR_MODEL is not None

    logger.info(f"Transcribing {wav_path}")
    t0 = time.time()

    _preload_cudnn_for_onnxruntime_gpu()

    # Prefer batched pipeline if available (much higher throughput on GPU).
    if _ASR_PIPELINE is not None:
        segments_iter, info = _ASR_PIPELINE.transcribe(
            wav_path,
            batch_size=batch_size,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
    else:
        segments_iter, info = _ASR_MODEL.transcribe(
            wav_path,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )

    segments_list = list(segments_iter)
    logger.info(f"ASR done in {time.time() - t0:.2f}s (segments={len(segments_list)}, language={getattr(info, 'language', None)})")

    transcript: list[dict[str, Any]] = []
    for seg in segments_list:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        transcript.append(
            {
                "start": float(getattr(seg, "start", 0.0)),
                "end": float(getattr(seg, "end", 0.0)),
                "text": text,
                "speaker": "SPEAKER_00",
            }
        )

    if diarization:
        try:
            load_diarize_model(device=device, settings=settings, model_manager=model_manager)
            assert _DIARIZATION_PIPELINE is not None
            logger.info("Starting diarization (pyannote)...")
            ann = _DIARIZATION_PIPELINE(wav_path, min_speakers=min_speakers, max_speakers=max_speakers)
            turns: list[dict[str, Any]] = []
            for seg, _, speaker in ann.itertracks(yield_label=True):
                turns.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(speaker)})
            _assign_speakers_by_overlap(transcript, turns)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"Diarization failed, falling back to single speaker: {exc}")
            for item in transcript:
                item["speaker"] = "SPEAKER_00"

    transcript = merge_segments(transcript)

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved transcript: {transcript_path}")

    generate_speaker_audio(folder, transcript)
    return True


def transcribe_all_audio_under_folder(
    folder: str,
    model_name: str | None = None,
    cpu_model_name: str | None = None,
    device: str = "auto",
    batch_size: int = 32,
    diarization: bool = True,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> str:
    count = 0
    for root, _, files in os.walk(folder):
        if "audio_vocals.wav" not in files:
            continue
        ok = transcribe_audio(
            root,
            model_name=model_name,
            cpu_model_name=cpu_model_name,
            device=device,
            batch_size=batch_size,
            diarization=diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            settings=settings,
            model_manager=model_manager,
        )
        if ok:
            count += 1
    msg = f"Transcribed all audio under {folder} (processed {count} files)"
    logger.info(msg)
    return msg
