from __future__ import annotations

import ctypes
import html
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from loguru import logger

from ..config import Settings
from ..models import ModelCheckError, ModelManager
from ..interrupts import check_cancelled
from ..utils import (
    read_speaker_ref_seconds,
    save_wav,
    torch_load_weights_only_compat,
    wav_duration_seconds,
)


def _import_faster_whisper():
    try:
        from faster_whisper import BatchedInferencePipeline, WhisperModel  # type: ignore
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "缺少依赖 faster-whisper，语音识别功能不可用。\n"
            "请在当前虚拟环境中安装：`pip install -U faster-whisper`（或使用 `uv sync`）。\n"
            f"原始错误: {exc}"
        ) from exc
    return WhisperModel, BatchedInferencePipeline


_DEFAULT_SETTINGS = Settings()
_DEFAULT_MODEL_MANAGER = ModelManager(_DEFAULT_SETTINGS)

_ASR_MODEL = None
_ASR_PIPELINE = None
_ASR_KEY: str | None = None

_QWEN_ASR_MODEL = None
_QWEN_ASR_KEY: str | None = None

_DIARIZATION_PIPELINE = None
_DIARIZATION_KEY: str | None = None
_PYANNOTE_DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-community-1"


_CUDNN_PRELOADED = False


def _import_qwen_asr():
    try:
        from qwen_asr import Qwen3ASRModel  # type: ignore
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "缺少依赖 qwen-asr，无法使用 Qwen3-ASR。\n"
            "请在当前虚拟环境中安装：`uv sync`（或 `uv add qwen-asr`）。\n"
            f"原始错误: {exc}"
        ) from exc
    return Qwen3ASRModel


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?\.])\s*")


def _split_text_with_timing(text: str, start_s: float, end_s: float) -> list[dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        return []
    duration = float(max(0.0, end_s - start_s))
    if duration <= 0:
        return [{"start": float(start_s), "end": float(start_s), "text": s, "speaker": "SPEAKER_00"}]

    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(s) if p and p.strip()]
    if not parts:
        return [{"start": float(start_s), "end": float(end_s), "text": s, "speaker": "SPEAKER_00"}]

    weights = [max(1, len(p.replace(" ", ""))) for p in parts]
    total = float(sum(weights)) or 1.0

    out: list[dict[str, Any]] = []
    cur = float(start_s)
    for i, (p, w) in enumerate(zip(parts, weights)):
        seg_dur = duration * (float(w) / total)
        seg_end = float(end_s) if i == (len(parts) - 1) else float(cur + seg_dur)
        out.append({"start": float(cur), "end": float(seg_end), "text": p, "speaker": "SPEAKER_00"})
        cur = seg_end
    return out


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
        logger.info(f"已预加载cuDNN用于onnxruntime-gpu: {cudnn_lib}")
    except Exception as exc:  # pylint: disable=broad-except
        # Never hard-fail here; downstream may still work if the system has cuDNN installed.
        logger.warning(f"预加载cuDNN失败 onnxruntime-gpu: {exc}")


def unload_all_models() -> None:
    global _ASR_MODEL, _ASR_PIPELINE, _QWEN_ASR_MODEL, _DIARIZATION_PIPELINE
    global _ASR_KEY, _QWEN_ASR_KEY, _DIARIZATION_KEY  # noqa: PLW0603

    cleared = False
    # IMPORTANT:
    # Do NOT `del _GLOBAL` for module-level caches. Under concurrency, another thread may
    # observe a brief window where the name is missing and crash with NameError.
    #
    # Instead: move the object to a local var, clear the global to None, then `del` the local.
    old_asr_pipeline = _ASR_PIPELINE
    old_asr_model = _ASR_MODEL
    old_qwen_model = _QWEN_ASR_MODEL
    old_diar = _DIARIZATION_PIPELINE

    if old_asr_pipeline is not None:
        _ASR_PIPELINE = None
        _ASR_KEY = None
        cleared = True
    if old_asr_model is not None:
        _ASR_MODEL = None
        _ASR_KEY = None
        cleared = True
    if old_qwen_model is not None:
        _QWEN_ASR_MODEL = None
        _QWEN_ASR_KEY = None
        cleared = True
    if old_diar is not None:
        _DIARIZATION_PIPELINE = None
        _DIARIZATION_KEY = None
        cleared = True

    if cleared:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()
        logger.info("ASR/说话人分离模型已卸载")

    # Drop references (best-effort) to encourage GC.
    try:
        del old_asr_pipeline
        del old_asr_model
        del old_qwen_model
        del old_diar
    except Exception:
        pass


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
    check_cancelled()
    global _ASR_MODEL, _ASR_PIPELINE, _ASR_KEY

    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER

    _ensure_assets(settings, model_manager, require_diarization=False)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if compute_type is None:
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE") or _default_compute_type(device)

    model_path = Path(model_dir)
    if not model_path.exists():
        raise ModelCheckError(
            "Whisper 模型目录不存在: "
            f"{model_path}。请下载 faster-whisper 的 CTranslate2 模型，并设置 "
            "WHISPER_MODEL_PATH（CPU 用 WHISPER_CPU_MODEL_PATH）。"
        )
    if model_path.is_dir() and not (model_path / "model.bin").exists():
        raise ModelCheckError(f"Whisper CTranslate2 模型缺少 model.bin: {model_path}")

    key = f"{model_path.absolute()}|device={device}|compute={compute_type}|batched={int(use_batched)}"
    if _ASR_KEY == key and _ASR_MODEL is not None:
        return

    WhisperModel, BatchedInferencePipeline = _import_faster_whisper()

    logger.info(f"加载Whisper模型 {model_path} (设备={device}, compute_type={compute_type})")
    t0 = time.time()
    _ASR_MODEL = WhisperModel(str(model_path), device=device, compute_type=compute_type)
    _ASR_PIPELINE = BatchedInferencePipeline(model=_ASR_MODEL) if use_batched else None
    _ASR_KEY = key
    logger.info(f"Whisper模型加载完成，耗时 {time.time() - t0:.2f}秒")


def _qwen_device_map(device: str) -> str:
    if device.startswith("cuda"):
        return "cuda:0"
    return "cpu"


def _qwen_default_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda"):
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def _looks_like_qwen_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    # Minimal heuristic: allow either config.json or any safetensors in the directory.
    if (path / "config.json").exists():
        return True
    for p in path.glob("*.safetensors"):
        if p.is_file():
            return True
    if (path / "model.safetensors.index.json").exists():
        return True
    # Fallback: non-empty dir.
    try:
        return any(path.iterdir())
    except Exception:
        return False


def load_qwen_asr_model(
    model_dir: str,
    device: str = "auto",
    dtype: torch.dtype | None = None,
    max_new_tokens: int | None = None,
    max_inference_batch_size: int | None = None,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> None:
    check_cancelled()
    global _QWEN_ASR_MODEL, _QWEN_ASR_KEY

    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER

    _ensure_assets(settings, model_manager, require_diarization=False)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype is None:
        dtype = _qwen_default_dtype(device)

    model_path = Path(str(model_dir)).expanduser()
    if not _looks_like_qwen_model_dir(model_path):
        raise ModelCheckError(
            "Qwen3-ASR 模型目录无效或不存在（需要离线模型目录）。"
            f"\n当前: {model_path}\n请下载模型并设置 QWEN_ASR_MODEL_PATH（或在 UI 中填写路径）。"
        )

    key = f"{model_path.absolute()}|device={device}|dtype={dtype}|max_new={max_new_tokens}|bs={max_inference_batch_size}"
    if _QWEN_ASR_KEY == key and _QWEN_ASR_MODEL is not None:
        return

    Qwen3ASRModel = _import_qwen_asr()
    logger.info(f"加载Qwen3-ASR模型 {model_path} (device={device}, dtype={dtype})")
    t0 = time.time()

    kwargs: dict[str, Any] = {"dtype": dtype, "device_map": _qwen_device_map(device)}
    if max_inference_batch_size is not None:
        kwargs["max_inference_batch_size"] = int(max_inference_batch_size)
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = int(max_new_tokens)

    try:
        _QWEN_ASR_MODEL = Qwen3ASRModel.from_pretrained(str(model_path), **kwargs)
    except TypeError:
        # Backward/forward compatibility: older versions may not accept some kwargs.
        kwargs.pop("max_inference_batch_size", None)
        kwargs.pop("max_new_tokens", None)
        _QWEN_ASR_MODEL = Qwen3ASRModel.from_pretrained(str(model_path), **kwargs)

    _QWEN_ASR_KEY = key
    logger.info(f"Qwen3-ASR模型加载完成，耗时 {time.time() - t0:.2f}秒")


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

    # Local clone layout (recommended by pyannote for offline use):
    #   <dir>/config.yaml
    direct = root_dir / "config.yaml"
    if direct.exists():
        return direct

    org, repo = _PYANNOTE_DIARIZATION_MODEL_ID.split("/", 1)

    # HF cache layout (new):
    #   <HF_HOME>/hub/models--ORG--REPO/snapshots/<rev>/config.yaml
    candidates = [
        root_dir / f"models--{org}--{repo}",
        root_dir / "hub" / f"models--{org}--{repo}",
    ]

    for base in candidates:
        if not base.exists() or not base.is_dir():
            continue

        cfg_direct = base / "config.yaml"
        if cfg_direct.exists():
            return cfg_direct

        snapshots = base / "snapshots"
        if snapshots.exists() and snapshots.is_dir():
            best: Path | None = None
            best_mtime = -1.0
            for snap in snapshots.iterdir():
                cfg = snap / "config.yaml"
                if not cfg.exists():
                    continue
                try:
                    mtime = float(cfg.stat().st_mtime)
                except Exception:
                    mtime = 0.0
                if mtime > best_mtime:
                    best = cfg
                    best_mtime = mtime
            if best is not None:
                return best

    return None


def load_diarize_model(
    device: str = "auto",
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> None:
    check_cancelled()
    global _DIARIZATION_PIPELINE, _DIARIZATION_KEY

    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER

    _ensure_assets(settings, model_manager, require_diarization=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def _read_float_env(name: str) -> float | None:
        raw = os.getenv(name)
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            logger.warning(f"环境变量 {name} 不是有效浮点数，将忽略: {raw!r}")
            return None

    # Optional: allow users to tune diarization hyperparameters without code changes.
    # These names intentionally match pipeline.parameters() keys:
    # - segmentation.min_duration_off
    # - clustering.threshold
    seg_min_off = _read_float_env("PYANNOTE_SEGMENTATION_MIN_DURATION_OFF")
    clust_thr = _read_float_env("PYANNOTE_CLUSTERING_THRESHOLD")

    diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
    diar_dir_str = str(diar_dir) if diar_dir else ""
    cache_dir = diar_dir_str or None
    key = (
        f"{device}|{diar_dir_str}|{_PYANNOTE_DIARIZATION_MODEL_ID}"
        f"|seg_min_off={seg_min_off if seg_min_off is not None else 'default'}"
        f"|clust_thr={clust_thr if clust_thr is not None else 'default'}"
    )
    if _DIARIZATION_PIPELINE is not None and _DIARIZATION_KEY == key:
        return

    try:
        from pyannote.audio import Pipeline  # type: ignore
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "缺少依赖 pyannote.audio，说话人分离不可用。\n"
            "请安装 `pyannote.audio` 并准备好离线模型缓存（或关闭 diarization）。\n"
            f"原始错误: {exc}"
        ) from exc

    token = settings.hf_token
    cfg = _find_pyannote_config(diar_dir) if diar_dir else None
    logger.info(f"加载说话人分离管道: {_PYANNOTE_DIARIZATION_MODEL_ID} (设备={device})")
    t0 = time.time()
    with torch_load_weights_only_compat():
        if cfg and cfg.exists():
            try:
                pipeline = Pipeline.from_pretrained(str(cfg), token=token, cache_dir=cache_dir)
            except TypeError:
                try:
                    pipeline = Pipeline.from_pretrained(str(cfg), use_auth_token=token, cache_dir=cache_dir)
                except TypeError:
                    # Older pyannote versions may not accept auth/cache kwargs for local checkpoints.
                    pipeline = Pipeline.from_pretrained(str(cfg))
        else:
            if not token:
                raise ModelCheckError(f"缺少 HF_TOKEN，无法加载 {_PYANNOTE_DIARIZATION_MODEL_ID}。")
            # pyannote.audio v4 uses `token=...`; older versions use `use_auth_token=...`.
            try:
                pipeline = Pipeline.from_pretrained(_PYANNOTE_DIARIZATION_MODEL_ID, token=token, cache_dir=cache_dir)
            except TypeError:
                try:
                    pipeline = Pipeline.from_pretrained(
                        _PYANNOTE_DIARIZATION_MODEL_ID, use_auth_token=token, cache_dir=cache_dir
                    )
                except TypeError:
                    pipeline = Pipeline.from_pretrained(_PYANNOTE_DIARIZATION_MODEL_ID, use_auth_token=token)
    logger.info(f"说话人分离管道加载完成，耗时 {time.time() - t0:.2f}秒")

    pipeline.to(torch.device(device))
    # Apply user-provided hyperparameters (best-effort).
    if seg_min_off is not None or clust_thr is not None:
        params: dict[str, dict[str, float]] = {}
        if seg_min_off is not None:
            params.setdefault("segmentation", {})["min_duration_off"] = float(seg_min_off)
        if clust_thr is not None:
            params.setdefault("clustering", {})["threshold"] = float(clust_thr)
        try:
            pipeline.instantiate(params)
            logger.info(f"已应用说话人分离参数: {params}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"应用说话人分离参数失败（忽略，使用默认值）: {exc}")
    _DIARIZATION_PIPELINE = pipeline
    _DIARIZATION_KEY = key


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
        # Merge word-level timestamps only when both sides have them.
        buf_words = buffer.get("words")
        seg_words = segment.get("words")
        if buf_words is None or seg_words is None:
            buffer["words"] = None
        else:
            buffer["words"] = list(buf_words) + list(seg_words)

    if buffer is not None:
        merged.append(buffer)
    return merged


def generate_speaker_audio(folder: str, transcript: list[dict[str, Any]]) -> None:
    check_cancelled()
    wav_path = os.path.join(folder, "audio_vocals.wav")
    if not os.path.exists(wav_path):
        logger.warning(f"未找到音频文件: {wav_path}")
        return

    target_sr = 24000
    max_ref_seconds = read_speaker_ref_seconds()
    max_ref_samples = int(max_ref_seconds * float(target_sr))
    delay = 0.05
    speakers = {str(seg.get("speaker") or "SPEAKER_00") for seg in transcript}
    speaker_dict: dict[str, np.ndarray] = {}
    for spk in speakers:
        check_cancelled()
        speaker_dict[spk] = np.zeros((0,), dtype=np.float32)

    for segment in transcript:
        check_cancelled()
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
            check_cancelled()
            chunk, _sr = librosa.load(wav_path, sr=target_sr, mono=True, offset=offset, duration=duration)
        except Exception as exc:
            logger.warning(f"加载说话人音频块失败 (speaker={speaker}, offset={offset:.2f}秒): {exc}")
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
        check_cancelled()
        if speaker_dict[speaker].size > 0:
            continue
        try:
            check_cancelled()
            chunk, _sr = librosa.load(wav_path, sr=target_sr, mono=True, offset=0.0, duration=max_ref_seconds)
            if chunk.size > 0:
                speaker_dict[speaker] = chunk.astype(np.float32)
        except Exception:
            # Best-effort: keep empty
            continue

    for speaker, audio in speaker_dict.items():
        check_cancelled()
        if audio.size <= 0:
            continue
        speaker_file_path = os.path.join(speaker_folder, f"{speaker}.wav")
        save_wav(audio, speaker_file_path, sample_rate=target_sr)
        if max_ref_samples > 0 and audio.shape[0] >= max_ref_samples:
            logger.info(f"已保存说话人参考 ({max_ref_seconds:.1f}秒): {speaker_file_path}")


def _assign_speakers_by_overlap(
    segments: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    default_speaker: str = "SPEAKER_00",
) -> None:
    if not segments or not turns:
        for seg in segments:
            check_cancelled()
            seg["speaker"] = default_speaker
        return

    turns_sorted = sorted(turns, key=lambda x: float(x["start"]))
    idx = 0

    for seg in segments:
        check_cancelled()
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        if seg_end <= seg_start:
            seg["speaker"] = default_speaker
            continue

        while idx < len(turns_sorted) and float(turns_sorted[idx]["end"]) <= seg_start:
            check_cancelled()
            idx += 1

        best_speaker = None
        best_overlap = 0.0
        j = idx
        while j < len(turns_sorted) and float(turns_sorted[j]["start"]) < seg_end:
            check_cancelled()
            t = turns_sorted[j]
            ov = max(0.0, min(seg_end, float(t["end"])) - max(seg_start, float(t["start"])))
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = str(t.get("speaker") or default_speaker)
            j += 1

        seg["speaker"] = best_speaker if best_speaker is not None else default_speaker


def _read_download_info(folder: str) -> dict[str, Any] | None:
    info_path = os.path.join(folder, "download.info.json")
    if not os.path.exists(info_path):
        return None
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _pick_preferred_manual_subtitle_lang(info: dict[str, Any]) -> str | None:
    subtitles = info.get("subtitles")
    if not isinstance(subtitles, dict) or not subtitles:
        return None

    keys = [str(k) for k in subtitles.keys() if str(k).strip()]
    if not keys:
        return None

    preferred: list[str] = []
    for k in (info.get("language"), info.get("original_language")):
        if isinstance(k, str) and k.strip():
            preferred.append(k.strip())

    # 1) Prefer the video's language / original_language when available.
    for p in preferred:
        if p in subtitles:
            return p

    # 2) Prefer English (en / en-*)
    if "en" in subtitles:
        return "en"
    for k in keys:
        if k.lower().startswith("en"):
            return k

    # 3) Fallback: first available language key (stable).
    return keys[0]


_VTT_TS_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})\.(\d{3})$")
_SRT_TS_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2}),(\d{3})$")
_SUB_TAG_RE = re.compile(r"<[^>]+>")
_SUB_WS_RE = re.compile(r"\s+")


def _parse_vtt_timestamp(ts: str) -> float | None:
    s = (ts or "").strip()
    m = _VTT_TS_RE.match(s)
    if not m:
        return None
    hh = int(m.group(1) or 0)
    mm = int(m.group(2) or 0)
    ss = int(m.group(3) or 0)
    ms = int(m.group(4) or 0)
    return float(hh * 3600 + mm * 60 + ss) + float(ms) / 1000.0


def _parse_srt_timestamp(ts: str) -> float | None:
    s = (ts or "").strip()
    m = _SRT_TS_RE.match(s)
    if not m:
        return None
    hh = int(m.group(1) or 0)
    mm = int(m.group(2) or 0)
    ss = int(m.group(3) or 0)
    ms = int(m.group(4) or 0)
    return float(hh * 3600 + mm * 60 + ss) + float(ms) / 1000.0


def _clean_subtitle_text(text: str) -> str:
    s = (text or "").replace("\r", "\n")
    # Unescape entities first so `&lt;...&gt;` can be stripped by tag regex.
    try:
        s = html.unescape(s)
    except Exception:
        pass
    # Strip basic tags (WebVTT often contains <c>, <v>, <i>, etc).
    s = _SUB_TAG_RE.sub("", s)
    # Collapse whitespace across lines.
    s = _SUB_WS_RE.sub(" ", s).strip()
    return s


def _parse_vtt_segments(content: str) -> list[dict[str, Any]]:
    lines = (content or "").splitlines()
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0].lstrip("\ufeff")

    segs: list[dict[str, Any]] = []
    i = 0
    n = len(lines)

    # Skip header (WEBVTT + optional metadata until blank line).
    if i < n and lines[i].strip().upper().startswith("WEBVTT"):
        i += 1
        while i < n and lines[i].strip() != "":
            i += 1
    while i < n and lines[i].strip() == "":
        i += 1

    while i < n:
        check_cancelled()
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Skip NOTE/STYLE/REGION blocks.
        if line.startswith("NOTE") or line.startswith("STYLE") or line.startswith("REGION"):
            i += 1
            while i < n and lines[i].strip() != "":
                i += 1
            continue

        # Optional cue identifier line.
        if "-->" not in line and (i + 1) < n and ("-->" in lines[i + 1]):
            i += 1
            line = lines[i].strip()

        if "-->" not in line:
            i += 1
            continue

        left, right = line.split("-->", 1)
        start_ts = left.strip().split()[0] if left.strip() else ""
        end_ts = right.strip().split()[0] if right.strip() else ""
        start_s = _parse_vtt_timestamp(start_ts)
        end_s = _parse_vtt_timestamp(end_ts)
        if start_s is None or end_s is None:
            i += 1
            continue

        i += 1
        text_lines: list[str] = []
        while i < n and lines[i].strip() != "":
            text_lines.append(lines[i])
            i += 1

        text = _clean_subtitle_text("\n".join(text_lines))
        if text and end_s > start_s:
            segs.append(
                {
                    "start": round(float(start_s), 3),
                    "end": round(float(end_s), 3),
                    "text": text,
                    "speaker": "SPEAKER_00",
                }
            )

        while i < n and lines[i].strip() == "":
            i += 1

    return segs


def _parse_srt_segments(content: str) -> list[dict[str, Any]]:
    lines = (content or "").splitlines()
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0].lstrip("\ufeff")

    segs: list[dict[str, Any]] = []
    i = 0
    n = len(lines)
    while i < n:
        check_cancelled()
        if not lines[i].strip():
            i += 1
            continue

        # Optional numeric index line
        if lines[i].strip().isdigit():
            i += 1
            if i >= n:
                break

        if "-->" not in lines[i]:
            i += 1
            continue

        left, right = lines[i].split("-->", 1)
        start_ts = left.strip()
        end_ts = right.strip().split()[0] if right.strip() else ""
        start_s = _parse_srt_timestamp(start_ts)
        end_s = _parse_srt_timestamp(end_ts)
        i += 1

        text_lines: list[str] = []
        while i < n and lines[i].strip() != "":
            text_lines.append(lines[i])
            i += 1

        if start_s is None or end_s is None:
            continue

        text = _clean_subtitle_text("\n".join(text_lines))
        if text and end_s > start_s:
            segs.append(
                {
                    "start": round(float(start_s), 3),
                    "end": round(float(end_s), 3),
                    "text": text,
                    "speaker": "SPEAKER_00",
                }
            )

    return segs


def _find_subtitle_file(folder: str, preferred_lang: str | None) -> tuple[str | None, str | None]:
    """
    Find a parseable subtitle file written by yt-dlp under `folder`.

    Returns: (file_path, detected_lang)
    """
    exts = (".vtt", ".srt")

    def _candidate(lang: str, ext: str) -> str:
        return os.path.join(folder, f"download.{lang}{ext}")

    if preferred_lang:
        for ext in exts:
            p = _candidate(preferred_lang, ext)
            if os.path.exists(p):
                return p, preferred_lang

    # Fallback: scan folder for any download.*.vtt/srt
    try:
        names = [
            n
            for n in os.listdir(folder)
            if n.startswith("download.")
            and n.lower().endswith(exts)
            # Ignore auto captions artifacts (yt-dlp typically uses *.auto.vtt).
            and (".auto." not in n.lower())
            and (not n.lower().endswith(".auto.vtt"))
            and (not n.lower().endswith(".auto.srt"))
        ]
    except Exception:
        names = []

    def _lang_from_name(name: str) -> str:
        low = name
        for ext in exts:
            if low.lower().endswith(ext):
                low = low[: -len(ext)]
                break
        # download.<lang>
        if low.startswith("download."):
            return low[len("download.") :]
        return ""

    candidates: list[tuple[str, str]] = []
    for n in sorted(names):
        lang = _lang_from_name(n).strip()
        candidates.append((os.path.join(folder, n), lang))

    if not candidates:
        return None, None

    if preferred_lang:
        # Case-insensitive match.
        for p, lang in candidates:
            if lang.lower() == preferred_lang.lower():
                return p, lang
        for p, lang in candidates:
            ll = lang.lower()
            pl = preferred_lang.lower()
            if ll.startswith(pl) and (len(ll) == len(pl) or ll[len(pl)] in {"-", "_"}):
                return p, lang

    # Prefer English if present.
    for p, lang in candidates:
        ll = lang.lower()
        if ll == "en" or (ll.startswith("en") and (len(ll) == 2 or ll[2] in {"-", "_"})):
            return p, lang

    return candidates[0]


def _parse_subtitle_file(path: str) -> list[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return []

    low = path.lower()
    if low.endswith(".vtt"):
        return _parse_vtt_segments(content)
    if low.endswith(".srt"):
        return _parse_srt_segments(content)
    return []


def _is_youtube_subtitle_transcript(transcript: Any, *, lang: str | None = None) -> bool:
    if not isinstance(transcript, list) or not transcript:
        return False
    # Best-effort marker: we store source/lang per-segment.
    head = transcript[0]
    if not isinstance(head, dict):
        return False
    if str(head.get("source") or "") != "youtube_subtitles":
        return False
    if lang is None:
        return True
    return str(head.get("subtitle_lang") or "") == str(lang)


def _save_youtube_subtitle_transcript(
    folder: str,
    segments: list[dict[str, Any]],
    *,
    subtitle_lang: str | None,
    diarization: bool,
    device: str,
    min_speakers: int | None,
    max_speakers: int | None,
    settings: Settings | None,
    model_manager: ModelManager | None,
) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = segments
    for seg in transcript:
        seg["source"] = "youtube_subtitles"
        seg["subtitle_lang"] = str(subtitle_lang or "")

    # If diarization requested, still compute speakers and align them to subtitle segments.
    if diarization:
        wav_path_existing = os.path.join(folder, "audio_vocals.wav")
        if not os.path.exists(wav_path_existing):
            logger.warning(f"检测到字幕但缺少人声轨，无法说话人分离: {wav_path_existing}")
            for seg in transcript:
                seg["speaker"] = "SPEAKER_00"
        else:
            try:
                check_cancelled()
                load_diarize_model(device=device, settings=settings, model_manager=model_manager)
                assert _DIARIZATION_PIPELINE is not None
                model_tag = (_DIARIZATION_KEY.split("|")[-1] if _DIARIZATION_KEY else "pyannote")
                logger.info(f"字幕转录：计算说话人分离 ({model_tag})...")
                check_cancelled()
                ann = _DIARIZATION_PIPELINE(
                    wav_path_existing, min_speakers=min_speakers, max_speakers=max_speakers
                )
                ann_view = getattr(ann, "exclusive_speaker_diarization", None) or ann
                turns: list[dict[str, Any]] = []
                for seg, _, speaker in ann_view.itertracks(yield_label=True):
                    check_cancelled()
                    turns.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(speaker)})
                _assign_speakers_by_overlap(transcript, turns)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(f"说话人分离失败，回退到单说话人: {exc}")
                for seg in transcript:
                    seg["speaker"] = "SPEAKER_00"

    check_cancelled()
    transcript = merge_segments(transcript)

    check_cancelled()
    transcript_path = os.path.join(folder, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    logger.info(f"已保存字幕转录: {transcript_path}")

    # Best-effort: refresh speaker reference files.
    try:
        check_cancelled()
        generate_speaker_audio(folder, transcript)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(f"生成说话人参考音频失败（忽略）: {exc}")

    return transcript


def _invalidate_downstream_cache_for_new_transcript(folder: str) -> None:
    """
    When transcript.json is replaced, downstream artifacts (translation/TTS) become invalid.

    Best-effort cleanup only; failures should not block the pipeline.
    """

    # Translation artifacts
    rm_files = [
        os.path.join(folder, "translation.json"),
        os.path.join(folder, "translation_raw.json"),
        os.path.join(folder, "summary.json"),
        os.path.join(folder, "transcript_punctuated.json"),
    ]
    removed = 0
    for p in rm_files:
        try:
            if os.path.exists(p):
                os.remove(p)
                removed += 1
        except Exception:
            continue

    # TTS artifacts under wavs/
    wavs_dir = os.path.join(folder, "wavs")
    wavs_removed = 0
    try:
        if os.path.isdir(wavs_dir):
            for ent in os.scandir(wavs_dir):
                if not ent.is_file():
                    continue
                name = ent.name.lower()
                if name.endswith(".wav") or name.endswith(".json"):
                    try:
                        os.remove(ent.path)
                        wavs_removed += 1
                    except Exception:
                        continue
    except Exception:
        wavs_removed = 0

    if removed or wavs_removed:
        logger.info(f"已清理下游缓存: {folder} (translation={removed}, wavs={wavs_removed})")


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
    asr_method: str | None = None,
    qwen_asr_model_dir: str | None = None,
    qwen_asr_num_threads: int | None = None,
    qwen_asr_vad_segment_threshold: int | None = None,
) -> bool:
    check_cancelled()
    transcript_path = os.path.join(folder, "transcript.json")

    # Prefer YouTube manual subtitles (NOT auto captions) when available.
    info = _read_download_info(folder)
    preferred_lang = _pick_preferred_manual_subtitle_lang(info) if info else None
    subtitle_path, detected_lang = _find_subtitle_file(folder, preferred_lang)
    subtitle_segments: list[dict[str, Any]] | None = None
    chosen_lang = detected_lang or preferred_lang
    if info and isinstance(info.get("subtitles"), dict) and info.get("subtitles") and subtitle_path:
        try:
            subtitle_segments = _parse_subtitle_file(subtitle_path)
        except Exception as exc:  # pylint: disable=broad-except
            subtitle_segments = None
            logger.warning(f"解析字幕失败，将回退到ASR: {subtitle_path} ({exc})")

    if os.path.exists(transcript_path):
        transcript: Any | None = None
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
            if not isinstance(transcript, list):
                raise ValueError("transcript.json is not a list")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"转录文件无效，将重新生成: {transcript_path} ({exc})")
            try:
                os.remove(transcript_path)
            except Exception:
                pass
        else:
            # If manual subtitles are present, override non-subtitle transcripts.
            if subtitle_segments:
                # No-op when transcript already came from the same subtitle language.
                if not _is_youtube_subtitle_transcript(transcript, lang=chosen_lang):
                    logger.info("检测到人工字幕，将使用字幕替换ASR转录")
                    _invalidate_downstream_cache_for_new_transcript(folder)
                    _save_youtube_subtitle_transcript(
                        folder,
                        subtitle_segments,
                        subtitle_lang=chosen_lang,
                        diarization=bool(diarization),
                        device=device,
                        min_speakers=min_speakers,
                        max_speakers=max_speakers,
                        settings=settings,
                        model_manager=model_manager,
                    )
                    return True

            logger.info(f"转录已存在于 {folder}")
            # Optional: re-run diarization on existing transcript to refresh speaker labels.
            # IMPORTANT: do not change segmentation/text here (keep downstream alignment stable).
            if diarization:
                wav_path_existing = os.path.join(folder, "audio_vocals.wav")
                if not os.path.exists(wav_path_existing):
                    logger.warning(f"转录已存在但缺少人声轨，无法重新说话人分离: {wav_path_existing}")
                else:
                    try:
                        check_cancelled()
                        load_diarize_model(device=device, settings=settings, model_manager=model_manager)
                        assert _DIARIZATION_PIPELINE is not None
                        model_tag = (_DIARIZATION_KEY.split("|")[-1] if _DIARIZATION_KEY else "pyannote")
                        logger.info(f"转录已存在，重新计算说话人分离 ({model_tag})...")
                        check_cancelled()
                        ann = _DIARIZATION_PIPELINE(
                            wav_path_existing, min_speakers=min_speakers, max_speakers=max_speakers
                        )
                        ann_view = getattr(ann, "exclusive_speaker_diarization", None) or ann
                        turns: list[dict[str, Any]] = []
                        for seg, _, speaker in ann_view.itertracks(yield_label=True):
                            check_cancelled()
                            turns.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(speaker)})
                        _assign_speakers_by_overlap(transcript, turns)

                        check_cancelled()
                        with open(transcript_path, "w", encoding="utf-8") as f:
                            json.dump(transcript, f, indent=2, ensure_ascii=False)
                        logger.info(f"已更新转录说话人标签: {transcript_path}")

                        # Speaker refs are derived from speaker labels; refresh them best-effort.
                        try:
                            check_cancelled()
                            generate_speaker_audio(folder, transcript)
                        except Exception as exc:  # pylint: disable=broad-except
                            logger.warning(f"生成说话人参考音频失败（忽略）: {exc}")
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.warning(f"重新说话人分离失败，保留原转录说话人标签: {exc}")

            # Ensure speaker reference files exist even when transcript step is skipped.
            try:
                speakers = {str(seg.get("speaker") or "SPEAKER_00") for seg in transcript}
                speaker_dir = os.path.join(folder, "SPEAKER")
                need = False
                for spk in speakers:
                    check_cancelled()
                    p = os.path.join(speaker_dir, f"{spk}.wav")
                    if not os.path.exists(p) or os.path.getsize(p) < 44:
                        need = True
                        break
                if need:
                    generate_speaker_audio(folder, transcript)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(f"确保说话人参考音频失败: {exc}")
            return True

    # transcript.json 不存在：若有人工字幕，直接用字幕生成转录并跳过 ASR。
    if subtitle_segments:
        _invalidate_downstream_cache_for_new_transcript(folder)
        _save_youtube_subtitle_transcript(
            folder,
            subtitle_segments,
            subtitle_lang=chosen_lang,
            diarization=bool(diarization),
            device=device,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            settings=settings,
            model_manager=model_manager,
        )
        return True

    wav_path = os.path.join(folder, "audio_vocals.wav")
    if not os.path.exists(wav_path):
        return False

    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER

    method = (asr_method or getattr(settings, "asr_method", "whisper") or "whisper").strip().lower()
    if method not in ("whisper", "qwen"):
        method = "whisper"

    if model_name is None:
        model_name = str(settings.whisper_model_path)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    transcript: list[dict[str, Any]] = []

    if method == "qwen":
        # Qwen3-ASR: chunk the audio (seconds) and approximate timestamps per sentence.
        qwen_model_dir = (qwen_asr_model_dir or str(getattr(settings, "qwen_asr_model_path", "") or "")).strip()
        qwen_threads = int(qwen_asr_num_threads or getattr(settings, "qwen_asr_num_threads", 1) or 1)
        qwen_chunk_s = int(qwen_asr_vad_segment_threshold or getattr(settings, "qwen_asr_vad_segment_threshold", 60) or 60)
        qwen_threads = max(1, min(qwen_threads, 32))
        qwen_chunk_s = max(5, qwen_chunk_s)

        load_qwen_asr_model(
            model_dir=qwen_model_dir,
            device=device,
            settings=settings,
            model_manager=model_manager,
            max_inference_batch_size=1,
        )
        assert _QWEN_ASR_MODEL is not None

        logger.info(f"转录中(Qwen3-ASR) {wav_path}")
        t0 = time.time()

        dur = wav_duration_seconds(wav_path)
        if dur is None:
            # Fallback: use librosa header-based duration.
            try:
                dur = float(librosa.get_duration(filename=wav_path))
            except Exception:
                dur = None

        if dur is None:
            # Last resort: treat whole file as one chunk without timestamps.
            dur = float(qwen_chunk_s)

        starts = [float(x) for x in np.arange(0.0, float(dur) + 1e-6, float(qwen_chunk_s)).tolist()]

        def _run_chunk(start_s: float) -> tuple[float, float, str, str | None]:
            check_cancelled()
            logger.info(f"Qwen3-ASR: 处理 chunk offset={start_s:.1f}s / {dur:.1f}s")
            # Load chunk as 16k mono.
            chunk, sr = librosa.load(wav_path, sr=16000, mono=True, offset=max(0.0, float(start_s)), duration=float(qwen_chunk_s))
            if chunk.size <= 0:
                return float(start_s), float(start_s), "", None
            chunk_dur = float(chunk.shape[0]) / float(sr)
            end_s = float(start_s) + chunk_dur
            check_cancelled()
            results = _QWEN_ASR_MODEL.transcribe(audio=(chunk, int(sr)), language=None)  # type: ignore[attr-defined]
            if not results:
                return float(start_s), float(end_s), "", None
            r0 = results[0]
            text = str(getattr(r0, "text", "") or "").strip()
            lang = getattr(r0, "language", None)
            lang_str = str(lang) if lang is not None else None
            return float(start_s), float(end_s), text, lang_str

        chunk_results: list[tuple[float, float, str, str | None]] = []
        if qwen_threads <= 1 or len(starts) <= 1:
            for st in starts:
                check_cancelled()
                chunk_results.append(_run_chunk(float(st)))
        else:
            with ThreadPoolExecutor(max_workers=qwen_threads) as ex:
                futs = [ex.submit(_run_chunk, float(st)) for st in starts]
                for fut in as_completed(futs):
                    check_cancelled()
                    chunk_results.append(fut.result())

        chunk_results.sort(key=lambda x: x[0])
        lang_seen: str | None = None
        for st, ed, txt, lang in chunk_results:
            check_cancelled()
            if lang and not lang_seen:
                lang_seen = lang
            if not txt:
                continue
            transcript.extend(_split_text_with_timing(txt, st, ed))

        logger.info(
            f"ASR完成(Qwen3-ASR)，耗时 {time.time() - t0:.2f}秒 (段数={len(transcript)}, 语言={lang_seen})"
        )
    else:
        if device == "cpu":
            # Prefer explicit CPU model path (UI arg) -> settings default -> fallback to model_name.
            cpu_candidate = (cpu_model_name or "").strip()
            if not cpu_candidate:
                cpu_dir = getattr(settings, "whisper_cpu_model_path", None)
                cpu_candidate = str(cpu_dir).strip() if cpu_dir else ""
            if cpu_candidate:
                model_name = cpu_candidate

        check_cancelled()
        load_asr_model(
            model_dir=model_name,
            device=device,
            settings=settings,
            model_manager=model_manager,
            use_batched=True,
        )

        assert _ASR_MODEL is not None

        logger.info(f"转录中 {wav_path}")
        t0 = time.time()

        _preload_cudnn_for_onnxruntime_gpu()

        # VAD: 使用 faster-whisper 默认参数（避免过度切分导致重复/幻觉）
        # Prefer batched pipeline if available (much higher throughput on GPU).
        if _ASR_PIPELINE is not None:
            check_cancelled()
            segments_iter, info = _ASR_PIPELINE.transcribe(
                wav_path,
                batch_size=batch_size,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                word_timestamps=True,  # 开启词级时间戳
                no_speech_threshold=0.8,  # 提高阈值，防止漏句
            )
        else:
            check_cancelled()
            segments_iter, info = _ASR_MODEL.transcribe(
                wav_path,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                word_timestamps=True,  # 开启词级时间戳
                no_speech_threshold=0.8,  # 提高阈值，防止漏句
            )

        segments_list = []
        for seg in segments_iter:
            check_cancelled()
            segments_list.append(seg)
        logger.info(
            f"ASR完成，耗时 {time.time() - t0:.2f}秒 (段数={len(segments_list)}, 语言={getattr(info, 'language', None)})"
        )

        for seg in segments_list:
            check_cancelled()
            text = (getattr(seg, "text", "") or "").strip()
            if not text:
                continue
            words = getattr(seg, "words", None)
            words_data: list[dict[str, Any]] | None = None
            if words:
                words_data = []
                for w in words:
                    if w is None:
                        continue
                    words_data.append(
                        {
                            "start": float(getattr(w, "start", 0.0) or 0.0),
                            "end": float(getattr(w, "end", 0.0) or 0.0),
                            "word": str(getattr(w, "word", "") or ""),
                            "probability": (
                                float(getattr(w, "probability", 0.0) or 0.0)
                                if getattr(w, "probability", None) is not None
                                else None
                            ),
                        }
                    )
            transcript.append(
                {
                    "start": float(getattr(seg, "start", 0.0)),
                    "end": float(getattr(seg, "end", 0.0)),
                    "text": text,
                    "speaker": "SPEAKER_00",
                    "words": words_data,
                }
            )

    if diarization:
        check_cancelled()
        try:
            load_diarize_model(device=device, settings=settings, model_manager=model_manager)
            assert _DIARIZATION_PIPELINE is not None
            model_tag = (_DIARIZATION_KEY.split("|")[-1] if _DIARIZATION_KEY else "pyannote")
            logger.info(f"开始说话人分离 ({model_tag})...")
            check_cancelled()
            ann = _DIARIZATION_PIPELINE(wav_path, min_speakers=min_speakers, max_speakers=max_speakers)
            # Some pyannote pipelines provide a non-overlapping ("exclusive") diarization view which
            # generally aligns better with ASR timestamps. Fall back to legacy tracks when unavailable.
            ann_view = getattr(ann, "exclusive_speaker_diarization", None) or ann
            turns: list[dict[str, Any]] = []
            for seg, _, speaker in ann_view.itertracks(yield_label=True):
                check_cancelled()
                turns.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(speaker)})
            _assign_speakers_by_overlap(transcript, turns)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"说话人分离失败，回退到单说话人: {exc}")
            for item in transcript:
                item["speaker"] = "SPEAKER_00"

    check_cancelled()
    transcript = merge_segments(transcript)

    check_cancelled()
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    logger.info(f"已保存转录: {transcript_path}")

    check_cancelled()
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
    asr_method: str | None = None,
    qwen_asr_model_dir: str | None = None,
    qwen_asr_num_threads: int | None = None,
    qwen_asr_vad_segment_threshold: int | None = None,
) -> str:
    check_cancelled()
    count = 0
    for root, _, files in os.walk(folder):
        check_cancelled()
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
            asr_method=asr_method,
            qwen_asr_model_dir=qwen_asr_model_dir,
            qwen_asr_num_threads=qwen_asr_num_threads,
            qwen_asr_vad_segment_threshold=qwen_asr_vad_segment_threshold,
        )
        if ok:
            count += 1
    msg = f"转录完成: {folder}（处理 {count} 个文件）"
    logger.info(msg)
    return msg
