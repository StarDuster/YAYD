from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from ..config import Settings
from ..interrupts import check_cancelled
from ..models import ModelCheckError, ModelManager
from ..utils import torch_load_weights_only_compat
from .transcribe_asr_run import run_qwen_asr, run_whisper_asr
from .transcribe_assets import ensure_assets
from .transcribe_compat import (
    patch_torchaudio_backend_compat as _patch_torchaudio_backend_compat,
    preload_cudnn_for_onnxruntime_gpu as _preload_cudnn_for_onnxruntime_gpu,
)
from .transcribe_segments import _assign_speakers_by_overlap, generate_speaker_audio, merge_segments
from .transcribe_speaker_repair import (
    _repair_speakers_by_embedding,
    _speaker_repair_enabled,
    _update_translation_speakers_from_transcript,
)
from .transcribe_subtitles import (
    _find_subtitle_file,
    _invalidate_downstream_cache_for_new_transcript,
    _is_youtube_subtitle_transcript,
    _parse_subtitle_file,
    _pick_preferred_manual_subtitle_lang,
    _read_download_info,
    _save_youtube_subtitle_transcript,
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

    ensure_assets(settings, model_manager, require_diarization=False)

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

    ensure_assets(settings, model_manager, require_diarization=False)

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

    ensure_assets(settings, model_manager, require_diarization=True)

    _patch_torchaudio_backend_compat()

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
    # - segmentation.threshold
    # - segmentation.min_duration_off
    # - clustering.threshold
    seg_thr = _read_float_env("PYANNOTE_SEGMENTATION_THRESHOLD")
    seg_min_off = _read_float_env("PYANNOTE_SEGMENTATION_MIN_DURATION_OFF")
    clust_thr = _read_float_env("PYANNOTE_CLUSTERING_THRESHOLD")

    diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
    diar_dir_str = str(diar_dir) if diar_dir else ""
    cache_dir = diar_dir_str or None
    key = (
        f"{device}|{diar_dir_str}|{_PYANNOTE_DIARIZATION_MODEL_ID}"
        f"|seg_thr={seg_thr if seg_thr is not None else 'default'}"
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
    if seg_thr is not None or seg_min_off is not None or clust_thr is not None:
        params: dict[str, dict[str, float]] = {}
        if seg_thr is not None:
            params.setdefault("segmentation", {})["threshold"] = float(seg_thr)
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


def _compute_diarization_turns(
    wav_path: str,
    *,
    device: str,
    min_speakers: int | None,
    max_speakers: int | None,
    settings: Settings | None,
    model_manager: ModelManager | None,
    log_hint: str,
) -> list[dict[str, Any]] | None:
    if not os.path.exists(wav_path):
        return None
    try:
        check_cancelled()
        load_diarize_model(device=device, settings=settings, model_manager=model_manager)
        assert _DIARIZATION_PIPELINE is not None
        model_tag = (_DIARIZATION_KEY.split("|")[-1] if _DIARIZATION_KEY else "pyannote")
        logger.info(f"{log_hint} ({model_tag})...")
        check_cancelled()
        ann = _DIARIZATION_PIPELINE(wav_path, min_speakers=min_speakers, max_speakers=max_speakers)
        ann_view = getattr(ann, "exclusive_speaker_diarization", None) or ann
        turns: list[dict[str, Any]] = []
        for seg, _, speaker in ann_view.itertracks(yield_label=True):
            check_cancelled()
            turns.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(speaker)})
        return turns
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(f"{log_hint}失败: {exc}")
        return None


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

                    turns: list[dict[str, Any]] | None = None
                    if diarization:
                        wav_path_existing = os.path.join(folder, "audio_vocals.wav")
                        if not os.path.exists(wav_path_existing):
                            logger.warning(f"检测到字幕但缺少人声轨，无法说话人分离: {wav_path_existing}")
                        else:
                            turns = _compute_diarization_turns(
                                wav_path_existing,
                                device=device,
                                min_speakers=min_speakers,
                                max_speakers=max_speakers,
                                settings=settings,
                                model_manager=model_manager,
                                log_hint="字幕转录：计算说话人分离",
                            )

                    _save_youtube_subtitle_transcript(
                        folder,
                        subtitle_segments,
                        subtitle_lang=chosen_lang,
                        turns=turns,
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
                        turns = _compute_diarization_turns(
                            wav_path_existing,
                            device=device,
                            min_speakers=min_speakers,
                            max_speakers=max_speakers,
                            settings=settings,
                            model_manager=model_manager,
                            log_hint="转录已存在，重新计算说话人分离",
                        )
                        if turns:
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

            # Optional: embedding-based speaker repair (can run even if re-diarization fails).
            if _speaker_repair_enabled():
                wav_path_existing = os.path.join(folder, "audio_vocals.wav")
                if not os.path.exists(wav_path_existing):
                    logger.warning(f"说话人修复: 缺少人声轨，跳过: {wav_path_existing}")
                else:
                    try:
                        check_cancelled()
                        settings2 = settings or _DEFAULT_SETTINGS
                        mm2 = model_manager or _DEFAULT_MODEL_MANAGER
                        _repair_speakers_by_embedding(
                            folder=folder,
                            transcript=transcript,  # type: ignore[arg-type]
                            wav_path=wav_path_existing,
                            settings=settings2,
                            model_manager=mm2,
                            device=device,
                        )
                        _update_translation_speakers_from_transcript(folder, transcript)  # type: ignore[arg-type]
                        # Persist repaired transcript (even if no changes were applied, keep formatting stable).
                        check_cancelled()
                        with open(transcript_path, "w", encoding="utf-8") as f:
                            json.dump(transcript, f, indent=2, ensure_ascii=False)
                        logger.info(f"说话人修复: 已写回转录: {transcript_path}")
                    except Exception as exc_rep:  # pylint: disable=broad-except
                        logger.warning(f"说话人修复失败（忽略，保留原转录）: {exc_rep}")

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
                    generate_speaker_audio(folder, transcript)  # type: ignore[arg-type]
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(f"确保说话人参考音频失败: {exc}")
            return True

    # transcript.json 不存在：若有人工字幕，直接用字幕生成转录并跳过 ASR。
    if subtitle_segments:
        _invalidate_downstream_cache_for_new_transcript(folder)
        turns: list[dict[str, Any]] | None = None
        if diarization:
            wav_path_existing = os.path.join(folder, "audio_vocals.wav")
            if not os.path.exists(wav_path_existing):
                logger.warning(f"检测到字幕但缺少人声轨，无法说话人分离: {wav_path_existing}")
            else:
                turns = _compute_diarization_turns(
                    wav_path_existing,
                    device=device,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    settings=settings,
                    model_manager=model_manager,
                    log_hint="字幕转录：计算说话人分离",
                )
        _save_youtube_subtitle_transcript(folder, subtitle_segments, subtitle_lang=chosen_lang, turns=turns)
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

    transcript_out: list[dict[str, Any]] = []

    if method == "qwen":
        qwen_model_dir2 = (qwen_asr_model_dir or str(getattr(settings, "qwen_asr_model_path", "") or "")).strip()
        qwen_threads = int(qwen_asr_num_threads or getattr(settings, "qwen_asr_num_threads", 1) or 1)
        qwen_chunk_s = int(qwen_asr_vad_segment_threshold or getattr(settings, "qwen_asr_vad_segment_threshold", 60) or 60)

        load_qwen_asr_model(
            model_dir=qwen_model_dir2,
            device=device,
            settings=settings,
            model_manager=model_manager,
            max_inference_batch_size=1,
        )
        assert _QWEN_ASR_MODEL is not None

        transcript_out, _lang_seen = run_qwen_asr(
            wav_path,
            model=_QWEN_ASR_MODEL,
            num_threads=qwen_threads,
            vad_segment_threshold_seconds=qwen_chunk_s,
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

        _preload_cudnn_for_onnxruntime_gpu()
        transcript_out, _lang = run_whisper_asr(
            wav_path,
            asr_model=_ASR_MODEL,
            asr_pipeline=_ASR_PIPELINE,
            batch_size=batch_size,
        )

    if diarization:
        turns = _compute_diarization_turns(
            wav_path,
            device=device,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            settings=settings,
            model_manager=model_manager,
            log_hint="开始说话人分离",
        )
        if turns:
            _assign_speakers_by_overlap(transcript_out, turns)
        else:
            for item in transcript_out:
                item["speaker"] = "SPEAKER_00"

    check_cancelled()
    transcript_out = merge_segments(transcript_out)

    # Optional: embedding-based speaker repair (must not change segmentation/count).
    if diarization and _speaker_repair_enabled():
        try:
            check_cancelled()
            _repair_speakers_by_embedding(
                folder=folder,
                transcript=transcript_out,
                wav_path=wav_path,
                settings=settings,
                model_manager=model_manager,
                device=device,
            )
            _update_translation_speakers_from_transcript(folder, transcript_out)
        except Exception as exc_rep:  # pylint: disable=broad-except
            logger.warning(f"说话人修复失败（忽略，保留 diarization 结果）: {exc_rep}")

    check_cancelled()
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript_out, f, indent=2, ensure_ascii=False)
    logger.info(f"已保存转录: {transcript_path}")

    check_cancelled()
    generate_speaker_audio(folder, transcript_out)
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

