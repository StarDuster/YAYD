import gc
import os
import time
from typing import Any

import torch
from loguru import logger

from ...config import Settings
from ...models import ModelManager
from ..utils import save_wav


class DemucsDependencyError(RuntimeError):
    pass


_DEMUCS_MODEL: Any | None = None
_DEMUCS_MODEL_NAME: str | None = None
_DEMUCS_DEVICE: str | None = None

_AUTO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _import_demucs_infer() -> tuple[Any, Any]:
    try:
        from demucs_infer.apply import apply_model
        from demucs_infer.pretrained import get_model
    except ModuleNotFoundError as exc:
        raise DemucsDependencyError(
            "缺少依赖 `demucs-infer`：请先安装后再使用“人声分离(Demucs)”功能。\n"
            "建议：`uv sync`（或 `pip install demucs-infer`）"
        ) from exc
    return get_model, apply_model


def _ensure_demucs_ready(settings: Settings, model_manager: ModelManager) -> None:
    model_manager.enforce_offline()
    model_manager.ensure_ready(names=[model_manager._demucs_requirement().name])


def _set_torch_hub_dir(settings: Settings) -> None:
    demucs_dir = settings.resolve_path(settings.demucs_model_dir)
    if not demucs_dir:
        return
    try:
        os.makedirs(demucs_dir, exist_ok=True)
        torch.hub.set_dir(str(demucs_dir))
        logger.info(f"Set torch hub dir to {demucs_dir}")
    except Exception as e:
        logger.warning(f"Failed to set torch hub dir: {e}")


def init_demucs(settings: Settings | None = None, model_manager: ModelManager | None = None) -> None:
    """Pre-load the Demucs model."""
    settings = settings or Settings()
    model_manager = model_manager or ModelManager(settings)
    _ensure_demucs_ready(settings, model_manager)
    load_model(settings.demucs_model_name, settings.demucs_device, settings=settings, model_manager=model_manager)


def load_model(
    model_name: str = "htdemucs_ft",
    device: str = "auto",
    progress: bool = True,  # kept for API compatibility
    shifts: int = 5,  # kept for API compatibility
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> Any:
    global _DEMUCS_MODEL, _DEMUCS_MODEL_NAME, _DEMUCS_DEVICE

    if settings and model_manager:
        _ensure_demucs_ready(settings, model_manager)
        _set_torch_hub_dir(settings)

    target_device = _AUTO_DEVICE if device == "auto" else device

    if _DEMUCS_MODEL is not None and _DEMUCS_MODEL_NAME == model_name and _DEMUCS_DEVICE == target_device:
        return _DEMUCS_MODEL

    unload_model()

    get_model, _apply_model = _import_demucs_infer()

    logger.info(f"Loading Demucs model: {model_name}")
    t_start = time.time()
    model = get_model(model_name)
    _DEMUCS_MODEL = model
    _DEMUCS_MODEL_NAME = model_name
    _DEMUCS_DEVICE = target_device
    logger.info(f"Demucs model loaded in {time.time() - t_start:.2f} seconds")
    return model


def unload_model() -> None:
    global _DEMUCS_MODEL, _DEMUCS_MODEL_NAME, _DEMUCS_DEVICE
    if _DEMUCS_MODEL is None:
        return

    try:
        del _DEMUCS_MODEL
    finally:
        _DEMUCS_MODEL = None
        _DEMUCS_MODEL_NAME = None
        _DEMUCS_DEVICE = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Demucs model unloaded")


def separate_audio(
    folder: str,
    model_name: str = "htdemucs_ft",
    device: str = "auto",
    progress: bool = True,
    shifts: int = 5,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> None:
    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        return

    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')
    
    if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
        logger.info(f'Audio already separated in {folder}')
        return
    
    settings = settings or Settings()
    model_manager = model_manager or ModelManager(settings)
    _ensure_demucs_ready(settings, model_manager)

    logger.info(f"Separating audio from {folder}")

    model = load_model(model_name, device, progress, shifts, settings=settings, model_manager=model_manager)
    _get_model, apply_model = _import_demucs_infer()

    import torchaudio

    # Try setting soundfile backend which is more stable than sox_io for large files
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass

    target_device = _AUTO_DEVICE if device == "auto" else device

    t_start = time.time()
    try:
        logger.info(f"Loading audio from {audio_path}...")
        wav, sr = torchaudio.load(audio_path)
        logger.info(f"Audio loaded. Shape: {wav.shape}, SR: {sr}")

        target_sr = int(getattr(model, "samplerate", sr))
        target_channels = int(getattr(model, "audio_channels", 2))

        if sr != target_sr:
            logger.info(f"Resampling from {sr} to {target_sr}...")
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)
            sr = target_sr

        if wav.shape[0] > target_channels:
            wav = wav[:target_channels, :]
        elif wav.shape[0] < target_channels:
            wav = wav.repeat(target_channels, 1)

        mix = wav.unsqueeze(0)  # [1, C, T]

        logger.info(f"Running Demucs ({model_name}) on {target_device} (shifts={shifts})...")
        # apply_model handles its own chunking/overlap; no need for our 300s chunk loop.
        sources = apply_model(model, mix, shifts=shifts, split=True, progress=progress, device=target_device)
        sources = sources[0]  # [S, C, T]

        source_names = list(getattr(model, "sources", []))
        if "vocals" in source_names:
            vocals_idx = source_names.index("vocals")
        else:
            # Fallback: assume last stem is vocals.
            vocals_idx = sources.shape[0] - 1

        vocals_tensor = sources[vocals_idx]
        instruments_tensor = sources.sum(dim=0) - vocals_tensor

        vocals = vocals_tensor.cpu().numpy().T
        instruments = instruments_tensor.cpu().numpy().T
    except Exception as e:
        logger.error(f"Error separating audio from {folder}: {e}")
        raise

    logger.info(f"Audio separated in {time.time() - t_start:.2f} seconds")

    logger.info(f"Saving vocals to {vocal_output_path}...")
    save_wav(vocals, vocal_output_path, sample_rate=sr)

    logger.info(f"Saving instruments to {instruments_output_path}...")
    save_wav(instruments, instruments_output_path, sample_rate=sr)

    logger.info("Separation task finished completely.")




def extract_audio_from_video(folder: str) -> bool:
    video_path = os.path.join(folder, 'download.mp4')
    if not os.path.exists(video_path):
        return False
        
    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        logger.info(f'Audio already extracted in {folder}')
        return True
        
    logger.info(f'Extracting audio from {folder}')

    # Use ffmpeg
    cmd = f'ffmpeg -loglevel error -y -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_path}"'
    ret = os.system(cmd)
    
    if ret != 0:
        logger.error(f"FFmpeg failed to extract audio from {video_path}")
        return False

    time.sleep(1)
    logger.info(f'Audio extracted from {folder}')
    return True


def separate_all_audio_under_folder(
    root_folder: str,
    model_name: str = "htdemucs_ft",
    device: str = 'auto',
    progress: bool = True,
    shifts: int = 5,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> str:
    if settings is None:
        settings = Settings()
    if model_manager is None:
        model_manager = ModelManager(settings)

    _ensure_demucs_ready(settings, model_manager)
    
    # Pre-load model once before iterating
    load_model(model_name, device, progress, shifts, settings=settings, model_manager=model_manager)

    count = 0
    for subdir, _dirs, files in os.walk(root_folder):
        if 'download.mp4' not in files:
            continue
            
        if 'audio.wav' not in files:
            extract_audio_from_video(subdir)
            
        if 'audio_vocals.wav' not in files:
            separate_audio(subdir, model_name, device, progress, shifts, settings=settings, model_manager=model_manager)
            count += 1

    msg = f'All audio separated under {root_folder} (processed {count} files)'
    logger.info(msg)
    return msg
