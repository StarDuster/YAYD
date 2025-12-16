import os
import time
import shutil
import importlib
from typing import Any, TYPE_CHECKING
import torch
from loguru import logger

from ...config import Settings
from ...models import ModelManager
from ..utils import save_wav

if TYPE_CHECKING:  # pragma: no cover
    from demucs.api import Separator  # type: ignore[import-not-found]


class DemucsDependencyError(RuntimeError):
    pass


def _get_demucs_separator_class() -> type[Any]:
    try:
        api = importlib.import_module("demucs.api")
    except ModuleNotFoundError as exc:
        if exc.name == "demucs":
            raise DemucsDependencyError(
                "缺少可选依赖 `demucs`：请先安装后再使用“人声分离(Demucs)”功能。\n"
                "建议：`pip install git+https://github.com/facebookresearch/demucs`"
            ) from exc
        if exc.name == "demucs.api":
            raise DemucsDependencyError(
                "已安装 `demucs` 但缺少 `demucs.api`（版本不兼容）。\n"
                "请升级/重装 Demucs 后再使用“人声分离(Demucs)”功能，例如：\n"
                "`pip install -U --force-reinstall git+https://github.com/facebookresearch/demucs`"
            ) from exc
        raise

    separator = getattr(api, "Separator", None)
    if separator is None:
        raise DemucsDependencyError(
            "`demucs.api` 中未找到 `Separator`（版本不兼容）。\n"
            "请升级/重装 Demucs 后再使用“人声分离(Demucs)”功能，例如：\n"
            "`pip install -U --force-reinstall git+https://github.com/facebookresearch/demucs`"
        )
    return separator

# Global separator instance for caching
_SEPARATOR: Any | None = None
_AUTO_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _ensure_demucs_ready(settings: Settings, model_manager: ModelManager) -> None:
    model_manager.enforce_offline()
    model_manager.ensure_ready(names=[model_manager._demucs_requirement().name])


def init_demucs(settings: Settings | None = None, model_manager: ModelManager | None = None) -> None:
    """Pre-load the Demucs model."""
    global _SEPARATOR
    if settings is None:
        settings = Settings()
    if model_manager is None:
        model_manager = ModelManager(settings)
        
    _ensure_demucs_ready(settings, model_manager)
    load_model(settings.demucs_model_name, settings.demucs_device, settings=settings, model_manager=model_manager)


def load_model(
    model_name: str = "htdemucs_ft",
    device: str = 'auto',
    progress: bool = True,
    shifts: int = 5,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> Any:
    global _SEPARATOR
    if _SEPARATOR is not None:
        # Check if we need to reload due to changed parameters if needed, 
        # but for now we assume global reuse is fine if model_name matches.
        # But `demucs.api.Separator` doesn't expose model name easily maybe.
        # Let's just return if cached for simplicity as in original code.
        return _SEPARATOR

    if settings and model_manager:
        _ensure_demucs_ready(settings, model_manager)

    logger.info(f'Loading Demucs model: {model_name}')
    t_start = time.time()
    
    target_device = _AUTO_DEVICE if device == 'auto' else device
    
    # Configure torch hub to use our custom demucs directory
    if settings:
        try:
            import torch
            demucs_dir = settings.resolve_path(settings.demucs_model_dir)
            if demucs_dir:
                torch.hub.set_dir(str(demucs_dir))
                logger.info(f"Set torch hub dir to {demucs_dir}")
        except Exception as e:
            logger.warning(f"Failed to set torch hub dir: {e}")

    separator_cls = _get_demucs_separator_class()
    _SEPARATOR = separator_cls(model_name, device=target_device, progress=progress, shifts=shifts)
    
    t_end = time.time()
    logger.info(f'Demucs model loaded in {t_end - t_start:.2f} seconds')
    return _SEPARATOR


def unload_model() -> None:
    global _SEPARATOR
    if _SEPARATOR is not None:
        del _SEPARATOR
        _SEPARATOR = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info("Demucs model unloaded")



def separate_audio(
    folder: str,
    model_name: str = "htdemucs_ft",
    device: str = 'auto',
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
    
    logger.info(f'Separating audio from {folder}')
    
    separator = load_model(model_name, device, progress, shifts, settings=settings, model_manager=model_manager)
    
    import torchaudio
    
    # Try setting soundfile backend which is more stable than sox_io for large files
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass

    t_start = time.time()
    try:
        logger.info(f"Loading audio from {audio_path}...")
        wav, sr = torchaudio.load(audio_path)
        logger.info(f"Audio loaded. Shape: {wav.shape}, SR: {sr}")

        if sr != separator.samplerate:
             logger.info(f"Resampling from {sr} to {separator.samplerate}...")
             resampler = torchaudio.transforms.Resample(sr, separator.samplerate)
             wav = resampler(wav)
             sr = separator.samplerate
             
        if wav.shape[0] > 2:
            wav = wav[:2, :] 
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1) 


        chunk_duration = 300 
        chunk_samples = chunk_duration * sr
        total_samples = wav.shape[-1]
        
        all_vocals = []
        all_instruments = []
        
        num_chunks = (total_samples + chunk_samples - 1) // chunk_samples
        logger.info(f"Processing in {num_chunks} chunks of {chunk_duration}s.")

        for i in range(num_chunks):
            start = i * chunk_samples
            end = min((i + 1) * chunk_samples, total_samples)
            chunk_wav = wav[:, start:end]
            
            # (origin, separated_dict)
            _, separated = separator.separate_tensor(chunk_wav)
            
            # Extract and move to CPU immediately
            vocals_chunk = separated['vocals'].cpu()
            
            instruments_chunk = None
            for k, v in separated.items():
                if k == 'vocals':
                    continue
                if instruments_chunk is None:
                    instruments_chunk = v.cpu()
                else:
                    instruments_chunk += v.cpu()
            
            all_vocals.append(vocals_chunk)
            all_instruments.append(instruments_chunk)
            
            if progress:
                logger.info(f"Processed chunk {i+1}/{num_chunks}")
            
            # Force cleanup
            del separated, _
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info("Concatenating chunks...")
        vocals = torch.cat(all_vocals, dim=-1).numpy().T
        instruments = torch.cat(all_instruments, dim=-1).numpy().T
        logger.info("Concatenation complete.")

    except Exception as e:
        logger.error(f'Error separating audio from {folder}: {e}')
        raise e

    t_end = time.time()
    logger.info(f'Audio separated in {t_end - t_start:.2f} seconds')
    
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
    for subdir, dirs, files in os.walk(root_folder):
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
