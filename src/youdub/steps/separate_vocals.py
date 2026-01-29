import gc
import os
import subprocess
import time
from typing import Any

import torch
from loguru import logger

from ..config import Settings
from ..models import ModelManager
from ..interrupts import check_cancelled, sleep_with_cancel
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
            "建议：运行 `uv sync`（或 `pip install demucs-infer`）"
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
        logger.info(f"设置 torch hub 目录: {demucs_dir}")
    except Exception as e:
        logger.warning(f"设置 torch hub 目录失败: {e}")


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

    logger.info(f"加载 Demucs 模型: {model_name}")
    t_start = time.time()
    model = get_model(model_name)
    _DEMUCS_MODEL = model
    _DEMUCS_MODEL_NAME = model_name
    _DEMUCS_DEVICE = target_device
    logger.info(f"Demucs 模型加载完成，耗时 {time.time() - t_start:.2f} 秒")
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
        logger.info("Demucs 模型已卸载")


def separate_audio(
    folder: str,
    model_name: str = "htdemucs_ft",
    device: str = "auto",
    progress: bool = True,
    shifts: int = 5,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> None:
    check_cancelled()
    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        return

    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')
    
    if os.path.exists(vocal_output_path) and os.path.exists(instruments_output_path):
        logger.info(f"已分离音频: {folder}")
        return
    
    settings = settings or Settings()
    model_manager = model_manager or ModelManager(settings)
    _ensure_demucs_ready(settings, model_manager)

    logger.info(f"开始分离音频: {folder}")

    model = load_model(model_name, device, progress, shifts, settings=settings, model_manager=model_manager)
    _, apply_model = _import_demucs_infer()

    target_device = _AUTO_DEVICE if device == "auto" else device
    target_sr = int(getattr(model, "samplerate", 44100))
    target_channels = int(getattr(model, "audio_channels", 2))
    source_names = list(getattr(model, "sources", []))
    vocals_idx = source_names.index("vocals") if "vocals" in source_names else -1

    t_start = time.time()
    try:
        # 保持简单：整段 load + save_wav。
        import torchaudio

        # Try setting soundfile backend which is more stable than sox_io for large files
        try:
            torchaudio.set_audio_backend("soundfile")
        except Exception:
            pass

        logger.info(f"加载音频: {audio_path}")
        check_cancelled()
        wav, sr = torchaudio.load(audio_path)
        logger.info(f"音频已加载: shape={wav.shape}, sr={sr}")

        if sr != target_sr:
            logger.info(f"重采样: {sr}->{target_sr}")
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)
            sr = target_sr

        if wav.shape[0] > target_channels:
            wav = wav[:target_channels, :]
        elif wav.shape[0] < target_channels:
            wav = wav.repeat(target_channels, 1)

        mix = wav.unsqueeze(0)  # [1, C, T]

        logger.info(f"运行 Demucs: {model_name} device={target_device} shifts={shifts}")
        check_cancelled()
        sources = apply_model(model, mix, shifts=shifts, split=True, progress=progress, device=target_device)
        sources = sources[0]  # [S, C, T]

        vocals_tensor = sources[vocals_idx]
        instruments_tensor = sources.sum(dim=0) - vocals_tensor

        vocals = vocals_tensor.cpu().numpy().T
        instruments = instruments_tensor.cpu().numpy().T
    except Exception as e:
        logger.error(f"分离失败: {folder} ({e})")
        raise

    logger.info(f"分离完成，耗时 {time.time() - t_start:.2f} 秒")

    logger.info(f"保存人声: {vocal_output_path}")
    save_wav(vocals, vocal_output_path, sample_rate=sr)

    logger.info(f"保存伴奏: {instruments_output_path}")
    save_wav(instruments, instruments_output_path, sample_rate=sr)

    logger.info("分离任务完成")




def extract_audio_from_video(folder: str) -> bool:
    video_path = os.path.join(folder, 'download.mp4')
    if not os.path.exists(video_path):
        return False
        
    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        logger.info(f"已提取音频: {folder}")
        return True
        
    logger.info(f"开始提取音频: {folder}")
    check_cancelled()

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        audio_path,
    ]

    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        while True:
            check_cancelled()
            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(0.2)
    except BaseException:
        # Best-effort: stop ffmpeg quickly on Ctrl+C.
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        raise

    stdout = ""
    stderr = ""
    try:
        stdout, stderr = proc.communicate(timeout=1)
    except Exception:
        pass

    if proc.returncode not in (0, None):
        logger.error(f"FFmpeg 提取音频失败: {video_path} (rc={proc.returncode})\n{stderr or stdout}")
        return False

    sleep_with_cancel(1)
    logger.info(f"音频已提取: {folder}")
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

    def _valid_audio_file(p: str) -> bool:
        try:
            return os.path.exists(p) and os.path.getsize(p) >= 44
        except Exception:
            return False

    # 先扫描一遍：如果没有任何需要处理的文件，就不要加载 Demucs。
    pending: list[str] = []
    for subdir, _dirs, files in os.walk(root_folder):
        check_cancelled()
        if "download.mp4" not in files:
            continue
        vocal_output_path = os.path.join(subdir, "audio_vocals.wav")
        instruments_output_path = os.path.join(subdir, "audio_instruments.wav")
        if _valid_audio_file(vocal_output_path) and _valid_audio_file(instruments_output_path):
            continue
        pending.append(subdir)

    if not pending:
        msg = f"音频分离完成: {root_folder}（processed 0 files / 处理 0 个文件）"
        logger.info(msg)
        return msg

    _ensure_demucs_ready(settings, model_manager)
    load_model(model_name, device, progress, shifts, settings=settings, model_manager=model_manager)

    count = 0
    for subdir in pending:
        check_cancelled()

        audio_wav = os.path.join(subdir, "audio.wav")
        if not os.path.exists(audio_wav):
            extract_audio_from_video(subdir)

        vocal_output_path = os.path.join(subdir, "audio_vocals.wav")
        instruments_output_path = os.path.join(subdir, "audio_instruments.wav")

        # Re-run separation if outputs are missing or look truncated/corrupted (common after interruptions).
        if not _valid_audio_file(vocal_output_path) or not _valid_audio_file(instruments_output_path):
            for p in (vocal_output_path, instruments_output_path):
                if os.path.exists(p) and not _valid_audio_file(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            separate_audio(subdir, model_name, device, progress, shifts, settings=settings, model_manager=model_manager)
            count += 1

    msg = f"音频分离完成: {root_folder}（processed {count} files / 处理 {count} 个文件）"
    logger.info(msg)
    return msg
