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
    _, apply_model = _import_demucs_infer()

    target_device = _AUTO_DEVICE if device == "auto" else device
    target_sr = int(getattr(model, "samplerate", 44100))
    target_channels = int(getattr(model, "audio_channels", 2))
    source_names = list(getattr(model, "sources", []))
    vocals_idx = source_names.index("vocals") if "vocals" in source_names else -1

    def _fix_channels(x):
        # x: [T, C]
        if x.size == 0:
            return x
        if x.shape[1] > target_channels:
            return x[:, :target_channels]
        if x.shape[1] < target_channels:
            if x.shape[1] == 1:
                import numpy as np

                return np.repeat(x, target_channels, axis=1)
            import numpy as np

            pad = np.zeros((x.shape[0], target_channels - x.shape[1]), dtype=x.dtype)
            return np.concatenate([x, pad], axis=1)
        return x

    def _pick_cut_frame(in_f, start_frame: int, nominal_end: int, total_frames: int, sr: int) -> int:
        """在 nominal_end 附近找一个更“安静”的切点，避免切在说话中间。"""
        # 参数尽量保守：chunk 做大，切点只做小范围微调。
        chunk_sec = 10 * 60
        min_chunk_frames = max(int(sr * 60), int(sr * 2))  # 至少 60s（也保证比 overlap 大）
        search_sec = 20
        hop_sec = 0.1

        if nominal_end >= total_frames:
            return total_frames

        search_frames = int(search_sec * sr)
        hop_frames = max(1, int(hop_sec * sr))

        min_cut = min(total_frames, start_frame + max(min_chunk_frames, int(0.5 * chunk_sec * sr)))
        search_start = max(min_cut, nominal_end - search_frames)
        search_end = min(total_frames, nominal_end + search_frames)
        if search_end - search_start < hop_frames * 2:
            return nominal_end

        # 读取搜索窗口，找 RMS 最小的 hop 位置
        in_f.seek(search_start)
        block = in_f.read(frames=search_end - search_start, dtype="float32", always_2d=True)
        block = _fix_channels(block)
        if block.size == 0:
            return nominal_end

        import numpy as np

        energy = np.mean(block.astype(np.float32) ** 2, axis=1)  # [T]
        n = int(len(energy) // hop_frames)
        if n <= 0:
            return nominal_end
        energy_hop = energy[: n * hop_frames].reshape(n, hop_frames).mean(axis=1)
        idx = int(np.argmin(energy_hop))
        cut = search_start + idx * hop_frames
        # 别让 cut 太靠近 start 导致不前进
        return int(max(min_cut, min(cut, search_end)))

    t_start = time.time()
    try:
        # 长音频：不要整段 load（会 OOM）。用 soundfile 流式读 + 大 chunk + 静音附近切点 + overlap 淡入淡出拼接。
        try:
            import soundfile as sf

            info = sf.info(audio_path)
            sr_info = int(info.samplerate)
            total_frames = int(info.frames)
            duration_sec = float(total_frames) / float(sr_info) if sr_info else 0.0
        except Exception:
            sr_info = 0
            total_frames = 0
            duration_sec = 0.0

        long_audio_threshold_sec = 15 * 60
        if duration_sec >= long_audio_threshold_sec:
            if sr_info != target_sr:
                raise RuntimeError(
                    f"音频过长（{duration_sec:.1f}s），采样率为 {sr_info}Hz，但 Demucs 期望 {target_sr}Hz。"
                    "为避免 OOM，本项目不会对长音频走“整段加载+重采样”。请先用 ffmpeg 重新抽取成 44100Hz/2ch 的 wav。"
                )

            import numpy as np
            import soundfile as sf

            chunk_sec = 10 * 60
            overlap_sec = 2
            chunk_frames = int(chunk_sec * target_sr)
            overlap_frames = int(overlap_sec * target_sr)
            if chunk_frames <= overlap_frames * 2:
                raise ValueError(f"Invalid chunk config: chunk_sec={chunk_sec}, overlap_sec={overlap_sec}")

            logger.warning(
                f"Audio is long ({duration_sec:.1f}s). Use chunked Demucs to avoid OOM "
                f"(chunk~{chunk_sec}s, overlap={overlap_sec}s, shifts={shifts}, cut=near-silence)."
            )

            prev_v_tail = None
            prev_i_tail = None

            start_frame = 0
            with (
                sf.SoundFile(audio_path, mode="r") as in_f,
                sf.SoundFile(vocal_output_path, mode="w", samplerate=target_sr, channels=target_channels, subtype="PCM_16") as out_v,
                sf.SoundFile(
                    instruments_output_path, mode="w", samplerate=target_sr, channels=target_channels, subtype="PCM_16"
                ) as out_i,
            ):
                while start_frame < total_frames:
                    nominal_end = min(total_frames, start_frame + chunk_frames)
                    end_frame = total_frames if nominal_end >= total_frames else _pick_cut_frame(
                        in_f, start_frame, nominal_end, total_frames, target_sr
                    )
                    # 避免极端情况下不前进
                    if end_frame <= start_frame + overlap_frames:
                        end_frame = nominal_end

                    in_f.seek(start_frame)
                    audio_chunk = in_f.read(frames=end_frame - start_frame, dtype="float32", always_2d=True)
                    audio_chunk = _fix_channels(audio_chunk)
                    if audio_chunk.size == 0:
                        break

                    wav_t = torch.from_numpy(np.asarray(audio_chunk.T, dtype=np.float32))  # [C, T]
                    mix = wav_t.unsqueeze(0)  # [1, C, T]

                    sources = apply_model(model, mix, shifts=shifts, split=True, progress=False, device=target_device)
                    sources = sources[0]  # [S, C, T]
                    vocals_tensor = sources[vocals_idx]
                    instruments_tensor = sources.sum(dim=0) - vocals_tensor

                    vocals = vocals_tensor.cpu().numpy().T  # [T, C]
                    instruments = instruments_tensor.cpu().numpy().T  # [T, C]
                    del sources, vocals_tensor, instruments_tensor, mix, wav_t, audio_chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                    chunk_len = int(vocals.shape[0])
                    has_next = end_frame < total_frames

                    if prev_v_tail is None:
                        if has_next and overlap_frames > 0:
                            keep_end = max(0, chunk_len - overlap_frames)
                            out_v.write(vocals[:keep_end])
                            out_i.write(instruments[:keep_end])
                            prev_v_tail = vocals[keep_end:]
                            prev_i_tail = instruments[keep_end:]
                        else:
                            out_v.write(vocals)
                            out_i.write(instruments)
                            prev_v_tail = None
                            prev_i_tail = None
                    else:
                        overlap_len = min(overlap_frames, chunk_len, int(prev_v_tail.shape[0]))
                        if overlap_len > 0:
                            fade_in = np.linspace(0.0, 1.0, overlap_len, dtype=np.float32)[:, None]
                            fade_out = 1.0 - fade_in
                            out_v.write(prev_v_tail[-overlap_len:] * fade_out + vocals[:overlap_len] * fade_in)
                            out_i.write(prev_i_tail[-overlap_len:] * fade_out + instruments[:overlap_len] * fade_in)

                        if has_next and overlap_frames > 0:
                            keep_end = max(overlap_len, chunk_len - overlap_frames)
                            out_v.write(vocals[overlap_len:keep_end])
                            out_i.write(instruments[overlap_len:keep_end])
                            prev_v_tail = vocals[keep_end:]
                            prev_i_tail = instruments[keep_end:]
                        else:
                            out_v.write(vocals[overlap_len:])
                            out_i.write(instruments[overlap_len:])
                            prev_v_tail = None
                            prev_i_tail = None

                    if not has_next:
                        break
                    start_frame = max(0, end_frame - overlap_frames)

            logger.info(f"Audio separated in {time.time() - t_start:.2f} seconds")
            logger.info("Separation task finished completely.")
            return

        # 短音频：保持简单（整段 load + save_wav）。
        import torchaudio

        # Try setting soundfile backend which is more stable than sox_io for large files
        try:
            torchaudio.set_audio_backend("soundfile")
        except Exception:
            pass

        logger.info(f"Loading audio from {audio_path}...")
        wav, sr = torchaudio.load(audio_path)
        logger.info(f"Audio loaded. Shape: {wav.shape}, SR: {sr}")

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
        sources = apply_model(model, mix, shifts=shifts, split=True, progress=progress, device=target_device)
        sources = sources[0]  # [S, C, T]

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
