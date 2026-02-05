import contextlib
import os
import wave

import numpy as np
from scipy.io import wavfile


def wav_duration_seconds(path: str) -> float | None:
    """Return WAV duration in seconds using stdlib `wave`.

    Returns None if header is invalid/unreadable.
    """
    try:
        with wave.open(path, "rb") as wf:
            rate = int(wf.getframerate() or 0)
            if rate <= 0:
                return None
            frames = int(wf.getnframes() or 0)
            if frames <= 0:
                return 0.0
            return frames / float(rate)
    except Exception:
        return None


def read_speaker_ref_seconds(default: float = 15.0) -> float:
    """
    Speaker reference audio duration (seconds) for voice cloning.

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


def valid_file(path: str, *, min_bytes: int = 1) -> bool:
    """检查文件是否存在且大小不小于 min_bytes。"""
    try:
        return os.path.exists(path) and os.path.getsize(path) >= int(min_bytes)
    except Exception:
        return False


def require_file(path: str, desc: str, *, min_bytes: int = 1) -> None:
    """检查文件是否存在且大小不小于 min_bytes，否则抛出 FileNotFoundError。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少{desc}: {path}")
    try:
        if os.path.getsize(path) < min_bytes:
            raise FileNotFoundError(f"{desc}文件过小/疑似损坏: {path}")
    except OSError:
        raise FileNotFoundError(f"无法读取{desc}: {path}") from None


def _peak_abs(wav: np.ndarray) -> float:
    """Return max absolute amplitude without allocating a full `abs(wav)` temp array."""
    if wav.size <= 0:
        return 0.0
    # Two-pass (max+min) but constant memory; avoids `np.abs(wav)` doubling peak memory for long audio.
    max_val = float(np.max(wav))
    min_val = float(np.min(wav))
    return max(abs(max_val), abs(min_val))


@contextlib.contextmanager
def torch_load_weights_only_compat():
    """Temporarily force `torch.load(weights_only=False)` for legacy checkpoints.

    PyTorch 2.6+ changed the default value of `weights_only` from False to True. Some
    Hugging Face checkpoints (including pyannote models) still rely on pickled objects
    and will fail to load unless `weights_only=False`.

    Scope this monkeypatch to model loading only.
    """

    try:
        import torch  # type: ignore
    except Exception:
        yield
        return

    orig_load = torch.load

    def _load(*args, **kwargs):  # type: ignore[no-untyped-def]
        # Force legacy behavior even if callers explicitly pass weights_only=True
        # (e.g. lightning_fabric cloud_io).
        patched = dict(kwargs)
        patched["weights_only"] = False
        try:
            return orig_load(*args, **patched)
        except TypeError:
            # Older torch versions may not accept `weights_only`.
            patched.pop("weights_only", None)
            return orig_load(*args, **patched)

    torch.load = _load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = orig_load  # type: ignore[assignment]


def save_wav(wav: np.ndarray, output_path: str, sample_rate: int = 24000) -> None:
    wav_i16 = wav * 32767
    wavfile.write(output_path, sample_rate, wav_i16.astype(np.int16))

def save_wav_norm(wav: np.ndarray, output_path: str, sample_rate: int = 24000) -> None:
    peak = _peak_abs(wav)
    wav_norm = wav * (32767 / max(0.01, peak))
    wavfile.write(output_path, sample_rate, wav_norm.astype(np.int16))


def soft_clip(wav: np.ndarray, threshold: float = 0.9, knee: float = 0.1) -> np.ndarray:
    """
    Soft clipper to reduce harsh peaks without hard clipping.

    Uses a smooth tanh-like curve to compress samples above the threshold.

    Args:
        wav: Input waveform (float, typically in [-1, 1])
        threshold: Amplitude above which soft clipping starts (default 0.9)
        knee: Softness of the knee (default 0.1)

    Returns:
        Soft-clipped waveform
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        return wav

    th = float(max(0.1, min(threshold, 0.99)))
    k = float(max(0.01, min(knee, 0.5)))

    # Use tanh for smooth compression above threshold
    out = np.copy(wav)
    above_th = np.abs(wav) > th

    if np.any(above_th):
        # Map [th, 1] -> [0, ~1] then apply tanh, then map back
        sign = np.sign(wav[above_th])
        excess = np.abs(wav[above_th]) - th
        # Soft compress the excess using tanh
        compressed = th + k * np.tanh(excess / k)
        out[above_th] = sign * compressed

    return out


def smooth_transients(wav: np.ndarray, max_diff: float = 0.3, alpha: float = 0.3) -> np.ndarray:
    """
    Smooth sudden transients (potential pops/clicks) in audio.

    Applies exponential smoothing to samples that change too rapidly.

    Args:
        wav: Input waveform (float)
        max_diff: Maximum allowed difference between consecutive samples (default 0.3)
        alpha: Smoothing factor when transient detected (default 0.3, lower = smoother)

    Returns:
        Smoothed waveform
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size <= 1:
        return wav

    out = np.copy(wav)
    md = float(max(0.01, max_diff))
    a = float(max(0.01, min(alpha, 1.0)))

    for i in range(1, len(out)):
        diff = out[i] - out[i - 1]
        if abs(diff) > md:
            # Exponential smoothing toward the new value
            out[i] = out[i - 1] + a * diff

    return out


def prepare_speaker_ref_audio(
    wav: np.ndarray,
    *,
    trim_silence: bool = True,
    trim_top_db: float = 30.0,
    apply_soft_clip: bool = True,
    clip_threshold: float = 0.85,
    apply_smooth: bool = True,
    smooth_max_diff: float = 0.25,
) -> np.ndarray:
    """
    Prepare speaker reference audio for voice cloning with anti-pop processing.

    Steps:
    1. Trim leading/trailing silence (optional)
    2. Apply soft clipper to reduce harsh peaks (optional)
    3. Smooth rapid transients to reduce pops/clicks (optional)
    4. Normalize to safe peak level

    Args:
        wav: Input waveform (float, mono)
        trim_silence: Whether to trim silence
        trim_top_db: Threshold for silence trimming
        apply_soft_clip: Whether to apply soft clipping
        clip_threshold: Soft clip threshold
        apply_smooth: Whether to smooth transients
        smooth_max_diff: Max diff for transient smoothing

    Returns:
        Processed waveform (float32)
    """
    import librosa

    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        return wav

    # Ensure mono
    if wav.ndim > 1:
        wav = wav.mean(axis=-1) if wav.shape[-1] <= 2 else wav[..., 0]
    wav = wav.flatten().astype(np.float32)

    # 1. Trim silence
    if trim_silence:
        try:
            trimmed, _ = librosa.effects.trim(wav, top_db=float(trim_top_db))
            if trimmed is not None and trimmed.size > 0:
                wav = trimmed.astype(np.float32)
        except Exception:
            pass

    # 2. Normalize to [-1, 1] before processing
    peak = _peak_abs(wav)
    if peak > 1e-6:
        wav = wav / peak

    # 3. Soft clip to reduce harsh peaks
    if apply_soft_clip:
        wav = soft_clip(wav, threshold=float(clip_threshold), knee=0.1)

    # 4. Smooth rapid transients
    if apply_smooth:
        wav = smooth_transients(wav, max_diff=float(smooth_max_diff), alpha=0.3)

    # 5. Final normalize to safe level (leave headroom)
    peak = _peak_abs(wav)
    if peak > 1e-6:
        wav = wav * (0.95 / peak)

    return wav.astype(np.float32)
