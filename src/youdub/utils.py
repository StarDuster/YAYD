import os
import re
import wave

import numpy as np
from scipy.io import wavfile
from loguru import logger


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


def ensure_torchaudio_backend_compat() -> None:
    """
    Compatibility shim for torchaudio / pyannote.audio backend APIs.

    `pyannote.audio==3.1.1` calls `torchaudio.set_audio_backend("soundfile")` and
    `torchaudio.get_audio_backend()` / `torchaudio.list_audio_backends()` at import time.
    Some torchaudio versions/builds may not expose these symbols, causing pyannote to fail
    importing or executing diarization.

    We provide best-effort no-op shims so pyannote can import and rely on the
    default audio I/O stack (torchcodec / soundfile).
    """

    try:
        import torchaudio  # type: ignore
    except Exception:
        return

    # torchaudio>=2.9 removed `torchaudio.info`, but pyannote.audio still calls it.
    # Provide a best-effort implementation backed by soundfile (already a project dependency).
    if not hasattr(torchaudio, "info"):
        try:
            import soundfile as _sf  # type: ignore
        except Exception:
            _sf = None

        try:
            from torchaudio.backend.common import AudioMetaData as _AudioMetaData  # type: ignore
        except Exception:
            from typing import NamedTuple

            class _AudioMetaData(NamedTuple):
                sample_rate: int
                num_frames: int
                num_channels: int
                bits_per_sample: int
                encoding: str

        def _guess_bits_per_sample(subtype: str | None) -> int:
            if not subtype:
                return 0
            m = re.search(r"(\d+)", str(subtype))
            return int(m.group(1)) if m else 0

        def _info(uri, *_, **__):  # type: ignore[no-untyped-def]
            if _sf is None:
                raise RuntimeError("torchaudio.info 不可用且未安装 soundfile，无法读取音频元信息。")
            try:
                si = _sf.info(uri)
            except Exception as exc:
                raise RuntimeError(f"无法读取音频元信息: {exc}") from exc

            return _AudioMetaData(
                int(getattr(si, "samplerate", 0) or 0),
                int(getattr(si, "frames", 0) or 0),
                int(getattr(si, "channels", 0) or 0),
                _guess_bits_per_sample(getattr(si, "subtype", None)),
                str(getattr(si, "subtype", "") or getattr(si, "format", "") or ""),
            )

        try:
            setattr(torchaudio, "info", _info)
        except Exception:
            pass

    # torchaudio>=2.x may remove the `torchaudio.backend` package entirely.
    # Some downstream libs/checkpoints still import it (e.g. during torch.load/unpickling),
    # which would crash diarization. Provide a minimal module tree for compatibility.
    try:
        import importlib.util
        import sys
        import types
        from typing import NamedTuple

        if importlib.util.find_spec("torchaudio.backend") is None:
            backend_mod = types.ModuleType("torchaudio.backend")
            # Mark as a package so `import torchaudio.backend.xxx` won't complain.
            backend_mod.__path__ = []  # type: ignore[attr-defined]

            class AudioMetaData(NamedTuple):
                sample_rate: int
                num_frames: int
                num_channels: int
                bits_per_sample: int
                encoding: str

            common_mod = types.ModuleType("torchaudio.backend.common")
            common_mod.AudioMetaData = AudioMetaData

            def _wrap_top_level(fn_name: str):
                fn = getattr(torchaudio, fn_name, None)
                if callable(fn):
                    return fn

                def _missing(*_args, **_kwargs):
                    raise RuntimeError(f"torchaudio.{fn_name} 不可用，无法兼容旧的 torchaudio.backend 调用。")

                return _missing

            # Commonly referenced legacy backend submodules.
            legacy_backend_names = [
                "sox_io_backend",
                "soundfile_backend",
                "ffmpeg_backend",
            ]

            for name in legacy_backend_names:
                mod = types.ModuleType(f"torchaudio.backend.{name}")
                mod.load = _wrap_top_level("load")  # type: ignore[attr-defined]
                mod.save = _wrap_top_level("save")  # type: ignore[attr-defined]
                mod.info = _wrap_top_level("info")  # type: ignore[attr-defined]
                sys.modules[f"torchaudio.backend.{name}"] = mod
                setattr(backend_mod, name, mod)

            setattr(backend_mod, "common", common_mod)
            sys.modules["torchaudio.backend"] = backend_mod
            sys.modules["torchaudio.backend.common"] = common_mod
            # Ensure attribute access works: `torchaudio.backend`.
            setattr(torchaudio, "backend", backend_mod)
    except Exception:
        # Best-effort only; never crash the main pipeline.
        pass

    # Keep a tiny per-process backend state so set/get/list are consistent.
    state_attr = "_youdub_audio_backend"
    if not hasattr(torchaudio, state_attr):
        try:
            setattr(torchaudio, state_attr, "soundfile")
        except Exception:
            # If we cannot set attributes on torchaudio module, shimming won't work reliably.
            return

    def _set_audio_backend(backend: str) -> None:
        try:
            setattr(torchaudio, state_attr, str(backend))
        except Exception:
            return None

    def _get_audio_backend() -> str:
        try:
            v = getattr(torchaudio, state_attr)
        except Exception:
            return "soundfile"
        return str(v) if v else "soundfile"

    def _list_audio_backends() -> list[str]:
        backends: list[str] = []

        # Best-effort: expose "soundfile" if installed (commonly used by pyannote).
        try:
            import soundfile  # type: ignore  # noqa: F401

            backends.append("soundfile")
        except Exception:
            pass

        # Also include current backend value so callers won't crash on unexpected strings.
        cur = _get_audio_backend()
        if cur and cur not in backends:
            backends.append(cur)

        # If none detected, still provide a non-empty list for older callers.
        return backends or ["soundfile"]

    # Patch set/get/list if missing
    if not hasattr(torchaudio, "set_audio_backend"):
        try:
            setattr(torchaudio, "set_audio_backend", _set_audio_backend)
        except Exception:
            pass

    if not hasattr(torchaudio, "get_audio_backend"):
        try:
            setattr(torchaudio, "get_audio_backend", _get_audio_backend)
        except Exception:
            pass

    if not hasattr(torchaudio, "list_audio_backends"):
        try:
            setattr(torchaudio, "list_audio_backends", _list_audio_backends)
        except Exception:
            pass


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
    sample_rate: int = 24000,
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
        sample_rate: Sample rate
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

def normalize_wav(wav_path: str) -> None:
    try:
        sample_rate, wav = wavfile.read(wav_path)
        peak = _peak_abs(wav)
        wav_norm = wav * (32767 / max(0.01, peak))
        wavfile.write(wav_path, sample_rate, wav_norm.astype(np.int16))
    except Exception as e:
        logger.warning(f"标准化音频失败 {wav_path}: {e}")

