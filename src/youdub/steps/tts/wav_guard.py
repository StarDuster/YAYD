from __future__ import annotations

import os
import wave

from ...utils import wav_duration_seconds


# Qwen3-TTS (Tokenizer-12Hz) output guard:
# If the model keeps generating until `max_new_tokens`, the audio duration will be ~max_new_tokens/12 seconds.
# This often indicates a degenerate loop / failure-to-stop. We treat it as a failure and retry.
_QWEN_TTS_TOKEN_HZ = 12.0
_QWEN_TTS_MAX_NEW_TOKENS_DEFAULT = 2048


def _read_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    raw = raw.strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    raw = raw.strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _qwen_tts_max_new_tokens() -> int:
    v = _read_env_int("YOUDUB_QWEN_TTS_MAX_NEW_TOKENS", _QWEN_TTS_MAX_NEW_TOKENS_DEFAULT)
    if v <= 0:
        return int(_QWEN_TTS_MAX_NEW_TOKENS_DEFAULT)
    return int(v)


def _qwen_tts_hit_max_tokens_seconds() -> float:
    return float(_qwen_tts_max_new_tokens()) / float(_QWEN_TTS_TOKEN_HZ)


def _qwen_tts_is_degenerate_hit_cap(*, wav_dur: float | None) -> bool:
    """
    Detect "keeps generating until max_new_tokens" by duration.

    We intentionally use a small tolerance because wav header rounding and model decoding
    may not land on the exact boundary.
    """
    if wav_dur is None:
        return False
    tol = _read_env_float("YOUDUB_QWEN_TTS_HIT_CAP_TOL_SEC", 2.0)
    if tol < 0:
        tol = 2.0
    cap = _qwen_tts_hit_max_tokens_seconds()
    return bool(float(wav_dur) >= float(cap) - float(tol))


def _qwen_resp_duration_seconds(resp: dict | None) -> float | None:
    if not isinstance(resp, dict):
        return None
    try:
        sr = resp.get("sr")
        n_samples = resp.get("n_samples")
        if sr is None or n_samples is None:
            return None
        sr_f = float(sr)
        ns_f = float(n_samples)
        if not (sr_f > 0) or not (ns_f >= 0):
            return None
        return ns_f / sr_f
    except Exception:
        return None


def is_valid_wav(path: str) -> bool:
    """Check if a file is a valid WAV file."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 44:  # Minimal WAV header size
        return False
    try:
        # NOTE:
        # We intentionally validate via stdlib `wave` here (instead of libsndfile/soundfile).
        # Reason: downstream time-stretch (`audiostretchy`) also uses `wave` and will crash
        # on files that libsndfile can decode (e.g. FLAC/OGG/MP3 content saved as *.wav).
        with wave.open(path, "rb") as wf:
            if int(wf.getframerate() or 0) <= 0:
                return False
            if int(wf.getnchannels() or 0) <= 0:
                return False
            if int(wf.getsampwidth() or 0) <= 0:
                return False
            # Allow zero-length wav (treated as valid but will be handled later).
            _ = int(wf.getnframes() or 0)
        return True
    except Exception:
        return False


def _tts_duration_guard_params() -> tuple[float, float, float, int]:
    """
    Guardrail for "TTS segment should not be significantly longer than source segment".

    Defaults are intentionally permissive (to tolerate language expansion) but will catch
    pathological outputs like ~10-minute clips for a ~10s segment.
    """
    ratio = _read_env_float("YOUDUB_TTS_MAX_SEGMENT_DURATION_RATIO", 3.0)
    extra = _read_env_float("YOUDUB_TTS_MAX_SEGMENT_DURATION_EXTRA_SEC", 8.0)
    abs_cap = _read_env_float("YOUDUB_TTS_MAX_SEGMENT_DURATION_ABS_SEC", 180.0)
    retries = _read_env_int("YOUDUB_TTS_SEGMENT_MAX_RETRIES", 3)

    if not (ratio >= 1.0):
        ratio = 3.0
    if not (extra >= 0.0):
        extra = 8.0
    # abs_cap <= 0 means "disable absolute cap"
    if not (retries >= 1):
        retries = 3
    retries = int(max(1, min(retries, 10)))
    return float(ratio), float(extra), float(abs_cap), int(retries)


def _tts_segment_allowed_max_seconds(seg_dur: float, ratio: float, extra: float, abs_cap: float) -> float | None:
    """
    Compute allowed max duration for a TTS segment.

    - seg_dur <= 0: only apply abs_cap (if configured), otherwise no guard.
    - otherwise: max(seg_dur * ratio, seg_dur + extra).
    """
    sd = float(seg_dur or 0.0)
    if sd <= 0.0:
        return float(abs_cap) if (abs_cap and abs_cap > 0) else None
    allowed = max(sd * float(ratio), sd + float(extra))
    return float(allowed)


def _tts_wav_ok_for_segment(path: str, seg_dur: float, ratio: float, extra: float, abs_cap: float) -> bool:
    if not (os.path.exists(path) and is_valid_wav(path)):
        return False
    dur = wav_duration_seconds(path)
    if dur is None:
        return False
    allowed = _tts_segment_allowed_max_seconds(seg_dur, ratio, extra, abs_cap)
    if allowed is None:
        return True
    # tolerate tiny header rounding errors
    return bool(dur <= allowed + 0.02)

