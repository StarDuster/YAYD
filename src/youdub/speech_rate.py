from __future__ import annotations

import math
import os
import re
import tempfile
import threading
import time
import uuid
from typing import Any

from cmudict import dict as cmudict_dict
import librosa
import numpy as np
from audiostretchy.stretch import stretch_audio
from loguru import logger
from pypinyin import Style, pinyin

from .utils import save_wav

_CMU_DICT = cmudict_dict()


def _clamp(v: float, lo: float, hi: float) -> float:
    x = float(v)
    a = float(lo)
    b = float(hi)
    if a > b:
        a, b = b, a
    return float(max(a, min(b, x)))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return float(x)


def _fallback_syllable_count(word: str) -> int:
    """
    Rule-based syllable count fallback for English OOV words.

    Heuristic:
    - count vowel groups [aeiouy]+
    - drop silent trailing 'e' (except *le endings)
    - always return at least 1 for non-empty words
    """
    w = re.sub(r"[^a-z]+", "", str(word or "").lower())
    if not w:
        return 0
    if len(w) > 2 and w.endswith("e") and not w.endswith("le"):
        w = w[:-1]
    groups = re.findall(r"[aeiouy]+", w)
    return int(max(1, len(groups)))


def count_en_syllables(text: str) -> int:
    """
    Count English syllables using CMUdict (via `cmudict`) with a rule-based fallback.
    """
    s = str(text or "")
    token_re = re.compile(r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?|[A-Za-z]+(?:'[A-Za-z]+)?")
    tokens = token_re.findall(s)
    if not tokens:
        return 0

    letter_syllables = {ch: 1 for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    letter_syllables["W"] = 3
    total = 0
    for tok in tokens:
        t = str(tok or "")
        if not t:
            continue

        # Numeric / currency / percent tokens.
        if any(ch.isdigit() for ch in t):
            raw = t.strip()
            is_money = raw.startswith("$")
            is_percent = raw.endswith("%")
            raw = raw[1:] if is_money else raw
            raw = raw[:-1] if is_percent else raw
            raw = raw.replace(",", "")

            num_syl = 0
            if raw:
                if "." in raw:
                    a, b = raw.split(".", 1)
                    num_syl += len([ch for ch in a if ch.isdigit()])
                    if b:
                        num_syl += 1  # "point"
                        num_syl += len([ch for ch in b if ch.isdigit()])
                else:
                    num_syl += len([ch for ch in raw if ch.isdigit()])

            if is_money:
                num_syl += 2  # "dollars"
            if is_percent:
                num_syl += 2  # "percent"

            total += int(max(1, num_syl)) if raw else 0
            continue

        # Alphabetic tokens: try CMUdict first.
        w = t.lower()
        prons = _CMU_DICT.get(w, [])
        if prons:
            try:
                total += int(sum(1 for p in prons[0] if p and p[-1].isdigit()))
                continue
            except Exception:
                pass

        # Initialism: all-caps, not in CMUdict => count letter names.
        if t.isalpha() and t.isupper() and len(t) >= 2:
            total += int(sum(int(letter_syllables.get(ch, 1)) for ch in t))
            continue

        total += _fallback_syllable_count(w)
    return int(max(1, total))


def count_zh_syllables(text: str) -> int:
    """
    Count Chinese syllables:
    - Hanzi: use pypinyin and count produced pinyin tokens (errors='ignore' skips non-Hanzi).
    - Latin words: count English syllables (CMUdict + fallback).
    - Digits: count each digit as 1 syllable (very rough but stable).
    """
    s = str(text or "")
    if not s.strip():
        return 0

    cjk_count = 0
    try:
        py = pinyin(s, style=Style.NORMAL, heteronym=False, errors="ignore")
        cjk_count = int(sum(1 for item in py if item and item[0]))
    except Exception:
        cjk_count = int(sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff"))

    latin_words = re.findall(r"[A-Za-z]+", s)
    latin_count = int(sum(count_en_syllables(w) for w in latin_words)) if latin_words else 0
    digit_count = int(sum(1 for ch in s if ch.isdigit()))
    return int(cjk_count + latin_count + digit_count)


def compute_en_speech_rate(
    text: str,
    duration: float,
) -> dict[str, Any]:
    """Compute English speech stats from subtitle text + segment duration."""
    dur = _safe_float(duration, default=0.0)
    dur = float(max(0.0, dur))
    syllables = int(count_en_syllables(text))
    syllable_rate = float(syllables / dur) if dur > 0 else 0.0
    return {"syllable_count": int(syllables), "duration": float(dur), "syllable_rate": float(syllable_rate)}


def compute_zh_speech_rate(
    text: str,
    duration: float,
) -> dict[str, Any]:
    """Compute Chinese speech stats from translated subtitle text + TTS segment duration."""
    dur = _safe_float(duration, default=0.0)
    dur = float(max(0.0, dur))
    syllables = int(count_zh_syllables(text))
    syllable_rate = float(syllables / dur) if dur > 0 else 0.0
    return {"syllable_count": int(syllables), "duration": float(dur), "syllable_rate": float(syllable_rate)}


def compute_scaling_ratio(
    en_stats: dict[str, Any],
    zh_stats: dict[str, Any],
    *,
    voice_min: float = 0.7,
    voice_max: float = 1.0,
) -> dict[str, Any]:
    """
    Compute scaling ratio for time-scale modification (TSM).

    ratio = new_duration / old_duration
    voice_ratio = clamp(zh_rate / en_rate, voice_min, voice_max)
    """
    en_rate = _safe_float(en_stats.get("syllable_rate", 0.0), default=0.0)
    zh_rate = _safe_float(zh_stats.get("syllable_rate", 0.0), default=0.0)
    if not (en_rate > 0.0 and zh_rate > 0.0):
        return {
            "voice_ratio": 1.0,
            "voice_ratio_raw": 1.0,
            "clamped": False,
        }

    voice_ratio_raw = float(zh_rate / en_rate)
    voice_ratio = _clamp(voice_ratio_raw, float(voice_min), float(voice_max))
    clamped = abs(voice_ratio - voice_ratio_raw) > 1e-9

    return {
        "voice_ratio": float(voice_ratio),
        "voice_ratio_raw": float(voice_ratio_raw),
        "clamped": bool(clamped),
    }


def _time_stretch(y: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """
    TSM with pitch preservation.

    Prefer audiostretchy (quality), fallback to librosa (in-memory).
    """
    r = _safe_float(ratio, default=1.0)
    if not (r > 0.0):
        r = 1.0
    r = float(_clamp(r, 0.01, 100.0))

    if abs(r - 1.0) <= 1e-6:
        return np.asarray(y, dtype=np.float32).reshape(-1)

    tag = f"youdub_tsm_{int(time.time() * 1000)}_{threading.get_ident()}_{uuid.uuid4().hex}"
    tmp_dir = tempfile.gettempdir()
    in_path = os.path.join(tmp_dir, f"{tag}_in.wav")
    out_path = os.path.join(tmp_dir, f"{tag}_out.wav")

    try:
        save_wav(np.asarray(y, dtype=np.float32).reshape(-1), in_path, sample_rate=int(sr))
        stretch_audio(in_path, out_path, ratio=float(r), sample_rate=int(sr))
        y2, _ = librosa.load(out_path, sr=int(sr), mono=True)
        return np.asarray(y2, dtype=np.float32).reshape(-1)
    except Exception as exc:
        logger.warning(f"TSM失败（audiostretchy），回退到librosa: {exc}")
        try:
            rate = float(1.0 / max(float(r), 1e-6))
            y2 = librosa.effects.time_stretch(np.asarray(y, dtype=np.float32).reshape(-1), rate=rate)
            return np.asarray(y2, dtype=np.float32).reshape(-1)
        except Exception as exc2:
            logger.warning(f"TSM失败（librosa），返回原音频: {exc2}")
            return np.asarray(y, dtype=np.float32).reshape(-1)
    finally:
        for p in (in_path, out_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def apply_scaling_ratio(
    audio: np.ndarray,
    sr: int,
    ratio: dict[str, Any],
    *,
    mode: str = "single",  # kept for backward compat (ignored)
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply voice_ratio TSM to audio (global time-stretch).
    """
    sample_rate = int(sr or 0)
    y0 = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate <= 0 or y0.size <= 0:
        return y0, {
            "original_duration": 0.0,
            "scaled_duration": 0.0,
            "voice_ratio_applied": 1.0,
        }

    voice_ratio = _safe_float(ratio.get("voice_ratio", 1.0), default=1.0)

    original_duration = float(y0.shape[0]) / float(sample_rate)

    target_samples = int(round(float(y0.shape[0]) * float(voice_ratio)))
    target_samples = int(max(0, target_samples))
    y_out = _time_stretch(y0, sample_rate, voice_ratio)
    y_out = np.asarray(y_out, dtype=np.float32).reshape(-1)
    if target_samples <= 0:
        y_out = np.zeros((0,), dtype=np.float32)
    elif y_out.shape[0] < target_samples:
        y_out = np.pad(y_out, (0, target_samples - int(y_out.shape[0])), mode="constant")
    else:
        y_out = y_out[:target_samples]

    scaled_duration = float(y_out.shape[0]) / float(sample_rate) if y_out.size > 0 else 0.0

    return y_out, {
        "original_duration": float(original_duration),
        "scaled_duration": float(scaled_duration),
        "voice_ratio_applied": float(voice_ratio),
    }
