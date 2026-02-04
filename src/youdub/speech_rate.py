from __future__ import annotations

import math
import os
import re
import tempfile
import threading
import time
import uuid
from typing import Any

import librosa
import numpy as np
import pronouncing
from audiostretchy.stretch import stretch_audio
from loguru import logger
from pypinyin import Style, pinyin

from .utils import save_wav


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


def compute_clamp_gap(raw_ratio: Any, applied_ratio: Any) -> float:
    """
    Measure how far a ratio was clamped.

    Returns a multiplicative gap >= 1.0:
      gap = max(raw/applied, applied/raw)

    - gap == 1.0 means no clamping.
    - larger gap means raw was further from the applied ratio.
    """
    raw = _safe_float(raw_ratio, default=1.0)
    applied = _safe_float(applied_ratio, default=1.0)
    if raw <= 0.0 or applied <= 0.0:
        # Degenerate input: treat as extreme.
        return float("inf") if abs(raw - applied) > 1e-12 else 1.0
    try:
        a = float(raw / applied)
        b = float(applied / raw)
        gap = float(max(a, b))
        return gap if math.isfinite(gap) and gap >= 1.0 else 1.0
    except Exception:
        return 1.0


def is_outlier_by_clamp_gap(
    raw_ratio: Any,
    applied_ratio: Any,
    *,
    clamp_gap_threshold: float = 1.7,
) -> bool:
    """
    Decide whether a per-segment ratio is unreliable (outlier) based on clamp gap.
    """
    th = _safe_float(clamp_gap_threshold, default=1.7)
    if not (th > 1.0):
        th = 1.0
    gap = compute_clamp_gap(raw_ratio, applied_ratio)
    return bool(gap > float(th) + 1e-9)


def compute_global_baseline(
    *,
    en_total_duration: Any | None = None,
    zh_total_duration: Any | None = None,
    en_total_syllables: Any | None = None,
    zh_total_syllables: Any | None = None,
    mode: str = "duration",
    default: float = 1.0,
) -> float:
    """
    Compute a global baseline scaling ratio for fallback.

    ratio definition:
        ratio = new_duration / old_duration

    Supported modes:
    - duration: en_total_duration / zh_total_duration
    - syllable: en_total_syllables / zh_total_syllables
    """
    m = str(mode or "duration").strip().lower()
    dflt = _safe_float(default, default=1.0)
    if not (dflt > 0.0):
        dflt = 1.0

    if m in {"duration", "dur", "time"}:
        en = _safe_float(en_total_duration, default=0.0)
        zh = _safe_float(zh_total_duration, default=0.0)
        if en > 0.0 and zh > 0.0:
            r = float(en / zh)
            return r if math.isfinite(r) and r > 0.0 else dflt
        return dflt

    if m in {"syllable", "syllables", "syl"}:
        try:
            en_n = int(en_total_syllables or 0)
            zh_n = int(zh_total_syllables or 0)
        except Exception:
            en_n, zh_n = 0, 0
        if en_n > 0 and zh_n > 0:
            r = float(float(en_n) / float(zh_n))
            return r if math.isfinite(r) and r > 0.0 else dflt
        return dflt

    return dflt


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
    Count English syllables using CMUdict (via `pronouncing`) with a rule-based fallback.
    """
    # Keep a small amount of normalization for numeric/currency/initialisms.
    # Examples:
    # - 2024 -> "2 0 2 4" (approx)
    # - $1.13 -> "1 point 1 3 dollars" (approx)
    # - QTS -> "Q T S" (initialism)
    s = str(text or "")
    token_re = re.compile(r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?|[A-Za-z]+(?:'[A-Za-z]+)?")
    tokens = token_re.findall(s)
    if not tokens:
        return 0

    # Letter-name syllables for initialisms (approx).
    # Most letter names are 1 syllable; W is the main exception.
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

            # Number syllables (very rough but better than dropping numbers).
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

            # Add spoken suffix approximation.
            if is_money:
                num_syl += 2  # "dollars" (2 syllables)
            if is_percent:
                num_syl += 2  # "percent" (2 syllables)

            total += int(max(1, num_syl)) if raw else 0
            continue

        # Alphabetic tokens: try CMUdict first.
        w = t.lower()
        try:
            phones = pronouncing.phones_for_word(w)
        except Exception:
            phones = []
        if phones:
            try:
                total += int(sum(1 for p in str(phones[0]).split() if p and p[-1].isdigit()))
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
        # Fallback: count CJK Unified Ideographs.
        cjk_count = int(sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff"))

    latin_words = re.findall(r"[A-Za-z]+", s)
    latin_count = int(sum(count_en_syllables(w) for w in latin_words)) if latin_words else 0
    digit_count = int(sum(1 for ch in s if ch.isdigit()))
    return int(cjk_count + latin_count + digit_count)


def compute_en_speech_rate(
    text: str,
    duration: float,
) -> dict[str, Any]:
    """
    Compute English speech stats from subtitle text + segment duration.
    """
    dur = _safe_float(duration, default=0.0)
    dur = float(max(0.0, dur))
    syllables = int(count_en_syllables(text))
    syllable_rate = float(syllables / dur) if dur > 0 else 0.0
    return {"syllable_count": int(syllables), "duration": float(dur), "syllable_rate": float(syllable_rate)}


def compute_zh_speech_rate(
    text: str,
    duration: float,
) -> dict[str, Any]:
    """
    Compute Chinese speech stats from translated subtitle text + TTS segment duration.
    """
    dur = _safe_float(duration, default=0.0)
    dur = float(max(0.0, dur))
    syllables = int(count_zh_syllables(text))
    syllable_rate = float(syllables / dur) if dur > 0 else 0.0
    return {"syllable_count": int(syllables), "duration": float(dur), "syllable_rate": float(syllable_rate)}


def compute_scaling_ratio(
    en_stats: dict[str, Any],
    zh_stats: dict[str, Any],
    *,
    mode: str = "single",
    voice_min: float = 0.7,
    voice_max: float = 1.3,
    silence_min: float = 0.3,
    silence_max: float = 3.0,
    overall_min: float = 0.5,
    overall_max: float = 2.0,
) -> dict[str, Any]:
    """
    Compute scaling ratios for time-scale modification (TSM).

    ratio definition:
        ratio = new_duration / old_duration

    By default, the *voice* ratio is derived from the speech-rate ratio:
        voice_ratio_rate_raw = zh_rate / en_rate
    """
    mode_norm = str(mode or "single").strip().lower()
    if mode_norm in {"two_stage", "two-stage"}:
        mode_norm = "two_stage"
    if mode_norm not in {"single", "two_stage"}:
        mode_norm = "single"

    en_rate = _safe_float(en_stats.get("syllable_rate", 0.0), default=0.0)
    zh_rate = _safe_float(zh_stats.get("syllable_rate", 0.0), default=0.0)
    if not (en_rate > 0.0 and zh_rate > 0.0):
        return {
            "mode": mode_norm,
            "voice_ratio": 1.0,
            "silence_ratio": 1.0,
            "overall_ratio": 1.0,
            "voice_ratio_raw": 1.0,
            "voice_ratio_rate_raw": 1.0,
            "voice_ratio_budget_raw": 1.0,
            "speech_rate_budget_weight": 0.0,
            "silence_ratio_raw": 1.0,
            "clamped": False,
        }

    voice_ratio_rate_raw = float(zh_rate / en_rate)
    voice_ratio_raw = float(voice_ratio_rate_raw)

    voice_ratio = _clamp(voice_ratio_raw, float(voice_min), float(voice_max))

    V_zh = _safe_float(zh_stats.get("voiced_duration", 0.0), default=0.0)
    S_zh = _safe_float(zh_stats.get("silence_duration", 0.0), default=0.0)
    T_zh = float(max(0.0, V_zh + S_zh))

    silence_ratio_raw = float(voice_ratio_raw)
    silence_ratio = float(voice_ratio)

    clamped = (abs(voice_ratio - voice_ratio_raw) > 1e-9)

    if mode_norm == "two_stage":
        p_en = _safe_float(en_stats.get("pause_ratio", 0.0), default=0.0)
        p_en = float(_clamp(p_en, 0.0, 1.0))
        if S_zh <= 1e-6:
            silence_ratio_raw = float(voice_ratio_raw)
            silence_ratio = float(voice_ratio)
        else:
            denom = float(max(1e-6, (1.0 - p_en) * S_zh))
            silence_ratio_raw = float((p_en * voice_ratio * V_zh) / denom) if p_en > 0 else 0.0
            silence_ratio = _clamp(silence_ratio_raw, float(silence_min), float(silence_max))
            if abs(silence_ratio - silence_ratio_raw) > 1e-9:
                clamped = True

    if T_zh <= 1e-6:
        overall_ratio = float(voice_ratio)
    else:
        overall_ratio = float((voice_ratio * V_zh + silence_ratio * S_zh) / T_zh)

    # Enforce overall bounds by adjusting silence ratio first (voice ratio is the critical one).
    if overall_ratio > float(overall_max) + 1e-9:
        if S_zh > 1e-6 and T_zh > 1e-6:
            target = float(overall_max)
            needed = float((target * T_zh - voice_ratio * V_zh) / S_zh)
            new_silence = _clamp(needed, float(silence_min), float(silence_max))
            if abs(new_silence - silence_ratio) > 1e-9:
                silence_ratio = float(new_silence)
                clamped = True
                overall_ratio = float((voice_ratio * V_zh + silence_ratio * S_zh) / T_zh)
        else:
            # No silence to adjust: clamp voice ratio within the overall bounds.
            new_voice = _clamp(voice_ratio, float(max(voice_min, overall_min)), float(min(voice_max, overall_max)))
            if abs(new_voice - voice_ratio) > 1e-9:
                voice_ratio = float(new_voice)
                clamped = True
            overall_ratio = float(voice_ratio)

    if overall_ratio < float(overall_min) - 1e-9:
        if S_zh > 1e-6 and T_zh > 1e-6:
            target = float(overall_min)
            needed = float((target * T_zh - voice_ratio * V_zh) / S_zh)
            new_silence = _clamp(needed, float(silence_min), float(silence_max))
            if abs(new_silence - silence_ratio) > 1e-9:
                silence_ratio = float(new_silence)
                clamped = True
                overall_ratio = float((voice_ratio * V_zh + silence_ratio * S_zh) / T_zh)
        else:
            new_voice = _clamp(voice_ratio, float(max(voice_min, overall_min)), float(min(voice_max, overall_max)))
            if abs(new_voice - voice_ratio) > 1e-9:
                voice_ratio = float(new_voice)
                clamped = True
            overall_ratio = float(voice_ratio)

    return {
        "mode": mode_norm,
        "voice_ratio": float(voice_ratio),
        "silence_ratio": float(silence_ratio),
        "overall_ratio": float(overall_ratio),
        "voice_ratio_raw": float(voice_ratio_raw),
        "voice_ratio_rate_raw": float(voice_ratio_rate_raw),
        # budget is intentionally not used (keep placeholder keys stable for downstream debug tools)
        "voice_ratio_budget_raw": 1.0,
        "speech_rate_budget_weight": 0.0,
        "silence_ratio_raw": float(silence_ratio_raw),
        "clamped": bool(clamped),
    }


def _apply_fade(wav: np.ndarray, sr: int, *, fade_ms: float = 5.0) -> np.ndarray:
    y = np.asarray(wav, dtype=np.float32)
    n = int(y.shape[0])
    if n <= 0:
        return y
    fade_n = int(round(float(sr) * float(fade_ms) / 1000.0))
    fade_n = int(max(0, min(fade_n, n // 4, 2048)))
    if fade_n <= 1:
        return y
    ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
    y[:fade_n] *= ramp
    y[-fade_n:] *= ramp[::-1]
    return y


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
            # librosa: new_duration = old_duration / rate  => rate = 1 / ratio
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
    mode: str = "single",
    vad_top_db: float = 30.0,
    fade_ms: float = 5.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Apply scaling ratios to audio.

    - single: global TSM with voice_ratio
    - two_stage:
        1) global TSM with voice_ratio
        2) rebuild timeline by re-scaling detected silence spans to silence_ratio
    """
    sample_rate = int(sr or 0)
    y0 = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate <= 0 or y0.size <= 0:
        return y0, {
            "original_duration": 0.0,
            "scaled_duration": 0.0,
            "voice_ratio_applied": 1.0,
            "silence_ratio_applied": 1.0,
        }

    voice_ratio = _safe_float(ratio.get("voice_ratio", 1.0), default=1.0)
    silence_ratio = _safe_float(ratio.get("silence_ratio", voice_ratio), default=voice_ratio)
    mode_norm = str(mode or "single").strip().lower()
    if mode_norm not in {"single", "two_stage"}:
        mode_norm = "single"

    original_duration = float(y0.shape[0]) / float(sample_rate)

    # Always do one global TSM for the voice ratio, then pad/trim to exact target samples.
    target_voice_samples = int(round(float(y0.shape[0]) * float(voice_ratio)))
    target_voice_samples = int(max(0, target_voice_samples))
    y_voice = _time_stretch(y0, sample_rate, voice_ratio)
    y_voice = np.asarray(y_voice, dtype=np.float32).reshape(-1)
    if target_voice_samples <= 0:
        y_voice = np.zeros((0,), dtype=np.float32)
    elif y_voice.shape[0] < target_voice_samples:
        y_voice = np.pad(y_voice, (0, target_voice_samples - int(y_voice.shape[0])), mode="constant")
    else:
        y_voice = y_voice[:target_voice_samples]

    scaled_duration = float(y_voice.shape[0]) / float(sample_rate) if y_voice.size > 0 else 0.0
    voice_ratio_applied = float(voice_ratio)

    if mode_norm == "single" or abs(float(silence_ratio) - float(voice_ratio)) <= 1e-6:
        return y_voice, {
            "original_duration": float(original_duration),
            "scaled_duration": float(scaled_duration),
            "voice_ratio_applied": float(voice_ratio_applied),
            "silence_ratio_applied": float(voice_ratio_applied),
        }

    # two_stage: adjust silence spans
    try:
        intervals = librosa.effects.split(y0, top_db=float(vad_top_db))
    except Exception:
        intervals = np.zeros((0, 2), dtype=np.int64)

    segments: list[tuple[str, int, int]] = []
    cursor = 0
    try:
        for st, ed in intervals:
            st_i = int(st)
            ed_i = int(ed)
            if ed_i <= st_i:
                continue
            if st_i > cursor:
                segments.append(("silence", int(cursor), int(st_i)))
            segments.append(("voice", int(st_i), int(ed_i)))
            cursor = int(ed_i)
    except Exception:
        segments = []
        cursor = 0

    if cursor < int(y0.shape[0]):
        segments.append(("silence", int(cursor), int(y0.shape[0])))

    if not segments:
        # All silence: just return scaled silence (zeros) by silence_ratio.
        target = int(round(float(y0.shape[0]) * float(silence_ratio)))
        target = int(max(0, target))
        return np.zeros((target,), dtype=np.float32), {
            "original_duration": float(original_duration),
            "scaled_duration": float(target) / float(sample_rate) if sample_rate > 0 else 0.0,
            "voice_ratio_applied": float(voice_ratio_applied),
            "silence_ratio_applied": float(silence_ratio),
        }

    # Map original sample indices to y_voice indices using the actually produced global scaling.
    idx_scale = float(y_voice.shape[0]) / float(y0.shape[0]) if y0.shape[0] > 0 else float(voice_ratio)
    idx_scale = float(max(1e-6, idx_scale))

    out_chunks: list[np.ndarray] = []
    for kind, s0, s1 in segments:
        s0 = int(max(0, min(s0, int(y0.shape[0]))))
        s1 = int(max(0, min(s1, int(y0.shape[0]))))
        if s1 <= s0:
            continue
        orig_len = int(s1 - s0)

        if kind == "voice":
            vs = int(round(float(s0) * idx_scale))
            ve = int(round(float(s1) * idx_scale))
            vs = int(max(0, min(vs, int(y_voice.shape[0]))))
            ve = int(max(0, min(ve, int(y_voice.shape[0]))))
            if ve <= vs:
                continue
            chunk = y_voice[vs:ve].astype(np.float32, copy=False)
            chunk = _apply_fade(chunk, sample_rate, fade_ms=float(fade_ms))
            out_chunks.append(chunk)
        else:
            # Silence: generate clean zeros at the desired scaled length.
            n = int(round(float(orig_len) * float(silence_ratio)))
            if n <= 0:
                continue
            out_chunks.append(np.zeros((n,), dtype=np.float32))

    y_out = np.concatenate(out_chunks).astype(np.float32, copy=False) if out_chunks else np.zeros((0,), dtype=np.float32)
    return y_out, {
        "original_duration": float(original_duration),
        "scaled_duration": float(y_out.shape[0]) / float(sample_rate) if sample_rate > 0 else 0.0,
        "voice_ratio_applied": float(voice_ratio_applied),
        "silence_ratio_applied": float(silence_ratio),
    }
