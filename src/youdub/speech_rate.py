from __future__ import annotations

import os
import tempfile
from typing import Any

import librosa
import numpy as np
from audiostretchy.stretch import stretch_audio
from loguru import logger

from .utils import save_wav


def _clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        lo, hi = hi, lo
    return float(min(max(float(x), float(lo)), float(hi)))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    den_f = float(den)
    if not (den_f > 0.0):
        return float(default)
    return float(num) / den_f


def _count_zh_chars(text: str) -> int:
    # "字≈音节" 的粗近似：过滤空白/标点，仅保留字母数字（含 CJK 字符）。
    return int(sum(1 for ch in (text or "") if ch.isalnum()))


def compute_en_speech_rate(words: list[dict[str, Any]]) -> dict[str, Any]:
    """
    从词级时间戳计算英文语速统计。

    说明：
    - total_duration 仅由 words 的 min(start)~max(end) 估计（不含段落边界外静音）。
    - voiced_duration 用 sum(word_dur) 估计（近似发声时长）。
    """
    if not isinstance(words, list) or not words:
        return {
            "word_count": 0,
            "voiced_duration": 0.0,
            "total_duration": 0.0,
            "silence_duration": 0.0,
            "speech_rate": 0.0,
            "syllable_rate": 0.0,
            "pause_ratio": 0.0,
        }

    starts: list[float] = []
    ends: list[float] = []
    voiced = 0.0
    count = 0
    for w in words:
        if not isinstance(w, dict):
            continue
        try:
            s = float(w.get("start", 0.0) or 0.0)
            e = float(w.get("end", 0.0) or 0.0)
        except Exception:
            continue
        if not (e > s):
            continue
        starts.append(s)
        ends.append(e)
        voiced += float(e - s)
        count += 1

    if count <= 0 or not starts or not ends:
        return {
            "word_count": 0,
            "voiced_duration": 0.0,
            "total_duration": 0.0,
            "silence_duration": 0.0,
            "speech_rate": 0.0,
            "syllable_rate": 0.0,
            "pause_ratio": 0.0,
        }

    total = float(max(ends) - min(starts))
    if total < 0.0:
        total = 0.0
    # Prevent pathological overlap (shouldn't happen, but be defensive).
    voiced = float(min(voiced, total)) if total > 0.0 else float(voiced)
    silence = float(max(0.0, total - voiced))

    word_rate = _safe_div(float(count), voiced, default=0.0)
    syll_rate = float(word_rate * 1.5)
    pause_ratio = _safe_div(silence, total, default=0.0)
    return {
        "word_count": int(count),
        "voiced_duration": float(voiced),
        "total_duration": float(total),
        "silence_duration": float(silence),
        "speech_rate": float(word_rate),
        "syllable_rate": float(syll_rate),
        "pause_ratio": float(pause_ratio),
    }


def compute_zh_speech_rate(audio: np.ndarray, sr: int, text: str, *, top_db: float = 30.0) -> dict[str, Any]:
    """从 TTS 音频与文本粗估中文语速（字≈音节）。"""
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr_i = int(sr)
    if sr_i <= 0:
        sr_i = 24000

    total = float(y.shape[0]) / float(sr_i) if y.size > 0 else 0.0
    voiced = 0.0
    if y.size > 0:
        try:
            intervals = librosa.effects.split(y, top_db=float(top_db))
            voiced = float(sum((int(e) - int(s)) for s, e in intervals)) / float(sr_i)
        except Exception:
            voiced = 0.0
    voiced = float(min(max(voiced, 0.0), total)) if total > 0.0 else float(max(voiced, 0.0))
    silence = float(max(0.0, total - voiced))

    char_count = _count_zh_chars(text)
    char_rate = _safe_div(float(char_count), voiced, default=0.0)
    pause_ratio = _safe_div(silence, total, default=0.0)
    return {
        "char_count": int(char_count),
        "voiced_duration": float(voiced),
        "total_duration": float(total),
        "silence_duration": float(silence),
        "speech_rate": float(char_rate),
        "syllable_rate": float(char_rate),
        "pause_ratio": float(pause_ratio),
    }


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
    计算缩放比例，并应用约束。

    ratio 定义：new_duration = ratio * old_duration
    - ratio < 1: 加速（压缩时长）
    - ratio > 1: 减速（拉伸时长）
    """
    mode_s = str(mode or "single").strip().lower()
    if mode_s not in {"single", "two_stage", "two-stage"}:
        mode_s = "single"
    mode_s = "two_stage" if mode_s in {"two_stage", "two-stage"} else "single"

    en_syll = float(en_stats.get("syllable_rate", 0.0) or 0.0) if isinstance(en_stats, dict) else 0.0
    zh_syll = float(zh_stats.get("syllable_rate", 0.0) or 0.0) if isinstance(zh_stats, dict) else 0.0

    voice_raw = _safe_div(zh_syll, en_syll, default=1.0) if (en_syll > 0.0 and zh_syll > 0.0) else 1.0
    voice = _clamp(voice_raw, float(voice_min), float(voice_max))

    # single: pause scaling == voice scaling
    if mode_s == "single":
        overall = _clamp(voice, float(overall_min), float(overall_max))
        voice = _clamp(overall, float(voice_min), float(voice_max))
        silence = voice
        return {
            "mode": "single",
            "voice_ratio": float(voice),
            "silence_ratio": float(silence),
            "overall_ratio": float(voice),
            "voice_ratio_raw": float(voice_raw),
            "silence_ratio_raw": float(voice_raw),
            "clamped": bool(abs(float(voice) - float(voice_raw)) > 1e-6),
        }

    # two-stage: compute pause ratio + solve for silence scaling
    p_en = float(en_stats.get("pause_ratio", 0.0) or 0.0) if isinstance(en_stats, dict) else 0.0
    p_en = float(_clamp(p_en, 0.0, 1.0))

    v_zh = float(zh_stats.get("voiced_duration", 0.0) or 0.0) if isinstance(zh_stats, dict) else 0.0
    s_zh = float(zh_stats.get("silence_duration", 0.0) or 0.0) if isinstance(zh_stats, dict) else 0.0
    t_zh = float(zh_stats.get("total_duration", 0.0) or 0.0) if isinstance(zh_stats, dict) else 0.0
    if t_zh <= 0.0:
        t_zh = float(max(v_zh + s_zh, 0.0))

    silence_raw = 1.0
    if (s_zh > 1e-6) and (0.0 < p_en < 1.0):
        silence_raw = float((p_en * voice_raw * v_zh) / ((1.0 - p_en) * s_zh))
    silence = _clamp(silence_raw, float(silence_min), float(silence_max))

    def _overall_ratio(v_ratio: float, s_ratio: float) -> float:
        if t_zh <= 0.0:
            return float(v_ratio)
        return float((float(v_ratio) * float(v_zh) + float(s_ratio) * float(s_zh)) / float(t_zh))

    overall = _overall_ratio(voice, silence)

    # Try to satisfy overall bounds by adjusting silence first (voice is more fragile perceptually).
    overall_lo = float(min(float(overall_min), float(overall_max)))
    overall_hi = float(max(float(overall_min), float(overall_max)))
    if (overall < overall_lo - 1e-9) or (overall > overall_hi + 1e-9):
        target = float(_clamp(overall, overall_lo, overall_hi))
        if s_zh > 1e-6 and t_zh > 0.0:
            need_s = _safe_div(target * t_zh - voice * v_zh, s_zh, default=silence)
            silence = _clamp(need_s, float(silence_min), float(silence_max))
            overall = _overall_ratio(voice, silence)
        if (overall < overall_lo - 1e-9) or (overall > overall_hi + 1e-9):
            if v_zh > 1e-6 and t_zh > 0.0:
                need_v = _safe_div(target * t_zh - silence * s_zh, v_zh, default=voice)
                voice = _clamp(need_v, float(voice_min), float(voice_max))
                overall = _overall_ratio(voice, silence)
        overall = float(_clamp(overall, overall_lo, overall_hi))

    clamped = bool(
        abs(float(voice) - float(voice_raw)) > 1e-6
        or abs(float(silence) - float(silence_raw)) > 1e-6
        or (overall < overall_lo - 1e-6)
        or (overall > overall_hi + 1e-6)
    )
    return {
        "mode": "two_stage",
        "voice_ratio": float(voice),
        "silence_ratio": float(silence),
        "overall_ratio": float(overall),
        "voice_ratio_raw": float(voice_raw),
        "silence_ratio_raw": float(silence_raw),
        "clamped": bool(clamped),
    }


def _read_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    s = str(raw).strip()
    if not s:
        return float(default)
    try:
        return float(s)
    except Exception:
        return float(default)


def _fade_edges(y: np.ndarray, sr: int, *, fade_ms: float = 5.0) -> np.ndarray:
    out = np.asarray(y, dtype=np.float32).reshape(-1)
    n = int(round(float(sr) * float(fade_ms) / 1000.0))
    if out.size <= 0 or n <= 0:
        return out
    n = int(min(n, out.size // 2))
    if n <= 0:
        return out
    ramp = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def _stretch_audio_array(y: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """
    TSM: new_duration = ratio * old_duration.
    Prefer audiostretchy; fallback to librosa phase-vocoder when needed.
    """
    y0 = np.asarray(y, dtype=np.float32).reshape(-1)
    if y0.size <= 0:
        return y0
    ratio_f = float(ratio)
    if not (ratio_f > 0.0):
        return y0
    if abs(ratio_f - 1.0) < 1e-6:
        return y0

    # audiostretchy works on WAV files (via stdlib wave); write PCM16 temp WAV for compatibility.
    with tempfile.TemporaryDirectory(prefix="youdub_speechrate_") as tmpdir:
        in_path = os.path.join(tmpdir, "in.wav")
        out_path = os.path.join(tmpdir, "out.wav")
        try:
            save_wav(y0, in_path, sample_rate=int(sr))
            stretch_audio(in_path, out_path, ratio=ratio_f, sample_rate=int(sr))
            y1, _ = librosa.load(out_path, sr=int(sr), mono=True)
            return np.asarray(y1, dtype=np.float32).reshape(-1)
        except Exception as exc:
            logger.warning(f"audiostretchy 拉伸失败 (ratio={ratio_f:.3f}): {exc} (回退到librosa)")
            try:
                rate = float(1.0 / max(ratio_f, 1e-6))
                y1 = librosa.effects.time_stretch(y0.astype(np.float32, copy=False), rate=rate)
                return np.asarray(y1, dtype=np.float32).reshape(-1)
            except Exception as exc2:
                logger.warning(f"librosa 拉伸失败 (ratio={ratio_f:.3f}): {exc2} (回退到原音频)")
                return y0


def apply_scaling_ratio(
    audio: np.ndarray,
    sr: int,
    ratio: dict[str, Any],
    *,
    mode: str = "single",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    对音频应用缩放比例。
    - single: 整段 TSM
    - two_stage: 发声段 TSM，静音段按比例裁剪/扩展（用零填充）
    """
    y = np.asarray(audio, dtype=np.float32).reshape(-1)
    sr_i = int(sr)
    if sr_i <= 0:
        sr_i = 24000

    voice_ratio = float(ratio.get("voice_ratio", 1.0) or 1.0) if isinstance(ratio, dict) else 1.0
    silence_ratio = float(ratio.get("silence_ratio", voice_ratio) or voice_ratio) if isinstance(ratio, dict) else voice_ratio
    mode_s = str(mode or ratio.get("mode", "single") if isinstance(ratio, dict) else "single").strip().lower()
    mode_s = "two_stage" if mode_s in {"two_stage", "two-stage"} else "single"

    original_duration = float(y.size) / float(sr_i) if y.size > 0 else 0.0

    if mode_s == "single":
        y1 = _stretch_audio_array(y, sr_i, voice_ratio)
        target_samples = int(round(float(y.size) * float(voice_ratio)))
        if target_samples <= 0:
            y1 = np.zeros((0,), dtype=np.float32)
        elif y1.size < target_samples:
            y1 = np.pad(y1, (0, target_samples - int(y1.size)), mode="constant")
        else:
            y1 = y1[:target_samples]
        scaled_duration = float(y1.size) / float(sr_i) if y1.size > 0 else 0.0
        return y1.astype(np.float32, copy=False), {
            "original_duration": float(original_duration),
            "scaled_duration": float(scaled_duration),
            "voice_ratio_applied": float(voice_ratio),
            "silence_ratio_applied": float(voice_ratio),
        }

    # two-stage
    top_db = _read_env_float("SPEECH_RATE_ZH_VAD_TOP_DB", 30.0)
    intervals: np.ndarray
    try:
        intervals = librosa.effects.split(y, top_db=float(top_db))
    except Exception:
        intervals = np.zeros((0, 2), dtype=np.int64)

    chunks: list[np.ndarray] = []
    pos = 0
    for s, e in intervals:
        s_i = int(max(0, int(s)))
        e_i = int(min(int(y.size), int(e)))
        if e_i <= s_i:
            continue

        if s_i > pos:
            sil = y[pos:s_i]
            new_n = int(round(float(sil.size) * float(silence_ratio)))
            chunks.append(np.zeros((max(0, new_n),), dtype=np.float32))
        voice = y[s_i:e_i]
        voice_stretched = _stretch_audio_array(voice, sr_i, voice_ratio)
        voice_target = int(round(float(voice.size) * float(voice_ratio)))
        if voice_target > 0:
            if voice_stretched.size < voice_target:
                voice_stretched = np.pad(voice_stretched, (0, voice_target - int(voice_stretched.size)), mode="constant")
            else:
                voice_stretched = voice_stretched[:voice_target]
        chunks.append(_fade_edges(voice_stretched, sr_i, fade_ms=5.0))
        pos = e_i

    if pos < int(y.size):
        sil = y[pos:]
        new_n = int(round(float(sil.size) * float(silence_ratio)))
        chunks.append(np.zeros((max(0, new_n),), dtype=np.float32))

    y1 = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
    scaled_duration = float(y1.size) / float(sr_i) if y1.size > 0 else 0.0
    return y1.astype(np.float32, copy=False), {
        "original_duration": float(original_duration),
        "scaled_duration": float(scaled_duration),
        "voice_ratio_applied": float(voice_ratio),
        "silence_ratio_applied": float(silence_ratio),
    }

