import numpy as np

from youdub.speech_rate import (
    apply_scaling_ratio,
    compute_en_speech_rate,
    compute_scaling_ratio,
    compute_zh_speech_rate,
)


def test_compute_en_speech_rate_basic():
    words = [
        {"start": 0.0, "end": 0.2, "word": "hello", "probability": 0.9},
        {"start": 0.3, "end": 0.5, "word": "world", "probability": 0.8},
    ]
    stats = compute_en_speech_rate(words)
    assert stats["word_count"] == 2
    assert abs(stats["voiced_duration"] - 0.4) < 1e-6
    assert abs(stats["total_duration"] - 0.5) < 1e-6
    assert abs(stats["silence_duration"] - 0.1) < 1e-6
    assert abs(stats["speech_rate"] - (2.0 / 0.4)) < 1e-6
    assert abs(stats["syllable_rate"] - (2.0 / 0.4) * 1.5) < 1e-6
    assert abs(stats["pause_ratio"] - (0.1 / 0.5)) < 1e-6


def test_compute_zh_speech_rate_counts_chars_and_durations():
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / float(sr)
    voice = 0.2 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    silence = np.zeros((sr,), dtype=np.float32)
    y = np.concatenate([voice, silence])

    stats = compute_zh_speech_rate(y, sr, "你好，世界！ 123", top_db=30.0)
    assert stats["char_count"] == 7  # 你好世界 + 123
    assert abs(stats["total_duration"] - 2.0) < 1e-6
    # librosa.effects.split is frame-based; allow some tolerance around the 1s boundary.
    assert 0.85 <= stats["voiced_duration"] <= 1.15
    assert 0.85 <= stats["silence_duration"] <= 1.15
    assert 0.0 <= stats["pause_ratio"] <= 1.0


def test_compute_scaling_ratio_single_clamps_voice():
    en = {"syllable_rate": 6.0, "pause_ratio": 0.2}
    zh = {"syllable_rate": 12.0, "voiced_duration": 1.0, "silence_duration": 0.5, "total_duration": 1.5}
    ratio = compute_scaling_ratio(en, zh, mode="single", voice_min=0.7, voice_max=1.3, overall_min=0.5, overall_max=2.0)
    assert ratio["mode"] == "single"
    assert abs(ratio["voice_ratio_raw"] - 2.0) < 1e-6
    assert ratio["voice_ratio"] == 1.3
    assert ratio["silence_ratio"] == 1.3
    assert ratio["clamped"] is True


def test_apply_scaling_ratio_single_length_exact():
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / float(sr)
    y = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    ratio = {"voice_ratio": 1.25, "silence_ratio": 1.25}
    y2, info = apply_scaling_ratio(y, sr, ratio, mode="single")
    assert y2.dtype == np.float32
    assert y2.shape[0] == int(round(y.shape[0] * 1.25))
    assert abs(info["voice_ratio_applied"] - 1.25) < 1e-6


def test_apply_scaling_ratio_two_stage_changes_only_silence():
    sr = 16000
    sil = np.zeros((sr // 2,), dtype=np.float32)
    t = np.arange(sr, dtype=np.float32) / float(sr)
    voice = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    y = np.concatenate([sil, voice, sil])

    ratio = {"voice_ratio": 1.0, "silence_ratio": 0.5}
    y2, info = apply_scaling_ratio(y, sr, ratio, mode="two_stage")
    assert y2.dtype == np.float32
    # Expected: voice unchanged, silence halved (approx; exact depends on split framing).
    assert info["voice_ratio_applied"] == 1.0
    assert info["silence_ratio_applied"] == 0.5
    assert 1.0 <= info["scaled_duration"] <= 2.0

