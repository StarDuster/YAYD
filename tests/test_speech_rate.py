import numpy as np

from youdub.speech_rate import (
    apply_scaling_ratio,
    count_en_syllables,
    count_zh_syllables,
    compute_en_speech_rate,
    compute_scaling_ratio,
    compute_zh_speech_rate,
)


def test_count_en_syllables_uses_cmudict_for_common_words():
    # hello(2) + world(1)
    assert count_en_syllables("hello world") == 3


def test_count_en_syllables_fallback_for_oov_words():
    # Make sure OOV doesn't crash and always returns >= 1 for non-empty tokens.
    assert count_en_syllables("blorb") >= 1


def test_count_zh_syllables_counts_hanzi_digits_and_latin_words():
    # 你好世界 (4) + 123 (3)
    assert count_zh_syllables("你好，世界！ 123") == 7
    # + QTS (fallback syllable count >= 1)
    assert count_zh_syllables("你好，世界！ 123 QTS") >= 8


def test_compute_en_speech_rate_from_text_and_duration():
    stats = compute_en_speech_rate("hello world", 2.0)
    assert stats["syllable_count"] == 3
    assert abs(stats["duration"] - 2.0) < 1e-9
    assert abs(stats["syllable_rate"] - 1.5) < 1e-9


def test_compute_zh_speech_rate_from_text_and_duration():
    stats = compute_zh_speech_rate("你好世界", 2.0)
    assert stats["syllable_count"] == 4
    assert abs(stats["duration"] - 2.0) < 1e-9
    assert abs(stats["syllable_rate"] - 2.0) < 1e-9


def test_compute_scaling_ratio_single_clamps_voice():
    en = {"syllable_rate": 6.0, "pause_ratio": 0.2}
    zh = {"syllable_rate": 12.0}
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

