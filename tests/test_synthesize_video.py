"""Synthesize video 步骤测试."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest


def test_synthesize_video_raises_when_missing_inputs(tmp_path: Path):
    import youdub.steps.synthesize_video as sv

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    # Missing translation.json/audio_combined.wav/download.mp4 should raise.
    with pytest.raises(FileNotFoundError):
        sv.synthesize_video(str(folder))


def test_ensure_audio_combined_adaptive_generates_plan_and_audio(tmp_path: Path, monkeypatch):
    import youdub.steps.synthesize_video as sv
    from youdub.utils import save_wav

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "wavs").mkdir(parents=True, exist_ok=True)

    # Minimal translation timeline (1 segment == 1 wav).
    (folder / "translation.json").write_text(
        json.dumps(
            [{"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_00", "translation": "好。"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Create a short TTS wav with leading/trailing silence so trim() is exercised.
    sr = 24000
    tone_s = 0.2
    n_tone = int(sr * tone_s)
    t = np.linspace(0.0, tone_s, n_tone, endpoint=False, dtype=np.float32)
    tone = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32, copy=False)
    silence = np.zeros((int(sr * 0.05),), dtype=np.float32)
    wav = np.concatenate([silence, tone, silence]).astype(np.float32, copy=False)
    save_wav(wav, str(folder / "wavs" / "0000.wav"), sample_rate=sr)
    (folder / "wavs" / ".tts_done.json").write_text(
        json.dumps({"tts_method": "bytedance"}, ensure_ascii=False),
        encoding="utf-8",
    )

    # Disable speech-rate alignment to keep test fast/deterministic.
    monkeypatch.setenv("SPEECH_RATE_ALIGN_ENABLED", "0")

    # Should generate plan + adaptive timeline + audio outputs without needing instruments/video.
    sv._ensure_audio_combined(str(folder), adaptive_segment_stretch=True, sample_rate=sr)  # noqa: SLF001

    assert (folder / "translation_adaptive.json").exists()
    assert (folder / "adaptive_plan.json").exists()
    assert (folder / "audio_tts.wav").exists()
    assert (folder / "audio_combined.wav").exists()

    meta = json.loads((folder / ".audio_combined.json").read_text(encoding="utf-8"))
    assert meta.get("adaptive_segment_stretch") is True

    with wave.open(str(folder / "audio_combined.wav"), "rb") as wf:
        assert int(wf.getframerate() or 0) == sr


def test_subtitle_style_params_1080p_not_tiny():
    import youdub.steps.synthesize_video as sv

    font_size, outline, margin_v, max_chars_zh, max_chars_en = sv._calc_subtitle_style_params(1920, 1080)
    assert font_size == 49
    assert outline >= 1
    assert margin_v >= 10
    assert max_chars_zh >= 1
    assert max_chars_en >= 1

    # Portrait outputs should be slightly smaller than landscape for the same 1080 short edge.
    font_size2, *_rest = sv._calc_subtitle_style_params(1080, 1920)
    assert 24 <= font_size2 <= 49
    assert font_size2 < font_size


def test_subtitle_style_params_ultrawide_not_smaller_than_16_9():
    import youdub.steps.synthesize_video as sv

    font_16_9, *_rest = sv._calc_subtitle_style_params(1920, 1080)
    font_21_9, *_rest2 = sv._calc_subtitle_style_params(2560, 1080)
    assert font_21_9 >= font_16_9


def test_subtitle_style_params_4k_not_huge():
    import youdub.steps.synthesize_video as sv

    # Guard against regressions that make subtitles excessively large on 4K outputs.
    # (e.g. accidentally using max(width,height) or an overly aggressive scale factor)
    font_size, *_rest = sv._calc_subtitle_style_params(3840, 2160)
    assert font_size < 100


def test_bilingual_source_text_keeps_multiple_sentences():
    import youdub.steps.synthesize_video_subtitles as subs

    src = "Hello world. How are you? I'm fine!"
    out = subs._bilingual_source_text(src)
    # Should not truncate to only the first sentence.
    assert out.startswith("Hello world.")
    assert "How are you?" in out
    assert "I'm fine!" in out


def test_bilingual_source_text_splits_overlong_sentence_by_clauses():
    import youdub.steps.synthesize_video_subtitles as subs

    src = (
        "This is clause one, this is clause two, this is clause three, "
        "this is clause four, this is clause five."
    )
    out = subs._bilingual_source_text(src, max_words=6, max_chars=10_000)
    # When a single sentence is too long, we split by clauses and insert newlines.
    assert "\n" in out
