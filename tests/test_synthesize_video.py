"""Synthesize video 步骤测试."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_synthesize_video_raises_when_missing_inputs(tmp_path: Path):
    import youdub.steps.synthesize_video as sv

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    # Missing translation.json/audio_combined.wav/download.mp4 should raise.
    with pytest.raises(FileNotFoundError):
        sv.synthesize_video(str(folder))


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
    import youdub.steps.synthesize_video as sv

    src = "Hello world. How are you? I'm fine!"
    out = sv._bilingual_source_text(src)
    # Should not truncate to only the first sentence.
    assert out.startswith("Hello world.")
    assert "How are you?" in out
    assert "I'm fine!" in out


def test_bilingual_source_text_splits_overlong_sentence_by_clauses():
    import youdub.steps.synthesize_video as sv

    src = (
        "This is clause one, this is clause two, this is clause three, "
        "this is clause four, this is clause five."
    )
    out = sv._bilingual_source_text(src, max_words=6, max_chars=10_000)
    # When a single sentence is too long, we split by clauses and insert newlines.
    assert "\n" in out
