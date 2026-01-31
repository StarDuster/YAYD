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
    assert 30 <= font_size <= 60
    assert outline >= 1
    assert margin_v >= 12
    assert max_chars_zh >= 1
    assert max_chars_en >= 1

    # Portrait outputs should use the shorter edge as base_dim, thus same font size for 1080p.
    font_size2, *_rest = sv._calc_subtitle_style_params(1080, 1920)
    assert font_size2 == font_size
