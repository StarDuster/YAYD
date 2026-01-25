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
