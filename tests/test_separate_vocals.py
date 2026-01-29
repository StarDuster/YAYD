"""Separate vocals 步骤测试."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _write_dummy_wav(path: Path, sr: int = 24000, seconds: float = 0.5) -> None:
    from youdub.utils import save_wav

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    wav = 0.1 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    save_wav(wav, str(path), sample_rate=sr)


def test_separate_all_audio_under_folder_regenerates_when_outputs_missing(tmp_path: Path, monkeypatch):
    import youdub.steps.separate_vocals as sv

    # Avoid model checks / real Demucs.
    monkeypatch.setattr(sv, "_ensure_demucs_ready", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sv, "load_model", lambda *_args, **_kwargs: None)

    def _fake_extract(folder: str) -> bool:
        p = Path(folder) / "audio.wav"
        p.write_bytes(b"RIFFxxxxWAVEfmt ")  # dummy placeholder
        return True

    def _fake_separate(folder: str, *_args, **_kwargs) -> None:
        _write_dummy_wav(Path(folder) / "audio_vocals.wav", seconds=0.5)
        _write_dummy_wav(Path(folder) / "audio_instruments.wav", seconds=0.5)

    monkeypatch.setattr(sv, "extract_audio_from_video", _fake_extract)
    monkeypatch.setattr(sv, "separate_audio", _fake_separate)

    job = tmp_path / "job"
    job.mkdir(parents=True, exist_ok=True)
    (job / "download.mp4").write_bytes(b"0" * 2048)

    msg = sv.separate_all_audio_under_folder(str(tmp_path))
    assert "processed" in msg
    assert (job / "audio_vocals.wav").exists()
    assert (job / "audio_instruments.wav").exists()


def test_separate_all_audio_under_folder_short_circuits_without_loading_model_when_outputs_exist(
    tmp_path: Path, monkeypatch
):
    import youdub.steps.separate_vocals as sv

    def _boom(*_args, **_kwargs):
        raise AssertionError("should not load/check Demucs when outputs already exist")

    monkeypatch.setattr(sv, "_ensure_demucs_ready", _boom)
    monkeypatch.setattr(sv, "load_model", _boom)
    monkeypatch.setattr(sv, "separate_audio", _boom)
    monkeypatch.setattr(sv, "extract_audio_from_video", _boom)

    job = tmp_path / "job"
    job.mkdir(parents=True, exist_ok=True)
    (job / "download.mp4").write_bytes(b"0" * 2048)
    _write_dummy_wav(job / "audio_vocals.wav", seconds=0.5)
    _write_dummy_wav(job / "audio_instruments.wav", seconds=0.5)

    msg = sv.separate_all_audio_under_folder(str(tmp_path))
    assert "processed 0 files" in msg
