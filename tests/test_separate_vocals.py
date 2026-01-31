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


def test_demucs_model_cache_globals_defined_and_load_model_does_not_nameerror(monkeypatch):
    """Guard against regressions like `NameError: _DEMUCS_MODEL is not defined`."""
    import youdub.steps.separate_vocals as sv

    assert hasattr(sv, "_DEMUCS_MODEL")
    assert hasattr(sv, "_DEMUCS_MODEL_NAME")
    assert hasattr(sv, "_DEMUCS_DEVICE")

    calls = {"get_model": 0}

    def _fake_get_model(_name: str):
        calls["get_model"] += 1
        return object()

    # Avoid importing / running the real demucs implementation in unit tests.
    monkeypatch.setattr(
        sv,
        "_import_demucs_infer",
        lambda: (_fake_get_model, lambda *_args, **_kwargs: None),
    )

    try:
        m1 = sv.load_model("htdemucs_ft", device="cpu")
        m2 = sv.load_model("htdemucs_ft", device="cpu")
        assert m1 is m2
        assert calls["get_model"] == 1

        sv.unload_model()
        assert sv._DEMUCS_MODEL is None
        assert sv._DEMUCS_MODEL_NAME is None
        assert sv._DEMUCS_DEVICE is None
    finally:
        # Ensure we don't leak module-level cache state to other tests.
        setattr(sv, "_DEMUCS_MODEL", None)
        setattr(sv, "_DEMUCS_MODEL_NAME", None)
        setattr(sv, "_DEMUCS_DEVICE", None)
