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


def test_demucs_model_cache_is_thread_safe_against_unload_race(monkeypatch):
    """
    Guard against regressions like `NameError: _DEMUCS_MODEL is not defined` under concurrency.

    Historical failure mode:
    - unload() temporarily removed the global name (via `del _DEMUCS_MODEL`) without a lock
    - another thread called load_model() at the wrong time -> NameError
    """
    import threading
    import time

    import youdub.steps.separate_vocals as sv

    # Avoid importing / running the real demucs implementation in unit tests.
    calls = {"get_model": 0}

    def _fake_get_model(_name: str):
        calls["get_model"] += 1
        return object()

    monkeypatch.setattr(
        sv,
        "_import_demucs_infer",
        lambda: (_fake_get_model, lambda *_args, **_kwargs: None),
    )

    # Seed a cached model so unload_model() has something to "unload".
    sv._DEMUCS_MODEL = object()  # noqa: SLF001
    sv._DEMUCS_MODEL_NAME = "htdemucs_ft"  # noqa: SLF001
    sv._DEMUCS_DEVICE = sv._AUTO_DEVICE  # noqa: SLF001

    deleted = threading.Event()
    resume = threading.Event()
    orig_unload = sv._unload_model_unlocked  # noqa: SLF001
    armed = {"once": True}

    def _patched_unload():  # noqa: ANN001
        # First call: simulate the old bug window by removing the global name while holding the lock.
        if armed["once"]:
            armed["once"] = False
            sv.__dict__.pop("_DEMUCS_MODEL", None)
            deleted.set()
            resume.wait(timeout=2)
            # Restore a valid state before releasing the lock.
            sv._DEMUCS_MODEL = None  # noqa: SLF001
            sv._DEMUCS_MODEL_NAME = None  # noqa: SLF001
            sv._DEMUCS_DEVICE = None  # noqa: SLF001
            return
        return orig_unload()

    monkeypatch.setattr(sv, "_unload_model_unlocked", _patched_unload)

    box: dict[str, object] = {}

    def _call_load() -> None:
        try:
            box["model"] = sv.load_model("htdemucs_ft", device="auto")
        except Exception as exc:  # noqa: BLE001
            box["exc"] = exc

    t_unload = threading.Thread(target=sv.unload_model, daemon=True)
    t_unload.start()
    assert deleted.wait(timeout=1.0)

    t_load = threading.Thread(target=_call_load, daemon=True)
    t_load.start()

    # If load_model isn't protected by the same lock, it may crash immediately with NameError.
    time.sleep(0.05)

    try:
        resume.set()
        t_unload.join(timeout=2.0)
        t_load.join(timeout=2.0)

        assert "exc" not in box
        assert "model" in box
    finally:
        # Ensure we don't leak module-level cache state to other tests.
        try:
            sv._DEMUCS_MODEL = None  # noqa: SLF001
            sv._DEMUCS_MODEL_NAME = None  # noqa: SLF001
            sv._DEMUCS_DEVICE = None  # noqa: SLF001
        except Exception:
            pass
