import json
import os
from pathlib import Path

import numpy as np
import pytest


def _write_dummy_wav(path: Path, sr: int = 24000, seconds: float = 1.0) -> None:
    from youdub.core.utils import save_wav

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    wav = 0.1 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    save_wav(wav, str(path), sample_rate=sr)


def test_download_single_video_returns_none_when_file_missing(tmp_path: Path, monkeypatch):
    import youdub.core.steps.download as dl

    class _StubYdl:
        def __init__(self, _opts):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def download(self, _urls):
            return 1

    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _StubYdl)

    info = {
        "title": "t",
        "uploader": "u",
        "upload_date": "20260101",
        "webpage_url": "https://example.com/v",
    }
    out = dl.download_single_video(info, str(tmp_path), resolution="360p")
    assert out is None


def test_separate_all_audio_under_folder_regenerates_when_outputs_missing(tmp_path: Path, monkeypatch):
    import youdub.core.steps.separate_vocals as sv

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


def test_translate_folder_backfills_summary_when_translation_exists(tmp_path: Path, monkeypatch):
    import youdub.core.steps.translate as tr

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    # Prereqs
    (folder / "download.info.json").write_text(json.dumps({"title": "t", "uploader": "u", "upload_date": "20260101"}), encoding="utf-8")
    (folder / "transcript.json").write_text(json.dumps([{"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_00"}]), encoding="utf-8")

    # Existing translation.json but missing summary.json
    (folder / "translation.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_00", "translation": "好"}], ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(tr, "summarize", lambda *_args, **_kwargs: {"title": "t", "author": "u", "summary": "s", "tags": []})

    ok = tr.translate_folder(str(folder), target_language="简体中文")
    assert ok is True
    assert (folder / "summary.json").exists()


def test_synthesize_video_raises_when_missing_inputs(tmp_path: Path):
    import youdub.core.steps.synthesize_video as sv

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    # Missing translation.json/audio_combined.wav/download.mp4 should raise.
    with pytest.raises(FileNotFoundError):
        sv.synthesize_video(str(folder))


def test_generate_all_wavs_requires_marker_not_just_audio_file(tmp_path: Path, monkeypatch):
    import youdub.core.steps.synthesize_speech as ss

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "translation.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "translation": "好"}], ensure_ascii=False),
        encoding="utf-8",
    )
    # Create a valid audio_combined.wav but no marker
    _write_dummy_wav(folder / "audio_combined.wav", seconds=0.5)

    called = {"n": 0}

    def _fake_generate_wavs(_folder: str, _tts_method: str = "bytedance") -> None:
        called["n"] += 1
        # Simulate success marker creation
        (Path(_folder) / ".tts_done.json").write_text(json.dumps({"tts_method": _tts_method}), encoding="utf-8")

    monkeypatch.setattr(ss, "generate_wavs", _fake_generate_wavs)

    ss.generate_all_wavs_under_folder(str(tmp_path), tts_method="bytedance")
    assert called["n"] == 1

