"""Download 步骤测试."""

from __future__ import annotations

from pathlib import Path

import youdub.steps.download as dl


def test_download_single_video_short_circuits_when_cached_mp4_exists(tmp_path: Path, monkeypatch):
    class _BoomYdl:
        def __init__(self, _opts):
            raise AssertionError("yt-dlp should not be invoked when cached download.mp4 is valid")

    monkeypatch.setattr(dl.yt_dlp, "YoutubeDL", _BoomYdl)

    info = {
        "title": "A/B?C: D*E|F",
        "uploader": "U<1>",
        "upload_date": "20260102",
        "webpage_url": "https://example.com/v",
    }
    folder = dl.get_target_folder(info, str(tmp_path))
    assert folder is not None
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "download.mp4").write_bytes(b"0" * 2048)
    # Avoid best-effort backfill calls on cache hit.
    (out_dir / "download.jpg").write_bytes(b"0")
    (out_dir / "download.en.vtt").write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nx\n", encoding="utf-8")

    out = dl.download_single_video(info, str(tmp_path), resolution="360p")
    assert out == folder


def test_download_single_video_returns_none_when_file_missing(tmp_path: Path, monkeypatch):
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
