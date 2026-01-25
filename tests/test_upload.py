"""Upload 步骤测试."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class _Line:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Line {self.name}>"


class _FakeStreamGears:
    class UploadLine:
        Bda = _Line("Bda")
        Bda2 = _Line("Bda2")
        Qn = _Line("Qn")
        Tx = _Line("Tx")
        Txa = _Line("Txa")
        Bldsa = _Line("Bldsa")

    def __init__(self):
        self.upload_calls: list[dict[str, object]] = []

    def upload(self, **kwargs) -> None:
        self.upload_calls.append(dict(kwargs))


def _make_minimal_upload_folder(folder: Path, *, with_cover: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "video.mp4").write_bytes(b"0" * 16)
    if with_cover:
        (folder / "video.png").write_bytes(b"png")
    (folder / "summary.json").write_text(
        json.dumps(
            {
                "title": "视频标题：Test Title",
                "author": "Tester",
                "summary": "视频摘要：Summary",
                "tags": ["AI", "AI", "ChatGPT", "中文配音"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (folder / "download.info.json").write_text(
        json.dumps(
            {
                "title": "English Title",
                "webpage_url": "https://example.com/v",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# stream_gears availability
# --------------------------------------------------------------------------- #


def test_upload_video_returns_false_when_stream_gears_missing(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "stream_gears", None, raising=False)
    monkeypatch.setattr(up, "_STREAM_GEARS_IMPORT_ERROR", RuntimeError("no"), raising=False)
    assert up.upload_video(str(tmp_path)) is False


def test_upload_all_videos_returns_error_when_stream_gears_missing(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "stream_gears", None, raising=False)
    monkeypatch.setattr(up, "_STREAM_GEARS_IMPORT_ERROR", RuntimeError("no"), raising=False)
    out = up.upload_all_videos_under_folder(str(tmp_path))
    assert out.startswith("错误：stream_gears 不可用")


# --------------------------------------------------------------------------- #
# Cookie file handling
# --------------------------------------------------------------------------- #


def test_ensure_cookie_file_uses_existing_nonempty_cookie(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    cookie_path = tmp_path / "cookies" / "bili_cookies.json"
    cookie_path.parent.mkdir(parents=True, exist_ok=True)
    cookie_path.write_text("ok", encoding="utf-8")

    cwd = Path.cwd()
    ok = up._ensure_cookie_file(cookie_path)  # noqa: SLF001
    assert ok is True
    assert Path.cwd() == cwd


def test_ensure_cookie_file_adopts_existing_cookies_json(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    cookie_dir = tmp_path / "cookies"
    cookie_dir.mkdir(parents=True, exist_ok=True)
    (cookie_dir / "cookies.json").write_text("cookie", encoding="utf-8")

    cookie_path = cookie_dir / "bili_cookies.json"
    ok = up._ensure_cookie_file(cookie_path)  # noqa: SLF001
    assert ok is True
    assert cookie_path.exists()
    assert cookie_path.read_text(encoding="utf-8") == "cookie"


def test_upload_video_cookie_missing_returns_false(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "stream_gears", _FakeStreamGears(), raising=False)
    monkeypatch.setenv("BILI_COOKIE_PATH", str(tmp_path / "missing.json"))
    assert up.upload_video(str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Biliup integration
# --------------------------------------------------------------------------- #


def test_upload_video_with_biliup_writes_marker_and_calls_stream_gears_upload(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    fake = _FakeStreamGears()
    monkeypatch.setattr(up, "stream_gears", fake, raising=False)

    # avoid waiting in retry loops
    monkeypatch.setattr(up, "sleep_with_cancel", lambda *_args, **_kwargs: None)

    monkeypatch.setattr(up, "_iter_upload_lines", lambda _cdn: [_Line("Auto")])

    folder = tmp_path / "job"
    _make_minimal_upload_folder(folder)
    cookie_path = tmp_path / "bili_cookies.json"
    cookie_path.write_text("cookie", encoding="utf-8")

    ok = up._upload_video_with_biliup(  # noqa: SLF001
        str(folder),
        proxy="http://127.0.0.1:7890",
        upload_cdn="bda2",
        cookie_path=cookie_path,
    )
    assert ok is True

    assert len(fake.upload_calls) == 1
    call = fake.upload_calls[0]

    assert call["video_path"] == [str(folder / "video.mp4")]
    assert call["cookie_file"] == str(cookie_path)
    assert call["tid"] == 201
    assert call["copyright"] == 2
    assert call["source"] == "https://example.com/v"
    assert call["proxy"] == "http://127.0.0.1:7890"
    assert call["cover"] == str(folder / "video.png")
    assert call["line"].__class__ is _Line

    marker = json.loads((folder / "bilibili.json").read_text(encoding="utf-8"))
    assert marker["results"][0]["code"] == 0
    assert marker["tool"] == "biliup"
    assert marker["tid"] == 201
    assert marker["source"] == "https://example.com/v"
    assert isinstance(marker["tag"], list)


def test_upload_video_with_biliup_skips_when_already_uploaded(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    fake = _FakeStreamGears()
    monkeypatch.setattr(up, "stream_gears", fake, raising=False)
    monkeypatch.setattr(up, "sleep_with_cancel", lambda *_args, **_kwargs: None)

    folder = tmp_path / "job"
    _make_minimal_upload_folder(folder)
    (folder / "bilibili.json").write_text(json.dumps({"results": [{"code": 0}]}), encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise AssertionError("should not upload when already uploaded marker exists")

    monkeypatch.setattr(fake, "upload", _boom)
    cookie_path = tmp_path / "bili_cookies.json"
    cookie_path.write_text("cookie", encoding="utf-8")

    ok = up._upload_video_with_biliup(  # noqa: SLF001
        str(folder),
        proxy=None,
        upload_cdn=None,
        cookie_path=cookie_path,
    )
    assert ok is True


def test_upload_video_reads_env_and_forwards_to_impl(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "stream_gears", _FakeStreamGears(), raising=False)
    monkeypatch.setenv("BILI_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("BILI_UPLOAD_CDN", "bda2")
    monkeypatch.setenv("BILI_COOKIE_PATH", str(tmp_path / "cookies.json"))
    (tmp_path / "cookies.json").write_text("cookie", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_impl(folder, proxy, upload_cdn, cookie_path):  # noqa: ANN001
        captured.update(
            {
                "folder": folder,
                "proxy": proxy,
                "upload_cdn": upload_cdn,
                "cookie_path": cookie_path,
            }
        )
        return True

    monkeypatch.setattr(up, "_upload_video_with_biliup", _fake_impl)

    assert up.upload_video(str(tmp_path)) is True
    assert captured["folder"] == str(tmp_path)
    assert captured["proxy"] == "http://127.0.0.1:7890"
    assert captured["upload_cdn"] == "bda2"
    assert str(captured["cookie_path"]) == str(tmp_path / "cookies.json")


# --------------------------------------------------------------------------- #
# Batch upload
# --------------------------------------------------------------------------- #


def test_upload_all_videos_counts_uploaded_folders(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "stream_gears", _FakeStreamGears(), raising=False)
    monkeypatch.setenv("BILI_COOKIE_PATH", str(tmp_path / "cookies.json"))
    (tmp_path / "cookies.json").write_text("cookie", encoding="utf-8")

    job1 = tmp_path / "job1"
    job2 = tmp_path / "job2"
    job1.mkdir(parents=True, exist_ok=True)
    job2.mkdir(parents=True, exist_ok=True)
    (job1 / "video.mp4").write_bytes(b"0")
    (job2 / "video.mp4").write_bytes(b"0")

    def _fake_impl(folder, *_args, **_kwargs):  # noqa: ANN001
        return folder.endswith("job1")

    monkeypatch.setattr(up, "_upload_video_with_biliup", _fake_impl)
    out = up.upload_all_videos_under_folder(str(tmp_path))
    assert "成功 1 个" in out


def test_upload_all_videos_waits_between_uploads(monkeypatch, tmp_path: Path):
    """Test that upload_all_videos waits between consecutive uploads."""
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "stream_gears", _FakeStreamGears(), raising=False)
    monkeypatch.setenv("BILI_COOKIE_PATH", str(tmp_path / "cookies.json"))
    monkeypatch.setenv("BILI_UPLOAD_INTERVAL", "30")
    (tmp_path / "cookies.json").write_text("cookie", encoding="utf-8")

    job1 = tmp_path / "job1"
    job2 = tmp_path / "job2"
    job1.mkdir(parents=True, exist_ok=True)
    job2.mkdir(parents=True, exist_ok=True)
    (job1 / "video.mp4").write_bytes(b"0")
    (job2 / "video.mp4").write_bytes(b"0")

    uploaded: list[str] = []
    waits: list[int] = []

    def _fake_impl(folder, *_args, **_kwargs):  # noqa: ANN001
        uploaded.append(folder)
        return True

    monkeypatch.setattr(up, "_upload_video_with_biliup", _fake_impl)
    monkeypatch.setattr(up, "sleep_with_cancel", lambda secs, **_kw: waits.append(secs))

    out = up.upload_all_videos_under_folder(str(tmp_path))
    assert "成功 2 个" in out
    assert len(uploaded) == 2
    # Should wait 30s between first and second upload
    assert waits == [30]


# --------------------------------------------------------------------------- #
# Retry and error handling
# --------------------------------------------------------------------------- #


def test_upload_video_with_biliup_retries_on_failure(monkeypatch, tmp_path: Path):
    """Test that upload retries on failure with exponential backoff."""
    import youdub.steps.upload as up

    call_count = 0
    waits: list[int] = []

    def _failing_upload(**kwargs):  # noqa: ANN003
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Unknown Error")

    fake = _FakeStreamGears()
    fake.upload = _failing_upload  # type: ignore[method-assign]
    monkeypatch.setattr(up, "stream_gears", fake, raising=False)
    monkeypatch.setattr(up, "sleep_with_cancel", lambda secs, **_kw: waits.append(secs))
    monkeypatch.setattr(up, "_iter_upload_lines", lambda _cdn: [_Line("Auto")])

    folder = tmp_path / "job"
    _make_minimal_upload_folder(folder)
    cookie_path = tmp_path / "bili_cookies.json"
    cookie_path.write_text("cookie", encoding="utf-8")

    ok = up._upload_video_with_biliup(  # noqa: SLF001
        str(folder),
        proxy=None,
        upload_cdn=None,
        cookie_path=cookie_path,
    )
    assert ok is False
    assert call_count == 5  # MAX_ATTEMPTS
    # Exponential backoff: 5, 10, 20, 40
    assert waits == [5, 10, 20, 40]


def test_upload_video_with_biliup_stops_on_auth_error(monkeypatch, tmp_path: Path):
    """Test that upload stops immediately on authentication errors."""
    import youdub.steps.upload as up

    call_count = 0

    def _auth_error_upload(**kwargs):  # noqa: ANN003
        nonlocal call_count
        call_count += 1
        raise RuntimeError("cookie expired")

    fake = _FakeStreamGears()
    fake.upload = _auth_error_upload  # type: ignore[method-assign]
    monkeypatch.setattr(up, "stream_gears", fake, raising=False)
    monkeypatch.setattr(up, "sleep_with_cancel", lambda *_args, **_kwargs: None)

    folder = tmp_path / "job"
    _make_minimal_upload_folder(folder)
    cookie_path = tmp_path / "bili_cookies.json"
    cookie_path.write_text("cookie", encoding="utf-8")

    ok = up._upload_video_with_biliup(  # noqa: SLF001
        str(folder),
        proxy=None,
        upload_cdn=None,
        cookie_path=cookie_path,
    )
    assert ok is False
    assert call_count == 1  # Should stop after first auth error


def test_upload_video_with_biliup_respects_cancellation(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up
    from youdub.interrupts import CancelledByUser

    fake = _FakeStreamGears()
    monkeypatch.setattr(up, "stream_gears", fake, raising=False)

    folder = tmp_path / "job"
    _make_minimal_upload_folder(folder, with_cover=False)

    monkeypatch.setattr(up, "check_cancelled", lambda *_args, **_kwargs: (_ for _ in ()).throw(CancelledByUser("SIGINT")))
    monkeypatch.setattr(up, "sleep_with_cancel", lambda *_args, **_kwargs: None)

    with pytest.raises(CancelledByUser):
        up._upload_video_with_biliup(  # noqa: SLF001
            str(folder),
            proxy=None,
            upload_cdn=None,
            cookie_path=tmp_path / "bili_cookies.json",
        )
    assert not (folder / "bilibili.json").exists()
