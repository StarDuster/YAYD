"""Upload 步骤测试."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


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
# biliAPI availability
# --------------------------------------------------------------------------- #


def test_upload_video_returns_false_when_biliapi_unavailable(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "_biliapi_availability_error", lambda: "错误：unavailable")
    assert up.upload_video(str(tmp_path)) is False


def test_upload_all_videos_returns_error_when_biliapi_unavailable(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "_biliapi_availability_error", lambda: "错误：unavailable")
    out = up.upload_all_videos_under_folder(str(tmp_path))
    assert out.startswith("错误：")


# --------------------------------------------------------------------------- #
# Cookie file handling
# --------------------------------------------------------------------------- #


def test_ensure_cookie_file_uses_existing_nonempty_cookie(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    cookie_path = tmp_path / "cookies" / "bili_cookies.json"
    cookie_path.parent.mkdir(parents=True, exist_ok=True)
    cookie_path.write_text("ok", encoding="utf-8")

    ok = up._ensure_cookie_file(cookie_path)  # noqa: SLF001
    assert ok is True


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


# --------------------------------------------------------------------------- #
# biliAPI integration (mocked runner)
# --------------------------------------------------------------------------- #


def test_upload_video_with_biliapi_writes_marker_and_calls_runner(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    captured: dict[str, object] = {}

    def _fake_run(payload):  # noqa: ANN001
        captured["payload"] = payload
        return {"ok": True, "submit": "web", "aid": 123, "bvid": "BV1xx411c7xx"}

    monkeypatch.setattr(up, "_run_biliapi_upload", _fake_run)

    folder = tmp_path / "job"
    _make_minimal_upload_folder(folder, with_cover=True)
    cookie_path = tmp_path / "bili_cookies.json"
    cookie_path.write_text("cookie", encoding="utf-8")

    success, actually_uploaded = up._upload_video_with_biliapi(  # noqa: SLF001
        str(folder),
        proxy="socks5h://127.0.0.1:1080",
        upload_cdn="bda2",
        cookie_path=cookie_path,
    )
    assert success is True
    assert actually_uploaded is True

    payload = captured.get("payload")
    assert isinstance(payload, dict)
    assert payload["cookieFile"] == str(cookie_path)
    assert payload["proxy"] == "socks5h://127.0.0.1:1080"
    assert payload["videoPaths"] == [str(folder / "video.mp4")]
    assert isinstance(payload["options"], dict)

    options = payload["options"]
    assert options["tid"] == 201
    assert options["copyright"] == 2
    assert options["source"] == "https://example.com/v"
    assert isinstance(options["tag"], str)
    assert "YouDub" in options["tag"]
    assert options["cover"] == str(folder / "video.png")

    marker = json.loads((folder / "bilibili.json").read_text(encoding="utf-8"))
    assert marker["results"][0]["code"] == 0
    assert marker["tool"] == "biliapi"
    assert marker["tid"] == 201
    assert marker["source"] == "https://example.com/v"
    assert marker["submit"] == "web"
    assert marker["aid"] == 123
    assert marker["bvid"] == "BV1xx411c7xx"
    assert isinstance(marker["tag"], list)


def test_upload_video_with_biliapi_skips_when_already_uploaded(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    folder = tmp_path / "job"
    _make_minimal_upload_folder(folder)
    (folder / "bilibili.json").write_text(json.dumps({"results": [{"code": 0}]}), encoding="utf-8")

    def _boom(_payload):  # noqa: ANN001
        raise AssertionError("should not run uploader when marker exists")

    monkeypatch.setattr(up, "_run_biliapi_upload", _boom)

    cookie_path = tmp_path / "bili_cookies.json"
    cookie_path.write_text("cookie", encoding="utf-8")

    success, actually_uploaded = up._upload_video_with_biliapi(  # noqa: SLF001
        str(folder),
        proxy=None,
        upload_cdn=None,
        cookie_path=cookie_path,
    )
    assert success is True
    assert actually_uploaded is False  # 跳过已上传的不算真正上传


def test_upload_video_reads_env_and_forwards_to_impl(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "_biliapi_availability_error", lambda: None)

    monkeypatch.setenv("BILI_PROXY", "socks5h://127.0.0.1:1080")
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
        return (True, True)  # 返回 tuple

    monkeypatch.setattr(up, "_upload_video_with_biliapi", _fake_impl)

    assert up.upload_video(str(tmp_path)) is True
    assert captured["folder"] == str(tmp_path)
    assert captured["proxy"] == "socks5h://127.0.0.1:1080"
    assert captured["upload_cdn"] == "bda2"
    assert str(captured["cookie_path"]) == str(tmp_path / "cookies.json")


# --------------------------------------------------------------------------- #
# Batch upload
# --------------------------------------------------------------------------- #


def test_upload_all_videos_counts_uploaded_folders(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "_biliapi_availability_error", lambda: None)
    monkeypatch.setenv("BILI_COOKIE_PATH", str(tmp_path / "cookies.json"))
    (tmp_path / "cookies.json").write_text("cookie", encoding="utf-8")

    job1 = tmp_path / "job1"
    job2 = tmp_path / "job2"
    job1.mkdir(parents=True, exist_ok=True)
    job2.mkdir(parents=True, exist_ok=True)
    (job1 / "video.mp4").write_bytes(b"0")
    (job2 / "video.mp4").write_bytes(b"0")

    def _fake_impl(folder, *_args, **_kwargs):  # noqa: ANN001
        # job1 成功，job2 失败
        return (folder.endswith("job1"), folder.endswith("job1"))

    monkeypatch.setattr(up, "_upload_video_with_biliapi", _fake_impl)
    out = up.upload_all_videos_under_folder(str(tmp_path))
    assert "成功 1 个" in out


def test_upload_all_videos_waits_between_uploads(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "_biliapi_availability_error", lambda: None)
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
        return (True, True)  # 返回 tuple

    monkeypatch.setattr(up, "_upload_video_with_biliapi", _fake_impl)
    monkeypatch.setattr(up, "sleep_with_cancel", lambda secs, **_kw: waits.append(secs))

    out = up.upload_all_videos_under_folder(str(tmp_path))
    assert "成功 2 个" in out
    assert len(uploaded) == 2
    assert waits == [30]


def test_upload_all_videos_skips_already_uploaded_no_wait(monkeypatch, tmp_path: Path):
    """已上传的视频跳过时不应该等待间隔"""
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "_biliapi_availability_error", lambda: None)
    monkeypatch.setenv("BILI_COOKIE_PATH", str(tmp_path / "cookies.json"))
    monkeypatch.setenv("BILI_UPLOAD_INTERVAL", "30")
    (tmp_path / "cookies.json").write_text("cookie", encoding="utf-8")

    job1 = tmp_path / "job1"
    job2 = tmp_path / "job2"
    job3 = tmp_path / "job3"
    for job in [job1, job2, job3]:
        job.mkdir(parents=True, exist_ok=True)
        (job / "video.mp4").write_bytes(b"0")
    # job2 已上传
    (job2 / "bilibili.json").write_text(json.dumps({"results": [{"code": 0}]}), encoding="utf-8")

    uploaded: list[str] = []
    waits: list[int] = []

    def _fake_impl(folder, *_args, **_kwargs):  # noqa: ANN001
        uploaded.append(folder)
        return (True, True)

    monkeypatch.setattr(up, "_upload_video_with_biliapi", _fake_impl)
    monkeypatch.setattr(up, "sleep_with_cancel", lambda secs, **_kw: waits.append(secs))

    out = up.upload_all_videos_under_folder(str(tmp_path))
    # job1 和 job3 被上传，job2 跳过
    assert len(uploaded) == 2
    # 只有一次等待（job1 和 job3 之间），跳过 job2 不产生等待
    assert waits == [30]


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #


def test_upload_video_with_biliapi_respects_cancellation(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up
    from youdub.interrupts import CancelledByUser

    folder = tmp_path / "job"
    _make_minimal_upload_folder(folder, with_cover=False)

    monkeypatch.setattr(up, "check_cancelled", lambda *_a, **_k: (_ for _ in ()).throw(CancelledByUser("SIGINT")))

    with pytest.raises(CancelledByUser):
        up._upload_video_with_biliapi(  # noqa: SLF001
            str(folder),
            proxy=None,
            upload_cdn=None,
            cookie_path=tmp_path / "bili_cookies.json",
        )
    assert not (folder / "bilibili.json").exists()


# --------------------------------------------------------------------------- #
# biliAPI runner error handling
# --------------------------------------------------------------------------- #


def test_run_biliapi_upload_returns_error_when_unavailable(monkeypatch):
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "_biliapi_availability_error", lambda: "错误：node 未安装")
    result = up._run_biliapi_upload({})  # noqa: SLF001
    assert result["ok"] is False
    assert "node" in result["error"]


def test_upload_video_returns_false_when_missing_video(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "_biliapi_availability_error", lambda: None)

    folder = tmp_path / "job"
    folder.mkdir(parents=True)
    # No video.mp4 created

    cookie_path = tmp_path / "cookies.json"
    cookie_path.write_text("cookie", encoding="utf-8")

    success, actually_uploaded = up._upload_video_with_biliapi(  # noqa: SLF001
        str(folder),
        proxy=None,
        upload_cdn=None,
        cookie_path=cookie_path,
    )
    assert success is False
    assert actually_uploaded is False


def test_upload_video_returns_false_when_missing_summary(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    monkeypatch.setattr(up, "_biliapi_availability_error", lambda: None)

    folder = tmp_path / "job"
    folder.mkdir(parents=True)
    (folder / "video.mp4").write_bytes(b"0" * 16)
    # No summary.json created

    cookie_path = tmp_path / "cookies.json"
    cookie_path.write_text("cookie", encoding="utf-8")

    success, actually_uploaded = up._upload_video_with_biliapi(  # noqa: SLF001
        str(folder),
        proxy=None,
        upload_cdn=None,
        cookie_path=cookie_path,
    )
    assert success is False
    assert actually_uploaded is False


def test_upload_handles_runner_failure(monkeypatch, tmp_path: Path):
    import youdub.steps.upload as up

    def _failing_runner(_payload):  # noqa: ANN001
        return {"ok": False, "error": "模拟上传失败"}

    monkeypatch.setattr(up, "_run_biliapi_upload", _failing_runner)

    folder = tmp_path / "job"
    _make_minimal_upload_folder(folder)

    cookie_path = tmp_path / "cookies.json"
    cookie_path.write_text("cookie", encoding="utf-8")

    success, actually_uploaded = up._upload_video_with_biliapi(  # noqa: SLF001
        str(folder),
        proxy=None,
        upload_cdn=None,
        cookie_path=cookie_path,
    )
    assert success is False
    assert actually_uploaded is False
    # No marker file should be created on failure
    assert not (folder / "bilibili.json").exists()


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def test_is_uploaded_checks_nested_results():
    from youdub.steps.upload import _is_uploaded

    assert _is_uploaded({"results": [{"code": 0}]}) is True
    assert _is_uploaded({"results": [{"code": -1}]}) is False
    assert _is_uploaded({"code": 0}) is True
    assert _is_uploaded({"code": -1}) is False
    assert _is_uploaded({}) is False
    assert _is_uploaded(None) is False
    assert _is_uploaded([]) is False
