import pytest

from youdub.config import Settings
from youdub.models import ModelManager


@pytest.mark.parametrize(
    ("tts_method", "diarization", "expect_gemini", "expect_qwen", "expect_bytedance", "expect_diar"),
    [
        ("bytedance", True, False, False, True, True),
        ("bytedance", False, False, False, True, False),
        ("gemini", True, True, False, False, True),
        ("qwen", False, False, True, False, False),
    ],
)
def test_pipeline_builds_required_model_list_for_method(
    tmp_path,
    monkeypatch,
    tts_method: str,
    diarization: bool,
    expect_gemini: bool,
    expect_qwen: bool,
    expect_bytedance: bool,
    expect_diar: bool,
):
    import youdub.core.pipeline as pl

    # Pipeline now validates Whisper model.bin paths early (outside ModelManager.ensure_ready),
    # so provide a minimal dummy model directory.
    whisper_dir = tmp_path / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    (whisper_dir / "model.bin").write_bytes(b"0")

    settings = Settings(root_folder=tmp_path, whisper_model_path=whisper_dir)
    manager = ModelManager(settings)

    captured: dict[str, list[str]] = {"names": []}

    def _ensure_ready(*, names=None):
        captured["names"] = list(names or [])

    # Avoid file-system/model checks: this test is only about selecting names.
    monkeypatch.setattr(manager, "ensure_ready", _ensure_ready)

    # Avoid warmup work and network I/O in pipeline.run()
    monkeypatch.setattr(pl.separate_vocals, "init_demucs", lambda *args, **kwargs: None)
    monkeypatch.setattr(pl.synthesize_speech, "init_TTS", lambda *args, **kwargs: None)
    monkeypatch.setattr(pl.transcribe, "init_asr", lambda *args, **kwargs: None)
    monkeypatch.setattr(pl.download, "get_info_list_from_url", lambda urls, num: [])

    pipe = pl.VideoPipeline(settings=settings, model_manager=manager)
    out = pipe.run(
        url="",
        num_videos=1,
        max_workers=1,
        whisper_diarization=diarization,
        tts_method=tts_method,
        auto_upload_video=False,
    )
    assert "Success: 0" in out

    names = captured["names"]
    assert any(n.startswith("Demucs") for n in names)

    assert any("Speaker Diarization" in n for n in names) is expect_diar
    assert any("Gemini TTS" in n for n in names) is expect_gemini

    qwen_flags = [("Qwen3-TTS Runtime" in n) or ("Qwen3-TTS Base" in n) for n in names]
    assert any(qwen_flags) is expect_qwen

    assert any("ByteDance TTS" in n for n in names) is expect_bytedance

