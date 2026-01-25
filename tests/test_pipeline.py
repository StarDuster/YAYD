"""Pipeline 整体流程测试."""

from __future__ import annotations

import json
import types
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from youdub.config import Settings
from youdub.models import ModelManager


def _write_dummy_wav(path: Path, sr: int = 24000, seconds: float = 0.5) -> None:
    from youdub.utils import save_wav

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    wav = 0.1 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    save_wav(wav, str(path), sample_rate=sr)


def _touch_model_bin(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.bin").write_bytes(b"0")


# --------------------------------------------------------------------------- #
# Pipeline required models
# --------------------------------------------------------------------------- #


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
    import youdub.pipeline as pl

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
    monkeypatch.setattr(pl.download, "get_info_list_from_url", lambda *_args, **_kwargs: [])

    pipe = pl.VideoPipeline(settings=settings, model_manager=manager)
    out = pipe.run(
        url="",
        num_videos=1,
        max_workers=1,
        whisper_diarization=diarization,
        tts_method=tts_method,
        auto_upload_video=False,
    )
    assert "成功: 0" in out

    names = captured["names"]
    assert any(n.startswith("Demucs") for n in names)

    assert any("Speaker Diarization" in n for n in names) is expect_diar
    assert any("Gemini TTS" in n for n in names) is expect_gemini

    qwen_flags = [("Qwen3-TTS Runtime" in n) or ("Qwen3-TTS Base" in n) for n in names]
    assert any(qwen_flags) is expect_qwen

    assert any("ByteDance TTS" in n for n in names) is expect_bytedance


# --------------------------------------------------------------------------- #
# Pipeline process_single interface contracts
# --------------------------------------------------------------------------- #


def test_pipeline_process_single_happy_path_interface_contracts(tmp_path: Path, monkeypatch):
    import youdub.pipeline as pl

    # Keep the test deterministic and fast.
    monkeypatch.setattr(pl, "check_cancelled", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pl, "sleep_with_cancel", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pl.separate_vocals, "unload_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pl.transcribe, "unload_all_models", lambda *_args, **_kwargs: None)

    job = tmp_path / "job"
    calls: list[str] = []

    def _stub_get_target_folder(_info: dict[str, Any], _root: str) -> str:
        return str(job)

    def _stub_download_single_video(_info: dict[str, Any], _root: str, _resolution: str, settings=None) -> str:
        calls.append("download")
        _ = settings
        job.mkdir(parents=True, exist_ok=True)
        (job / "download.mp4").write_bytes(b"0" * 2048)
        (job / "download.info.json").write_text(json.dumps(_info, ensure_ascii=False), encoding="utf-8")
        return str(job)

    def _stub_separate_all(_folder: str, **_kwargs) -> str:
        calls.append("separate")
        assert (job / "download.mp4").exists()
        _write_dummy_wav(job / "audio_vocals.wav", seconds=0.5)
        _write_dummy_wav(job / "audio_instruments.wav", seconds=0.5)
        return "ok"

    def _stub_transcribe_all(_folder: str, **_kwargs) -> str:
        calls.append("transcribe")
        assert (job / "audio_vocals.wav").exists()
        (job / "transcript.json").write_text(
            json.dumps([{"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_00"}], ensure_ascii=False),
            encoding="utf-8",
        )
        return "ok"

    def _stub_translate_all(_folder: str, **_kwargs) -> str:
        calls.append("translate")
        assert (job / "transcript.json").exists()
        (job / "translation.json").write_text(
            json.dumps(
                [{"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_00", "translation": "好"}],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (job / "summary.json").write_text(
            json.dumps({"title": "t", "author": "u", "summary": "s", "tags": [], "translation_model": "dummy"}, ensure_ascii=False),
            encoding="utf-8",
        )
        return "ok"

    def _stub_tts_all(_folder: str, **_kwargs) -> str:
        calls.append("tts")
        assert (job / "translation.json").exists()
        _write_dummy_wav(job / "audio_combined.wav", seconds=0.5)
        (job / ".tts_done.json").write_text(json.dumps({"tts_method": "bytedance"}, ensure_ascii=False), encoding="utf-8")
        return "ok"

    def _stub_video_all(_folder: str, **_kwargs) -> str:
        calls.append("video")
        assert (job / "download.mp4").exists()
        assert (job / "audio_combined.wav").exists()
        assert (job / "translation.json").exists()
        (job / "video.mp4").write_bytes(b"0" * 2048)
        return "ok"

    def _stub_generate_info_all(_folder: str) -> str:
        calls.append("info")
        assert (job / "summary.json").exists()
        # marker-only: avoid PIL work here (covered by dedicated tests)
        (job / "video.txt").write_text("ok", encoding="utf-8")
        return "ok"

    monkeypatch.setattr(pl.download, "get_target_folder", _stub_get_target_folder)
    monkeypatch.setattr(pl.download, "download_single_video", _stub_download_single_video)
    monkeypatch.setattr(pl.separate_vocals, "separate_all_audio_under_folder", _stub_separate_all)
    monkeypatch.setattr(pl.transcribe, "transcribe_all_audio_under_folder", _stub_transcribe_all)
    monkeypatch.setattr(pl.translate, "translate_all_transcript_under_folder", _stub_translate_all)
    monkeypatch.setattr(pl.synthesize_speech, "generate_all_wavs_under_folder", _stub_tts_all)
    monkeypatch.setattr(pl, "synthesize_all_video_under_folder", _stub_video_all)
    monkeypatch.setattr(pl, "generate_all_info_under_folder", _stub_generate_info_all)

    pipe = pl.VideoPipeline(settings=Settings(root_folder=tmp_path))
    ok = pipe.process_single(
        info={"title": "t"},
        root_folder=str(tmp_path),
        resolution="360p",
        demucs_model="htdemucs_ft",
        device="cpu",
        shifts=1,
        whisper_model="/fake/whisper",
        whisper_batch_size=1,
        whisper_diarization=False,
        whisper_min_speakers=None,
        whisper_max_speakers=None,
        translation_target_language="简体中文",
        tts_method="bytedance",
        qwen_tts_batch_size=1,
        tts_adaptive_segment_stretch=False,
        subtitles=False,
        speed_up=1.0,
        fps=30,
        target_resolution="360p",
        max_retries=1,
        auto_upload_video=False,
        use_nvenc=False,
        whisper_device=None,
        whisper_cpu_model=None,
    )
    assert ok is True
    assert calls == ["download", "separate", "transcribe", "translate", "tts", "video", "info"]

    assert (job / "download.mp4").exists()
    assert (job / "audio_vocals.wav").exists()
    assert (job / "transcript.json").exists()
    assert (job / "translation.json").exists()
    assert (job / "audio_combined.wav").exists()
    assert (job / "video.mp4").exists()

    # Minimal sanity: audio files are valid wavs.
    with wave.open(str(job / "audio_combined.wav"), "rb") as wf:
        assert int(wf.getframerate() or 0) > 0


# --------------------------------------------------------------------------- #
# Pipeline warmup model selection
# --------------------------------------------------------------------------- #


def test_pipeline_warmup_loads_cpu_model_when_whisper_device_cpu(tmp_path: Path, monkeypatch):
    import youdub.pipeline as pl

    gpu_dir = tmp_path / "gpu"
    cpu_dir = tmp_path / "cpu"
    _touch_model_bin(gpu_dir)
    _touch_model_bin(cpu_dir)

    settings = Settings(
        root_folder=tmp_path,
        whisper_model_path=gpu_dir,
        whisper_cpu_model_path=cpu_dir,
    )
    manager = ModelManager(settings)

    monkeypatch.setattr(manager, "ensure_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(pl.separate_vocals, "init_demucs", lambda *args, **kwargs: None)
    monkeypatch.setattr(pl.synthesize_speech, "init_TTS", lambda *args, **kwargs: None)
    monkeypatch.setattr(pl.download, "get_info_list_from_url", lambda *_args, **_kwargs: [])

    calls: list[tuple[str, str]] = []

    def _fake_load_asr_model(model_dir: str, device: str = "auto", **_kwargs):
        calls.append((str(model_dir), str(device)))

    monkeypatch.setattr(pl.transcribe, "load_asr_model", _fake_load_asr_model)

    pipe = pl.VideoPipeline(settings=settings, model_manager=manager)
    out = pipe.run(
        url="",
        num_videos=1,
        max_workers=1,
        whisper_diarization=False,
        auto_upload_video=False,
        whisper_device="cpu",
        whisper_model=str(gpu_dir),
        whisper_cpu_model=str(cpu_dir),
    )
    assert "成功: 0" in out
    assert calls == [(str(cpu_dir), "cpu")]


def test_pipeline_warmup_loads_gpu_model_when_whisper_device_cuda(tmp_path: Path, monkeypatch):
    import youdub.pipeline as pl

    gpu_dir = tmp_path / "gpu"
    cpu_dir = tmp_path / "cpu"
    _touch_model_bin(gpu_dir)
    _touch_model_bin(cpu_dir)

    settings = Settings(
        root_folder=tmp_path,
        whisper_model_path=gpu_dir,
        whisper_cpu_model_path=cpu_dir,
    )
    manager = ModelManager(settings)

    monkeypatch.setattr(manager, "ensure_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(pl.separate_vocals, "init_demucs", lambda *args, **kwargs: None)
    monkeypatch.setattr(pl.synthesize_speech, "init_TTS", lambda *args, **kwargs: None)
    monkeypatch.setattr(pl.download, "get_info_list_from_url", lambda *_args, **_kwargs: [])

    calls: list[tuple[str, str]] = []

    def _fake_load_asr_model(model_dir: str, device: str = "auto", **_kwargs):
        calls.append((str(model_dir), str(device)))

    monkeypatch.setattr(pl.transcribe, "load_asr_model", _fake_load_asr_model)

    pipe = pl.VideoPipeline(settings=settings, model_manager=manager)
    out = pipe.run(
        url="",
        num_videos=1,
        max_workers=1,
        whisper_diarization=False,
        auto_upload_video=False,
        whisper_device="cuda",
        whisper_model=str(gpu_dir),
        whisper_cpu_model=str(cpu_dir),
    )
    assert "成功: 0" in out
    assert calls == [(str(gpu_dir), "cuda")]
