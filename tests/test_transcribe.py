"""Transcribe 步骤测试."""

from __future__ import annotations

import json
import types
import wave
from pathlib import Path

import numpy as np
import pytest

from youdub.config import Settings
from youdub.models import ModelCheckError, ModelManager


def _write_dummy_wav(path: Path, sr: int = 24000, seconds: float = 4.0) -> None:
    from youdub.utils import save_wav

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    wav = 0.1 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    save_wav(wav, str(path), sample_rate=sr)


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        rate = int(wf.getframerate() or 0)
        frames = int(wf.getnframes() or 0)
    return 0.0 if rate <= 0 else float(frames) / float(rate)


def _touch_model_bin(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.bin").write_bytes(b"0")


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #


def test_merge_segments_respects_speaker_and_punctuation():
    import youdub.steps.transcribe as tr

    transcript = [
        {"start": 0.0, "end": 0.5, "text": "hello", "speaker": "S1"},
        {"start": 0.5, "end": 1.0, "text": "world", "speaker": "S1"},
        {"start": 1.0, "end": 1.5, "text": "ok.", "speaker": "S1"},
        {"start": 1.5, "end": 2.0, "text": "next", "speaker": "S1"},
        {"start": 2.0, "end": 3.0, "text": "diff", "speaker": "S2"},
        {"start": 3.0, "end": 4.0, "text": "speaker2", "speaker": "S2"},
    ]

    merged = tr.merge_segments(transcript)
    assert len(merged) == 3

    assert merged[0]["speaker"] == "S1"
    assert merged[0]["start"] == 0.0
    assert merged[0]["end"] == 1.5
    assert merged[0]["text"] == "hello world ok."

    assert merged[1]["speaker"] == "S1"
    assert merged[1]["start"] == 1.5
    assert merged[1]["end"] == 2.0
    assert merged[1]["text"] == "next"

    assert merged[2]["speaker"] == "S2"
    assert merged[2]["start"] == 2.0
    assert merged[2]["end"] == 4.0
    assert merged[2]["text"] == "diff speaker2"


def test_assign_speakers_by_overlap_picks_best_match():
    import youdub.steps.transcribe as tr

    segments = [
        {"start": 0.0, "end": 1.0, "text": "a"},
        {"start": 1.0, "end": 2.0, "text": "b"},
    ]
    turns = [
        {"start": 0.2, "end": 1.2, "speaker": "SPEAKER_01"},
        {"start": 1.2, "end": 2.2, "speaker": "SPEAKER_02"},
    ]

    tr._assign_speakers_by_overlap(segments, turns, default_speaker="SPEAKER_00")
    assert segments[0]["speaker"] == "SPEAKER_01"
    assert segments[1]["speaker"] == "SPEAKER_02"


def test_assign_speakers_by_overlap_defaults_when_no_turns_or_bad_segment():
    import youdub.steps.transcribe as tr

    segments = [
        {"start": 0.0, "end": 1.0, "text": "a"},
        {"start": 1.0, "end": 1.0, "text": "bad"},  # end <= start
    ]

    tr._assign_speakers_by_overlap(segments, turns=[], default_speaker="SPEAKER_00")
    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[1]["speaker"] == "SPEAKER_00"


# --------------------------------------------------------------------------- #
# Short-circuit behavior
# --------------------------------------------------------------------------- #


def test_transcribe_audio_short_circuits_when_transcript_exists(tmp_path):
    import youdub.steps.transcribe as tr

    folder = tmp_path / "job"
    folder.mkdir()
    (folder / "transcript.json").write_text(json.dumps([], ensure_ascii=False), encoding="utf-8")

    # Should not require audio file or model availability.
    assert tr.transcribe_audio(str(folder), diarization=False) is True


# --------------------------------------------------------------------------- #
# Model loading validation
# --------------------------------------------------------------------------- #


def test_load_asr_model_requires_model_bin_even_if_dir_nonempty(tmp_path):
    import youdub.steps.transcribe as tr

    model_dir = tmp_path / "whisper_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "dummy.txt").write_text("ok", encoding="utf-8")  # make the dir "exist" for ModelManager

    settings = Settings(whisper_model_path=model_dir)
    manager = ModelManager(settings)

    with pytest.raises(ModelCheckError, match=r"model\.bin"):
        tr.load_asr_model(
            model_dir=str(model_dir),
            device="cpu",
            settings=settings,
            model_manager=manager,
            use_batched=False,
        )


# --------------------------------------------------------------------------- #
# Speaker reference regeneration
# --------------------------------------------------------------------------- #


def test_transcribe_audio_existing_transcript_regenerates_speaker_wavs(tmp_path: Path):
    import youdub.steps.transcribe as tr

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    # Minimal inputs: transcript.json exists, but SPEAKER/*.wav is missing.
    _write_dummy_wav(folder / "audio_vocals.wav", seconds=6.0)
    transcript = [{"start": 0.0, "end": 2.0, "text": "x", "speaker": "SPEAKER_00"}]
    (folder / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = tr.transcribe_audio(str(folder), diarization=False)
    assert ok is True

    spk = folder / "SPEAKER" / "SPEAKER_00.wav"
    assert spk.exists()
    assert spk.stat().st_size >= 44


# --------------------------------------------------------------------------- #
# Device selection (auto/cpu/cuda)
# --------------------------------------------------------------------------- #


def test_init_asr_auto_uses_cuda_when_available(tmp_path: Path, monkeypatch):
    import youdub.steps.transcribe as tr

    gpu_dir = tmp_path / "gpu"
    cpu_dir = tmp_path / "cpu"
    _touch_model_bin(gpu_dir)
    _touch_model_bin(cpu_dir)

    settings = Settings(
        root_folder=tmp_path,
        whisper_model_path=gpu_dir,
        whisper_cpu_model_path=cpu_dir,
        whisper_device="auto",
    )
    manager = ModelManager(settings)

    monkeypatch.setattr(tr.torch.cuda, "is_available", lambda: True)

    captured: dict[str, str] = {}

    def _fake_load_asr_model(model_dir: str, device: str = "auto", **_kwargs):
        captured["model_dir"] = str(model_dir)
        captured["device"] = str(device)

    monkeypatch.setattr(tr, "load_asr_model", _fake_load_asr_model)

    tr.init_asr(settings=settings, model_manager=manager)
    assert captured["device"] == "cuda"
    assert captured["model_dir"] == str(gpu_dir)


def test_init_asr_auto_uses_cpu_model_path_when_no_cuda(tmp_path: Path, monkeypatch):
    import youdub.steps.transcribe as tr

    gpu_dir = tmp_path / "gpu"
    cpu_dir = tmp_path / "cpu"
    _touch_model_bin(gpu_dir)
    _touch_model_bin(cpu_dir)

    settings = Settings(
        root_folder=tmp_path,
        whisper_model_path=gpu_dir,
        whisper_cpu_model_path=cpu_dir,
        whisper_device="auto",
    )
    manager = ModelManager(settings)

    monkeypatch.setattr(tr.torch.cuda, "is_available", lambda: False)

    captured: dict[str, str] = {}

    def _fake_load_asr_model(model_dir: str, device: str = "auto", **_kwargs):
        captured["model_dir"] = str(model_dir)
        captured["device"] = str(device)

    monkeypatch.setattr(tr, "load_asr_model", _fake_load_asr_model)

    tr.init_asr(settings=settings, model_manager=manager)
    assert captured["device"] == "cpu"
    assert captured["model_dir"] == str(cpu_dir)


def test_init_asr_cpu_falls_back_to_whisper_model_path_when_cpu_model_unset(tmp_path: Path, monkeypatch):
    import youdub.steps.transcribe as tr

    gpu_dir = tmp_path / "gpu"
    _touch_model_bin(gpu_dir)

    settings = Settings(
        root_folder=tmp_path,
        whisper_model_path=gpu_dir,
        whisper_cpu_model_path=None,
        whisper_device="cpu",
    )
    manager = ModelManager(settings)

    captured: dict[str, str] = {}

    def _fake_load_asr_model(model_dir: str, device: str = "auto", **_kwargs):
        captured["model_dir"] = str(model_dir)
        captured["device"] = str(device)

    monkeypatch.setattr(tr, "load_asr_model", _fake_load_asr_model)

    tr.init_asr(settings=settings, model_manager=manager)
    assert captured["device"] == "cpu"
    assert captured["model_dir"] == str(gpu_dir)


def test_transcribe_audio_prefers_cpu_model_when_device_cpu(tmp_path: Path, monkeypatch):
    import youdub.steps.transcribe as tr

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "audio_vocals.wav").write_bytes(b"0")

    settings = Settings(root_folder=tmp_path, whisper_device="cpu")
    manager = ModelManager(settings)

    captured: dict[str, str] = {}

    class _DummyModel:
        def transcribe(self, _wav_path: str, **_kwargs):
            seg = types.SimpleNamespace(text="hello", start=0.0, end=1.0)
            info = types.SimpleNamespace(language="en")
            return [seg], info

    def _fake_load_asr_model(model_dir: str, device: str = "auto", **_kwargs):
        captured["model_dir"] = str(model_dir)
        captured["device"] = str(device)
        tr._ASR_MODEL = _DummyModel()
        tr._ASR_PIPELINE = None

    monkeypatch.setattr(tr, "load_asr_model", _fake_load_asr_model)
    monkeypatch.setattr(tr, "_preload_cudnn_for_onnxruntime_gpu", lambda: None)
    monkeypatch.setattr(tr, "generate_speaker_audio", lambda *_args, **_kwargs: None)

    ok = tr.transcribe_audio(
        str(folder),
        model_name="/fake/gpu/model",
        cpu_model_name="/fake/cpu/model",
        device="cpu",
        diarization=False,
        settings=settings,
        model_manager=manager,
    )
    assert ok is True
    assert captured["device"] == "cpu"
    assert captured["model_dir"] == "/fake/cpu/model"
    assert (folder / "transcript.json").exists()


def test_transcribe_audio_uses_youtube_manual_subtitles_when_available(tmp_path: Path, monkeypatch):
    import youdub.steps.transcribe as tr

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    # yt-dlp infojson: manual subtitles are under "subtitles" (NOT "automatic_captions")
    (folder / "download.info.json").write_text(
        json.dumps(
            {
                "language": "en",
                "original_language": "en",
                "subtitles": {"en": [{"ext": "vtt"}]},
                "automatic_captions": {"en": [{"ext": "vtt"}]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (folder / "download.en.vtt").write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:00.500\n"
        "Hello.\n\n"
        "00:00:00.500 --> 00:00:01.000\n"
        "World.\n",
        encoding="utf-8",
    )

    # ASR should not be invoked when manual subtitles exist.
    monkeypatch.setattr(tr, "load_asr_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ASR should not run")))
    monkeypatch.setattr(
        tr, "load_qwen_asr_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ASR should not run"))
    )
    monkeypatch.setattr(tr, "generate_speaker_audio", lambda *_args, **_kwargs: None)

    ok = tr.transcribe_audio(str(folder), diarization=False)
    assert ok is True

    out = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
    assert isinstance(out, list) and out
    assert out[0]["source"] == "youtube_subtitles"
    assert out[0]["subtitle_lang"] == "en"
    assert out[0]["speaker"] == "SPEAKER_00"


def test_transcribe_audio_overwrites_asr_transcript_and_clears_downstream_cache_when_manual_subs_present(
    tmp_path: Path, monkeypatch
):
    import youdub.steps.transcribe as tr

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    # Existing ASR transcript + downstream artifacts
    (folder / "transcript.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "asr", "speaker": "SPEAKER_00"}], ensure_ascii=False),
        encoding="utf-8",
    )
    (folder / "translation.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_00", "translation": "好"}], ensure_ascii=False),
        encoding="utf-8",
    )
    (folder / "translation_raw.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_00", "translation": "好"}], ensure_ascii=False),
        encoding="utf-8",
    )
    (folder / "summary.json").write_text(json.dumps({"summary": "s"}, ensure_ascii=False), encoding="utf-8")
    (folder / "transcript_punctuated.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "asr.", "speaker": "SPEAKER_00"}], ensure_ascii=False),
        encoding="utf-8",
    )
    wavs = folder / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    (wavs / ".tts_done.json").write_text(json.dumps({"tts_method": "bytedance"}, ensure_ascii=False), encoding="utf-8")
    (wavs / "0000.wav").write_bytes(b"RIFFxxxxWAVE")  # dummy header bytes

    # Manual subtitles available
    (folder / "download.info.json").write_text(
        json.dumps(
            {"language": "en", "original_language": "en", "subtitles": {"en": [{"ext": "vtt"}]}}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    (folder / "download.en.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello.\n",
        encoding="utf-8",
    )

    # ASR should not run; cache should be cleared.
    monkeypatch.setattr(tr, "load_asr_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ASR should not run")))
    monkeypatch.setattr(
        tr, "load_qwen_asr_model", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ASR should not run"))
    )
    monkeypatch.setattr(tr, "generate_speaker_audio", lambda *_args, **_kwargs: None)

    ok = tr.transcribe_audio(str(folder), diarization=False)
    assert ok is True

    # Downstream artifacts should be removed.
    assert not (folder / "translation.json").exists()
    assert not (folder / "translation_raw.json").exists()
    assert not (folder / "summary.json").exists()
    assert not (folder / "transcript_punctuated.json").exists()
    assert not (wavs / ".tts_done.json").exists()
    assert not (wavs / "0000.wav").exists()

    out = json.loads((folder / "transcript.json").read_text(encoding="utf-8"))
    assert isinstance(out, list) and out
    assert out[0]["source"] == "youtube_subtitles"
