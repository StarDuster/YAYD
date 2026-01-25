import json
import sys
from pathlib import Path

import numpy as np


def _write_dummy_wav(path: Path, sr: int = 24000, seconds: float = 1.0) -> None:
    # Use project utility to keep format consistent (int16 PCM WAV).
    from youdub.utils import save_wav

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    wav = 0.1 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    save_wav(wav, str(path), sample_rate=sr)


def test_generate_wavs_qwen_path_smoke(tmp_path, monkeypatch):
    """
    End-to-end-ish smoke test for the qwen TTS path:
    - No large Qwen weights are needed: we run the worker in stub mode.
    - We still exercise: worker IPC, per-segment wav writing, stitching, and output files.
    """

    import youdub.steps.synthesize_speech as ss
    from youdub.config import Settings
    from youdub.models import ModelManager

    # Avoid depending on external time-stretch details in this test.
    def _adjust_audio_length_no_stretch(
        wav_path: str,
        desired_length: float,
        sample_rate: int = 24000,
        min_speed_factor: float = 0.6,
        max_speed_factor: float = 1.1,
    ):
        wav, _ = ss.librosa.load(wav_path, sr=sample_rate)
        target_samples = max(0, int(desired_length * sample_rate))
        if target_samples == 0:
            return np.zeros((0,), dtype=np.float32), 0.0
        if len(wav) >= target_samples:
            out = wav[:target_samples]
        else:
            out = np.pad(wav, (0, target_samples - len(wav)), mode="constant")
        return out.astype(np.float32), float(len(out) / sample_rate)

    monkeypatch.setattr(ss, "adjust_audio_length", _adjust_audio_length_no_stretch)

    # Run Qwen worker in stub mode (no qwen-tts import).
    monkeypatch.setenv("YOUDUB_QWEN_WORKER_STUB", "1")

    # Point Qwen model path to a dummy local directory (worker stub doesn't read it, but the path must exist).
    model_dir = tmp_path / "qwen_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "dummy.txt").write_text("ok", encoding="utf-8")

    test_settings = Settings(qwen_tts_model_path=model_dir, qwen_tts_python_path=Path(sys.executable))
    monkeypatch.setattr(ss, "_DEFAULT_SETTINGS", test_settings)
    monkeypatch.setattr(ss, "_DEFAULT_MODEL_MANAGER", ModelManager(test_settings))

    # Build a minimal folder structure expected by generate_wavs()
    folder = tmp_path / "job"
    speaker_dir = folder / "SPEAKER"
    speaker_dir.mkdir(parents=True, exist_ok=True)

    speaker = "S1"
    speaker_wav = speaker_dir / f"{speaker}.wav"
    _write_dummy_wav(speaker_wav)

    transcript = [
        {"start": 0.0, "end": 1.0, "speaker": speaker, "translation": "你好"},
        {"start": 1.0, "end": 2.2, "speaker": speaker, "translation": "世界"},
    ]
    (folder / "translation.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

    ss.generate_wavs(str(folder), tts_method="qwen")

    assert (folder / "audio_tts.wav").exists()
    assert (folder / "audio_combined.wav").exists()
    assert (folder / "wavs" / "0000.wav").exists()
    assert (folder / "wavs" / "0001.wav").exists()


def test_generate_wavs_qwen_icl_path_smoke(tmp_path, monkeypatch):
    """
    Smoke test for Qwen ICL path (stub worker):
    - Enable QWEN_TTS_ICL=1.
    - Provide audio_vocals.wav + per-segment `text` so the pipeline can build ICL refs.
    - Worker runs in stub mode (no qwen-tts import).
    """
    import youdub.steps.synthesize_speech as ss
    from youdub.config import Settings
    from youdub.models import ModelManager

    # Avoid depending on external time-stretch details in this test.
    def _adjust_audio_length_no_stretch(
        wav_path: str,
        desired_length: float,
        sample_rate: int = 24000,
        min_speed_factor: float = 0.6,
        max_speed_factor: float = 1.1,
    ):
        wav, _ = ss.librosa.load(wav_path, sr=sample_rate)
        target_samples = max(0, int(desired_length * sample_rate))
        if target_samples == 0:
            return np.zeros((0,), dtype=np.float32), 0.0
        if len(wav) >= target_samples:
            out = wav[:target_samples]
        else:
            out = np.pad(wav, (0, target_samples - len(wav)), mode="constant")
        return out.astype(np.float32), float(len(out) / sample_rate)

    monkeypatch.setattr(ss, "adjust_audio_length", _adjust_audio_length_no_stretch)

    # Run Qwen worker in stub mode (no qwen-tts import).
    monkeypatch.setenv("YOUDUB_QWEN_WORKER_STUB", "1")
    monkeypatch.setenv("QWEN_TTS_ICL", "1")
    monkeypatch.setenv("TTS_SPEAKER_REF_SECONDS", "3")

    # Point Qwen model path to a dummy local directory (worker stub doesn't read it, but the path must exist).
    model_dir = tmp_path / "qwen_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "dummy.txt").write_text("ok", encoding="utf-8")

    test_settings = Settings(qwen_tts_model_path=model_dir, qwen_tts_python_path=Path(sys.executable))
    monkeypatch.setattr(ss, "_DEFAULT_SETTINGS", test_settings)
    monkeypatch.setattr(ss, "_DEFAULT_MODEL_MANAGER", ModelManager(test_settings))

    folder = tmp_path / "job_icl"
    (folder / "SPEAKER").mkdir(parents=True, exist_ok=True)

    # Provide vocals audio so ICL can extract anchor + reference segments.
    _write_dummy_wav(folder / "audio_vocals.wav", seconds=6.0)

    speaker = "S1"
    transcript = [
        # Simulate split_sentences(): consecutive items share same original `text`.
        {"start": 0.0, "end": 1.0, "speaker": speaker, "text": "Hello world.", "translation": "你好"},
        {"start": 1.0, "end": 2.2, "speaker": speaker, "text": "Hello world.", "translation": "世界"},
    ]
    (folder / "translation.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

    ss.generate_wavs(str(folder), tts_method="qwen")

    assert (folder / ".qwen_speaker_anchor.wav").exists()
    assert (folder / ".qwen_icl_ref" / "0000.wav").exists()
    assert (folder / "audio_tts.wav").exists()
    assert (folder / "audio_combined.wav").exists()
    assert (folder / "wavs" / "0000.wav").exists()
    assert (folder / "wavs" / "0001.wav").exists()

