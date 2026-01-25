import json
import os
import wave
from pathlib import Path

import numpy as np


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        rate = int(wf.getframerate() or 0)
        frames = int(wf.getnframes() or 0)
    return 0.0 if rate <= 0 else float(frames) / float(rate)


def _write_dummy_wav(path: Path, sr: int = 24000, seconds: float = 4.0) -> None:
    from youdub.utils import save_wav

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    wav = 0.1 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    save_wav(wav, str(path), sample_rate=sr)


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


def test_generate_wavs_qwen_creates_missing_speaker_ref(tmp_path: Path, monkeypatch):
    """
    Regression:
    - When translation.json exists but SPEAKER/<speaker>.wav is missing,
      qwen TTS path should generate a short reference wav and continue.
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

    # Limit reference duration to make assertions deterministic.
    monkeypatch.setenv("TTS_SPEAKER_REF_SECONDS", "10")

    model_dir = tmp_path / "qwen_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "dummy.txt").write_text("ok", encoding="utf-8")

    test_settings = Settings(qwen_tts_model_path=model_dir, qwen_tts_python_path=Path(os.sys.executable))
    monkeypatch.setattr(ss, "_DEFAULT_SETTINGS", test_settings)
    monkeypatch.setattr(ss, "_DEFAULT_MODEL_MANAGER", ModelManager(test_settings))

    folder = tmp_path / "job2"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SPEAKER").mkdir(parents=True, exist_ok=True)  # keep empty on purpose

    # Provide vocals audio so generate_wavs() can auto-generate speaker refs.
    _write_dummy_wav(folder / "audio_vocals.wav", seconds=20.0)

    speaker = "SPEAKER_00"
    transcript = [
        {"start": 0.0, "end": 1.0, "speaker": speaker, "translation": "你好"},
        {"start": 1.0, "end": 2.0, "speaker": speaker, "translation": "世界"},
    ]
    (folder / "translation.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

    ss.generate_wavs(str(folder), tts_method="qwen")

    spk_ref = folder / "SPEAKER" / f"{speaker}.wav"
    assert spk_ref.exists()
    assert spk_ref.stat().st_size >= 44
    assert _wav_duration_seconds(spk_ref) <= 10.1

    assert (folder / "wavs" / "0000.wav").exists()
    assert (folder / "wavs" / "0001.wav").exists()
    assert (folder / "audio_tts.wav").exists()
    assert (folder / "audio_combined.wav").exists()

