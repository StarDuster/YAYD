"""Synthesize speech (TTS) 步骤测试."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from youdub.config import Settings
from youdub.models import ModelManager


def _write_dummy_wav(path: Path, sr: int = 24000, seconds: float = 1.0) -> None:
    from youdub.utils import save_wav

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    wav = 0.1 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    save_wav(wav, str(path), sample_rate=sr)


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        rate = int(wf.getframerate() or 0)
        frames = int(wf.getnframes() or 0)
    return 0.0 if rate <= 0 else float(frames) / float(rate)


def _assert_translation_items_schema(items: list[dict[str, Any]]) -> None:
    assert isinstance(items, list)
    assert items, "translation.json must be a non-empty list"
    for it in items[:50]:
        assert isinstance(it, dict)
        assert isinstance(it.get("start"), (int, float))
        assert isinstance(it.get("end"), (int, float))
        assert isinstance(it.get("speaker"), str)
        assert isinstance(it.get("text"), str)
        assert isinstance(it.get("translation"), str)


# --------------------------------------------------------------------------- #
# WAV validation
# --------------------------------------------------------------------------- #


def test_is_valid_wav_rejects_non_wav_content_even_if_soundfile_can_read(tmp_path: Path):
    # Regression: cached "*.wav" files might actually be FLAC/OGG/etc (content-based readable by libsndfile),
    # but downstream `audiostretchy` uses stdlib `wave` and will crash. We should treat those as invalid.
    from youdub.steps.synthesize_speech import is_valid_wav
    from youdub.utils import save_wav

    if "FLAC" not in sf.available_formats():
        pytest.skip("libsndfile without FLAC support")

    sr = 24000
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False, dtype=np.float32)
    wav = 0.1 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    real_wav = tmp_path / "real.wav"
    save_wav(wav, str(real_wav), sample_rate=sr)
    assert is_valid_wav(str(real_wav)) is True

    fake_wav = tmp_path / "fake.wav"
    sf.write(str(fake_wav), wav, sr, format="FLAC")  # content is FLAC, name is *.wav
    assert is_valid_wav(str(fake_wav)) is False


# --------------------------------------------------------------------------- #
# Marker-based skip logic
# --------------------------------------------------------------------------- #


def test_generate_all_wavs_requires_marker_not_just_audio_file(tmp_path: Path, monkeypatch):
    import youdub.steps.synthesize_speech as ss

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "translation.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "translation": "好"}], ensure_ascii=False),
        encoding="utf-8",
    )
    # Create a valid per-segment wav but no marker
    (folder / "wavs").mkdir(parents=True, exist_ok=True)
    _write_dummy_wav(folder / "wavs" / "0000.wav", seconds=0.2)

    called = {"n": 0}

    def _fake_generate_wavs(_folder: str, _tts_method: str = "bytedance", **_kwargs) -> None:
        called["n"] += 1
        # Simulate success marker creation (marker now lives in wavs/)
        wavs_dir = Path(_folder) / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)
        (wavs_dir / ".tts_done.json").write_text(json.dumps({"tts_method": _tts_method}), encoding="utf-8")

    monkeypatch.setattr(ss, "generate_wavs", _fake_generate_wavs)

    ss.generate_all_wavs_under_folder(str(tmp_path), tts_method="bytedance")
    assert called["n"] == 1


def test_generate_all_wavs_reruns_when_marker_exists_but_wavs_incomplete(tmp_path: Path, monkeypatch):
    import youdub.steps.synthesize_speech as ss

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    transcript = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "translation": "一"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00", "translation": "二"},
        {"start": 2.0, "end": 3.0, "speaker": "SPEAKER_00", "translation": "三"},
    ]
    (folder / "translation.json").write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")

    # Create an incomplete wav cache (only 0000.wav exists).
    (folder / "wavs").mkdir(parents=True, exist_ok=True)
    _write_dummy_wav(folder / "wavs" / "0000.wav", seconds=0.2)

    # Create marker after translation.json so done_mtime >= tr_mtime.
    # Marker now lives in wavs/
    (folder / "wavs" / ".tts_done.json").write_text(
        json.dumps({"tts_method": "bytedance", "segments": 3}, ensure_ascii=False),
        encoding="utf-8",
    )

    called = {"n": 0}

    def _fake_generate_wavs(_folder: str, _tts_method: str = "bytedance", **_kwargs) -> None:
        called["n"] += 1
        job = Path(_folder)
        (job / "wavs").mkdir(parents=True, exist_ok=True)
        _write_dummy_wav(job / "wavs" / "0001.wav", seconds=0.2)
        _write_dummy_wav(job / "wavs" / "0002.wav", seconds=0.2)
        (job / "wavs" / ".tts_done.json").write_text(
            json.dumps({"tts_method": _tts_method, "segments": 3}, ensure_ascii=False),
            encoding="utf-8",
        )

    monkeypatch.setattr(ss, "generate_wavs", _fake_generate_wavs)

    ss.generate_all_wavs_under_folder(str(tmp_path), tts_method="bytedance")
    assert called["n"] == 1


def test_generate_all_wavs_reruns_when_marker_exists_but_wav_too_long(tmp_path: Path, monkeypatch):
    import youdub.steps.synthesize_speech as ss

    # Make the guard strict so the test is deterministic.
    monkeypatch.setenv("YOUDUB_TTS_MAX_SEGMENT_DURATION_RATIO", "2")
    monkeypatch.setenv("YOUDUB_TTS_MAX_SEGMENT_DURATION_EXTRA_SEC", "0")

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "translation.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "translation": "好"}], ensure_ascii=False),
        encoding="utf-8",
    )
    (folder / "wavs").mkdir(parents=True, exist_ok=True)
    # Create an abnormally long segment wav (should trigger re-run even if marker exists).
    _write_dummy_wav(folder / "wavs" / "0000.wav", seconds=5.0)

    # Marker now lives in wavs/
    (folder / "wavs" / ".tts_done.json").write_text(
        json.dumps({"tts_method": "bytedance", "segments": 1}, ensure_ascii=False),
        encoding="utf-8",
    )

    called = {"n": 0}

    def _fake_generate_wavs(_folder: str, _tts_method: str = "bytedance", **_kwargs) -> None:
        called["n"] += 1
        job = Path(_folder)
        (job / "wavs").mkdir(parents=True, exist_ok=True)
        _write_dummy_wav(job / "wavs" / "0000.wav", seconds=0.2)
        (job / "wavs" / ".tts_done.json").write_text(
            json.dumps({"tts_method": _tts_method, "segments": 1}, ensure_ascii=False),
            encoding="utf-8",
        )

    monkeypatch.setattr(ss, "generate_wavs", _fake_generate_wavs)

    ss.generate_all_wavs_under_folder(str(tmp_path), tts_method="bytedance")
    assert called["n"] == 1


# --------------------------------------------------------------------------- #
# ByteDance TTS integration
# --------------------------------------------------------------------------- #


def test_generate_wavs_bytedance_produces_audio_combined_and_srt(tmp_path: Path, monkeypatch):
    import youdub.steps.synthesize_speech as ss
    import youdub.steps.synthesize_video as sv

    # Avoid sleeps in unit tests.
    monkeypatch.setattr(ss, "sleep_with_cancel", lambda *_args, **_kwargs: None)

    # Avoid depending on external time-stretch details in this contract test.
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

    # Stub ByteDance network call: just write a small valid wav.
    def _fake_bytedance_tts(_text: str, output_path: str, _speaker_wav: str, voice_type: str | None = None) -> None:
        _ = voice_type
        _write_dummy_wav(Path(output_path), seconds=0.2)

    monkeypatch.setattr(ss, "bytedance_tts", _fake_bytedance_tts)

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    # Provide vocals/instruments so the audio-build step can do scaling/mixing.
    _write_dummy_wav(folder / "audio_vocals.wav", seconds=4.0)
    _write_dummy_wav(folder / "audio_instruments.wav", seconds=4.0)

    transcript = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "hello", "translation": "你好"},
        {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "world", "translation": "世界"},
    ]
    (folder / "translation.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    orig_translation_json = (folder / "translation.json").read_text(encoding="utf-8")

    ss.generate_wavs(str(folder), tts_method="bytedance")

    assert (folder / "wavs" / "0000.wav").exists()
    assert (folder / "wavs" / "0001.wav").exists()
    assert ss.is_valid_wav(str(folder / "wavs" / "0000.wav")) is True
    assert ss.is_valid_wav(str(folder / "wavs" / "0001.wav")) is True

    assert (folder / "wavs" / ".tts_done.json").exists()
    state = json.loads((folder / "wavs" / ".tts_done.json").read_text(encoding="utf-8"))
    assert state.get("tts_method") == "bytedance"

    # Build aligned voice + combined audio (video synthesis stage).
    sv._ensure_audio_combined(str(folder), adaptive_segment_stretch=False)
    assert (folder / "audio_tts.wav").exists()
    assert (folder / "audio_combined.wav").exists()
    assert ss.is_valid_wav(str(folder / "audio_combined.wav")) is True

    # translation.json should not be modified by TTS/audio build.
    assert (folder / "translation.json").read_text(encoding="utf-8") == orig_translation_json

    # translation.json should remain usable by the next module (video synthesis / SRT generation).
    translation = json.loads((folder / "translation.json").read_text(encoding="utf-8"))
    _assert_translation_items_schema(translation)

    srt_path = folder / "subtitles.srt"
    sv.generate_srt(translation, str(srt_path))
    assert srt_path.exists()
    content = srt_path.read_text(encoding="utf-8")
    assert "-->" in content


def test_generate_wavs_retries_when_segment_too_long(tmp_path: Path, monkeypatch):
    import youdub.steps.synthesize_speech as ss

    # Avoid sleeps in unit tests.
    monkeypatch.setattr(ss, "sleep_with_cancel", lambda *_args, **_kwargs: None)

    # Make the guard strict so the test is deterministic.
    monkeypatch.setenv("YOUDUB_TTS_MAX_SEGMENT_DURATION_RATIO", "2")
    monkeypatch.setenv("YOUDUB_TTS_MAX_SEGMENT_DURATION_EXTRA_SEC", "0")
    monkeypatch.setenv("YOUDUB_TTS_SEGMENT_MAX_RETRIES", "2")

    folder = tmp_path / "job_retry"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SPEAKER").mkdir(parents=True, exist_ok=True)
    _write_dummy_wav(folder / "SPEAKER" / "SPEAKER_00.wav", seconds=1.0)

    (folder / "translation.json").write_text(
        json.dumps(
            [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "x", "translation": "你好"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calls = {"n": 0}

    def _fake_bytedance_tts(_text: str, output_path: str, _speaker_wav: str, voice_type: str | None = None) -> None:
        _ = voice_type
        calls["n"] += 1
        if calls["n"] == 1:
            # First attempt: produce a wav longer than allowed -> should be deleted and retried.
            _write_dummy_wav(Path(output_path), seconds=5.0)
        else:
            _write_dummy_wav(Path(output_path), seconds=0.2)

    monkeypatch.setattr(ss, "bytedance_tts", _fake_bytedance_tts)

    ss.generate_wavs(str(folder), tts_method="bytedance")
    assert calls["n"] == 2

    out = folder / "wavs" / "0000.wav"
    assert out.exists()
    assert ss.is_valid_wav(str(out)) is True
    assert _wav_duration_seconds(out) <= 2.05


# --------------------------------------------------------------------------- #
# Qwen TTS integration
# --------------------------------------------------------------------------- #


def test_generate_wavs_qwen_path_smoke(tmp_path, monkeypatch):
    """
    End-to-end-ish smoke test for the qwen TTS path:
    - No large Qwen weights are needed: we run the worker in stub mode.
    - We still exercise: worker IPC, per-segment wav writing, stitching, and output files.
    """
    import youdub.steps.synthesize_speech as ss
    import youdub.steps.synthesize_video as sv

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

    assert (folder / "wavs" / "0000.wav").exists()
    assert (folder / "wavs" / "0001.wav").exists()

    sv._ensure_audio_combined(str(folder), adaptive_segment_stretch=False)
    assert (folder / "audio_tts.wav").exists()
    assert (folder / "audio_combined.wav").exists()


# --------------------------------------------------------------------------- #
# Qwen worker protocol
# --------------------------------------------------------------------------- #


def test_qwen_worker_protocol_tolerates_stdout_noise(tmp_path, monkeypatch):
    """
    Regression test:
    - Worker may print non-protocol lines to stdout before __READY__ (or between request/response).
    - Parent must ignore noise and still complete handshake + JSON response parsing.
    """
    import youdub.steps.synthesize_speech as ss

    # Ensure we don't accidentally pass --stub from environment.
    monkeypatch.delenv("YOUDUB_QWEN_WORKER_STUB", raising=False)

    # Create a tiny "noisy worker" compatible with the same argv protocol.
    worker_path = tmp_path / "noisy_qwen_worker.py"
    worker_path.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import wave

READY_LINE = "__READY__"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--stub", action="store_true")
    return p.parse_args()


def _write_wav(path: str, sr: int = 24000, seconds: float = 0.2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = int(sr * seconds)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\\x00\\x00" * n)


def main() -> int:
    _ = _parse_args()

    # Noise before READY (stdout + stderr)
    print("noise-before-ready", flush=True)
    print("stderr-noise-before-ready", file=sys.stderr, flush=True)
    print(READY_LINE, flush=True)

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        req = json.loads(raw)
        cmd = req.get("cmd")
        if cmd == "shutdown":
            break
        if cmd != "synthesize":
            print(json.dumps({"ok": False, "error": f"未知命令: {cmd}"}), flush=True)
            continue

        # Noise between request and response (stdout)
        print("noise-between-request-and-response", flush=True)

        out = str(req.get("output_path", ""))
        _write_wav(out)
        print(json.dumps({"ok": True, "output_path": out}), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )

    # Use our worker script for this test only.
    monkeypatch.setattr(ss, "_get_qwen_worker_script_path", lambda: worker_path)

    # Minimal settings: python exists; model dir exists (worker ignores it but arg is required).
    model_dir = tmp_path / "dummy_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "dummy.txt").write_text("ok", encoding="utf-8")

    test_settings = Settings(qwen_tts_model_path=model_dir, qwen_tts_python_path=Path(sys.executable))
    monkeypatch.setattr(ss, "_DEFAULT_SETTINGS", test_settings)
    monkeypatch.setattr(ss, "_DEFAULT_MODEL_MANAGER", ModelManager(test_settings))

    w = ss._QwenTtsWorker.from_settings(test_settings)  # noqa: SLF001
    out = tmp_path / "out.wav"
    w.synthesize("hello", speaker_wav=str(tmp_path / "spk.wav"), output_path=str(out), language="Auto")
    w.close()

    assert out.exists()
    assert out.stat().st_size >= 44


# --------------------------------------------------------------------------- #
# Speaker reference auto-generation
# --------------------------------------------------------------------------- #


def test_generate_wavs_qwen_creates_missing_speaker_ref(tmp_path: Path, monkeypatch):
    """
    Regression:
    - When translation.json exists but SPEAKER/<speaker>.wav is missing,
      qwen TTS path should generate a short reference wav and continue.
    """
    import os

    import youdub.steps.synthesize_speech as ss
    import youdub.steps.synthesize_video as sv

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

    sv._ensure_audio_combined(str(folder), adaptive_segment_stretch=False)
    assert (folder / "audio_tts.wav").exists()
    assert (folder / "audio_combined.wav").exists()


# --------------------------------------------------------------------------- #
# TTS prompt text normalization (Python tokens)
# --------------------------------------------------------------------------- #


def test_tts_text_for_attempt_normalizes_python_tokens_for_tts_only():
    import youdub.steps.synthesize_speech as ss

    s1 = "接着导入 matplotlib.pyplot 简写为 plt，用来绘制幅度谱"
    out1 = ss._tts_text_for_attempt(s1, 0)  # noqa: SLF001
    assert "matplotlib dot pyplot" in out1
    assert "matplotlib.pyplot" not in out1

    s2 = "我们会调用 os.path.join"
    out2 = ss._tts_text_for_attempt(s2, 0)  # noqa: SLF001
    assert "os dot path dot join" in out2
    assert "os.path.join" not in out2

    s3 = "传入 base_dir 以及小提琴的文件路径"
    out3 = ss._tts_text_for_attempt(s3, 0)  # noqa: SLF001
    assert "base dir" in out3
    assert "base_dir" not in out3

    # Do not mis-handle decimals like 3.14 as "dot".
    s4 = "圆周率是 3.14"
    out4 = ss._tts_text_for_attempt(s4, 0)  # noqa: SLF001
    assert " dot " not in out4
