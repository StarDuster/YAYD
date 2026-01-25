import types
from pathlib import Path

import pytest

from youdub.config import Settings
from youdub.models import ModelManager


def _touch_model_bin(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.bin").write_bytes(b"0")


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

