import pytest

from youdub.config import Settings
from youdub.models import ModelCheckError, ModelManager


def test_missing_models_reports_all(tmp_path):
    settings = Settings(
        demucs_model_dir=tmp_path / "demucs",
        whisper_model_path=tmp_path / "whisper",
        whisper_diarization_model_dir=tmp_path / "diar",
        qwen_tts_model_path=tmp_path / "qwen",
        qwen_tts_python_path=tmp_path / "qwen_py",
    )
    manager = ModelManager(settings)
    missing = manager.missing()
    names = {req.name for req in missing}
    assert any(name.startswith("Whisper ASR") for name in names)
    assert any("Qwen3-TTS" in name for name in names)
    assert any(name.startswith("Demucs") for name in names)


def test_ensure_ready_raises_on_missing(tmp_path):
    settings = Settings(
        demucs_model_dir=tmp_path / "demucs",
        whisper_model_path=tmp_path / "whisper",
        whisper_diarization_model_dir=tmp_path / "diar",
        qwen_tts_model_path=tmp_path / "qwen",
        qwen_tts_python_path=tmp_path / "qwen_py",
    )
    manager = ModelManager(settings)
    with pytest.raises(ModelCheckError):
        manager.ensure_ready()
