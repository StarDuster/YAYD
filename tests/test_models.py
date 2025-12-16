import pytest

from youdub.config import Settings
from youdub.models import ModelCheckError, ModelManager


def test_missing_models_reports_all(tmp_path):
    settings = Settings(
        demucs_model_dir=tmp_path / "demucs",
        whisper_model_path=tmp_path / "whisper",
        whisper_align_model_dir=tmp_path / "align",
        whisper_diarization_model_dir=tmp_path / "diar",
        xtts_model_path=tmp_path / "xtts",
    )
    manager = ModelManager(settings)
    missing = manager.missing()
    names = {req.name for req in missing}
    assert any(name.startswith("WhisperX ASR") for name in names)
    assert any("XTTS" in name for name in names)
    assert any(name.startswith("Demucs") for name in names)


def test_ensure_ready_raises_on_missing(tmp_path):
    settings = Settings(
        demucs_model_dir=tmp_path / "demucs",
        whisper_model_path=tmp_path / "whisper",
        whisper_align_model_dir=tmp_path / "align",
        whisper_diarization_model_dir=tmp_path / "diar",
        xtts_model_path=tmp_path / "xtts",
    )
    manager = ModelManager(settings)
    with pytest.raises(ModelCheckError):
        manager.ensure_ready()
