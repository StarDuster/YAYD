from __future__ import annotations

from ..models import ModelManager


def ensure_assets(model_manager: ModelManager, *, require_diarization: bool) -> None:
    """
    Ensure required model assets exist in offline mode.

    NOTE:
    - Whisper ASR model path is often provided via UI arguments (not only Settings).
      Avoid enforcing a fixed Settings-based path here; `load_asr_model()` will validate the
      actual model_dir passed in (expects model.bin).
    """
    model_manager.enforce_offline()
    if require_diarization:
        model_manager.ensure_ready(
            names=[model_manager._whisper_diarization_requirement().name]  # type: ignore[attr-defined]
        )

