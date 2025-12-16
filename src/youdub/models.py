from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from loguru import logger

from .config import Settings


class ModelCheckError(RuntimeError):
    """Raised when required models or credentials are missing."""


@dataclass
class ModelRequirement:
    name: str
    path: Optional[Path]
    hint: str
    required: bool = True
    env_keys: Optional[list[str]] = None

    def exists(self) -> bool:
        if self.env_keys:
            # All listed env keys must be present for this requirement to be considered satisfied.
            if all(os.getenv(k) for k in self.env_keys):
                return True
        # Path-based requirement
        if self.path is None:
            return False
        path = Path(self.path).expanduser()
        if path.is_file():
            return True
        if path.is_dir():
            # Consider non-empty directory as present
            return any(path.iterdir())
        return False


class ModelManager:
    """Central place to validate required models and surface user-friendly hints."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

    def _demucs_requirement(self) -> ModelRequirement:
        path = self.settings.resolve_path(self.settings.demucs_model_dir)
        hint = (
            f"请提前下载 Demucs 权重（{self.settings.demucs_model_name}），"
            f"放到 {path}，否则 Demucs 会尝试在线下载。"
        )
        return ModelRequirement(
            name=f"Demucs ({self.settings.demucs_model_name})",
            path=path,
            hint=hint,
            required=True,
        )

    def _whisper_requirement(self) -> ModelRequirement:
        path = self.settings.resolve_path(self.settings.whisper_model_path)
        hint = (
            "需要 WhisperX CTranslate2 离线模型（例如 large-v3 转换后的 model.bin）。"
            f" 请下载后放在 {path} 或设置 WHISPERX_MODEL_PATH。"
        )
        return ModelRequirement(
            name=f"WhisperX ASR ({self.settings.whisper_model_name})",
            path=path,
            hint=hint,
        )

    def _whisper_align_requirement(self) -> ModelRequirement:
        path = self.settings.resolve_path(self.settings.whisper_align_model_dir)
        hint = (
            "需要 WhisperX 对齐模型缓存（wav2vec2 等）。"
            " 参考 WhisperX 文档下载到本地并设置 WHISPERX_ALIGN_MODEL_DIR。"
        )
        return ModelRequirement(
            name="WhisperX Alignment",
            path=path,
            hint=hint,
        )

    def _whisper_diarization_requirement(self) -> ModelRequirement:
        path = self.settings.resolve_path(self.settings.whisper_diarization_model_dir)
        hint = (
            "需要 WhisperX 说话人分离模型缓存（pyannote）。"
            " 参考 WhisperX 文档下载到本地并设置 WHISPERX_DIARIZATION_MODEL_DIR / HF_TOKEN。"
        )
        return ModelRequirement(
            name="WhisperX Diarization",
            path=path,
            hint=hint,
        )

    def _xtts_requirement(self) -> ModelRequirement:
        path = self.settings.resolve_path(self.settings.xtts_model_path)
        hint = (
            "需要本地 XTTS v2 模型目录（包含 config.json / model.pth 等），"
            "请提前下载并设置 XTTS_MODEL_PATH。若不用 XTTS，请将 TTS Method 设为 bytedance。"
        )
        return ModelRequirement(
            name="XTTS v2",
            path=path,
            hint=hint,
        )

    def _bytedance_requirement(self) -> ModelRequirement:
        hint = "配置 BYTEDANCE_APPID 与 BYTEDANCE_ACCESS_TOKEN 用于火山 TTS。"
        env_present = all(
            [
                bool(self.settings.bytedance_appid),
                bool(self.settings.bytedance_access_token),
            ]
        )
        path = Path(".bytedance_credentials_ok") if env_present else None
        return ModelRequirement(
            name="ByteDance TTS Credentials",
            path=path,
            hint=hint,
            env_keys=["BYTEDANCE_APPID", "BYTEDANCE_ACCESS_TOKEN"],
        )

    def _gemini_tts_requirement(self) -> ModelRequirement:
        hint = "配置 GEMINI_API_KEY 用于 Gemini TTS。"
        env_present = bool(self.settings.gemini_api_key)
        path = Path(".gemini_credentials_ok") if env_present else None
        return ModelRequirement(
            name="Gemini TTS Credentials",
            path=path,
            hint=hint,
            env_keys=["GEMINI_API_KEY"],
        )

    def list_requirements(self) -> list[ModelRequirement]:
        return [
            self._demucs_requirement(),
            self._whisper_requirement(),
            self._whisper_align_requirement(),
            self._whisper_diarization_requirement(),
            self._xtts_requirement(),
            self._gemini_tts_requirement(),
            self._bytedance_requirement(),
        ]

    def _has_demucs_weights(self, path: Path | None) -> bool:
        if path is None:
            return False
        path = Path(path)
        if not path.exists() or not path.is_dir():
            return False
        for suffix in (".ckpt", ".pth", ".pt", ".th"):
            for _ in path.rglob(f"*{self.settings.demucs_model_name}*{suffix}"):
                return True
        # Fallback: non-empty dir
        return any(path.iterdir())

    def missing(self, names: Iterable[str] | None = None) -> list[ModelRequirement]:
        requirements = self.list_requirements()
        if names:
            selected = {name for name in names}
            requirements = [req for req in requirements if req.name in selected]
        missing_requirements: list[ModelRequirement] = []
        for req in requirements:
            if req.name.startswith("Demucs") and not self._has_demucs_weights(req.path):
                missing_requirements.append(req)
                continue
            if not req.exists():
                missing_requirements.append(req)
        return missing_requirements

    def ensure_ready(self, names: Iterable[str] | None = None) -> None:
        missing = self.missing(names=names)
        if missing:
            raise ModelCheckError(self.format_missing(missing))

    def format_missing(self, missing: list[ModelRequirement]) -> str:
        if not missing:
            return "模型已就绪。"
        lines = ["模型未准备就绪，请先手动下载或配置："]
        for req in missing:
            path_text = f"期望路径: {req.path}" if req.path else "缺少必需的凭据或路径"
            lines.append(f"- {req.name}: {path_text}。{req.hint}")
        return "\n".join(lines)

    def describe_status(self) -> str:
        statuses = []
        for req in self.list_requirements():
            status = "✅ 已找到" if req.exists() else "❌ 未找到"
            location = f"（{req.path}）" if req.path else ""
            statuses.append(f"{status} {req.name} {location}")
        return "\n".join(statuses)

    def enforce_offline(self) -> None:
        """Set environment flags to avoid accidental online downloads."""
        import os

        if os.getenv("HF_HUB_OFFLINE") is None:
            os.environ["HF_HUB_OFFLINE"] = "1"
        if os.getenv("TRANSFORMERS_OFFLINE") is None:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        if os.getenv("WHISPERX_LOCAL_FILES_ONLY") is None:
            os.environ["WHISPERX_LOCAL_FILES_ONLY"] = "1"
        logger.debug("Offline mode enforced for Hugging Face/WhisperX downloads.")
