from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger

from .config import Settings


class ModelCheckError(RuntimeError):
    """模型/凭据缺失时抛出（用于 UI 友好提示）。"""


@dataclass(frozen=True)
class ModelRequirement:
    """一个“资源要求”（文件/目录/配置占位），用于模型检查与提示。"""

    name: str
    path: Optional[Path]
    hint: str

    def exists(self) -> bool:
        if self.path is None:
            return False
        path = Path(self.path).expanduser()
        if path.is_file():
            return True
        if path.is_dir():
            # Consider non-empty directory as present
            try:
                return any(path.iterdir())
            except PermissionError:
                return False
        return False


class ModelManager:
    """集中管理：依赖资源是否就绪 + 对用户的提示文案。"""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()

    def _demucs_requirement(self) -> ModelRequirement:
        path = self.settings.resolve_path(self.settings.demucs_model_dir)
        hint = (
            f"需要 Demucs 权重（{self.settings.demucs_model_name}）。"
            f"请先把权重文件放到：{path}（或设置 DEMUCS_MODEL_DIR）。"
        )
        return ModelRequirement(
            name=f"Demucs ({self.settings.demucs_model_name})",
            path=path,
            hint=hint,
        )

    def _whisper_requirement(self) -> ModelRequirement:
        path = self.settings.resolve_path(self.settings.whisper_model_path)
        hint = (
            "需要 faster-whisper 的 CTranslate2 离线模型目录（目录内应包含 model.bin）。"
            f"请下载后放到：{path}（或设置 WHISPER_MODEL_PATH）。"
        )
        return ModelRequirement(
            name=f"Whisper ASR ({self.settings.whisper_model_name})",
            path=path,
            hint=hint,
        )

    def _whisper_diarization_requirement(self) -> ModelRequirement:
        path = self.settings.resolve_path(self.settings.whisper_diarization_model_dir)
        hint = (
            "需要 pyannote 的说话人分离缓存（speaker-diarization-3.1 + segmentation-3.0）。"
            "请先离线下载到 WHISPER_DIARIZATION_MODEL_DIR，并配置 HF_TOKEN（用于首次下载/校验）。"
        )
        return ModelRequirement(
            name="Speaker Diarization (pyannote)",
            path=path,
            hint=hint,
        )

    def _qwen_tts_runtime_requirement(self) -> ModelRequirement:
        path = self.settings.resolve_path(self.settings.qwen_tts_python_path)
        hint = (
            "需要一个能运行 Qwen3-TTS worker 的 Python 解释器。默认可直接用主工程 .venv；"
            "如果你自己改了 QWEN_TTS_PYTHON，请确保该环境已安装 qwen-tts（能 import qwen_tts）。"
        )
        return ModelRequirement(
            name="Qwen3-TTS Runtime (python)",
            path=path,
            hint=hint,
        )

    def _qwen_tts_weights_requirement(self) -> ModelRequirement:
        path = self.settings.resolve_path(self.settings.qwen_tts_model_path)
        hint = (
            "需要本地 Qwen3-TTS Base 权重目录（例如包含 config.json / model.safetensors）。"
            f"请下载后放到：{path}（或设置 QWEN_TTS_MODEL_PATH）。"
        )
        return ModelRequirement(
            name="Qwen3-TTS Base Weights",
            path=path,
            hint=hint,
        )

    def _bytedance_requirement(self) -> ModelRequirement:
        ok = bool(self.settings.bytedance_appid) and bool(self.settings.bytedance_access_token)
        # Use an always-existing marker when credentials are present. Avoid relying on os.environ (Settings reads .env).
        path = Path(__file__) if ok else None
        hint = "需要火山引擎 TTS 凭据：BYTEDANCE_APPID / BYTEDANCE_ACCESS_TOKEN（可写在 .env）。"
        return ModelRequirement(name="ByteDance TTS Credentials", path=path, hint=hint)

    def _gemini_tts_requirement(self) -> ModelRequirement:
        ok = bool(self.settings.gemini_api_key)
        path = Path(__file__) if ok else None
        hint = "需要 Gemini TTS 凭据：GEMINI_API_KEY（可写在 .env）。"
        return ModelRequirement(name="Gemini TTS Credentials", path=path, hint=hint)

    def list_requirements(self) -> list[ModelRequirement]:
        return [
            self._demucs_requirement(),
            self._whisper_requirement(),
            self._whisper_diarization_requirement(),
            self._qwen_tts_runtime_requirement(),
            self._qwen_tts_weights_requirement(),
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
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        logger.debug("Offline mode enforced for Hugging Face downloads.")
