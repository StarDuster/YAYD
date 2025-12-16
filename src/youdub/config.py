from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration loaded from environment variables and .env."""

    # General paths
    root_folder: Path = Field(default=Path("videos"), description="Base folder for downloads and outputs")

    # Demucs
    demucs_model_name: str = Field(
        default="htdemucs_ft",
        description="Demucs model identifier",
        alias="DEMUCS_MODEL_NAME",
    )
    demucs_model_dir: Path = Field(
        default=Path("models/demucs"),
        description="Folder containing Demucs weights (no auto-download)",
        alias="DEMUCS_MODEL_DIR",
    )
    demucs_device: str = Field(default="auto", description="cuda/cpu/auto for Demucs")
    demucs_shifts: int = Field(default=5, description="Number of shifts for Demucs separator")

    # WhisperX / ASR
    whisper_model_path: Path = Field(
        default=Path("models/ASR/whisper"),
        description="Path to the locally downloaded WhisperX CTranslate2 model",
        alias="WHISPERX_MODEL_PATH",
    )
    whisper_download_root: Path = Field(
        default=Path("models/ASR/whisper"),
        description="Root path where WhisperX assets are stored",
        alias="WHISPERX_DOWNLOAD_ROOT",
    )
    whisper_align_model_dir: Optional[Path] = Field(
        default=Path("models/ASR/whisper/align"),
        description="Path to alignment model cache (offline, no auto-download)",
        alias="WHISPERX_ALIGN_MODEL_DIR",
    )
    whisper_diarization_model_dir: Optional[Path] = Field(
        default=Path("models/ASR/whisper/diarization"),
        description="Path to diarization model cache (offline, no auto-download)",
        alias="WHISPERX_DIARIZATION_MODEL_DIR",
    )
    whisper_model_name: str = Field(default="large-v3", description="WhisperX model name", alias="WHISPERX_MODEL_NAME")
    whisper_batch_size: int = Field(default=32, description="Batch size for WhisperX")
    whisper_local_files_only: bool = Field(
        default=True, description="Force WhisperX to run in offline/local-only mode"
    )

    # Translation
    translation_target_language: str = Field(default="简体中文", description="Default translation target language")

    # TTS
    xtts_model_path: Optional[Path] = Field(
        default=Path("models/TTS/xtts_v2"),
        description="Local path to XTTS v2 model directory (no auto-download)",
        alias="XTTS_MODEL_PATH",
    )
    tts_method: str = Field(
        default="bytedance",
        description="TTS Engine to use (bytedance or xtts)",
        alias="TTS_METHOD"
    )

    # API tokens / credentials
    hf_token: Optional[str] = Field(default=None, description="HuggingFace token", alias="HF_TOKEN")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key", alias="OPENAI_API_KEY")
    openai_api_base: Optional[str] = Field(default=None, description="OpenAI compatible base URL", alias="OPENAI_API_BASE")
    bytedance_appid: Optional[str] = Field(default=None, description="ByteDance appid", alias="BYTEDANCE_APPID")
    bytedance_access_token: Optional[str] = Field(default=None, description="ByteDance access token", alias="BYTEDANCE_ACCESS_TOKEN")
    model_name: str = Field(default="gpt-3.5-turbo", description="Model name for summary/translation", alias="MODEL_NAME")
    bili_sessdata: Optional[str] = Field(default=None, description="Bilibili SESSDATA cookie", alias="BILI_SESSDATA")
    bili_jct: Optional[str] = Field(default=None, description="Bilibili bili_jct token", alias="BILI_BILI_JCT")
    
    # Gemini TTS
    gemini_api_key: Optional[str] = Field(
        default=None, 
        description="Gemini API key for TTS",
        alias="GEMINI_API_KEY"
    )
    gemini_tts_voice: str = Field(
        default="Kore",
        description="Gemini TTS voice name (30 options)",
        alias="GEMINI_TTS_VOICE"
    )
    gemini_tts_model: str = Field(
        default="gemini-2.5-flash-preview-tts",
        description="Gemini TTS model identifier",
        alias="GEMINI_TTS_MODEL"
    )


    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        env_prefix="",
        case_sensitive=False,
        populate_by_name=True,
    )

    def resolve_path(self, path: Optional[Path]) -> Optional[Path]:
        if path is None:
            return None
        return Path(path).expanduser().resolve()
