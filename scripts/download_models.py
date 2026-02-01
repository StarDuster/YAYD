
import os
import sys
from pathlib import Path
from loguru import logger

# Add src to path to import settings
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from youdub.config import Settings

try:
    from huggingface_hub import snapshot_download
except ImportError:
    logger.error("缺少 huggingface_hub：请运行 `pip install huggingface-hub`（或 `uv sync`）")
    sys.exit(1)

# --- REPRODUCIBILITY CONFIG ---
# Pin model revisions (commit hashes) to strict versions to ensure environment reproducibility.
# If revision is None, it uses the latest 'main' branch (not reproducible over time).
MODELS_TO_DOWNLOAD = {
    # Whisper (faster-whisper) CTranslate2 Model
    # Repo: https://huggingface.co/Systran/faster-whisper-large-v3
    "whisper": {
        "repo_id": "Systran/faster-whisper-large-v3",
        "revision": "edc79942a0352e00c3b03657b4943f293cf0f1d0", # Pinned to a known good state
        "type": "direct_download"
    },
    # Pyannote Diarization (Pipeline)
    # Repo: https://huggingface.co/pyannote/speaker-diarization-community-1
    "diarization": {
        "repo_id": "pyannote/speaker-diarization-community-1",
        # NOTE: community-1 is a gated model (needs accepting conditions + HF token).
        # Pin to a recent known revision for reproducibility.
        "revision": "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
        "type": "hf_cache"
    },
}

def verify_offline_readiness():
    """Ensure that enforced offline mode variables will work with what we downloaded."""
    logger.info("--- 检查离线模型就绪情况 ---")
    
    # 1. Check Whisper model
    settings = Settings()
    whisper_path = settings.resolve_path(settings.whisper_model_path)
    if not (whisper_path / "model.bin").exists():
        logger.warning(f" [失败] 未找到 Whisper model.bin: {whisper_path}")
    else:
        logger.info(f" [成功] Whisper 模型已就绪: {whisper_path}")

    # 2. Check Diarization Cache
    # We can't easily check HF cache structure without library, but we can check if directory is not empty
    diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
    if not diar_dir.exists() or not any(diar_dir.iterdir()):
         logger.warning(f" [失败] 说话人分离缓存为空: {diar_dir}")
    else:
         logger.info(f" [成功] 说话人分离缓存已就绪: {diar_dir}")
         
    # 3. Check Demucs
    # Harder to verify location as it's hidden in torch hub, but we trust the download step.
    
    logger.info("离线环境检查完成。")


def download_models():
    settings = Settings()
    
    # Ensure offline mode is OFF for downloading
    # We want to use the network now to prepare for later offline usage
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    env_vars_to_clear = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]
    for var in env_vars_to_clear:
        if var in os.environ:
            logger.info(f"临时清除环境变量: {var}（用于下载）")
            del os.environ[var]

    logger.info("开始下载离线模型...")
    logger.info(f"目标目录: {settings.root_folder}")
    
    # 1. Demucs
    logger.info("\n=== 1. Demucs 模型 (htdemucs_ft) ===")
    try:
        import torch
        from demucs_infer.pretrained import get_model

        # Ensure demucs weights go to our configured offline directory.
        demucs_dir = settings.resolve_path(settings.demucs_model_dir)
        if demucs_dir:
            demucs_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(demucs_dir))
            logger.info(f"设置 torch hub 目录: {demucs_dir}")

        model_name = settings.demucs_model_name
        logger.info(f"下载 Demucs 模型: {model_name} ...")
        # demucs-infer downloads the same official weights via torch.hub cache.
        get_model(model_name)
        logger.info("Demucs 模型下载完成。")
    except Exception as e:
        logger.error(f"Demucs 下载失败: {e}")

    # 2. Hugging Face Models (Pinned)
    logger.info("\n=== 2. Hugging Face 模型 ===")
    
    token = settings.hf_token
    if not token:
        logger.warning("缺少 HF_TOKEN：下载 pyannote 可能会 401（需先同意协议并设置 token）。")

    # A. Whisper model (Direct Download)
    wx_conf = MODELS_TO_DOWNLOAD["whisper"]
    wx_path = settings.resolve_path(settings.whisper_model_path)
    logger.info(f"下载 Whisper 模型 ({wx_conf['revision'][:7]}) -> {wx_path} ...")
    try:
        snapshot_download(
            repo_id=wx_conf["repo_id"],
            revision=wx_conf["revision"],
            local_dir=wx_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
    except Exception as e:
        logger.error(f"Whisper 下载失败: {e}")

    # B. Diarization (HF Cache Style)
    logger.info("\n=== 3. Pyannote 说话人分离（community-1 / HF Cache） ===")
    diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
    diar_dir.mkdir(parents=True, exist_ok=True)
    
    conf = MODELS_TO_DOWNLOAD["diarization"]
    logger.info(f"下载 {conf['repo_id']} ({conf['revision'][:7]}) ...")
    try:
        snapshot_download(
            repo_id=conf["repo_id"],
            revision=conf["revision"],
            token=token,
            # Cache under WHISPER_DIARIZATION_MODEL_DIR/models--ORG--REPO/...
            cache_dir=str(diar_dir),
            resume_download=True,
        )
    except Exception as e:
        logger.error(f"下载失败 {conf['repo_id']}: {e}")

    logger.info("\n下载流程已结束。")
    verify_offline_readiness()

if __name__ == "__main__":
    download_models()
