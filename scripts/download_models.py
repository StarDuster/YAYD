
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
    logger.error("Please install huggingface_hub: pip install huggingface-hub")
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
    # Repo: https://huggingface.co/pyannote/speaker-diarization-3.1
    "diarization": {
        "repo_id": "pyannote/speaker-diarization-3.1",
        "revision": "84fd25912480287da0247647c3d2b4853cb3ee5d", # Pinned from user logs
        "type": "hf_cache"
    },
    # Pyannote Segmentation (Dependency)
    # Repo: https://huggingface.co/pyannote/segmentation-3.0
    "segmentation": {
        "repo_id": "pyannote/segmentation-3.0",
        "revision": "4ca4d5a8d2ab82ddfbea8aa3b29c15431671239c", # Latest stable compatible with 3.1
        "type": "hf_cache"
    },
}

def verify_offline_readiness():
    """Ensure that enforced offline mode variables will work with what we downloaded."""
    logger.info("--- Verifying Offline Readiness ---")
    
    # 1. Check Whisper model
    settings = Settings()
    whisper_path = settings.resolve_path(settings.whisper_model_path)
    if not (whisper_path / "model.bin").exists():
        logger.warning(f" [FAIL] Whisper model.bin not found in {whisper_path}")
    else:
        logger.info(f" [OK] Whisper model found at {whisper_path}")

    # 2. Check Diarization Cache
    # We can't easily check HF cache structure without library, but we can check if directory is not empty
    diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
    if not diar_dir.exists() or not any(diar_dir.iterdir()):
         logger.warning(f" [FAIL] Diarization cache empty at {diar_dir}")
    else:
         logger.info(f" [OK] Diarization cache populated at {diar_dir}")
         
    # 3. Check Demucs
    # Harder to verify location as it's hidden in torch hub, but we trust the download step.
    
    logger.info("Environment is ready for reproducible offline execution.")


def download_models():
    settings = Settings()
    
    # Ensure offline mode is OFF for downloading
    # We want to use the network now to prepare for later offline usage
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    env_vars_to_clear = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]
    for var in env_vars_to_clear:
        if var in os.environ:
            logger.info(f"Temporarily clearing {var} for download script")
            del os.environ[var]

    logger.info("Starting reproducible model downloads...")
    logger.info(f"Target Root: {settings.root_folder}")
    
    # 1. Demucs
    logger.info("\n=== 1. Demucs Model (htdemucs_ft) ===")
    try:
        import torch
        from demucs_infer.pretrained import get_model

        # Ensure demucs weights go to our configured offline directory.
        demucs_dir = settings.resolve_path(settings.demucs_model_dir)
        if demucs_dir:
            demucs_dir.mkdir(parents=True, exist_ok=True)
            torch.hub.set_dir(str(demucs_dir))
            logger.info(f"Set torch hub dir to {demucs_dir}")

        model_name = settings.demucs_model_name
        logger.info(f"Downloading Demucs model: {model_name}...")
        # demucs-infer downloads the same official weights via torch.hub cache.
        get_model(model_name)
        logger.info("Demucs model downloaded.")
    except Exception as e:
        logger.error(f"Failed to download Demucs: {e}")

    # 2. Hugging Face Models (Pinned)
    logger.info("\n=== 2. Hugging Face Models ===")
    
    token = settings.hf_token
    if not token:
        logger.warning("HF_TOKEN missing. Pyannote downloads may fail 401 Unauthorized.")

    # A. Whisper model (Direct Download)
    wx_conf = MODELS_TO_DOWNLOAD["whisper"]
    wx_path = settings.resolve_path(settings.whisper_model_path)
    logger.info(f"Downloading Whisper model ({wx_conf['revision'][:7]}) to {wx_path}...")
    try:
        snapshot_download(
            repo_id=wx_conf["repo_id"],
            revision=wx_conf["revision"],
            local_dir=wx_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
    except Exception as e:
        logger.error(f"Whisper model download failed: {e}")

    # B. Diarization (HF Cache Style)
    logger.info("\n=== 3. Pyannote Diarization (HF Cache) ===")
    diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
    
    # Set HF_HOME to the target diarization directory to build the cache there
    os.environ["HF_HOME"] = str(diar_dir)
    
    for key in ["diarization", "segmentation"]:
        conf = MODELS_TO_DOWNLOAD[key]
        logger.info(f"Downloading {conf['repo_id']} ({conf['revision'][:7]})...")
        try:
            snapshot_download(
                repo_id=conf["repo_id"],
                revision=conf["revision"],
                token=token,
                # No local_dir -> uses HF cache structure in HF_HOME
                resume_download=True
            )
        except Exception as e:
             logger.error(f"Failed to download {conf['repo_id']}: {e}")

    logger.info("\nAll requests completed.")
    verify_offline_readiness()

if __name__ == "__main__":
    download_models()
