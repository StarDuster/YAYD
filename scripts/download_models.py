#!/usr/bin/env python3
"""下载 WhisperX 和 XTTS 所需的模型"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# 从环境变量读取 HF_TOKEN
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("❌ 错误: 请在 .env 文件中设置 HF_TOKEN")
    print("   示例: HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    sys.exit(1)

# 模型目录
ALIGN_DIR = str(PROJECT_ROOT / "models" / "ASR" / "whisper" / "align")
DIARIZATION_DIR = str(PROJECT_ROOT / "models" / "ASR" / "whisper" / "diarization")
XTTS_DIR = str(PROJECT_ROOT / "models" / "TTS" / "xtts_v2")

# 创建目录
os.makedirs(ALIGN_DIR, exist_ok=True)
os.makedirs(DIARIZATION_DIR, exist_ok=True)
os.makedirs(XTTS_DIR, exist_ok=True)

# 设置环境变量，确保 transformers 和 pyannote 使用正确的缓存/token
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


def download_xtts_models():
    """下载 XTTS v2 模型"""
    print("\n" + "=" * 60)
    print("正在下载 XTTS v2 模型 (coqui/XTTS-v2)...")
    print("=" * 60)
    print("⚠️  注意: 您需要接受 Coqui Public Model License (CPML)")
    print("   - https://huggingface.co/coqui/XTTS-v2")

    try:
        snapshot_download(
            repo_id="coqui/XTTS-v2",
            local_dir=XTTS_DIR,
            token=HF_TOKEN,
            ignore_patterns=["*.bin", "*onnx*"], # 忽略一些非必要的大文件，只保留 .pth 和 configs
        )
        print(f"✓ XTTS v2 模型下载成功! 路径: {XTTS_DIR}")
    except Exception as e:
        print(f"✗ XTTS v2 模型下载失败: {e}")
        print("\n请确保:")
        print("1. 已登录 Hugging Face 并接受了 coqui/XTTS-v2 的许可协议")
        print("2. HF_TOKEN 有效")


def download_align_models():
    """下载对齐模型 (wav2vec2)"""
    print("\n" + "=" * 60)
    print("正在下载对齐模型 (wav2vec2)...")
    print("=" * 60)
    
    # 临时设置 HF_HOME 以绕过一些默认路径问题
    os.environ["HF_HOME"] = ALIGN_DIR
    os.environ["TRANSFORMERS_CACHE"] = ALIGN_DIR
    
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    
    # 中文对齐模型
    print("\n[1/2] 下载中文对齐模型...")
    model_name_zh = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    try:
        # 使用 cache_dir 强制下载到指定目录
        processor_zh = Wav2Vec2Processor.from_pretrained(model_name_zh, cache_dir=ALIGN_DIR)
        model_zh = Wav2Vec2ForCTC.from_pretrained(model_name_zh, cache_dir=ALIGN_DIR)
        print(f"✓ 中文对齐模型下载成功: {model_name_zh}")
    except Exception as e:
        print(f"✗ 中文对齐模型下载失败: {e}")
    
    # 英文对齐模型
    print("\n[2/2] 下载英文对齐模型...")
    model_name_en = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
    try:
        processor_en = Wav2Vec2Processor.from_pretrained(model_name_en, cache_dir=ALIGN_DIR)
        model_en = Wav2Vec2ForCTC.from_pretrained(model_name_en, cache_dir=ALIGN_DIR)
        print(f"✓ 英文对齐模型下载成功: {model_name_en}")
    except Exception as e:
        print(f"✗ 英文对齐模型下载失败: {e}")


def download_diarization_models():
    """下载说话人分离模型 (pyannote)"""
    print("\n" + "=" * 60)
    print("正在下载说话人分离模型 (pyannote)...")
    print("=" * 60)
    print("\n⚠️  注意: 如果下载失败，请确保已在 Hugging Face 上接受许可协议:")
    print("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("   - https://huggingface.co/pyannote/segmentation-3.0")
    
    os.environ["HF_HOME"] = DIARIZATION_DIR
    os.environ["TRANSFORMERS_CACHE"] = DIARIZATION_DIR
    
    try:
        from pyannote.audio import Pipeline
        
        print("\n正在下载 pyannote/speaker-diarization-3.1...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
            cache_dir=DIARIZATION_DIR
        )
        print("✓ 说话人分离模型下载成功!")
    except Exception as e:
        print(f"✗ 说话人分离模型下载失败: {e}")
        print("\n请确保:")
        print("1. 已登录 Hugging Face 并接受了 pyannote 模型的许可协议")
        print("2. HF_TOKEN 有效")


def main():
    print("=" * 60)
    print("YouDub 模型下载工具")
    print("=" * 60)
    print(f"对齐模型目录: {ALIGN_DIR}")
    print(f"分离模型目录: {DIARIZATION_DIR}")
    print(f"XTTS 模型目录: {XTTS_DIR}")
    print()
    
    # 下载对齐模型
    download_align_models()
    
    # 下载说话人分离模型
    download_diarization_models()
    
    # 下载 XTTS 模型
    download_xtts_models()
    
    print("\n" + "=" * 60)
    print("下载流程结束!")
    print("=" * 60)


if __name__ == "__main__":
    main()
