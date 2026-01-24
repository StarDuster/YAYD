# YAYD: Yet another YouDub

`YouDub-webui` 是一个基于Gradio的开源项目，它集成了多种AI技术，提供一个可视化的操作界面，用于处理视频的翻译和配音任务。

本项目 fork 并重构自 https://github.com/liuzhao1225/YouDub-webui/blob/master/README.md 。

## 核心特性

本项目实现了一条模块化、端到端的处理流水线：

1.  **视频获取**: 集成 `yt-dlp`，支持从 YouTube 及其他主流平台下载视频，自动提取元数据。
2.  **人声分离**: 使用 **Demucs (htdemucs_ft)** 模型，高质量分离人声与背景音乐/音效，确保配音后保留原视频的背景氛围。
3.  **语音识别 (ASR) 与 说话人分离 (Diarization)**:
    *   采用 **faster-whisper** (Large-v3) 进行高精度、快速的语音识别。
    *   集成 **pyannote.audio**（使用 `pyannote/speaker-diarization-3.1` pipeline）进行说话人区分，支持多角色识别。
4.  **文本翻译**: 支持 OpenAI 兼容格式的 API (如 OpenAI GPT-4o, DeepSeek, Claude 等)，实现上下文感知的精准翻译。
5.  **语音合成 (TTS)** (通过 `TTS_METHOD` 配置选择):
    *   **ByteDance (火山引擎)**: 高质量云端 TTS，支持 **ICL 2.0 声音克隆**（需配置 API）。
    *   **Qwen3-TTS**: 本地离线 TTS，通过 Worker 子进程运行，无需网络。
    *   **Google Gemini**: 实验性支持 Gemini TTS API。
6.  **视频合成**: 智能音频对齐、变速处理，自动混音背景音轨，生成带字幕的最终视频。

## 系统要求

*   **操作系统**: Linux (推荐 Ubuntu 22.04+) 或 WSL2 (Windows Subsystem for Linux)。
*   **Python**: 3.10 或 3.11。
*   **GPU**: 推荐 NVIDIA GPU，显存 8GB+ (如需运行本地 LLM 或 TTS 可能需要更多)。
*   **CUDA**: 推荐版本 12.8（主环境默认使用 PyTorch `cu128` 轮子）。
*   **FFmpeg**: 必须安装并配置在系统 PATH 中。

## 安装指南

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 进行高效的依赖管理。

### 1. 克隆仓库

```bash
git clone https://github.com/StarDuster/YAYD.git
cd YAYD
```

### 2. 环境配置与安装

确保已安装 `uv`，然后同步依赖：

```bash
# 创建虚拟环境并安装依赖
uv sync
```

或者使用传统的 pip 安装方式（建议在虚拟环境中）：

```bash
pip install -e .
```

### 3. 安装外部依赖

**FFmpeg** (必需):
```bash
sudo apt update && sudo apt install ffmpeg
```

**yt-dlp 插件支持** (可选，解决部分 YouTube 下载限速问题):
需安装 Deno 运行时以支持 yt-dlp 的某些解密插件：
```bash
curl -fsSL https://deno.land/install.sh | sh
# 确保 ~/.deno/bin 在 PATH 中
```

**PO Token 服务** (可选，解决 YouTube 403 Forbidden 错误):
如果下载 YouTube 视频时遇到 `HTTP Error 403: Forbidden`，可能需要运行 PO Token 服务来绕过 bot 检查：
```bash
docker run --name bgutil-provider -d -p 4416:4416 brainicism/bgutil-ytdlp-pot-provider
```
服务启动后 yt-dlp 会自动从 `127.0.0.1:4416` 获取 token。更多信息参考 [bgutil-ytdlp-pot-provider](https://github.com/Brainicism/bgutil-ytdlp-pot-provider)。

### 4. 下载模型

项目提供了一键脚本下载所需的离线模型（Whisper, Demucs, Pyannote 等）。

**注意**：下载 `pyannote` 模型需要 Hugging Face 访问令牌，并同意相关模型的使用协议。
1.  访问 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) 和 [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) 接受用户协议。
2.  在环境变量或 `.env` 中设置 `HF_TOKEN`。

运行下载脚本：
```bash
uv run python scripts/download_models.py
```
此脚本将下载并验证以下模型：
*   `Systran/faster-whisper-large-v3`
*   `pyannote/speaker-diarization-3.1`
*   `demucs (htdemucs_ft)`

## 配置指南

复制 `.env.example` 为 `.env` 并填入关键配置：

```bash
cp .env.example .env
```

### 核心配置项

```ini
# --- 基础配置 ---
# Hugging Face Token (用于下载 Pyannote 模型)
HF_TOKEN=hf_xxxxxxxxxxxxxxx

# OpenAI 兼容 API (用于翻译)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxx
OPENAI_API_BASE=https://api.openai.com/v1
MODEL_NAME=gpt-4o

# --- TTS 配置 ---
# 选择 TTS 引擎: bytedance / gemini / qwen
TTS_METHOD=bytedance

# 1. ByteDance (火山引擎) - 推荐，支持声音克隆
BYTEDANCE_APPID=xxxxxxxxxx
BYTEDANCE_ACCESS_TOKEN=xxxxxxxxxx
# 可选：指定声音克隆的 speaker IDs (逗号分隔)
VOLCANO_CLONE_SPEAKER_IDS=

# 2. Gemini TTS
GEMINI_API_KEY=AIzaSyxxxxxxxxxx
GEMINI_TTS_MODEL=gemini-2.5-flash-preview-tts
GEMINI_TTS_VOICE=Kore

# 3. Qwen3-TTS (本地离线模式)
QWEN_TTS_MODEL_PATH=models/TTS/Qwen3-TTS-12Hz-1.7B-Base
# 默认使用主工程 .venv，也可指定独立环境
QWEN_TTS_PYTHON=.venv/bin/python
```

## 使用说明

启动 Web UI 服务：

```bash
uv run youdub
```

服务启动后，在浏览器访问显示的地址（默认 `http://127.0.0.1:7860`）。

### 操作流程

1.  **模型检查**: 首次运行建议在“模型检查”标签页确认所有模型加载正常。
2.  **全自动模式**:
    *   输入视频 URL。
    *   选择目标语言。
    *   点击“提交”，系统将自动执行下载、分离、识别、翻译、合成全流程。
3.  **分步模式**:
    *   可在各标签页单独执行特定步骤（如仅下载、仅翻译、仅 TTS），便于调试或人工修正中间结果（如修正 `translation.json`）。

## 高级功能：Qwen3-TTS (本地离线 TTS)

若需使用本地 Qwen3-TTS 模型进行语音合成：

1.  **环境准备**: 主工程 `.venv` 已包含 `qwen-tts` 依赖，可直接使用。如需隔离环境，可另建 venv 并安装 `qwen-tts`。
2.  **下载模型**: 从 Hugging Face 下载 [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) 到 `models/TTS/` 目录。
3.  **配置 `.env`**:
    ```ini
    TTS_METHOD=qwen
    QWEN_TTS_MODEL_PATH=models/TTS/Qwen3-TTS-12Hz-1.7B-Base
    QWEN_TTS_PYTHON=.venv/bin/python
    ```
4.  程序运行时会自动拉起 `scripts/qwen_tts_worker.py` 子进程进行推理。

## 许可证

MIT License. 详见 `pyproject.toml`。
