# YAYD: Yet another YouDub

`YouDub-webui` 是一个基于Gradio的开源项目，它集成了多种AI技术，提供一个可视化的操作界面，用于处理视频的翻译和配音任务。

本项目 fork 并重构自 https://github.com/liuzhao1225/YouDub-webui/blob/master/README.md 。

## 主要功能

- **双工作流模式**:
    - **自动流水线**: 提供视频URL后，可自动执行从下载到合成的完整流程。
    - **分步执行**: 提供独立的UI选项卡，用于分别执行下载、人声分离、语音识别、翻译、语音合成和视频合成等步骤。
- **AI模型支持**:
    - **语音识别**: 使用 `WhisperX` 进行语音转录，支持时间戳和说话人识别。
    - **机器翻译**: 通过 `OpenAI` API 调用语言模型进行文本翻译。
    - **语音合成 (TTS)**: 支持 `XTTS` (用于声音克隆) 和 字节跳动TTS。
- **项目配置**: 通过 `.env` 环境文件对API密钥、模型路径等参数进行配置。
- **模型管理**: 程序包含模型检查功能，并在首次使用时自动下载运行所需的模型。

## 安装指南

请遵循以下步骤完成安装。项目推荐使用 Python 3.10 或更高版本。

#### 第1步: 克隆仓库并进入目录

```bash
git clone https://github.com/liuzhao1225/YouDub-webui.git
cd YouDub-webui
```

#### 第2步: 创建虚拟环境

```bash
# (推荐) 使用 uv
uv venv
```
激活虚拟环境:
```bash
source .venv/bin/activate
```

#### 第3步: 安装 yt-dlp 所需运行环境（Deno / 插件）

由于 https://github.com/yt-dlp/yt-dlp/issues/15012 所提及的限制，下载 YouTube 等站点时，`yt-dlp` 可能需要可用的 JavaScript 运行时（推荐 `deno`）以及 `yt-dlp-ejs` 等插件支持。

```bash
curl -fsSL https://deno.land/install.sh | sh
deno --version
```

#### 第4步: 安装主程序和依赖

```bash
# (推荐) 使用 uv
uv pip install -e .
```

#### 第5步: (可选) 安装额外功能

如果您需要 **人声分离(Demucs)** 或 **声音克隆(XTTS)** 功能，请分别安装它们：

- **安装 Demucs:**
  ```bash
  uv pip install git+https://github.com/facebookresearch/demucs
  ```

- **安装 XTTS:**
  ```bash
  uv pip install TTS --no-deps
  ```

#### 第6步: 下载 WhisperX 模型

WhisperX 需要额外的对齐模型和说话人分离模型。

> ⚠️ **重要**: 说话人分离模型 (pyannote) 需要您在 Hugging Face 上**接受许可协议**后才能下载。

1. **接受 pyannote 模型许可协议**:
   - 访问 https://huggingface.co/pyannote/speaker-diarization-3.1 并点击 "Agree"
   - 访问 https://huggingface.co/pyannote/segmentation-3.0 并点击 "Agree"

2. **获取 Hugging Face Token**:
   - 访问 https://huggingface.co/settings/tokens
   - 创建一个 Access Token 并复制

3. **配置 HF_TOKEN**:
   在 `.env` 文件中添加:
   ```env
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

4. **运行下载脚本**:
   ```bash
   uv run python scripts/download_models.py
   ```
   
   此脚本将自动下载:
   - 中文对齐模型 (wav2vec2-large-xlsr-53-chinese-zh-cn)
   - 英文对齐模型 (wav2vec2-large-xlsr-53-english)
   - 说话人分离模型 (pyannote/speaker-diarization-3.1)

## 配置

在运行程序之前，需要配置必要的API密钥。

1.  **创建 `.env` 文件**:
    复制 `.env.example` 文件并重命名为 `.env`。
    ```bash
    cp .env.example .env
    ```

2.  **编辑 `.env` 文件**:
    打开 `.env` 文件并填入以下信息。

    **核心配置 (建议填写):**
    ```env
    # 用于翻译
    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

    # 用于从Hugging Face下载说话人识别模型
    HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```

    **可选功能配置:**
    ```env
    # 如果你想使用字节跳动TTS
    BYTEDANCE_APPID=xxxxxxxxxx
    BYTEDANCE_ACCESS_TOKEN=xxxxxxxxxx

    # 如果你想自动上传到Bilibili
    BILI_SESSDATA=xxxxxxxxxx
    BILI_BILI_JCT=xxxxxxxxxx
    ```

    **高级配置 (模型本地路径):**
    为避免程序自动下载，可预先下载模型并在此处指定本地路径。
    ```env
    WHISPERX_MODEL_PATH=models/ASR/whisper
    XTTS_MODEL_PATH=models/TTS/xtts_v2
    # ... 其他模型路径
    ```

## 如何使用

1.  **启动Web服务器**:
    在您的终端中（确保虚拟环境已激活），运行以下命令：
    ```bash
    youdub
    ```

2.  **访问Web界面**:
    打开浏览器，访问终端中显示的URL (通常是 `http://1.2.3.4:7860`)。

3.  **开始使用**:

    - **模型检查**: 访问 **“模型检查”** 标签页，可查看模型是否已准备就绪。

    - **自动模式**:
        - 切换到 **“全自动”** 标签页。
        - 输入一个视频或播放列表的 **URL**。
        - 调整下方的配置参数。
        - 点击 **“提交”** 按钮，流水线将开始运行。

    - **手动模式**:
        - 按照标签页的顺序，从 **“下载视频”** 开始。
        - 在每个标签页中，指定要处理的文件夹。
        - 完成一步后，进入下一个标签页继续处理。

## 许可协议

本项目遵循 `Apache-2.0` 许可协议。使用本工具时，请务必尊重并遵守原始内容的版权，以及相关的法律法规。
