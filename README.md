# YAYD: Yet another YouDub

`YouDub-webui` 是一个基于Gradio的开源项目，它集成了多种AI技术，提供一个可视化的操作界面，用于处理视频的翻译和配音任务。

本项目 fork 并重构自 https://github.com/liuzhao1225/YouDub-webui/blob/master/README.md 。

## 工作原理

本项目实现了一条端到端的视频本地化流水线，将外语视频自动转换为目标语言配音版本。核心处理流程如下：

1.  **视频获取**: 使用 `yt-dlp` 从 YouTube 等平台下载源视频及元数据。
2.  **音频分离**: 通过 **Demucs** (Facebook Research) 将音轨分离为人声与背景音乐/音效。
3.  **语音识别与说话人分离**: 使用 **WhisperX** 对人声进行自动语音识别 (ASR)，生成带时间戳的逐句文本；同时调用 **Pyannote** 进行说话人分离 (Speaker Diarization)，标记每句话的说话人身份。
4.  **文本翻译**: 通过 OpenAI 兼容 API 调用大语言模型，逐句将原文翻译为目标语言。
5.  **语音合成**: 使用 TTS 引擎（支持 **火山引擎 TTS** 或本地 **XTTS v2**）将翻译后的文本合成为语音，并尝试通过音色匹配模拟原说话人声音特征。
6.  **视频合成**: 将合成语音与原背景音轨混音，叠加字幕轨道，输出最终的本地化视频文件。

项目提供 Gradio Web UI，支持全自动流水线执行或分步手动操作。

## 安装指南

项目中含有较古老的依赖，避免使用过高版本的 Python

#### 第1步: 克隆仓库并进入目录

```bash
git clone git@github.com:StarDuster/YAYD.git
cd YAYD
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

- **安装 Demucs:**
  ```bash
  uv pip install git+https://github.com/facebookresearch/demucs
  ```

- **安装 XTTS:**
  ```bash
  uv pip install TTS --no-deps
  ```


#### 第5步: 运行模型下载脚本

本项目为保证环境一致性与离线可用性，提供了一键下载脚本。此脚本会下载所有必需的 AI 模型（Demucs, WhisperX, Pyannote Diarization, XTTS v2）并锁定到验证过的版本。

1. **配置 Hugging Face 权限**:
   - 访问 https://huggingface.co/pyannote/speaker-diarization-3.1 并接受许可协议。
   - 访问 https://huggingface.co/pyannote/segmentation-3.0 并接受许可协议。
   - 在 `.env` 文件中填入你的 `HF_TOKEN`。

2. **运行下载脚本**:
   ```bash
   uv run python scripts/download_models.py
   ```
   
   脚本会自动执行以下操作：
   - 从 PyTorch Hub 下载 **Demucs (htdemucs_ft)** 模型。
   - 从 Hugging Face 下载 **WhisperX (large-v3)** 的 CTranslate2 格式模型。
   - 下载并缓存 **Pyannote Speaker Diarization 3.1** 及其依赖。
   - 下载 **WhisperX Alignment** 对齐模型 (en, zh)。
   - 下载 **XTTS v2** 模型 (用于本地语音合成)。
   
   > 脚本运行完成后，您可以再次运行 `uv run youdub`，此时“模型检查”页面应全部通过，且程序可在无网络环境下（除翻译/TTS API外）运行。

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
