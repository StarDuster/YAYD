# YAYD: Yet another YouDub

`YouDub-webui` 是一个基于Gradio的开源项目，它集成了多种AI技术，提供一个可视化的操作界面，用于处理视频的翻译和配音任务。

本项目 fork 并重构自 https://github.com/liuzhao1225/YouDub-webui/blob/master/README.md ，替换了缺少维护的组件（如 bili-toolman），解决在新版本 CUDA 环境的依赖冲突，并引入 Qwen3-TTS 作为本地 TTS 方案。

## 核心特性

本项目实现了一条模块化、端到端的处理流水线：

1.  **视频获取**: 集成 `yt-dlp`，支持从 YouTube 及其他主流平台下载视频，自动提取元数据。
2.  **人声分离**: 使用 `demucs-infer` (htdemucs_ft)，支持流式处理长音频避免显存溢出。
3.  **语音识别 (ASR) 与 说话人分离 (Diarization)**:
    *   采用 `faster-whisper` (Large-v3) + `ctranslate2` 进行高精度语音识别。
    *   集成 `pyannote.audio` (兼容 v3.1-v4) 进行说话人区分，支持多角色识别。
4.  **文本翻译**: 支持 OpenAI 兼容 API 和并发翻译。
5.  **语音合成 (TTS)** (通过 `TTS_METHOD` 配置):
    *   ByteDance (火山引擎豆包语音大模型)
    *   Google Gemini TTS
    *   Qwen3-TTS (本地模型，Qwen3 TTS 实测效果已足够好，推荐使用)
6.  **视频合成**: 音频对齐、变速处理，自动混音背景音轨，生成带字幕的最终视频。
    *   **加速倍率**: 英语和中文的平均每分钟字/词数存在差别，默认使用 1.2 倍加速以避免大范围无声片段。
    *   **自适应时长**: 可在 TTS 阶段启用"按段自适应拉伸语音"，根据每段原始时长动态调整 TTS 语速，减少无声间隙（启用后加速倍率设置无效）。

## 系统要求

*   **操作系统**: 本项目只在 Linux（含 WSL2）上进行适配和测试
*   **Python**: >=3.10
*   **CUDA**: 可选（由于 Qwen3-TTS 依赖，建议 CUDA 12.8）。
*   **FFmpeg**: 必须安装并配置在系统 PATH 中。

## 安装指南

推荐使用 [uv](https://github.com/astral-sh/uv) 进行高效的依赖管理。

### 1. 克隆仓库

```bash
git clone https://github.com/StarDuster/YAYD.git
cd YAYD
```

### 2. 环境配置与安装

项目将依赖拆分为多个 extras：

| Extra | 说明 | 包含内容 |
|-------|------|----------|
| `cpu` | CPU 运行时 | `onnxruntime` (CPU 版) |
| `gpu` | GPU 加速栈 (Linux) | `onnxruntime-gpu`, `nvidia-cudnn-cu12`, PyTorch cu128 轮子 |
| `dev` | 开发工具 | `pytest`, `black`, `ruff` |

> **注意**: `onnxruntime` 和 `onnxruntime-gpu` 不能共存。选择 `--extra cpu` 或 `--extra gpu` 其中之一。

#### 安装命令

```bash
# CPU 环境（默认，适合无 NVIDIA GPU 或仅测试）
uv sync --extra cpu

# GPU 环境（推荐生产使用，Linux + CUDA 12.8）
uv sync --extra gpu

# GPU + 开发工具（开发/测试）
uv sync --extra gpu --extra dev
```

#### PyTorch GPU/CPU 自动选择

`torch` 和 `torchaudio` 会根据 extras 自动选择安装源：
- `--extra gpu` → 从 `pytorch-cu128` index 安装 CUDA 12.8 版本
- 无 `--extra gpu` → 从 `pytorch-cpu` index 安装 CPU 版本


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
此脚本将下载：
*   `Systran/faster-whisper-large-v3` (CTranslate2 格式)
*   `pyannote/speaker-diarization-3.1` + `segmentation-3.0`
*   `demucs htdemucs_ft` (via torch.hub)

**Qwen3-TTS 模型** (仅当 `TTS_METHOD=qwen` 时需要)：

需手动从 Hugging Face 下载 [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) 到 `models/TTS/` 目录：
```bash
# 使用 huggingface-cli（推荐）
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir models/TTS/Qwen3-TTS-12Hz-1.7B-Base

# 或使用 git clone（需安装 git-lfs）
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base models/TTS/Qwen3-TTS-12Hz-1.7B-Base
```

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

# --- 翻译配置 (可选) ---
# 翻译策略: history (串行带上下文) / guide_parallel (默认，先生成翻译指南，再并发翻译)
TRANSLATION_STRATEGY=parallel
# guide_parallel 模式下的并发数
TRANSLATION_MAX_CONCURRENCY=4

# --- yt-dlp 下载认证 (可选) ---
# 需要登录才能访问的内容（私有播放列表、年龄限制视频等）可配置 cookies。
# 优先级：YTDLP_COOKIE_PATH > YTDLP_COOKIES_FROM_BROWSER
#
# 方案 A（推荐）：使用浏览器导出的 cookies.txt（Netscape 格式）
YTDLP_COOKIE_PATH=
#
# 方案 B：直接从浏览器读取 cookies（在部分环境可能不可用）
YTDLP_COOKIES_FROM_BROWSER=
YTDLP_COOKIES_FROM_BROWSER_PROFILE=

# --- B站上传配置 (可选) ---
# cookie 文件路径（由 biliup login 生成）
BILI_COOKIE_PATH=bili_cookies.json
# 上传代理（可选，如 socks5h://127.0.0.1:1080）
BILI_PROXY=
# 首选上传线路：bda / bda2 / tx / txa / bldsa（留空则自动选择）
BILI_UPLOAD_CDN=
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

### yt-dlp 使用 Cookie（可选）

当下载 YouTube 遇到需要登录/年龄限制/私有内容等情况，可配置 cookies，使用 cookie 也是回避 Youtube 限流最可靠的方法：

1.  导出 cookies.txt（Netscape 格式）并设置：

```ini
YTDLP_COOKIE_PATH=yt_cookies.txt
```

2.  或直接从浏览器读取（不建议作为首选，兼容性依赖环境）：

```ini
YTDLP_COOKIES_FROM_BROWSER=chrome
YTDLP_COOKIES_FROM_BROWSER_PROFILE=Default
```

> 提醒：cookies 属于敏感信息，不要提交到仓库；本项目已在 `.gitignore` 中忽略常见的 cookies 文件名。

### B 站上传

本项目使用 [biliup](https://github.com/biliup/biliup) 进行 B 站视频上传。

#### 1. 登录获取 Cookie

首次使用前需要登录 B 站账号生成 `cookies.json`：

```bash
# 扫码登录（推荐）
uv run biliup login

# 登录成功后会在当前目录生成 cookies.json
```

#### 2. 配置 Cookie 路径

将生成的 `cookies.json` 放到项目根目录，或在 `.env` 中指定路径：

```ini
BILI_COOKIE_PATH=cookies.json
```

#### 3. 上传视频

在 Web UI 的"上传B站"标签页，输入包含已处理视频的文件夹路径，点击提交即可批量上传。

**注意事项**：
- 每个视频文件夹需包含 `video.mp4`、`summary.json`、`download.info.json`
- 上传成功后会生成 `bilibili.json` 标记，避免重复上传
- 如遇 Cookie 过期，重新运行 `uv run biliup login` 刷新
- 如需代理，设置 `BILI_PROXY=socks5h://127.0.0.1:1080`

## 许可证

MIT License. 详见 [LICENSE](./LICENSE) 文件。
