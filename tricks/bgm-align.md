# BGM 对齐策略文档

本文档详细说明了 YouDub-webui 项目中背景音乐（BGM）的对齐与处理机制。核心逻辑位于 `src/youdub/steps/synthesize_video.py`。

## 1. 音频分离 (Separation)

在处理 BGM 之前，项目首先使用 **Demucs** 模型将原始视频的音频分离为两条独立的音轨：

- **人声 (`audio_vocals.wav`)**：用于后续参考音量、音色克隆以及说话人识别。
- **伴奏/BGM (`audio_instruments.wav`)**：包含背景音乐和环境音效，是后续对齐操作的目标对象。

## 2. 对齐策略 (Alignment)

对齐的方式取决于视频合成时选择的模式（`adaptive_segment_stretch` 参数）。

### 2.1 自适应模式 (Adaptive Mode)

这是高级对齐模式，旨在确保画面口型、视频节奏与翻译后的 TTS 语音长度完美匹配。

- **适用场景**：开启 `adaptive_segment_stretch=True` 时。
- **核心原理**：BGM 会跟随视频的每一句台词进行“切片”和“变速”，以保持与画面同步。
- **详细流程**：
    1. **生成计划**：系统根据 TTS 语音时长与原视频片段时长的比例，生成 `adaptive_plan.json`。该文件定义了每一个语音片段（Speech）和停顿片段（Pause）的**原始时间段**（Source Start/End）和**目标时长**（Target Duration）。
    2. **逐段处理**：代码（`_render_adaptive_instruments`）会读取 `adaptive_plan.json`，并从 `audio_instruments.wav` 中截取对应的原始 BGM 片段。
    3. **变速对齐**：使用 FFmpeg 的 `atempo` 滤镜对每一小段 BGM 进行变速处理（Time Stretch）。
        - 如果翻译后的语音比原话长，BGM 会被拉伸（变慢）。
        - 如果翻译后的语音比原话短，BGM 会被压缩（变快）。
    4. **无缝拼接**：将所有处理后的 BGM 片段按顺序重新拼接。
- **优势**：
    - 确保背景音乐的节奏变化与视频画面的变速完全同步。
    - 避免出现“画面变慢了但音乐还在原速跑”导致的音画错位。

## 3. 混合与音量平衡 (Mixing)

完成对齐后，系统会将处理好的 BGM 与 TTS 语音进行混合，生成最终用于视频封装的音频文件 `audio_combined.wav`。

- **音量校准**：
    - 系统会分析原视频中“人声”与“BGM”的相对响度比例（基于 RMS）。
    - 自动调整 TTS 的音量，使其在融入背景音乐时听起来自然，保持原视频的混音平衡。
    - 避免出现 BGM 盖过人声或人声过于突兀的情况。
- **最终合成**：将调整好音量的 TTS 轨道与对齐后的 BGM 轨道进行叠加混合。
