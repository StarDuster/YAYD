# 自适应对轴算法 (Adaptive Alignment)

本文档描述 YouDub 中用于将中文 TTS 配音与原视频时间轴对齐的自适应算法。

## 概述

自适应对轴的核心思想是：**不对视频做全局加速，而是逐段调整 TTS 语速和视频片段时长，使配音与原始节奏尽量匹配**。

启用方式：`adaptive_segment_stretch=True`

## 算法流程

### 1. 全局 Baseline 计算

在处理具体段落前，先统计全局信息用于后续回退：

```
输入:
  - translation.json: 所有字幕段落（含原始时间戳）
  - wavs/*.wav: TTS 生成的中文音频

计算:
  total_en_duration = Σ (seg.end - seg.start)        # 英文总时长
  total_zh_duration = Σ trimmed_wav_duration         # 中文 TTS 总时长（裁剪静音后）
  total_en_syllables = Σ count_en_syllables(text)    # 英文总音节数
  total_zh_syllables = Σ count_zh_syllables(translation)  # 中文总音节数

Baseline 模式（SPEECH_RATE_GLOBAL_BASELINE_MODE）:
  - "duration": baseline = total_en_duration / total_zh_duration
  - "syllable": baseline = total_en_syllables / total_zh_syllables  [默认]
  - "min": 取两者较小值（倾向加速）
  - "blend": 几何平均 sqrt(duration * syllable)
```

### 2. 逐段语速对齐

对每个字幕段落执行：

#### 2.1 英文语速估算

```python
# 从原始人声轨提取该段音频
en_audio = audio_vocals[orig_start:orig_end]

# VAD 估算有声时长（多阈值探测，选择最接近目标语速的）
en_voiced_duration = estimate_voiced_duration(en_audio, syllables, target_rate=4.5)

# 计算英文语速（音节/秒）
en_rate = en_syllables / en_voiced_duration
```

#### 2.2 中文语速估算

```python
# 裁剪 TTS 首尾静音
tts_trimmed = librosa.effects.trim(tts_audio, top_db=35)

# VAD 估算有声时长
zh_voiced_duration = sum(voiced_intervals)

# 计算中文语速
zh_rate = zh_syllables / zh_voiced_duration
```

#### 2.3 缩放比率计算

```python
# 应用全局 bias（用于整体加速）
en_rate_adjusted = en_rate / global_bias

# 计算原始 voice_ratio
voice_ratio_raw = zh_rate / en_rate_adjusted

# Clamp 到安全范围 [voice_min, voice_max]
# 默认: voice_min=0.7, voice_max=1.0（只加速不减速）
voice_ratio = clamp(voice_ratio_raw, voice_min, voice_max)
```

#### 2.4 异常值回退

当单段估算不可靠时，回退到全局 baseline：

```python
outlier_conditions = [
    clamp_gap(raw, applied) > 1.7,           # clamp 幅度过大
    en_rate < 1.5 or en_rate > 12.0,         # 英文语速异常
    zh_rate < 1.5 or zh_rate > 12.0,         # 中文语速异常
]

if any(outlier_conditions):
    voice_ratio = global_baseline_ratio
    log_warning("段落 {i}: 回退到全局 baseline")
```

#### 2.5 应用 TSM (Time-Scale Modification)

```python
if abs(voice_ratio - 1.0) > threshold:
    # 使用 audiostretchy 进行保音高变速
    tts_audio = time_stretch(tts_audio, ratio=voice_ratio)
```

### 3. 停顿补偿 (Pause Compensation)

避免配音过快导致节奏不自然：

```python
# 原始时间预算
budget_B = orig_end - orig_start

# 基础停顿（根据标点类型）
post_base = {
    "。！？.!?": 0.25s,  # 句末
    "，、,;；:：": 0.18s,  # 分句
    default: 0.12s
}

# 如果语音比预算短，补偿部分差值
if speech_duration + post_base < budget_B:
    extra = pause_comp_weight * (budget_B - speech_duration - post_base)
    post_silence = clamp(post_base + extra, min=0.1s, max=2.0s)
```

### 4. 输出文件

#### translation_adaptive.json
```json
[
  {
    "start": 0.0,      // 新时间轴起点
    "end": 2.5,        // 新时间轴终点
    "text": "原文",
    "translation": "翻译",
    "speaker": "SPEAKER_00"
  }
]
```

#### adaptive_plan.json
```json
{
  "segments": [
    {
      "kind": "segment",
      "index": 0,
      "src_start": 0.0,           // 原视频裁剪起点
      "src_end": 3.2,             // 原视频裁剪终点（含后续间隙）
      "target_duration": 2.8,     // 目标总时长（语音+静音）
      "speech_target_duration": 2.5,
      "post_silence_duration": 0.3,
      "voice_ratio": 0.85,        // 实际应用的语速缩放
      "voice_ratio_raw": 0.72,    // 原始计算值
      "fallback_used": false,
      "clamp_gap": 1.18
    }
  ]
}
```

### 5. 视频合成

根据 `adaptive_plan.json` 逐段处理：

```
对每个 segment:
  1. 从原视频裁剪 [src_start, src_end]
  2. 使用 setpts 调整为 target_duration
  3. concat 拼接所有片段
```

## 环境变量配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `SPEECH_RATE_ALIGN_ENABLED` | `true` | 启用语速对齐 |
| `SPEECH_RATE_ALIGN_MODE` | `single` | 对齐模式 (single/two_stage) |
| `SPEECH_RATE_VOICE_MIN_RATIO` | `0.7` | 最小语速比（加速上限） |
| `SPEECH_RATE_VOICE_MAX_RATIO` | `1.0` | 最大语速比（不减速） |
| `SPEECH_RATE_GLOBAL_BASELINE_MODE` | `syllable` | 全局 baseline 计算模式 |
| `SPEECH_RATE_GLOBAL_BASELINE_CAP_AT_1` | `true` | baseline 是否限制为 ≤1（只加速） |
| `SPEECH_RATE_PAUSE_COMPENSATION_ENABLED` | `true` | 启用停顿补偿 |
| `SPEECH_RATE_PAUSE_COMP_WEIGHT` | `0.4` | 停顿补偿权重 [0,1] |
| `SPEECH_RATE_FALLBACK_ENABLED` | `true` | 启用异常值回退 |
| `SPEECH_RATE_FALLBACK_CLAMP_GAP` | `1.7` | 触发回退的 clamp gap 阈值 |

## 核心模块

- `src/youdub/speech_rate.py`: 语速计算、音节统计、TSM 应用
- `src/youdub/steps/synthesize_video.py`: 自适应音频合成、视频拼接

## 设计原则

1. **只加速不减速**: TTS 减速听感差，默认 `voice_max=1.0`
2. **保守回退**: 单段估算不可靠时使用全局 baseline
3. **停顿补偿**: 宁可稍慢也不过快，保持自然节奏
4. **逐段拼接**: 避免全局变速导致的累积漂移
