# YouDub Tricks 技巧集

本文档记录了本项目中“带计算/规则”的核心算法与工程策略（字幕字号/换行、时间轴、对齐、兜底等）。
注意：为了避免文档与实现漂移，优先写 **关键规则/公式 + 对应函数名**，不要长期依赖拷贝出来的旧代码片段。

## 专题文档

| 文档 | 说明 |
|------|------|
| [adaptive-alignment.md](./adaptive-alignment.md) | 自适应对轴算法详解 |
| [bgm-align.md](./bgm-align.md) | BGM 对齐策略 |
| [diarization-tuning.md](./diarization-tuning.md) | 说话人分离调参指南 |
| [qwen-tts.md](./qwen-tts.md) | Qwen TTS 防劣化与优化机制 |
| [srt-render.md](./srt-render.md) | 字幕渲染策略 |
| [translate.md](./translate.md) | 翻译策略文档 |
| [youtube-srt-align.md](./youtube-srt-align.md) | YouTube SRT 说话人对齐 |

---

## 一、双语字幕对轴

### 1. 自适应拉伸 (Adaptive Stretch)

**背景**: 早期的“非自适应全局加速”（例如 1.2x，现已移除）会导致 TTS 语音与视频画面不同步，口型对不上。

**解决方案**: 逐段拉伸视频以匹配TTS音频时长。

```python
# synthesize_video.py - 核心思路
# 1. 不使用相位声码器 time_stretch（避免"回音/空洞"感）；可选用 audiostretchy 做逐段 TSM 语速对齐
# 2. 裁剪TTS首尾静音，按自然停顿插入间隙
# 3. 生成 adaptive_plan.json 记录每段的 src_start/src_end/target_duration
# 4. 用FFmpeg filter_complex 逐段 trim+setpts 拉伸原视频
```

关键文件:
- `translation_adaptive.json` - 对轴后的字幕时间轴
- `adaptive_plan.json` - 视频合成计划（speech + pause 段）

补充要点（实现见 `src/youdub/steps/synthesize_video.py::_ensure_audio_combined` + 合成阶段滤镜生成）：
- 音频侧：自适应模式下不使用相位声码器 `time_stretch`；默认裁剪 TTS 首尾静音，并在启用语速对齐（`SPEECH_RATE_ALIGN_ENABLED`）时对每段 TTS 做可控的 TSM（`audiostretchy`，默认只加速不减速，异常值可回退全局 baseline）。
- 停顿时长：基础停顿按译文末尾标点估计（句号/问号/感叹号更长，逗号/分号/冒号次之），并可按原始时间预算做停顿补偿（`SPEECH_RATE_PAUSE_COMPENSATION_ENABLED` / `SPEECH_RATE_PAUSE_COMP_WEIGHT`），最终再 clamp 到最小/最大停顿范围。
- 视频侧：按 `adaptive_plan.json` 逐段 `trim + setpts`（倍率 `factor = target_duration / src_duration`），然后 `concat`；避免强制固定 FPS（使用 `-vsync vfr`）。

### 2. 字幕样式参数（字号/描边/边距/换行阈值）

**问题**: 固定的换行字符数在不同分辨率/宽高比下表现不一致（竖屏视频字幕过长，横屏过短）。

**解决方案**: 根据输出分辨率动态计算：字号、描边、底边距，以及 CJK/英文换行阈值（实现见 `src/youdub/steps/synthesize_video.py::_calc_subtitle_style_params` / `_calc_subtitle_wrap_chars`）。

```python
def _calc_subtitle_style_params(width, height, *, en_font_scale=0.75):
    # base_dim 取短边：横屏取高度，竖屏取宽度
    base_dim = min(width, height)

    # 经验比例：横屏略大(4.5%)，竖屏略小(4.0%)
    font_scale = 0.045 if (width >= height) else 0.040
    font_size = clamp(round(base_dim * font_scale), 18, 120)

    outline = max(1, round(font_size * 0.08))     # 描边约为字号的 8%
    margin_v = max(10, round(font_size * 0.20))   # 底边距约为字号的 20%

    # 换行阈值：按宽度扣掉左右安全边距后，再按“平均字形宽度”估算能容纳的字符数
    margin_x = max(10, round(width * 0.04))       # 左右各 ~4%，保守避免裁切
    safe_w = width - 2 * margin_x
    max_chars_zh = max(1, int(safe_w / (font_size * 0.90)))
    en_fs = max(1, round(font_size * en_font_scale))
    max_chars_en = max(1, int(safe_w / (en_fs * 0.65)))
```

### 3. 双语字幕原文智能排版

**问题**: 直接显示整段英文原文会铺满屏幕，影响可读性。

**解决方案**: `_bilingual_source_text()` 完整保留原文，智能换行（详见 [srt-render.md](./srt-render.md)）。

- 短句：单行显示
- 多句：按句号换行
- 长句：在逗号/分号处智能换行，保持多行平衡

### 4. 双语字幕原文补齐（旧文件兼容）

**问题**: 旧版 `translation.json` 可能缺少 `text` 字段（原文）。

**解决方案**: 从 `transcript.json` 按时间重叠自动回填。

```python
def _ensure_bilingual_source_text(folder, translation, *, adaptive_segment_stretch):
    # 读取 transcript.json
    # 按时间重叠找到最匹配的原文
    # 回写到 translation.json 持久化
```

### 5. 原文到译文数量映射（避免重复长段落）

**问题**: 译文按句切分后数量可能多于原文句子数，导致重复显示原文。

**解决方案**: 智能映射原文句子到译文数量。

```python
def _map_source_text_to_translation_count(text, target_count):
    # 1. 先按句号/问号/感叹号严格切分
    # 2. 如果不够，按逗号/分号/冒号宽松切分
    # 3. 还不够，按单词均分
    # 目标: 产出恰好 target_count 个原文块，避免重复
```

### 6. 翻译分句后的时间轴再分配（对齐字幕可读性）

实现见 `src/youdub/steps/translate.py::split_sentences`：
- 将每段译文按句切分（`split_text_into_sentences`）
- 将原文映射到相同数量（`_map_source_text_to_translation_count`）
- 按每句字符数分配时间：`duration_per_char = (end - start) / sum(len(sentence))`
- **强制最后一句的 end 等于原段 end**，避免累计误差导致时间轴漂移

### 7. 时间戳格式化（SRT/ASS）

实现见 `src/youdub/steps/synthesize_video.py`：
- `format_timestamp(seconds)`：SRT 格式 `HH:MM:SS,mmm`
- `format_timestamp_ass(seconds)`：ASS 格式 `H:MM:SS.cc`（厘秒，做四舍五入并处理 `cs==100` 进位）

---

## 二、TTS语音生成兜底

### 1. TTS输出时长校验（防止EOS预测失败）

**问题**: TTS模型（尤其是Qwen3-TTS）有时无法正确采样到结束符（EOS），导致持续输出直到触及token上限，产生异常长的音频（如10秒原文生成170秒音频）。

**解决方案**: 双层检测机制，任一触发即视为失败并重试。

```python
# 层1: 相对上限 - 基于原始片段时长
def _tts_duration_guard_params():
    ratio = 3.0      # 最大倍数
    extra = 8.0      # 额外秒数
    abs_cap = 180.0  # 绝对上限
    retries = 3

def _tts_segment_allowed_max_seconds(seg_dur, ratio, extra, abs_cap):
    # allowed = max(seg_dur * ratio, seg_dur + extra)
    # 例: 10秒原文 -> 允许最多 max(30, 18) = 30秒

# 层2: 绝对上限 - 检测是否命中Qwen的max_new_tokens
_QWEN_TTS_TOKEN_HZ = 12.0  # Tokenizer-12Hz
_QWEN_TTS_MAX_NEW_TOKENS_DEFAULT = 2048

def _qwen_tts_is_degenerate_hit_cap(*, wav_dur):
    cap = float(_qwen_tts_max_new_tokens()) / 12.0  # ≈ 170秒
    tol = 2.0  # 容差
    return wav_dur >= cap - tol  # 接近上限 = 退化循环
```

### 2. TTS提示词规范化

**问题**: Python代码token（如 `os.path.join`）和 `snake_case` 标识符会被TTS错误发音。

**解决方案**: 在送入TTS前转换为可读形式。

```python
def _tts_prompt_normalize_python_tokens(text):
    # matplotlib.pyplot -> matplotlib dot pyplot
    # base_dir -> base dir
    # 注意: 不影响小数如 3.14
```

### 3. 重试时逐步清理文本

**问题**: 含有markdown/代码块的文本可能导致TTS失败。

**解决方案**: 重试时逐步激进地清理文本。

```python
def _tts_text_for_attempt(raw_text, attempt):
    if attempt > 0:
        # 移除代码块 ```...``` 和反引号
        t = re.sub(r"```[\s\S]*?```", " ", t)
        t = t.replace("`", "")
    
    if attempt >= 2:
        # 只保留 CJK、ASCII字母数字、基本标点
        t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s，。！？,.!?:：]", " ", t)
```

### 4. 说话人参考音频裁剪

**问题**: 历史遗留的超长 SPEAKER/*.wav 会导致voice cloning显存/时间爆炸。

**解决方案**: 参考音频写入/复用时统一做“时长上限 + 防爆音处理”（默认15秒，可配），避免超长与爆音干扰克隆效果（实现见 `src/youdub/steps/synthesize_speech.py::_ensure_wav_max_duration` + `src/youdub/utils.py::prepare_speaker_ref_audio`）。

```python
def _ensure_wav_max_duration(path: str, max_seconds: float, sample_rate: int = 24000) -> None:
    # 1) 裁剪到 max_seconds（默认来自 TTS_SPEAKER_REF_SECONDS / read_speaker_ref_seconds）
    # 2) prepare_speaker_ref_audio: trim_silence + soft_clip + smooth_transients + normalize
    # 3) save_wav_norm 写回（必要时也会对“急剧瞬态/爆音”的旧文件做重处理）
```

---

## 三、翻译相关技巧

### 1. 翻译前标点修复

**问题**: Whisper转写常常缺少标点，输出长run-on sentences。

**解决方案**: 用LLM修复标点后再翻译。

```python
def _punct_fix_chunk(settings, payload, *, attempt_limit=5):
    system = """
    你是专业的 ASR 转录文本修复专家：
    1. 修正显著拼写错误、同音词混淆
    2. 长句断句：在合适位置插入句号/分号
    3. 补充语气标点：问号、感叹号、省略号
    4. 不改写/润色/总结，保持原意
    """
```

### 2. 翻译校验逻辑

**问题**: LLM有时会输出解释性文本而非纯译文，或译文过长/过短，也不适合直接送给TTS。

**解决方案**: 使用精确正则模式匹配检测异常输出。

```python
explanation_patterns = [
    # 使用精确正则（带位置锚点和特定上下文），区分"模型解释"和"原文内容"
    
    # "翻译" - 拒绝元解释，允许内容讨论
    (r'^翻译[：:]\s*', ...),              # 拒绝 "翻译：你好"（开头+冒号=元解释）
    (r'翻译(结果|如下|为)[：:]?', ...),   # 拒绝 "翻译结果是..."
    # 但允许 "机器翻译技术" / "这是一个翻译软件"（无特定模式=内容）
    
    # "中文" - 拒绝元解释，允许内容讨论
    (r'^(简体)?中文[：:]\s*', ...),       # 拒绝 "中文：你好"（开头+冒号）
    # 但允许 "学习中文" / "中文版本"（无冒号=内容）
    
    # "这句" - 拒绝解释句意，允许正常翻译
    (r'这句.{0,3}(的翻译|的意思)', ...),  # 拒绝 "这句话的翻译是..."
    # 但允许 "这句话很有道理"（无"翻译/意思"后缀=正常译文）
]
```

### 3. 智能分段翻译

**问题**: 长文本翻译时LLM可能丢失内容（"too short"错误）。

**解决方案**: 检测内容丢失后自动拆分重试。

```python
if "too short" in str(exc) and attempt == attempt_limit - 1:
    parts = _pack_source_text_chunks(_split_source_text_into_sentences(text))
    if len(parts) > 1:
        # 逐段翻译并合并
        for sub_text in parts:
            sub_translation = _translate_single_text(sub_text, ..., enable_fallback=False)
            full_translation_parts.append(sub_translation)
        return "".join(full_translation_parts)
```

### 4. 全局术语表

**问题**: 并发翻译时术语不一致（同一术语不同段落翻译不同）。

**解决方案**: 翻译前生成全局术语表，注入每个翻译请求。

```python
def _build_global_glossary(summary, transcript, target_language, settings):
    # 抽取专有名词/术语/人名/机构/产品名
    # 硬性要求: agent->智能体, Q-Learning, Transformer
    return {
        "glossary": {"term": "translation"},
        "dont_translate": ["API", "GPU", "FFT", ...],
        "notes": "..."
    }
```

---

## 四、其他健壮性措施

### 1. 音频/视频产物自动重建

基于元数据（`mix_version`、`adaptive_segment_stretch`等）判断是否需要重建：

```python
def _audio_combined_needs_rebuild(folder, *, adaptive_segment_stretch, sample_rate):
    meta = _read_audio_combined_meta(folder)
    if int(meta.get("mix_version")) != _AUDIO_COMBINED_MIX_VERSION:
        return True
    if bool(meta.get("adaptive_segment_stretch")) != bool(adaptive_segment_stretch):
        return True
    # 检查依赖文件的 mtime
    return _is_stale(audio_combined_path, deps)
```

### 2. TTS音量匹配原始人声

实现见 `src/youdub/steps/synthesize_video.py::_ensure_audio_combined`：
- 先用 `audio.wav` 对 demucs stems 做线性拟合，估计更接近原始混音的比例：`mix ≈ a*vocals + b*instruments`（多窗口采样，取 Top-K 中位数，避免落在“纯人声/纯音乐”窗口导致 BGM 被拉没）。
- 再用“有效样本 active RMS”（排除接近静音的样本）把 TTS 匹配到原人声响度，并把缩放系数 clamp 到合理范围，避免过度放大/压小（比 peak 更贴近听感）。

```python
# 概念示意：匹配 TTS 到原人声响度（active RMS，而不是 peak）
tts_scale = clamp(orig_voice_active_rms / tts_active_rms, 0.2, 3.0)
audio_tts = audio_tts * tts_scale
```

### 3. Qwen Worker重启策略

```python
def _should_restart_qwen_worker(err: str) -> bool:
    s = (err or "").lower()
    return (
        ("已退出" in err)
        or ("无输出" in err)
        or ("超时" in err)
        or ("broken pipe" in s)
        or ("eof" in s)
    )

# 策略（简化）：
# - 第一次遇到明显的 worker 异常：重启 worker 并重试一次
# - 对单段生成：若连续两次判定为 invalid/超长，再重启 worker（避免坏状态持续）
if attempt >= 1 and not qwen_restarted_for_this_segment:
    _restart_qwen_worker("segment invalid twice")
    qwen_restarted_for_this_segment = True
```

### 4. 输出分辨率计算（按宽高比 + 偶数对齐）

实现见 `src/youdub/steps/synthesize_video.py::convert_resolution`：
- 输入：视频宽高比 `aspect_ratio` + 目标档位（如 `1080p`）
- 规则：短边固定为 `base_res`，长边按宽高比推算；最后把宽高都对齐为偶数（FFmpeg 编码更稳）

### 5. 下载后视频有效性快速检查（ffprobe 时长兜底）

实现见 `src/youdub/steps/download.py::_probe_video_valid`：
- 优先探测 `stream=duration`（有视频流且可读时长）
- 失败则回退到 `format=duration`
- 只要可解析且 `duration >= min_duration` 即判定有效；超时/异常直接视为无效

### 6. WAV 写入归一化（峰值缩放到 int16）

实现见 `src/youdub/utils.py::save_wav_norm`：
- 计算峰值 `peak = max(abs(max(wav)), abs(min(wav)))`
- 缩放系数 `32767 / max(0.01, peak)`，避免除 0
- 写入 int16，防止削波并保持音量可听

---

## 五、ASR 转写 / 说话人相关算法

### 1. 句子切分并按权重分配时间（无词级时间戳的兜底）

实现见 `src/youdub/steps/transcribe.py::_split_text_with_timing`：
- 用句末标点切分：`(?<=[。！？!?\.])\s*`
- 权重：每段去空格后的字符数（最小 1）
- 按权重线性分配时长：`seg_dur = duration * (w / sum(w))`
- 最后一段强制贴到原 `end_s`（避免浮点误差）

### 2. 转写段合并（按说话人边界 + 句末标点）

实现见 `src/youdub/steps/transcribe.py::merge_segments`：
- **永不跨 speaker 合并**
- buffer 末尾字符如果属于 `ending='!"\\').:;?]}~'` 则截断（认为一句结束）
- 否则把下一段拼到 buffer，并延长 `end`

### 3. 说话人分配：与 diarization turns 最大重叠

实现见 `src/youdub/steps/transcribe.py::_assign_speakers_by_overlap`：
- 对每个 ASR segment，在与其时间窗口相交的 turns 中，选择 **重叠时长最大** 的 speaker
- turns 先按 start 排序，并用游标跳过完全在 segment 之前的 turn，避免 `O(n^2)` 扫描退化

---

## 六、通用小工具（用于“去重复实现”）

目前已集中到 `src/youdub/utils.py`：
- `wav_duration_seconds(path)`：用 stdlib `wave` 读 wav header 计算时长（返回 `float | None`）
- `read_speaker_ref_seconds(default=15.0)`：读取 `TTS_SPEAKER_REF_SECONDS` 并 clamp 到 `[3, 60]`