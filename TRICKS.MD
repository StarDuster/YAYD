# YouDub Tricks 技巧集

本文档记录了为实现双语字幕对轴和TTS语音生成兜底所采用的各种技巧和策略。

---

## 一、双语字幕对轴

### 1. 自适应拉伸 (Adaptive Stretch)

**问题**: 传统全局加速（如1.2x）会导致TTS语音和视频画面不同步，口型对不上。

**解决方案**: 逐段拉伸视频以匹配TTS音频时长。

```python
# synthesize_video.py - 核心思路
# 1. 不对TTS音频做相位声码器time_stretch（避免"回音/空洞"感）
# 2. 裁剪TTS首尾静音，按自然停顿插入间隙
# 3. 生成 adaptive_plan.json 记录每段的 src_start/src_end/target_duration
# 4. 用FFmpeg filter_complex 逐段 trim+setpts 拉伸原视频
```

关键文件:
- `translation_adaptive.json` - 对轴后的字幕时间轴
- `adaptive_plan.json` - 视频合成计划（speech + pause 段）

### 2. 字幕换行阈值按分辨率计算

**问题**: 固定的换行字符数在不同分辨率/宽高比下表现不一致（竖屏视频字幕过长，横屏过短）。

**解决方案**: 根据输出宽度和字号动态计算每行最大字符数。

```python
def _calc_subtitle_wrap_chars(width, font_size, *, en_font_scale=0.75):
    # 水平安全边距：~3% 每侧，至少10px
    margin_x = max(10, int(round(w * 0.03)))
    safe_w = max(1, w - 2 * margin_x)
    
    # CJK字符宽度约等于字号
    max_chars_zh = max(1, int(safe_w / (float(fs) * 0.85)))
    # 英文字符更窄
    max_chars_en = max(1, int(safe_w / (float(en_fs) * 0.62)))
```

### 3. 双语字幕英文仅保留一句

**问题**: 直接显示整段英文原文会铺满屏幕，影响可读性。

**解决方案**: `_first_sentence()` 只提取原文第一句作为双语字幕的英文部分。

```python
def _first_sentence(text, *, max_chars=220):
    # 只保留第一行非空内容
    # 按句号/问号/感叹号截断
    # 避免把长段落全部显示
```

### 4. 双语字幕原文补齐

**问题**: 旧版 `translation.json` 可能缺少 `text` 字段（原文）。

**解决方案**: 从 `transcript.json` 按时间重叠自动回填。

```python
def _ensure_bilingual_source_text(folder, translation, *, adaptive_segment_stretch):
    # 读取 transcript.json
    # 按时间重叠找到最匹配的原文
    # 回写到 translation.json 持久化
```

### 5. 原文到译文数量映射

**问题**: 译文按句切分后数量可能多于原文句子数，导致重复显示原文。

**解决方案**: 智能映射原文句子到译文数量。

```python
def _map_source_text_to_translation_count(text, target_count):
    # 1. 先按句号/问号/感叹号严格切分
    # 2. 如果不够，按逗号/分号/冒号宽松切分
    # 3. 还不够，按单词均分
    # 目标: 产出恰好 target_count 个原文块，避免重复
```

---

## 二、TTS语音生成兜底

### 1. Qwen TTS Token上限检测

**问题**: Qwen3-TTS 有时会陷入退化循环，一直生成到 `max_new_tokens`（约170秒）。

**解决方案**: 检测输出时长是否接近token上限，视为失败并重试。

```python
_QWEN_TTS_TOKEN_HZ = 12.0  # Tokenizer-12Hz
_QWEN_TTS_MAX_NEW_TOKENS_DEFAULT = 2048

def _qwen_tts_is_degenerate_hit_cap(*, wav_dur):
    cap = float(_qwen_tts_max_new_tokens()) / 12.0  # ≈ 170秒
    tol = 2.0  # 容差
    return wav_dur >= cap - tol
```

### 2. TTS片段时长校验

**问题**: TTS偶尔输出异常长的音频（如10秒原文生成10分钟音频）。

**解决方案**: 基于原始片段时长设置上限，超出则删除重试。

```python
def _tts_duration_guard_params():
    ratio = 3.0      # 最大倍数
    extra = 8.0      # 额外秒数
    abs_cap = 180.0  # 绝对上限
    retries = 3

def _tts_segment_allowed_max_seconds(seg_dur, ratio, extra, abs_cap):
    # allowed = max(seg_dur * ratio, seg_dur + extra)
    # 例: 10秒原文 -> 允许最多 max(30, 18) = 30秒
```

### 3. TTS提示词规范化

**问题**: Python代码token（如 `os.path.join`）和 `snake_case` 标识符会被TTS错误发音。

**解决方案**: 在送入TTS前转换为可读形式。

```python
def _tts_prompt_normalize_python_tokens(text):
    # matplotlib.pyplot -> matplotlib dot pyplot
    # base_dir -> base dir
    # 注意: 不影响小数如 3.14
```

### 4. 重试时逐步清理文本

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

### 5. 说话人参考音频裁剪

**问题**: 历史遗留的超长 SPEAKER/*.wav 会导致voice cloning显存/时间爆炸。

**解决方案**: 自动裁剪到合理长度（默认15秒）。

```python
def _ensure_wav_max_duration(path, max_seconds, sample_rate=24000):
    # 官方推荐10-20秒
    # 超出则用librosa裁剪
```

### 6. 缺失说话人音频兜底

**问题**: 旧任务目录或中途清理后 SPEAKER/*.wav 丢失。

**解决方案**: 从 `audio_vocals.wav` 自动生成。

```python
# 如果 SPEAKER/*.wav 丢失，从 audio_vocals.wav 截取前 max_ref_seconds 秒
wav, _ = librosa.load(vocals_path, sr=24000, duration=max_ref_seconds)
save_wav(wav, spk_path)
```

### 7. Marker存在但wavs不完整时继续生成

**问题**: `.tts_done.json` 存在但 `wavs/` 目录被部分清理。

**解决方案**: 校验所有分段wav文件的完整性。

```python
def _tts_segment_wavs_complete(folder, expected_segments):
    # 检查 first/last segment
    # 遍历检查每个 wav 是否存在且有效
    # 检查时长是否在合理范围内
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

**问题**: LLM有时会输出解释性文本而非纯译文，或译文过长/过短。

**解决方案**: 使用精确正则模式匹配检测异常输出。

```python
explanation_patterns = [
    (r'^翻译[：:]\s*', '翻译：'),           # 拒绝 "翻译：你好"
    (r'翻译(结果|如下|为)[：:]?', '...'),   # 拒绝 "翻译结果是..."
    (r'^(简体)?中文[：:]\s*', '中文：'),    # 拒绝 "中文：你好"
    (r'[Tt]ranslat(e|ion)[：:]\s*', '...'), # 拒绝 "Translation: ..."
]
# 注意: 允许正常翻译中提到"这句话"/"中文版本"等
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

```python
# 将TTS音量缩放到与原始人声峰值一致
vocal_peak = max(abs(np.max(vocal_wav)), abs(np.min(vocal_wav)))
tts_peak = max(abs(np.max(audio_tts)), abs(np.min(audio_tts)))
scale = vocal_peak / tts_peak
audio_tts = audio_tts * scale
```

### 3. Qwen Worker重启策略

```python
def _should_restart_qwen_worker(err):
    return ("已退出" in err) or ("超时" in err) or ("broken pipe" in s)

# 策略: 先在同一worker重试一次，仍失败再重启worker
if attempt >= 1 and not qwen_restarted_for_this_segment:
    _restart_qwen_worker("segment invalid twice")
    qwen_restarted_for_this_segment = True
```

---

## 相关提交

| Commit | 描述 |
|--------|------|
| `5a301b2` | feat(synth): implement advanced adaptive stretch and bilingual ASS subtitles |
| `b7f43e6` | fix(synth): prevent adaptive stretch drift via timeline anchoring |
| `187fc66` | feat(synth): improve adaptive stretch with timeline compaction |
| `470e1f0` | fix(synth): 限制Qwen3-TTS最大token并对命中上限输出重试 |
| `77ced1b` | fix(synth): 增加TTS片段时长校验与重试 |
| `0e429ef` | fix(synth): 规范化TTS提示词中的python点链与下划线 |
| `72eec85` | feat(translate): 翻译前增加转写标点修复 |
| `4d6a9e1` | fix(translate): 改进所有翻译校验逻辑，使用精确模式匹配 |
| `082c511` | fix(video): 双语字幕英文仅保留一句 |
| `bbe03c1` | fix(synth): 按视频分辨率等比例计算字幕换行阈值 |
| `16b0df5` | fix(video): 缩放后再渲染字幕并补齐双语原文 |
