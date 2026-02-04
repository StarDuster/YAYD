# 说话人分离 (Speaker Diarization) 调参指南

当 `pyannote` 说话人分离结果“太灵敏”或“切分太碎”（同一人的一句话被切成多段，导致 TTS 频繁换人/语调不连贯）时，通常需要调整其内部超参。

## 1. 关键参数说明

这些参数属于 `pyannote.audio` Pipeline 的内部参数，通过 `pipeline.instantiate({...})` 设置。

### `segmentation.threshold` (分割阈值)
- **作用**：控制"检测到有人在说话"的敏感度。
- **默认值**：约 0.5
- **调整方向**：
  - **调高 (如 0.6~0.7)**：更严格，减少误切分（噪音不当人声）。
  - **调低 (如 0.3~0.4)**：更敏感，不漏掉轻微的语音。

### `clustering.threshold` (聚类阈值)
- **作用**：控制"判断两段语音是否属于同一说话人"的严格程度。
- **默认值**：约 0.7
- **调整方向**：
  - **调高 (如 0.8~0.9)**：更严格，容易把同一人切成多个 SPEAKER（过分区分）。
  - **调低 (如 0.5~0.6)**：更宽松，容易把不同人的声音合并为同一个 SPEAKER（过分合并）。

### `segmentation.min_duration_off` (静音合并阈值) —— **最关键**
- **作用**：当两段语音之间的静音小于此值时，合并为一段连续语音。
- **默认值**：0.0 秒（不自动合并）。
- **调整方向**：
  - **调高 (如 0.5~1.0)**：短暂停顿（如说话人喘气、思考）不会切分，显著减少碎片。
  - **副作用**：如果调太高，可能会把紧接着插话的另一个人的声音也吞进来（导致一条字幕里出现多个人）。

---

## 2. 针对较多说话人的节目视频的搜索结果

针对视频 **"I Live 400 Yards From Mark Zuckerbergs Massive Data Center"**，我们使用 `ground_truth.en.srt` 作为基准，以“每条字幕应尽量只包含一个说话人”为目标进行了网格搜索。

**最优参数组合：**

```python
pipeline.instantiate({
    "segmentation": {
        "min_duration_off": 0.0,  # 保持默认，不要激进合并
    },
    "clustering": {
        "threshold": 0.7,         # 默认值附近，区分度适中
    }
})
```

**评估指标：**
- **Multi-speaker cues**（单条字幕内出现多说话人的比例）：**1.69%** (4/237)
- **Turns**（总切分段数）：130（相比默认参数略有优化）

**结论：**
对于该视频，激进提高 `min_duration_off` 虽然能减少 Turn 数，但显著增加了 Multi-speaker error（把别人的话吞进来了）。因此，**保持 `min_duration_off=0` + `threshold=0.7` 是在准确性和连贯性之间的最佳平衡**。

---

## 3. 如何在代码中应用

在 `resplit_by_diarization.py` 或 `transcribe.py` 中：

```python
from pyannote.audio import Pipeline

# ... 加载 pipeline ...

# 应用超参
pipeline.instantiate({
    "segmentation": {
        "min_duration_off": 0.0,  # 根据需要调整
    },
    "clustering": {
        "threshold": 0.7,         # 根据需要调整
    }
})

# 运行分离
ann = pipeline(wav_path, min_speakers=..., max_speakers=...)
```
