# YouTube SRT 对齐说话人分离（假设 1 cue = 1 人）

你已有 YouTube 提供的字幕（例如 `ground_truth.en.srt`）。目标是生成一个**带说话人标记**的新字幕（例如 `ground_truth.en.speaker.srt`）。

这里的“对齐”指：用说话人分离（diarization）的 **turns（谁在什么时候说话）**，按时间重叠把每条 YouTube cue 贴上 speaker 标签。

> 你假设“1 条 cue 只对应 1 个说话人”。我按这个假设写流程。现实里这不总成立：如果一条 cue 内发生换人，你只能得到“主说话人”，会错是正常的。

---

## 前置条件

- 需要与字幕时间轴一致的音频文件：
  - 推荐：`audio_vocals.wav`（人声分离后的轨道，仓库流程通常会产出）
  - 兜底：`audio.wav`（原始音频）
- 说话人分离依赖 `pyannote.audio`，并且需要模型缓存：
  - 环境变量 `WHISPER_DIARIZATION_MODEL_DIR` 指向缓存目录（仓库默认：`models/ASR/whisper/diarization`）
  - 若没有离线缓存，需要设置 `HF_TOKEN`（并在 HuggingFace 同意 gated 模型协议）
- **用 uv 环境运行**（本仓库约定）。

---

## 脚本运行说明（人话版）

这个脚本虽然看起来长，但核心逻辑其实非常简单，主要就干了三件事：

1.  **听声音**：利用 `pyannote` 模型把音频文件从头到尾听一遍，记下“第几秒到第几秒是谁在说话”（这叫 Diarization）。
2.  **读字幕**：把你的 SRT 字幕文件读进来，解析出每一句话的开始时间和结束时间。
3.  **连连看（对齐）**：
    *   拿着每一句字幕的时间段，去和第 1 步里的“说话人时间表”做对比。
    *   看看这段时间里，哪个说话人出现的时长最长（重叠最多）。
    *   那就认定这句话是这个人说的。

最后，脚本会生成一个新的 SRT 文件，内容和原来一模一样，只是在每一句的最前面加上了 `SPEAKER_01: ` 这样的标记。

---

## 最小可用脚本

你可以直接复制下面的代码块，保存为 `align_speaker.py` 然后运行，或者直接在终端里粘贴运行。

请注意修改 `main()` 函数里的 `video_dir` 路径为你实际的视频文件夹。

```python
cd /home/stardust/source/YouDub-webui

uv run python - <<'PY'
import re
import sys
from pathlib import Path

# === 0. 准备工作：复用仓库里现成的轮子 ===
# 我们直接借用 YouDub 仓库里已经写好的两个功能：
# 1. load_diarize_model: 负责加载那个死沉死沉的 pyannote 模型
# 2. _assign_speakers_by_overlap: 负责算“谁说话时间最长”这个数学题
sys.path.insert(0, str(Path.cwd() / "src"))
from youdub.steps.transcribe import load_diarize_model, _assign_speakers_by_overlap, _DIARIZATION_PIPELINE

# SRT 时间戳的正则格式 (00:00:00,000 --> 00:00:00,000)
TIME = re.compile(
    r"(\d\d):(\d\d):(\d\d),(\d\d\d)\s*-->\s*(\d\d):(\d\d):(\d\d),(\d\d\d)"
)

def to_seconds(h: str, m: str, s: str, ms: str) -> float:
    """把时分秒毫秒转成总秒数"""
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def fmt_srt_time(t: float) -> str:
    """把秒数转回 SRT 的时间格式"""
    ms = int(round(max(0.0, t) * 1000.0))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def parse_srt(path: Path) -> list[dict]:
    """
    读取 SRT 文件，把它变成一个 Python 列表。
    每一项长这样：{'start': 1.5, 'end': 4.2, 'text_lines': ['Hello world'], 'speaker': 'SPEAKER_00'}
    """
    print(f"📖 正在解析字幕文件: {path.name} ...")
    # utf-8-sig 是为了处理可能存在的 BOM 头
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    cues: list[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        # 跳过纯数字的序号行
        if line.isdigit():
            i += 1
            if i >= len(lines):
                break
            line = lines[i].strip()

        # 匹配时间轴行
        m = TIME.match(line)
        if not m:
            i += 1
            continue

        start = to_seconds(*m.group(1, 2, 3, 4))
        end = to_seconds(*m.group(5, 6, 7, 8))

        i += 1
        text_lines: list[str] = []
        # 读取接下来的文本行，直到遇到空行
        while i < len(lines) and lines[i].strip() != "":
            text_lines.append(lines[i])
            i += 1

        cues.append(
            {
                "start": float(start),
                "end": float(end),
                "text_lines": text_lines,
                "speaker": "SPEAKER_00",  # 暂时先填个默认的，一会儿改
            }
        )
        i += 1
    print(f"✅ 解析完成，共找到 {len(cues)} 条字幕。")
    return cues

def write_srt(cues: list[dict], out_path: Path) -> None:
    """把处理好的列表写回成 SRT 文件"""
    out_lines: list[str] = []
    seq = 0
    for cue in cues:
        seq += 1
        out_lines.append(str(seq))
        out_lines.append(f"{fmt_srt_time(cue['start'])} --> {fmt_srt_time(cue['end'])}")

        lines = list(cue.get("text_lines") or [""])
        if not lines:
            lines = [""]

        # 只在第一行加说话人前缀，这样既能看清是谁说的，又不破坏多行排版
        spk = str(cue.get("speaker") or "SPEAKER_00")
        lines[0] = f"{spk}: {lines[0]}".rstrip()

        out_lines.extend(lines)
        out_lines.append("")

    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"💾 已保存带说话人标记的字幕: {out_path}")

def main() -> None:
    # ---------------------------------------------------------
    # 👇👇👇 只需要改这一行路径 👇👇👇
    video_dir = Path(
        "/home/stardust/source/YouDub-webui/videos/More Perfect Union/20250327 I Live 400 Yards From Mark Zuckerbergs Massive Data Center"
    )
    # ---------------------------------------------------------
    
    srt_in = video_dir / "ground_truth.en.srt"
    
    # 优先找人声分离后的 wav，如果没有就找原始音频
    wav = video_dir / "audio_vocals.wav"
    if not wav.exists():
        wav = video_dir / "audio.wav"

    srt_out = video_dir / "ground_truth.en.speaker.srt"

    if not srt_in.exists():
        print(f"❌ 找不到字幕文件: {srt_in}")
        return
    if not wav.exists():
        print(f"❌ 找不到音频文件: {wav}")
        return

    # 1. 解析现有的 SRT
    cues = parse_srt(srt_in)
    if not cues:
        print("❌ 字幕文件是空的或者解析失败了。")
        return

    # 2. 跑 Pyannote 说话人分离 (Diarization)
    print("🤖 正在加载说话人分离模型 (可能需要一点时间)...")
    load_diarize_model(device="auto")
    
    print(f"🎧 正在分析音频中的说话人: {wav.name} ...")
    # 调用模型处理音频
    ann = _DIARIZATION_PIPELINE(str(wav))
    # 获取不重叠的说话人轨道 (Timeline)
    ann_view = getattr(ann, "exclusive_speaker_diarization", None) or ann
    
    # 把模型结果转换成简单的列表
    turns = [
        {"start": float(seg.start), "end": float(seg.end), "speaker": str(spk)}
        for seg, _, spk in ann_view.itertracks(yield_label=True)
    ]
    print(f"✅ 音频分析完成，识别到 {len(turns)} 个说话片段。")

    # 3. 核心步骤：对齐 (Mapping)
    # 用“时间重叠最大”原则，把每条字幕分配给一个说话人
    print("🔄 正在进行字幕与说话人的对齐...")
    _assign_speakers_by_overlap(cues, turns, default_speaker="SPEAKER_00")

    # 4. 保存结果
    write_srt(cues, srt_out)
    print("\n✨ 全部搞定！快去看看生成的文件吧。")

if __name__ == "__main__":
    main()
PY
```

---

## 这套对齐在什么情况下会明显不准

- 你的 YouTube `ground_truth.en.srt` 的时间轴与本地 `audio_vocals.wav` / `audio.wav` **不是同一个版本**（例如被加速、剪辑、重编码偏移）。
- 一条 cue 内确实包含多个人说话（你当前假设不成立）。
- diarization 本身分错人（尤其是背景噪声大、多人重叠讲话、强混响的片段）。

