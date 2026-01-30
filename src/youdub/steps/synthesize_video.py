import json
import os
import re
import subprocess
import time
from typing import Any

import librosa
import numpy as np
from loguru import logger

from ..interrupts import check_cancelled, sleep_with_cancel
from ..utils import save_wav


_AUDIO_COMBINED_META_NAME = ".audio_combined.json"
# Bump when the mixing/output semantics change.
# v2: audio_combined = audio_tts + instruments (NO original vocals mixed in)
_AUDIO_COMBINED_MIX_VERSION = 2


def _valid_file(path: str, *, min_bytes: int = 1) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) >= int(min_bytes)
    except Exception:
        return False


def _mtime(path: str) -> float | None:
    try:
        return float(os.path.getmtime(path))
    except Exception:
        return None


def _is_stale(target: str, deps: list[str]) -> bool:
    t = _mtime(target)
    if t is None:
        return True
    for p in deps:
        mt = _mtime(p)
        if mt is None:
            continue
        if mt > t:
            return True
    return False


def _read_audio_combined_meta(folder: str) -> dict[str, Any] | None:
    meta_path = os.path.join(folder, _AUDIO_COMBINED_META_NAME)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else None
    except Exception:
        return None


def _write_audio_combined_meta(
    folder: str,
    *,
    adaptive_segment_stretch: bool,
    sample_rate: int,
) -> None:
    meta_path = os.path.join(folder, _AUDIO_COMBINED_META_NAME)
    payload: dict[str, Any] = {
        "mix_version": int(_AUDIO_COMBINED_MIX_VERSION),
        "adaptive_segment_stretch": bool(adaptive_segment_stretch),
        "sample_rate": int(sample_rate),
        "created_at": float(time.time()),
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # Best-effort only; don't fail the pipeline due to metadata I/O.
        logger.debug(f"写入 audio_combined 元数据失败: {e}")


def _audio_combined_needs_rebuild(
    folder: str,
    *,
    adaptive_segment_stretch: bool,
    sample_rate: int = 24000,
) -> bool:
    audio_combined_path = os.path.join(folder, "audio_combined.wav")
    if not _valid_file(audio_combined_path, min_bytes=44):
        return True

    meta = _read_audio_combined_meta(folder)
    if not meta:
        return True
    if int(meta.get("mix_version") or 0) != int(_AUDIO_COMBINED_MIX_VERSION):
        return True
    if bool(meta.get("adaptive_segment_stretch")) != bool(adaptive_segment_stretch):
        return True
    if int(meta.get("sample_rate") or 0) != int(sample_rate):
        return True

    deps = [
        os.path.join(folder, "wavs", ".tts_done.json"),
        os.path.join(folder, "audio_instruments.wav"),
    ]
    return _is_stale(audio_combined_path, deps)


def _video_up_to_date(
    folder: str,
    *,
    subtitles: bool,
    adaptive_segment_stretch: bool,
    sample_rate: int = 24000,
) -> bool:
    output_video = os.path.join(folder, "video.mp4")
    if not _valid_file(output_video, min_bytes=1024):
        return False

    # If audio needs rebuild (e.g. old mixing semantics), video is also stale.
    if _audio_combined_needs_rebuild(
        folder, adaptive_segment_stretch=adaptive_segment_stretch, sample_rate=sample_rate
    ):
        return False

    deps = [os.path.join(folder, "download.mp4"), os.path.join(folder, "audio_combined.wav")]
    if subtitles:
        tr = "translation_adaptive.json" if adaptive_segment_stretch else "translation.json"
        deps.append(os.path.join(folder, tr))
    return not _is_stale(output_video, deps)


def _is_cjk_char(ch: str) -> bool:
    # CJK Unified Ideographs + Extension A (covers most Chinese/Japanese Kanji).
    return ("\u4e00" <= ch <= "\u9fff") or ("\u3400" <= ch <= "\u4dbf")


def _count_cjk_chars(text: str) -> int:
    s = (text or "").strip()
    if not s:
        return 0
    return sum(1 for ch in s if _is_cjk_char(ch))


_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _count_en_words(text: str) -> int:
    s = " ".join((text or "").split()).strip()
    if not s:
        return 0
    return len(_EN_WORD_RE.findall(s))


def _compute_global_x(
    translation: list[dict[str, Any]],
    *,
    video_duration_s: float,
    cn_chars_per_min: float = 220.0,
    en_wpm: float = 130.0,
    eps: float = 1e-6,
) -> tuple[float, dict[str, float]]:
    """Compute the global ratio X = A/B described in the plan.

    - A: original total duration (video_duration_s)
    - B: predicted Chinese speech duration (scaled from actual English speech duration)
    """
    # Counts from translation text.
    cn_chars = 0
    en_words = 0
    for seg in translation:
        cn_chars += _count_cjk_chars(str(seg.get("translation") or ""))
        en_words += _count_en_words(str(seg.get("text") or ""))

    cn_est_s = (float(cn_chars) / max(float(cn_chars_per_min), eps)) * 60.0
    en_est_s = (float(en_words) / max(float(en_wpm), eps)) * 60.0

    # Actual English speech duration (ignoring pauses) from ASR timestamps.
    a_speech = 0.0
    for seg in translation:
        try:
            s0 = float(seg.get("start", 0.0) or 0.0)
            s1 = float(seg.get("end", 0.0) or 0.0)
        except Exception:
            continue
        if s1 > s0:
            a_speech += (s1 - s0)

    # Predicted Chinese speech duration, mapped from actual speech duration.
    ratio = cn_est_s / max(en_est_s, eps) if en_est_s > eps else 1.0
    b_speech = a_speech * ratio

    x = float(video_duration_s) / max(b_speech, eps) if video_duration_s > 0 else 1.0
    if not np.isfinite(x) or x <= 0:
        x = 1.0

    stats = {
        "cn_chars": float(cn_chars),
        "en_words": float(en_words),
        "cn_est_s": float(cn_est_s),
        "en_est_s": float(en_est_s),
        "a_speech": float(a_speech),
        "b_speech": float(b_speech),
        "x": float(x),
        "video_duration_s": float(video_duration_s),
    }
    return x, stats


def split_text(
    input_data: list[dict[str, Any]], 
    punctuations: list[str] | None = None
) -> list[dict[str, Any]]:
    puncts = set(punctuations or ['，', '；', '：', '。', '？', '！', '\n', '”'])

    def is_punctuation(char: str) -> bool:
        return char in puncts

    output_data = []
    for item in input_data:
        start = item["start"]
        text = item["translation"]
        speaker = item.get("speaker", "SPEAKER_00")
        original_text = item["text"]
        sentence_start = 0

        if not text:
            output_data.append(item)
            continue

        duration_per_char = (item["end"] - item["start"]) / len(text)
        
        for i, char in enumerate(text):
            if not is_punctuation(char) and i != len(text) - 1:
                continue
            if i - sentence_start < 5 and i != len(text) - 1:
                continue
            if i < len(text) - 1 and is_punctuation(text[i+1]):
                continue
                
            sentence = text[sentence_start:i+1]
            sentence_end = start + duration_per_char * len(sentence)

            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": original_text,
                "translation": sentence,
                "speaker": speaker
            })

            start = sentence_end
            sentence_start = i + 1

    return output_data


def format_timestamp(seconds: float) -> str:
    millisec = int((seconds - int(seconds)) * 1000)
    hours, seconds_int = divmod(int(seconds), 3600)
    minutes, seconds_int = divmod(seconds_int, 60)
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{millisec:03}"


def format_timestamp_ass(seconds: float) -> str:
    """ASS timestamp: H:MM:SS.cc (centiseconds)."""
    if seconds < 0:
        seconds = 0.0
    total = int(seconds)
    frac = seconds - float(total)
    cs = int(round(frac * 100.0))
    if cs >= 100:
        total += 1
        cs = 0
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02}:{secs:02}.{cs:02}"


def _wrap_cjk(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [text]
    s = (text or "").strip()
    if not s:
        return []
    return [s[i : i + max_chars] for i in range(0, len(s), max_chars)]


def _wrap_words(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [text]
    s = " ".join((text or "").split()).strip()
    if not s:
        return []
    words = s.split(" ")
    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
            continue
        if len(cur) + 1 + len(w) <= max_chars:
            cur = f"{cur} {w}"
            continue
        lines.append(cur)
        cur = w
    if cur:
        lines.append(cur)
    return lines


def wrap_text(text: str, *, max_chars_zh: int = 55, max_chars_en: int = 100) -> str:
    """Language-aware wrapping for subtitles."""
    raw = (text or "").strip()
    if not raw:
        return ""
    out_lines: list[str] = []
    for para in raw.splitlines():
        p = para.strip()
        if not p:
            continue
        if any(_is_cjk_char(ch) for ch in p if not ch.isspace()):
            out_lines.extend(_wrap_cjk(p, max_chars_zh))
        else:
            out_lines.extend(_wrap_words(p, max_chars_en))
    return "\n".join(out_lines)


def _ass_escape_text(text: str) -> str:
    # Escape override braces so content doesn't break ASS parsing.
    s = str(text or "").replace("\r", "").strip()
    s = s.replace("{", r"\{").replace("}", r"\}")
    return s


def generate_bilingual_ass(
    translation: list[dict[str, Any]],
    ass_path: str,
    *,
    speed_up: float = 1.0,
    play_res_x: int = 1920,
    play_res_y: int = 1080,
    font_name: str = "Arial",
    font_size: int = 49,
    outline: int = 4,
    margin_l: int = 10,
    margin_r: int = 10,
    margin_v: int = 10,
    en_font_scale: float = 0.75,
    max_chars_zh: int = 55,
    max_chars_en: int = 100,
) -> None:
    en_font_size = max(10, int(round(float(font_size) * float(en_font_scale))))

    lines: list[str] = []
    lines.extend(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            f"PlayResX: {int(play_res_x)}",
            f"PlayResY: {int(play_res_y)}",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{font_name},{int(font_size)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{int(outline)},0,2,{int(margin_l)},{int(margin_r)},{int(margin_v)},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )

    for seg in translation:
        check_cancelled()
        zh_raw = str(seg.get("translation") or "").strip()
        en_raw = str(seg.get("text") or "").strip()
        if not zh_raw and not en_raw:
            continue

        start = format_timestamp_ass(float(seg.get("start", 0.0) or 0.0) / float(speed_up))
        end = format_timestamp_ass(float(seg.get("end", 0.0) or 0.0) / float(speed_up))

        zh = _ass_escape_text(wrap_text(zh_raw, max_chars_zh=max_chars_zh, max_chars_en=max_chars_en)).replace(
            "\n", r"\N"
        )
        en = _ass_escape_text(wrap_text(en_raw, max_chars_zh=max_chars_zh, max_chars_en=max_chars_en)).replace(
            "\n", r"\N"
        )

        if zh and en:
            text = f"{zh}" + r"\N" + r"{\fs" + str(en_font_size) + "}" + f"{en}"
        else:
            text = zh or (r"{\fs" + str(en_font_size) + "}" + en)

        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_srt(
    translation: list[dict[str, Any]], 
    srt_path: str, 
    speed_up: float = 1.0, 
    max_line_char: int = 55,
    bilingual_subtitle: bool = False,
) -> None:
    # 默认行为：按译文标点再切一遍，避免单条字幕太长。
    # 双语模式：translation.json 已在翻译阶段做过按句切分并尽量对齐原文；这里不再二次切分，
    # 否则会导致原文行重复（同一句英文被拆成多段译文时会重复显示）。
    if not bilingual_subtitle:
        translation = split_text(translation)

    def _wrap_lines(s: str) -> list[str]:
        s = str(s or "").strip()
        if not s:
            return []
        lines_count = len(s) // (max_line_char + 1) + 1
        avg_chars = min(max(1, round(len(s) / lines_count)), max_line_char)
        return [s[j * avg_chars : (j + 1) * avg_chars] for j in range(lines_count)]

    with open(srt_path, 'w', encoding='utf-8') as f:
        seq = 0
        for line in translation:
            start = format_timestamp(line['start'] / speed_up)
            end = format_timestamp(line['end'] / speed_up)
            tr_text = str(line.get('translation', '') or '').strip()
            src_text = str(line.get('text', '') or '').strip()

            if not tr_text and not (bilingual_subtitle and src_text):
                continue
                
            lines: list[str] = []
            if tr_text:
                lines.extend(_wrap_lines(tr_text))
            if bilingual_subtitle and src_text:
                lines.extend(_wrap_lines(src_text))
            if not lines:
                continue

            seq += 1
            wrapped_text = '\n'.join(lines)

            f.write(f'{seq}\n')
            f.write(f'{start} --> {end}\n')
            f.write(f'{wrapped_text}\n\n')


def get_aspect_ratio(video_path: str) -> float:
    check_cancelled()
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'json', video_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        dimensions = json.loads(result.stdout)['streams'][0]
        return dimensions['width'] / dimensions['height']
    except Exception as e:
        logger.error(f"获取视频宽高比失败: {e}")
        return 16/9


def convert_resolution(aspect_ratio: float, resolution: str = '1080p') -> tuple[int, int]:
    base_res = int(resolution.replace('p', ''))
    
    if aspect_ratio < 1:
        width = base_res
        height = int(width / aspect_ratio)
    else:
        height = base_res
        width = int(height * aspect_ratio)
        
    width = width - (width % 2)
    height = height - (height % 2)
    
    return width, height


def _get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    check_cancelled()
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=duration', '-of', 'json', video_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data.get('streams', [{}])[0].get('duration', 0))
        if duration > 0:
            return duration
    except Exception as e:
        logger.warning(f"获取视频时长失败: {e}")
    
    # 回退：尝试从音频流获取
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=duration', '-of', 'json', video_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data.get('streams', [{}])[0].get('duration', 0))
        if duration > 0:
            return duration
    except Exception:
        pass
    
    return 0.0


def _ensure_audio_combined(
    folder: str,
    adaptive_segment_stretch: bool = False,
    speed_up: float = 1.0,
    sample_rate: int = 24000,
) -> None:
    """
    生成 audio_combined.wav 和 audio_tts.wav
    
    如果 adaptive_segment_stretch=True:
        - 按原视频时间轴做全局缩放，每一段都放到它在缩放后应在的位置
        - 如果这段TTS更短，就在段内补静音把时长补齐
    否则:
        - 按顺序拼接TTS音频片段
    """
    check_cancelled()
    
    audio_combined_path = os.path.join(folder, 'audio_combined.wav')
    audio_tts_path = os.path.join(folder, 'audio_tts.wav')
    wavs_folder = os.path.join(folder, 'wavs')
    translation_path = os.path.join(folder, 'translation.json')
    translation_adaptive_path = os.path.join(folder, 'translation_adaptive.json')
    adaptive_plan_path = os.path.join(folder, "adaptive_plan.json")
    video_path = os.path.join(folder, 'download.mp4')
    audio_instruments_path = os.path.join(folder, 'audio_instruments.wav')
    
    # 检查必要文件
    if not os.path.exists(wavs_folder):
        raise FileNotFoundError(f"缺少 wavs 目录: {wavs_folder}")
    
    # 自适应模式的输入时间轴必须来自 translation.json（原始 ASR 时间戳）。
    # translation_adaptive.json 是本函数在自适应模式下生成的输出时间轴。
    if not os.path.exists(translation_path):
        raise FileNotFoundError(f"缺少翻译文件: {translation_path}")
    
    with open(translation_path, 'r', encoding='utf-8') as f:
        translation = json.load(f)
    
    if not translation:
        raise ValueError(f"翻译文件为空: {translation_path}")
    
    # 获取所有wav文件
    wav_files = sorted([f for f in os.listdir(wavs_folder) if f.endswith('.wav')])
    if not wav_files:
        raise ValueError(f"wavs 目录为空: {wavs_folder}")
    
    # 确保wav文件数量与translation段数一致
    expected_count = len(translation)
    if len(wav_files) < expected_count:
        raise FileNotFoundError(
            f"wav文件数量({len(wav_files)})少于翻译段数({expected_count})，请先确保TTS已完整生成"
        )
    
    if adaptive_segment_stretch:
        # 自适应模式（新算法）：
        # - 计算全局比例 X（基于字数/语速 + 原视频时长）
        # - 以 ASR 段 start 为锚点，生成 speech + pause 的输出时间轴
        # - speech 段按规则决定：最多 1.2x 提速音频；仍过长则放慢视频（通过 target_duration 体现）
        # - pause 段来自相邻 ASR 段 gap（>1s 标记为停顿片段），并插入 0.x 秒微停顿避免过密
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"自适应模式需要视频文件: {video_path}")
        
        video_duration_s = _get_video_duration(video_path)
        if video_duration_s <= 0:
            raise ValueError(f"无法获取视频时长: {video_path}")

        # Compute global X.
        x, x_stats = _compute_global_x(translation, video_duration_s=video_duration_s)
        logger.info(
            f"自适应缩放：X={x_stats['x']:.3f} "
            f"(cn_chars={int(x_stats['cn_chars'])}, en_words={int(x_stats['en_words'])}, "
            f"A_speech={x_stats['a_speech']:.1f}s, B_speech={x_stats['b_speech']:.1f}s, "
            f"A_total={x_stats['video_duration_s']:.1f}s)"
        )

        # Planner knobs (per the plan).
        pause_threshold_s = 1.0
        inter_pause_s = 0.25
        max_pause_s = 3.0
        max_audio_speed = 1.2
        eps = 1e-6

        audio_chunks: list[np.ndarray] = []
        adaptive_translation: list[dict[str, Any]] = []
        adaptive_plan: list[dict[str, Any]] = []

        t_cursor_samples = 0  # output timeline cursor (samples)

        def _append_silence(seconds: float) -> None:
            nonlocal t_cursor_samples
            sec = float(seconds)
            if sec <= 0:
                return
            n = int(round(sec * sample_rate))
            if n <= 0:
                return
            audio_chunks.append(np.zeros(n, dtype=np.float32))
            t_cursor_samples += n

        for i, seg in enumerate(translation):
            check_cancelled()
            # Load per-segment TTS audio.
            wav_path = os.path.join(wavs_folder, wav_files[i])
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"TTS音频文件不存在: {wav_path}")

            tts_audio, _ = librosa.load(wav_path, sr=sample_rate)
            tts_audio = tts_audio.astype(np.float32, copy=False)
            d_tts = float(len(tts_audio)) / float(sample_rate)

            # Original segment duration from ASR timestamps.
            orig_start = float(seg.get("start", 0.0) or 0.0)
            orig_end = float(seg.get("end", 0.0) or 0.0)
            if orig_end < orig_start:
                orig_start, orig_end = orig_end, orig_start
            d_orig = max(0.0, orig_end - orig_start)
            if d_orig <= 0:
                # Degenerate segment; treat as minimal to avoid div-by-zero.
                d_orig = 0.001

            d_base = d_orig / max(float(x), eps)

            # Decide audio speed-up (<=1.2) and target duration.
            if d_tts > d_base and d_tts > 0:
                audio_speed = min(float(max_audio_speed), d_tts / max(d_base, eps))
            else:
                audio_speed = 1.0
            target_duration = (d_tts / audio_speed) if d_tts > 0 else d_base
            target_samples = int(round(target_duration * sample_rate))
            if target_samples <= 0:
                target_samples = 1
            target_duration = float(target_samples) / float(sample_rate)

            # Time-stretch audio if needed, then hard-fit to target_samples.
            if audio_speed > 1.0001 and len(tts_audio) > 1:
                try:
                    stretched = librosa.effects.time_stretch(tts_audio, rate=audio_speed)
                except Exception as e:
                    # Fallback: simple resample by interpolation (may shift pitch).
                    logger.warning(f"librosa time_stretch 失败，改用插值加速: seg={i}, speed={audio_speed:.3f}, err={e}")
                    new_len = max(1, int(round(len(tts_audio) / audio_speed)))
                    x_idx = np.linspace(0.0, float(len(tts_audio) - 1), num=new_len, dtype=np.float64)
                    stretched = np.interp(x_idx, np.arange(len(tts_audio), dtype=np.float64), tts_audio).astype(
                        np.float32
                    )
                tts_out = stretched.astype(np.float32, copy=False)
            else:
                tts_out = tts_audio

            if len(tts_out) < target_samples:
                tts_out = np.pad(tts_out, (0, target_samples - len(tts_out)), mode="constant")
            elif len(tts_out) > target_samples:
                tts_out = tts_out[:target_samples]

            audio_chunks.append(tts_out.astype(np.float32, copy=False))

            # Record adaptive translation (speech segments only).
            start_s = float(t_cursor_samples) / float(sample_rate)
            end_s = float(t_cursor_samples + target_samples) / float(sample_rate)
            out_seg = dict(seg)
            out_seg["start"] = round(start_s, 3)
            out_seg["end"] = round(end_s, 3)
            adaptive_translation.append(out_seg)

            # Record plan for video composition.
            adaptive_plan.append(
                {
                    "kind": "speech",
                    "index": i,
                    "src_start": round(orig_start, 6),
                    "src_end": round(orig_end, 6),
                    "target_duration": round(target_duration, 6),
                    "audio_speed": round(float(audio_speed), 6),
                }
            )

            t_cursor_samples += target_samples

            # Insert pause between speech segments (except after last).
            if i < (len(translation) - 1):
                try:
                    next_start = float(translation[i + 1].get("start", orig_end) or orig_end)
                except Exception:
                    next_start = orig_end
                gap = float(next_start - orig_end)
                if gap > 0:
                    pause_src_start = orig_end
                    pause_src_end = next_start
                    pause_duration = max(float(inter_pause_s), gap / max(float(x), eps))
                else:
                    # Overlap/adjacent: take a small tail slice from the previous segment as "pause" visuals.
                    tail = 0.08
                    pause_src_end = orig_end
                    pause_src_start = max(0.0, pause_src_end - tail)
                    pause_duration = float(inter_pause_s)

                pause_duration = min(float(max_pause_s), float(pause_duration)) if max_pause_s > 0 else float(
                    pause_duration
                )

                _append_silence(pause_duration)
                adaptive_plan.append(
                    {
                        "kind": "pause",
                        "src_start": round(float(pause_src_start), 6),
                        "src_end": round(float(pause_src_end), 6),
                        "target_duration": round(float(pause_duration), 6),
                    }
                )

        if not audio_chunks:
            raise ValueError("没有有效的TTS音频片段")

        audio_tts = np.concatenate(audio_chunks).astype(np.float32, copy=False)

        # Save translation_adaptive.json (speech-only cues).
        with open(translation_adaptive_path, "w", encoding="utf-8") as f:
            json.dump(adaptive_translation, f, ensure_ascii=False, indent=2)
        logger.info(f"已生成 translation_adaptive.json: {translation_adaptive_path}")

        # Save plan for video composition (speech + pause segments).
        plan_payload = {"x_stats": x_stats, "segments": adaptive_plan}
        with open(adaptive_plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"已生成 adaptive_plan.json: {adaptive_plan_path}")

        # Save audio_tts.wav.
        save_wav(audio_tts, audio_tts_path, sample_rate=sample_rate)
        logger.info(f"已生成 audio_tts.wav (自适应模式): {audio_tts_path}")
        
    else:
        # 非自适应模式：按顺序拼接
        audio_segments = []
        for i, wav_file in enumerate(wav_files[:len(translation)]):
            check_cancelled()
            wav_path = os.path.join(wavs_folder, wav_file)
            if not os.path.exists(wav_path):
                logger.warning(f"TTS音频文件不存在: {wav_path}")
                continue
            
            try:
                audio, _ = librosa.load(wav_path, sr=sample_rate)
                audio_segments.append(audio)
            except Exception as e:
                logger.warning(f"加载TTS音频失败 {wav_path}: {e}")
                continue
        
        if not audio_segments:
            raise ValueError("没有有效的TTS音频片段")
        
        audio_tts = np.concatenate(audio_segments)
        save_wav(audio_tts, audio_tts_path, sample_rate=sample_rate)
        logger.info(f"已生成 audio_tts.wav: {audio_tts_path}")
    
    # 混合 TTS 音频和背景伴奏（不混入原人声，避免回声/双声）
    check_cancelled()
    if os.path.exists(audio_instruments_path):
        try:
            # 加载伴奏
            instruments, _ = librosa.load(audio_instruments_path, sr=sample_rate)
            
            # 对齐长度（以TTS音频为准）
            tts_len = len(audio_tts)
            instruments_len = len(instruments)
            
            # 如果伴奏更长，裁剪；如果更短，循环填充
            if instruments_len < tts_len:
                repeat_count = (tts_len + instruments_len - 1) // instruments_len
                instruments = np.tile(instruments, repeat_count)[:tts_len]
            else:
                instruments = instruments[:tts_len]
            
            # 混合：TTS + 伴奏（降低伴奏音量）
            audio_combined = audio_tts + 0.2 * instruments
            
            # 归一化防止削波
            max_val = np.abs(audio_combined).max()
            if max_val > 1.0:
                audio_combined = audio_combined / max_val * 0.95
            
            save_wav(audio_combined.astype(np.float32), audio_combined_path, sample_rate=sample_rate)
            _write_audio_combined_meta(
                folder,
                adaptive_segment_stretch=adaptive_segment_stretch,
                sample_rate=sample_rate,
            )
            logger.info(f"已生成 audio_combined.wav: {audio_combined_path}")
        except Exception as e:
            logger.warning(f"混合背景音乐失败，仅使用TTS音频: {e}")
            save_wav(audio_tts, audio_combined_path, sample_rate=sample_rate)
            _write_audio_combined_meta(
                folder,
                adaptive_segment_stretch=adaptive_segment_stretch,
                sample_rate=sample_rate,
            )
    else:
        # 没有背景音乐，直接使用TTS音频
        save_wav(audio_tts, audio_combined_path, sample_rate=sample_rate)
        _write_audio_combined_meta(
            folder,
            adaptive_segment_stretch=adaptive_segment_stretch,
            sample_rate=sample_rate,
        )
        logger.info(f"已生成 audio_combined.wav (无背景音乐): {audio_combined_path}")


def synthesize_video(
    folder: str, 
    subtitles: bool = True, 
    speed_up: float = 1.2, 
    fps: int = 30, 
    resolution: str = '1080p',
    use_nvenc: bool = False,
    adaptive_segment_stretch: bool = False,
    bilingual_subtitle: bool = False,
) -> None:
    check_cancelled()
    output_video = os.path.join(folder, "video.mp4")

    # NOTE:
    # 以前只要 video.mp4 存在就直接跳过，这会导致：
    # - 混音逻辑修复后仍复用旧的 audio_combined/video（用户听感依旧“回声/双声”）
    # - translation/音频更新后视频也不重建
    # 因此这里改为基于 mtime + 元数据判断是否需要重建。
    if _video_up_to_date(folder, subtitles=subtitles, adaptive_segment_stretch=adaptive_segment_stretch):
        logger.info(f"已合成视频: {folder}")
        return
    if os.path.exists(output_video):
        try:
            if os.path.getsize(output_video) < 1024:
                logger.warning(f"video.mp4 疑似无效(过小)，将重新生成: {output_video}")
            else:
                logger.info(f"video.mp4 已存在但已过期，将重新生成: {output_video}")
            os.remove(output_video)
        except Exception:
            # Best-effort: proceed to re-generate
            pass
    
    translation_path = os.path.join(folder, 'translation.json')
    translation_adaptive_path = os.path.join(folder, 'translation_adaptive.json')
    adaptive_plan_path = os.path.join(folder, "adaptive_plan.json")
    input_audio = os.path.join(folder, 'audio_combined.wav')
    input_video = os.path.join(folder, 'download.mp4')
    
    # 自适应模式下，除了 audio_combined.wav，还需要 translation_adaptive.json / adaptive_plan.json。
    if adaptive_segment_stretch:
        if _audio_combined_needs_rebuild(folder, adaptive_segment_stretch=True) or not (
            os.path.exists(input_audio) and os.path.exists(translation_adaptive_path)
        ):
            check_cancelled()
            logger.info("自适应模式所需文件不存在，正在生成 audio/translation_adaptive/plan ...")
            _ensure_audio_combined(folder, adaptive_segment_stretch=True, speed_up=1.0)
        # adaptive_plan.json 缺失时，可由 translation.json + translation_adaptive.json 重建（无需重跑TTS）。
        if not os.path.exists(adaptive_plan_path) and os.path.exists(translation_path) and os.path.exists(
            translation_adaptive_path
        ):
            try:
                with open(translation_path, "r", encoding="utf-8") as f:
                    _orig = json.load(f) or []
                with open(translation_adaptive_path, "r", encoding="utf-8") as f:
                    _adapt = json.load(f) or []
                segs: list[dict[str, Any]] = []
                for i in range(min(len(_orig), len(_adapt))):
                    o0 = float(_orig[i].get("start", 0.0) or 0.0)
                    o1 = float(_orig[i].get("end", 0.0) or 0.0)
                    if o1 < o0:
                        o0, o1 = o1, o0
                    a0 = float(_adapt[i].get("start", 0.0) or 0.0)
                    a1 = float(_adapt[i].get("end", 0.0) or 0.0)
                    if a1 < a0:
                        a0, a1 = a1, a0
                    segs.append(
                        {
                            "kind": "speech",
                            "index": i,
                            "src_start": round(o0, 6),
                            "src_end": round(o1, 6),
                            "target_duration": round(max(0.001, a1 - a0), 6),
                        }
                    )
                    if i < (min(len(_orig), len(_adapt)) - 1):
                        # Pause duration in output timeline comes from adaptive timestamps.
                        next_a0 = float(_adapt[i + 1].get("start", a1) or a1)
                        pause_dur = max(0.0, next_a0 - a1)
                        # Pause visuals from original gap if possible; else take a small tail slice.
                        next_o0 = float(_orig[i + 1].get("start", o1) or o1)
                        gap = float(next_o0 - o1)
                        if gap > 0:
                            p0, p1 = o1, next_o0
                        else:
                            p1 = o1
                            p0 = max(0.0, p1 - 0.08)
                        if pause_dur > 1e-6:
                            segs.append(
                                {
                                    "kind": "pause",
                                    "src_start": round(p0, 6),
                                    "src_end": round(p1, 6),
                                    "target_duration": round(pause_dur, 6),
                                }
                            )
                with open(adaptive_plan_path, "w", encoding="utf-8") as f:
                    json.dump({"segments": segs}, f, ensure_ascii=False, indent=2)
                logger.info(f"已重建 adaptive_plan.json: {adaptive_plan_path}")
            except Exception as e:
                logger.warning(f"重建 adaptive_plan.json 失败，将继续尝试直接合成视频: {e}")
    else:
        # 如果 audio_combined.wav 不存在，先生成它
        if _audio_combined_needs_rebuild(folder, adaptive_segment_stretch=False) or not os.path.exists(input_audio):
            check_cancelled()
            logger.info("audio_combined.wav 不存在/已过期，正在生成...")
            _ensure_audio_combined(folder, adaptive_segment_stretch=False, speed_up=speed_up)
    
    # 字幕在自适应模式下应跟随 translation_adaptive.json。
    translation_path_to_use = translation_adaptive_path if adaptive_segment_stretch else translation_path

    missing = [p for p in (translation_path_to_use, input_audio, input_video) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"缺少合成视频所需文件：{missing}")
    
    with open(translation_path_to_use, 'r', encoding='utf-8') as f:
        translation = json.load(f)
        
    srt_path = os.path.join(folder, 'subtitles.srt')
    if subtitles:
        # 自适应模式下，translation_adaptive.json 已经在输出时间轴，不应再做 speed_up 缩放。
        subs_speed = 1.0 if adaptive_segment_stretch else speed_up
        generate_srt(translation, srt_path, subs_speed, bilingual_subtitle=bilingual_subtitle)

    check_cancelled()
    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    res_string = f'{width}x{height}'
    
    srt_path_filter = srt_path.replace('\\', '/')
    
    if os.name == 'nt' and ':' in srt_path_filter:
        srt_path_filter = srt_path_filter.replace(':', '\\:')

    # Subtitle font size: readable across resolutions (1080p -> ~49).
    font_size = int(round(height * 0.045))
    font_size = max(18, min(font_size, 120))
    outline = int(round(font_size / 12))
    outline = max(2, outline)

    subtitle_filter = ""
    if subtitles:
        if bilingual_subtitle:
            ass_path = os.path.join(folder, "subtitles.ass")
            generate_bilingual_ass(
                translation,
                ass_path,
                speed_up=(1.0 if adaptive_segment_stretch else speed_up),
                play_res_x=width,
                play_res_y=height,
                font_name="Arial",
                font_size=font_size,
                outline=outline,
            )
            ass_path_filter = ass_path.replace("\\", "/")
            if os.name == "nt" and ":" in ass_path_filter:
                ass_path_filter = ass_path_filter.replace(":", "\\:")
            subtitle_filter = f"ass=filename='{ass_path_filter}':original_size={res_string}"
        else:
            subtitle_filter = (
                f"subtitles='{srt_path_filter}':force_style="
                f"'FontName=Arial,FontSize={font_size},PrimaryColour=&HFFFFFF,"
                f"OutlineColour=&H000000,Outline={outline},WrapStyle=2'"
            )

    # Build ffmpeg filtergraph.
    filter_complex: str | None = None
    filter_script_path: str | None = None

    if adaptive_segment_stretch:
        # Per-segment trim + setpts + concat (align each utterance start frame).
        segments: list[dict[str, Any]] = []
        try:
            if os.path.exists(adaptive_plan_path):
                with open(adaptive_plan_path, "r", encoding="utf-8") as f:
                    payload = json.load(f) or {}
                segments = list(payload.get("segments") or [])
        except Exception:
            segments = []

        if not segments:
            raise FileNotFoundError(f"自适应模式缺少有效的 adaptive_plan.json: {adaptive_plan_path}")

        video_dur = _get_video_duration(input_video)
        if video_dur <= 0:
            video_dur = None

        # Write filter_complex_script to avoid command length limits.
        lines: list[str] = []
        v_labels: list[str] = []
        for idx, seg in enumerate(segments):
            check_cancelled()
            s0 = float(seg.get("src_start", 0.0) or 0.0)
            s1 = float(seg.get("src_end", 0.0) or 0.0)
            if s1 < s0:
                s0, s1 = s1, s0
            if video_dur is not None:
                s0 = max(0.0, min(s0, float(video_dur)))
                s1 = max(0.0, min(s1, float(video_dur)))
            src_dur = max(0.001, s1 - s0)
            out_dur = float(seg.get("target_duration", src_dur) or src_dur)
            if out_dur <= 0:
                out_dur = src_dur
            factor = out_dur / src_dur
            v = f"v{idx}"
            v_labels.append(f"[{v}]")
            lines.append(
                f"[0:v]trim=start={s0:.6f}:end={s1:.6f},setpts=(PTS-STARTPTS)*{factor:.9f}[{v}]"
            )

        concat_in = "".join(v_labels)
        lines.append(f"{concat_in}concat=n={len(v_labels)}:v=1:a=0[vcat]")

        if subtitles:
            lines.append(f"[vcat]{subtitle_filter}[v]")
        else:
            lines.append("[vcat]null[v]")

        filter_script_path = os.path.join(folder, "ffmpeg_filter_complex.txt")
        with open(filter_script_path, "w", encoding="utf-8") as f:
            f.write(";\n".join(lines) + "\n")
    else:
        # Legacy global speed-up path (video+audio).
        video_speed_filter = f"setpts=PTS/{speed_up}"
        audio_speed_filter = f"atempo={speed_up}"
        if subtitles:
            filter_complex = f"[0:v]{video_speed_filter},{subtitle_filter}[v];[1:a]{audio_speed_filter}[a]"
        else:
            filter_complex = f"[0:v]{video_speed_filter}[v];[1:a]{audio_speed_filter}[a]"
        
    video_encoder = "h264_nvenc" if use_nvenc else "libx264"
    if adaptive_segment_stretch:
        if not filter_script_path:
            raise RuntimeError("adaptive_segment_stretch=True 但未生成滤镜脚本")
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            input_video,
            "-i",
            input_audio,
            "-filter_complex_script",
            filter_script_path,
            "-map",
            "[v]",
            "-map",
            "1:a",
            # Adaptive mode uses per-segment setpts, so we preserve timestamps and
            # avoid enforcing a fixed FPS here (which can introduce dup/drop drift).
            "-vsync",
            "vfr",
            "-s",
            res_string,
            "-c:v",
            video_encoder,
            "-c:a",
            "aac",
            "-shortest",
            output_video,
            "-y",
        ]
    else:
        if filter_complex is None:
            raise RuntimeError("未生成 filter_complex")
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            input_video,
            "-i",
            input_audio,
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-r",
            str(fps),
            "-s",
            res_string,
            "-c:v",
            video_encoder,
            "-c:a",
            "aac",
            output_video,
            "-y",
        ]
    
    def _run_ffmpeg(cmd: list[str]) -> None:
        import sys
        import select
        import io

        # Capture ffmpeg output and forward to sys.stderr so Gradio can display it
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )  # noqa: S603
        try:
            # Non-blocking read from stderr (ffmpeg progress goes there)
            if proc.stderr is not None:
                import os
                fd = proc.stderr.fileno()
                try:
                    import fcntl
                    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                except (ImportError, OSError):
                    pass  # Windows or other platforms without fcntl

            buf = b""
            while True:
                check_cancelled()
                rc = proc.poll()

                # Read available stderr output
                if proc.stderr is not None:
                    try:
                        chunk = proc.stderr.read(4096)
                        if chunk:
                            buf += chunk
                            # Process complete lines
                            while b"\n" in buf or b"\r" in buf:
                                # Find the first line terminator
                                idx_n = buf.find(b"\n")
                                idx_r = buf.find(b"\r")
                                if idx_n == -1:
                                    idx = idx_r
                                elif idx_r == -1:
                                    idx = idx_n
                                else:
                                    idx = min(idx_n, idx_r)
                                line = buf[:idx]
                                buf = buf[idx + 1:]
                                if line.strip():
                                    try:
                                        text = line.decode("utf-8", errors="replace").rstrip()
                                        # Use \r prefix for progress lines (ffmpeg uses carriage return)
                                        if idx_r != -1 and (idx_n == -1 or idx_r < idx_n):
                                            sys.stderr.write("\r" + text + "\n")
                                        else:
                                            sys.stderr.write(text + "\n")
                                        sys.stderr.flush()
                                    except Exception:
                                        pass
                    except (BlockingIOError, IOError):
                        pass  # No data available yet

                if rc is not None:
                    # Process remaining output
                    if proc.stderr is not None:
                        try:
                            remaining = proc.stderr.read()
                            if remaining:
                                for line in remaining.decode("utf-8", errors="replace").splitlines():
                                    if line.strip():
                                        sys.stderr.write(line + "\n")
                                sys.stderr.flush()
                        except Exception:
                            pass
                    if rc != 0:
                        raise subprocess.CalledProcessError(rc, cmd)
                    return
                time.sleep(0.1)
        except BaseException:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass
            raise

    try:
        _run_ffmpeg(ffmpeg_command)
        sleep_with_cancel(1)
        logger.info(f"视频已生成: {output_video}")
    except subprocess.CalledProcessError as e:
        if use_nvenc:
            logger.warning(f"NVENC({video_encoder}) 失败，回退到 libx264: {e}")
            ffmpeg_command_fallback = ffmpeg_command.copy()
            try:
                idx = ffmpeg_command_fallback.index("-c:v") + 1
                ffmpeg_command_fallback[idx] = "libx264"
            except ValueError:
                # Should not happen: keep best-effort fallback.
                pass
            _run_ffmpeg(ffmpeg_command_fallback)
            sleep_with_cancel(1)
            logger.info(f"视频已生成(回退libx264): {output_video}")
            return

        logger.error(f"FFmpeg 失败: {e}")
        raise


def synthesize_all_video_under_folder(
    folder: str, 
    subtitles: bool = True, 
    speed_up: float = 1.2, 
    fps: int = 30, 
    resolution: str = '1080p',
    use_nvenc: bool = False,
    adaptive_segment_stretch: bool = False,
    bilingual_subtitle: bool = False,
) -> str:
    count = 0
    for root, _dirs, files in os.walk(folder):
        check_cancelled()
        if 'download.mp4' not in files:
            continue
        # Use the same freshness logic as synthesize_video() to avoid stale reuse.
        if _video_up_to_date(root, subtitles=subtitles, adaptive_segment_stretch=adaptive_segment_stretch):
            continue
        synthesize_video(
            root,
            subtitles=subtitles,
            bilingual_subtitle=bilingual_subtitle,
            speed_up=speed_up,
            fps=fps,
            resolution=resolution,
            use_nvenc=use_nvenc,
            adaptive_segment_stretch=adaptive_segment_stretch,
        )
        count += 1
    msg = f"视频合成完成: {folder}（处理 {count} 个文件）"
    logger.info(msg)
    return msg
