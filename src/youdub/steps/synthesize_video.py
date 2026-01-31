import json
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Any

import librosa
import numpy as np
from loguru import logger

from ..interrupts import check_cancelled, sleep_with_cancel
from ..utils import save_wav, save_wav_norm, valid_file


_AUDIO_COMBINED_META_NAME = ".audio_combined.json"
# Bump when the mixing/output semantics change.
# v2: audio_combined = audio_tts + instruments (NO original vocals mixed in)
# v3: match TTS loudness to original vocals peak + normalize output;
#     adaptive mode no longer phase-vocoder stretches TTS (avoid “回音/空洞”感)
_AUDIO_COMBINED_MIX_VERSION = 3

_VIDEO_META_NAME = ".video_synth.json"
# Bump when the video output semantics/config keys change.
# v3: subtitles style/scale tweaks (font size, wrap, original_size for libass)
# v4: make wrap heuristic more conservative; shrink default font size further
# v5: shrink non-bilingual subtitle font size further (1080p -> ~26)
# v6: shrink non-bilingual subtitle size further (1080p -> ~19) and reduce outline
# v7: use ASS for monolingual subtitles too (ensures font size is respected via PlayRes)
# v8: restore subtitle font scaling (1080p -> ~36) for readability
_VIDEO_META_VERSION = 8

# Video output audio encoding (keep high enough to avoid AAC artifacts).
_VIDEO_AUDIO_SAMPLE_RATE = 48000
_VIDEO_AUDIO_BITRATE = "128k"

# NVENC 并发限制：全自动多视频时避免同时起太多 h264_nvenc 实例导致失败/性能抖动
_NVENC_MAX_CONCURRENCY = 8
_NVENC_SEMAPHORE = threading.BoundedSemaphore(_NVENC_MAX_CONCURRENCY)


@contextmanager
def _nvenc_slot():
    acquired = False
    try:
        # Use short timeouts so cancellation remains responsive while waiting.
        while not acquired:
            check_cancelled()
            acquired = _NVENC_SEMAPHORE.acquire(timeout=0.2)
        yield
    finally:
        if acquired:
            try:
                _NVENC_SEMAPHORE.release()
            except Exception:
                # Best-effort; should never fail with BoundedSemaphore.
                pass


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


def _read_video_meta(folder: str) -> dict[str, Any] | None:
    meta_path = os.path.join(folder, _VIDEO_META_NAME)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else None
    except Exception:
        return None


def _write_video_meta(
    folder: str,
    *,
    subtitles: bool,
    bilingual_subtitle: bool,
    adaptive_segment_stretch: bool,
    speed_up: float,
    fps: int,
    resolution: str,
    use_nvenc: bool,
) -> None:
    meta_path = os.path.join(folder, _VIDEO_META_NAME)
    payload: dict[str, Any] = {
        "version": int(_VIDEO_META_VERSION),
        "subtitles": bool(subtitles),
        "bilingual_subtitle": bool(bilingual_subtitle),
        "adaptive_segment_stretch": bool(adaptive_segment_stretch),
        "speed_up": float(speed_up),
        "fps": int(fps),
        "resolution": str(resolution),
        "use_nvenc": bool(use_nvenc),
        "audio_sample_rate": int(_VIDEO_AUDIO_SAMPLE_RATE),
        "audio_bitrate": str(_VIDEO_AUDIO_BITRATE),
        "created_at": float(time.time()),
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"写入 video 合成元数据失败: {e}")


def _audio_combined_needs_rebuild(
    folder: str,
    *,
    adaptive_segment_stretch: bool,
    sample_rate: int = 24000,
) -> bool:
    audio_combined_path = os.path.join(folder, "audio_combined.wav")
    if not valid_file(audio_combined_path, min_bytes=44):
        logger.debug(f"audio_combined 需要重建: 文件不存在或过小")
        return True

    meta = _read_audio_combined_meta(folder)
    if not meta:
        logger.debug(f"audio_combined 需要重建: 元数据文件不存在")
        return True
    if int(meta.get("mix_version") or 0) != int(_AUDIO_COMBINED_MIX_VERSION):
        logger.debug(f"audio_combined 需要重建: mix_version 不匹配 ({meta.get('mix_version')} != {_AUDIO_COMBINED_MIX_VERSION})")
        return True
    if bool(meta.get("adaptive_segment_stretch")) != bool(adaptive_segment_stretch):
        logger.debug(f"audio_combined 需要重建: adaptive_segment_stretch 参数不匹配")
        return True
    if int(meta.get("sample_rate") or 0) != int(sample_rate):
        logger.debug(f"audio_combined 需要重建: sample_rate 不匹配 ({meta.get('sample_rate')} != {sample_rate})")
        return True

    # audio_combined depends on: TTS wavs, translation timeline/text (pauses), instruments, and vocals (for loudness match)
    deps = [
        os.path.join(folder, "translation.json"),
        os.path.join(folder, "wavs", ".tts_done.json"),
        os.path.join(folder, "audio_instruments.wav"),
        os.path.join(folder, "audio_vocals.wav"),
    ]
    if _is_stale(audio_combined_path, deps):
        logger.debug(f"audio_combined 需要重建: 依赖文件比 audio_combined.wav 更新")
        return True
    return False


def _video_up_to_date(
    folder: str,
    *,
    subtitles: bool,
    bilingual_subtitle: bool,
    adaptive_segment_stretch: bool,
    speed_up: float,
    fps: int,
    resolution: str,
    use_nvenc: bool,
    sample_rate: int = 24000,
) -> bool:
    output_video = os.path.join(folder, "video.mp4")
    if not valid_file(output_video, min_bytes=1024):
        logger.debug(f"video 过期原因: video.mp4 不存在或过小")
        return False

    # If audio needs rebuild (e.g. old mixing semantics), video is also stale.
    if _audio_combined_needs_rebuild(
        folder, adaptive_segment_stretch=adaptive_segment_stretch, sample_rate=sample_rate
    ):
        logger.debug(f"video 过期原因: audio_combined 需要重建")
        return False

    meta = _read_video_meta(folder)
    if not meta:
        logger.debug(f"video 过期原因: 元数据文件不存在")
        return False
    if int(meta.get("version") or 0) != int(_VIDEO_META_VERSION):
        logger.debug(f"video 过期原因: 元数据版本不匹配 ({meta.get('version')} != {_VIDEO_META_VERSION})")
        return False
    if bool(meta.get("subtitles")) != bool(subtitles):
        logger.debug(f"video 过期原因: subtitles 参数不匹配")
        return False
    if bool(meta.get("bilingual_subtitle")) != bool(bilingual_subtitle):
        logger.debug(f"video 过期原因: bilingual_subtitle 参数不匹配")
        return False
    if bool(meta.get("adaptive_segment_stretch")) != bool(adaptive_segment_stretch):
        logger.debug(f"video 过期原因: adaptive_segment_stretch 参数不匹配")
        return False
    try:
        if abs(float(meta.get("speed_up")) - float(speed_up)) > 1e-6:
            logger.debug(f"video 过期原因: speed_up 参数不匹配 ({meta.get('speed_up')} != {speed_up})")
            return False
    except Exception:
        logger.debug(f"video 过期原因: speed_up 参数解析失败")
        return False
    if int(meta.get("fps") or 0) != int(fps):
        logger.debug(f"video 过期原因: fps 参数不匹配 ({meta.get('fps')} != {fps})")
        return False
    if str(meta.get("resolution") or "") != str(resolution):
        logger.debug(f"video 过期原因: resolution 参数不匹配 ({meta.get('resolution')} != {resolution})")
        return False
    if bool(meta.get("use_nvenc")) != bool(use_nvenc):
        logger.debug(f"video 过期原因: use_nvenc 参数不匹配")
        return False
    if int(meta.get("audio_sample_rate") or 0) != int(_VIDEO_AUDIO_SAMPLE_RATE):
        logger.debug(f"video 过期原因: audio_sample_rate 不匹配 ({meta.get('audio_sample_rate')} != {_VIDEO_AUDIO_SAMPLE_RATE})")
        return False
    if str(meta.get("audio_bitrate") or "") != str(_VIDEO_AUDIO_BITRATE):
        logger.debug(f"video 过期原因: audio_bitrate 不匹配 ({meta.get('audio_bitrate')} != {_VIDEO_AUDIO_BITRATE})")
        return False

    deps = [
        os.path.join(folder, "download.mp4"),
        os.path.join(folder, "audio_combined.wav"),
    ]
    # NOTE:
    # Do NOT include _VIDEO_META_NAME in deps: we write it after video.mp4, which would
    # make the video look "stale" forever due to mtime ordering on modern filesystems.
    if adaptive_segment_stretch:
        deps.append(os.path.join(folder, "adaptive_plan.json"))
    if subtitles:
        tr = "translation_adaptive.json" if adaptive_segment_stretch else "translation.json"
        deps.append(os.path.join(folder, tr))
    if _is_stale(output_video, deps):
        logger.debug(f"video 过期原因: 依赖文件比 video.mp4 更新")
        return False
    return True


def _is_cjk_char(ch: str) -> bool:
    # CJK Unified Ideographs + Extension A (covers most Chinese/Japanese Kanji).
    return ("\u4e00" <= ch <= "\u9fff") or ("\u3400" <= ch <= "\u4dbf")


def _read_transcript_segments(folder: str) -> list[tuple[float, float, str]]:
    """Read transcript.json and return sorted (start,end,text) segments."""
    transcript_path = os.path.join(folder, "transcript.json")
    if not os.path.exists(transcript_path):
        return []
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            raw = json.load(f) or []
    except Exception:
        return []

    segs: list[tuple[float, float, str]] = []
    if not isinstance(raw, list):
        return segs
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            s0 = float(item.get("start", 0.0) or 0.0)
            s1 = float(item.get("end", 0.0) or 0.0)
        except Exception:
            continue
        if s1 < s0:
            s0, s1 = s1, s0
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        segs.append((s0, s1, text))
    segs.sort(key=lambda x: (x[0], x[1]))
    return segs


def _best_text_by_overlap(
    segs: list[tuple[float, float, str]],
    start: float,
    end: float,
    *,
    cursor: int,
) -> tuple[str, int]:
    """Pick the best-matching transcript text for [start,end] and return (text,next_cursor)."""
    if not segs:
        return "", cursor
    s0 = float(start)
    s1 = float(end)
    if s1 < s0:
        s0, s1 = s1, s0
    mid = (s0 + s1) / 2.0

    n = len(segs)
    j = max(0, min(int(cursor), n - 1))
    # Advance cursor so segs[j].end >= s0 when possible.
    while j < n and segs[j][1] < s0:
        j += 1
    if j >= n:
        j = n - 1

    # Evaluate a small window around j.
    best_text = ""
    best_ov = 0.0
    best_dist = float("inf")
    for k in range(max(0, j - 4), min(n, j + 6)):
        t0, t1, txt = segs[k]
        ov = max(0.0, min(s1, t1) - max(s0, t0))
        if ov > best_ov + 1e-9:
            best_ov = ov
            best_text = txt
            best_dist = abs(((t0 + t1) / 2.0) - mid)
            continue
        if ov <= 0.0 and best_ov <= 0.0:
            dist = abs(((t0 + t1) / 2.0) - mid)
            if dist < best_dist:
                best_dist = dist
                best_text = txt

    return best_text, j


def _ensure_bilingual_source_text(
    folder: str,
    translation: list[dict[str, Any]],
    *,
    adaptive_segment_stretch: bool,
) -> list[dict[str, Any]]:
    """
    Bilingual subtitle needs `text` (source) per segment.

    Old translation.json might miss `text`. We best-effort backfill it from transcript.json
    by time overlap, and write back translation.json so the fix persists.
    """
    if not translation:
        return translation

    def _missing_count(items: list[dict[str, Any]]) -> int:
        c = 0
        for it in items:
            if not str(it.get("text", "") or "").strip():
                c += 1
        return c

    # Prefer updating original translation.json (timestamps match transcript.json).
    src_items: list[dict[str, Any]] | None = None
    src_path: str | None = None
    if adaptive_segment_stretch:
        p = os.path.join(folder, "translation.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f) or []
                if isinstance(obj, list) and obj:
                    src_items = [x for x in obj if isinstance(x, dict)]
                    src_path = p
            except Exception:
                src_items = None
                src_path = None
    else:
        src_items = translation
        src_path = os.path.join(folder, "translation.json")

    if not src_items:
        return translation

    missing_before = _missing_count(src_items)
    if missing_before <= 0:
        return translation

    segs = _read_transcript_segments(folder)
    if not segs:
        logger.warning("双语字幕需要原文，但缺少 transcript.json 或内容为空；将只显示译文。")
        return translation

    filled = 0
    cursor = 0
    for it in src_items:
        if str(it.get("text", "") or "").strip():
            continue
        try:
            s0 = float(it.get("start", 0.0) or 0.0)
            s1 = float(it.get("end", 0.0) or 0.0)
        except Exception:
            continue
        txt, cursor = _best_text_by_overlap(segs, s0, s1, cursor=cursor)
        if txt:
            it["text"] = txt
            filled += 1

    if filled > 0 and src_path and src_items is not translation:
        # Persist backfill for translation.json so future runs work without this step.
        try:
            with open(src_path, "w", encoding="utf-8") as f:
                json.dump(src_items, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # If we patched translation.json (src_items != translation) in adaptive mode, copy text by index.
    if adaptive_segment_stretch and src_items is not translation and len(src_items) == len(translation):
        for i in range(len(translation)):
            if not str(translation[i].get("text", "") or "").strip():
                translation[i]["text"] = str(src_items[i].get("text", "") or "")

    missing_after = _missing_count(translation)
    if missing_after > 0:
        logger.warning(
            f"双语字幕原文缺失：已补齐 {filled} 条，但仍有 {missing_after} 条缺少原文，可能是旧文件格式或转写缺失。"
        )
    return translation


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


def _calc_subtitle_wrap_chars(
    width: int,
    font_size: int,
    *,
    en_font_scale: float = 0.75,
) -> tuple[int, int]:
    """
    Estimate wrap thresholds (character counts) based on output width and font size.

    Goal: keep the subtitle text inside a "safe" width even for portrait videos.
    This is a heuristic, not an exact text-measurement.
    """
    w = max(1, int(width))
    fs = max(1, int(font_size))
    # Horizontal safe margin: ~4% each side, at least 10px.
    # Keep conservative to avoid clipping with libass metrics differences.
    margin_x = max(10, int(round(w * 0.04)))
    safe_w = max(1, w - 2 * margin_x)

    # Empirical average glyph widths (Arial-ish + libass):
    # - CJK: close to a square, slightly narrower than font size
    # - Latin: narrower, plus spaces; tuned to match previous defaults on 16:9
    max_chars_zh = max(1, int(safe_w / (float(fs) * 0.90)))
    en_fs = max(1, int(round(float(fs) * float(en_font_scale))))
    max_chars_en = max(1, int(safe_w / (float(en_fs) * 0.65)))
    return max_chars_zh, max_chars_en


def _calc_subtitle_style_params(
    width: int,
    height: int,
    *,
    en_font_scale: float = 0.75,
) -> tuple[int, int, int, int, int]:
    """
    Calculate subtitle style parameters based on output resolution.

    Returns: (font_size, outline, margin_v, max_chars_zh, max_chars_en)
    """
    # Subtitle font size: readable across resolutions (1080p -> ~36).
    # Use the shorter edge to avoid huge fonts on portrait videos (e.g. 1080x1920).
    base_dim = min(int(width), int(height))
    # Keep monolingual/bilingual consistent; bilingual already uses two lines + wrapping.
    font_size = int(round(base_dim * 0.033))
    font_size = max(18, min(font_size, 120))
    outline = max(1, int(round(font_size / 20)))
    # Increase bottom margin to avoid clipping (esp. bilingual / multi-line).
    margin_v = max(12, int(round(font_size * 0.80)))
    max_chars_zh, max_chars_en = _calc_subtitle_wrap_chars(
        int(width), int(font_size), en_font_scale=float(en_font_scale)
    )
    return int(font_size), int(outline), int(margin_v), int(max_chars_zh), int(max_chars_en)


def _first_sentence(text: str, *, max_chars: int = 220) -> str:
    """
    Return a short preview of source text for bilingual subtitles.
    We intentionally keep ONLY one sentence to avoid dumping long paragraphs on screen.
    """
    raw = str(text or "").replace("\r", "").strip()
    if not raw:
        return ""

    # Keep first non-empty line only.
    first_line = ""
    for ln in raw.splitlines():
        ln = ln.strip()
        if ln:
            first_line = ln
            break
    if not first_line:
        return ""

    s = " ".join(first_line.split()).strip()
    if not s:
        return ""

    # Sentence terminators. For '.' we require a following whitespace/end/quote/bracket to avoid decimals.
    m = re.search(r'(?:[。！？!?]|\.)(?=(?:\s|$|["\')\]]))', s)
    if m:
        return s[: m.end()].strip()
    if len(s) > int(max_chars):
        return s[: int(max_chars)].rstrip()
    return s


def _generate_ass_header(
    *,
    play_res_x: int = 1920,
    play_res_y: int = 1080,
    font_name: str = "Arial",
    font_size: int = 49,
    outline: int = 4,
    margin_l: int = 10,
    margin_r: int = 10,
    margin_v: int = 10,
) -> list[str]:
    """Generate common ASS header lines."""
    return [
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


def generate_monolingual_ass(
    translation: list[dict[str, Any]],
    ass_path: str,
    *,
    speed_up: float = 1.0,
    play_res_x: int = 1920,
    play_res_y: int = 1080,
    font_name: str = "Arial",
    font_size: int = 26,
    outline: int = 2,
    margin_l: int = 10,
    margin_r: int = 10,
    margin_v: int = 20,
    max_chars_zh: int = 55,
) -> None:
    """Generate ASS subtitle file with only translated (Chinese) text."""
    # Split long segments for readability.
    translation = split_text(translation)

    lines: list[str] = _generate_ass_header(
        play_res_x=play_res_x,
        play_res_y=play_res_y,
        font_name=font_name,
        font_size=font_size,
        outline=outline,
        margin_l=margin_l,
        margin_r=margin_r,
        margin_v=margin_v,
    )

    for seg in translation:
        check_cancelled()
        zh_raw = str(seg.get("translation") or "").strip()
        if not zh_raw:
            continue

        start = format_timestamp_ass(float(seg.get("start", 0.0) or 0.0) / float(speed_up))
        end = format_timestamp_ass(float(seg.get("end", 0.0) or 0.0) / float(speed_up))

        zh = _ass_escape_text(wrap_text(zh_raw, max_chars_zh=max_chars_zh, max_chars_en=max_chars_zh)).replace(
            "\n", r"\N"
        )
        if not zh:
            continue

        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{zh}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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

    lines: list[str] = _generate_ass_header(
        play_res_x=play_res_x,
        play_res_y=play_res_y,
        font_name=font_name,
        font_size=font_size,
        outline=outline,
        margin_l=margin_l,
        margin_r=margin_r,
        margin_v=margin_v,
    )

    for seg in translation:
        check_cancelled()
        zh_raw = str(seg.get("translation") or "").strip()
        en_raw = _first_sentence(str(seg.get("text") or ""))
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
    max_chars_zh: int | None = None,
    max_chars_en: int | None = None,
) -> None:
    # 默认行为：按译文标点再切一遍，避免单条字幕太长。
    # 双语模式：translation.json 已在翻译阶段做过按句切分并尽量对齐原文；这里不再二次切分，
    # 否则会导致原文行重复（同一句英文被拆成多段译文时会重复显示）。
    if not bilingual_subtitle:
        translation = split_text(translation)

    if max_chars_zh is None:
        max_chars_zh = int(max_line_char)
    if max_chars_en is None:
        # Keep roughly consistent with the old defaults: 55 (zh) vs 100 (en).
        max_chars_en = max(1, int(round(float(max_line_char) * 1.8)))

    def _wrap_lines(s: str) -> list[str]:
        wrapped = wrap_text(s, max_chars_zh=int(max_chars_zh), max_chars_en=int(max_chars_en))
        return [ln for ln in wrapped.splitlines() if ln.strip()]

    with open(srt_path, 'w', encoding='utf-8') as f:
        seq = 0
        for line in translation:
            start = format_timestamp(line['start'] / speed_up)
            end = format_timestamp(line['end'] / speed_up)
            tr_text = str(line.get('translation', '') or '').strip()
            src_text = _first_sentence(str(line.get("text", "") or "")) if bilingual_subtitle else str(line.get('text', '') or '').strip()

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
    sample_rate: int = 24000,
) -> None:
    """
    生成 audio_combined.wav 和 audio_tts.wav
    
    如果 adaptive_segment_stretch=True:
        - 不对 TTS 做相位声码器 time_stretch（避免“回音/空洞”感）
        - 逐段拼接（可裁剪首尾静音）并插入短停顿，生成 translation_adaptive.json
        - 同时生成 adaptive_plan.json，用于逐段拉伸/裁剪原视频并 concat
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
    # 优先只取纯数字命名的分段 wav（避免误把 *_adjusted.wav 等中间文件算进去）
    wav_files = sorted([f for f in os.listdir(wavs_folder) if re.fullmatch(r"\d+\.wav", f)])
    if not wav_files:
        wav_files = sorted([f for f in os.listdir(wavs_folder) if f.endswith(".wav")])
    if not wav_files:
        raise ValueError(f"wavs 目录为空: {wavs_folder}")
    
    # 确保wav文件数量与translation段数一致
    expected_count = len(translation)
    if len(wav_files) < expected_count:
        raise FileNotFoundError(
            f"wav文件数量({len(wav_files)})少于翻译段数({expected_count})，请先确保TTS已完整生成"
        )
    
    if adaptive_segment_stretch:
        # 自适应模式：保持 TTS 基本原速（仅做静音裁剪），通过“拉伸视频段”来匹配每段音频时长。
        trim_top_db = 35.0
        gap_default_s = 0.12
        gap_clause_s = 0.18
        gap_sentence_s = 0.25

        def _gap_seconds_for_text(text: str) -> float:
            s = (text or "").strip()
            if not s:
                return float(gap_default_s)
            tail = s[-1]
            if tail in {"。", "！", "？", ".", "!", "?"}:
                return float(gap_sentence_s)
            if tail in {"，", "、", ",", ";", "；", ":", "："}:
                return float(gap_clause_s)
            return float(gap_default_s)

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

            tts_audio, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
            tts_audio = tts_audio.astype(np.float32, copy=False)

            # Trim leading/trailing silence to reduce dead air from TTS.
            if tts_audio.size > 0:
                try:
                    trimmed, _idx = librosa.effects.trim(tts_audio, top_db=float(trim_top_db))
                    if trimmed is not None and trimmed.size > 0:
                        tts_audio = trimmed.astype(np.float32, copy=False)
                except Exception:
                    pass

            # Original segment duration from ASR timestamps.
            orig_start = float(seg.get("start", 0.0) or 0.0)
            orig_end = float(seg.get("end", 0.0) or 0.0)
            if orig_end < orig_start:
                orig_start, orig_end = orig_end, orig_start

            # Output speech duration is exactly the (trimmed) TTS segment duration.
            target_samples = int(tts_audio.shape[0])
            target_duration = float(target_samples) / float(sample_rate) if target_samples > 0 else 0.0

            if target_samples > 0:
                audio_chunks.append(tts_audio.astype(np.float32, copy=False))

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
                else:
                    # Overlap/adjacent: take a small tail slice from the previous segment as "pause" visuals.
                    tail = 0.08
                    pause_src_end = orig_end
                    pause_src_start = max(0.0, pause_src_end - tail)
                pause_duration = _gap_seconds_for_text(str(seg.get("translation") or ""))

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
        plan_payload = {"segments": adaptive_plan}
        with open(adaptive_plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"已生成 adaptive_plan.json: {adaptive_plan_path}")
        
    else:
        # 非自适应模式：按顺序拼接
        audio_segments: list[np.ndarray] = []
        for _i, wav_file in enumerate(wav_files[:len(translation)]):
            check_cancelled()
            wav_path = os.path.join(wavs_folder, wav_file)
            if not os.path.exists(wav_path):
                logger.warning(f"TTS音频文件不存在: {wav_path}")
                continue
            
            try:
                audio, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
                audio_segments.append(audio.astype(np.float32, copy=False))
            except Exception as e:
                logger.warning(f"加载TTS音频失败 {wav_path}: {e}")
                continue
        
        if not audio_segments:
            raise ValueError("没有有效的TTS音频片段")
        
        audio_tts = np.concatenate(audio_segments).astype(np.float32, copy=False)
    
    # 将 TTS 音量匹配到原始人声峰值（确保音量与原视频相近）
    audio_vocals_path = os.path.join(folder, 'audio_vocals.wav')
    if os.path.exists(audio_vocals_path) and len(audio_tts) > 0:
        try:
            check_cancelled()
            vocal_wav, _ = librosa.load(audio_vocals_path, sr=sample_rate, mono=True)
            tts_peak = max(abs(float(np.max(audio_tts))), abs(float(np.min(audio_tts))))
            if tts_peak > 0.0 and len(vocal_wav) > 0:
                vocal_peak = max(abs(float(np.max(vocal_wav))), abs(float(np.min(vocal_wav))))
                if vocal_peak > 0.0:
                    scale = vocal_peak / tts_peak
                    audio_tts = (audio_tts * np.float32(scale)).astype(np.float32)
                    logger.info(f"TTS 音量已匹配到原始人声峰值 (scale={scale:.3f})")
        except Exception as e:
            logger.warning(f"匹配音量失败（将跳过缩放）: {e}")

    # 保存音量匹配后的音轨，便于排查/试听
    check_cancelled()
    save_wav(audio_tts.astype(np.float32, copy=False), audio_tts_path, sample_rate=sample_rate)
    logger.info(f"已生成 audio_tts.wav: {audio_tts_path}")
    
    # 混合 TTS 音频和背景伴奏（不混入原人声，避免回声/双声）
    check_cancelled()
    if os.path.exists(audio_instruments_path):
        try:
            # 加载伴奏
            instruments, _ = librosa.load(audio_instruments_path, sr=sample_rate, mono=True)
            instruments = instruments.astype(np.float32, copy=False)
            
            # 对齐长度（以TTS音频为准）
            tts_len = len(audio_tts)
            instruments_len = len(instruments)
            
            # 如果伴奏更长，裁剪；如果更短，补零
            if instruments_len < tts_len:
                instruments = np.pad(instruments, (0, tts_len - instruments_len), mode="constant")
            else:
                instruments = instruments[:tts_len]
            
            # 混合：TTS + 伴奏（1:1 混合，与 origin/master 保持一致）
            audio_combined = (audio_tts + instruments).astype(np.float32, copy=False)
            
            # 归一化到峰值，确保音量正常且不削波
            save_wav_norm(audio_combined, audio_combined_path, sample_rate=sample_rate)
            _write_audio_combined_meta(
                folder,
                adaptive_segment_stretch=adaptive_segment_stretch,
                sample_rate=sample_rate,
            )
            logger.info(f"已生成 audio_combined.wav: {audio_combined_path}")
        except Exception as e:
            logger.warning(f"混合背景音乐失败，仅使用TTS音频: {e}")
            save_wav_norm(audio_tts, audio_combined_path, sample_rate=sample_rate)
            _write_audio_combined_meta(
                folder,
                adaptive_segment_stretch=adaptive_segment_stretch,
                sample_rate=sample_rate,
            )
    else:
        # 没有背景音乐，直接使用TTS音频（归一化以确保音量正常）
        save_wav_norm(audio_tts, audio_combined_path, sample_rate=sample_rate)
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

    # 缓存元数据中的“有效参数”（自适应模式下 speed_up/fps 不参与输出）
    meta_speed_up = 1.0 if adaptive_segment_stretch else float(speed_up)
    meta_fps = 0 if adaptive_segment_stretch else int(fps)

    # NOTE:
    # 以前只要 video.mp4 存在就直接跳过，这会导致：
    # - 混音逻辑修复后仍复用旧的 audio_combined/video（用户听感依旧“回声/双声”）
    # - translation/音频更新后视频也不重建
    # 因此这里改为基于 mtime + 元数据判断是否需要重建。
    if _video_up_to_date(
        folder,
        subtitles=subtitles,
        bilingual_subtitle=bilingual_subtitle,
        adaptive_segment_stretch=adaptive_segment_stretch,
        speed_up=meta_speed_up,
        fps=meta_fps,
        resolution=resolution,
        use_nvenc=use_nvenc,
    ):
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
            _ensure_audio_combined(folder, adaptive_segment_stretch=True)
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
            _ensure_audio_combined(folder, adaptive_segment_stretch=False)
    
    # 字幕在自适应模式下应跟随 translation_adaptive.json。
    translation_path_to_use = translation_adaptive_path if adaptive_segment_stretch else translation_path

    missing = [p for p in (translation_path_to_use, input_audio, input_video) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"缺少合成视频所需文件：{missing}")
    
    with open(translation_path_to_use, 'r', encoding='utf-8') as f:
        translation = json.load(f)

    if subtitles and bilingual_subtitle:
        translation = _ensure_bilingual_source_text(
            folder, translation, adaptive_segment_stretch=adaptive_segment_stretch
        )

    check_cancelled()
    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    res_string = f'{width}x{height}'
    
    font_size, outline, margin_v, max_chars_zh, max_chars_en = _calc_subtitle_style_params(
        width, height, en_font_scale=0.75
    )

    srt_path = os.path.join(folder, 'subtitles.srt')
    if subtitles:
        # 自适应模式下，translation_adaptive.json 已经在输出时间轴，不应再做 speed_up 缩放。
        subs_speed = 1.0 if adaptive_segment_stretch else speed_up
        generate_srt(
            translation,
            srt_path,
            subs_speed,
            bilingual_subtitle=bilingual_subtitle,
            max_chars_zh=max_chars_zh,
            max_chars_en=max_chars_en,
        )

    srt_path_filter = srt_path.replace('\\', '/')
    
    if os.name == 'nt' and ':' in srt_path_filter:
        srt_path_filter = srt_path_filter.replace(':', '\\:')

    subtitle_filter = ""
    if subtitles:
        ass_path = os.path.join(folder, "subtitles.ass")
        if bilingual_subtitle:
            generate_bilingual_ass(
                translation,
                ass_path,
                speed_up=(1.0 if adaptive_segment_stretch else speed_up),
                play_res_x=width,
                play_res_y=height,
                font_name="Arial",
                font_size=font_size,
                outline=outline,
                margin_v=margin_v,
                max_chars_zh=max_chars_zh,
                max_chars_en=max_chars_en,
            )
        else:
            # Use ASS for monolingual too (ensures font size is respected via PlayRes).
            generate_monolingual_ass(
                translation,
                ass_path,
                speed_up=(1.0 if adaptive_segment_stretch else speed_up),
                play_res_x=width,
                play_res_y=height,
                font_name="Arial",
                font_size=font_size,
                outline=outline,
                margin_v=margin_v,
                max_chars_zh=max_chars_zh,
            )
        ass_path_filter = ass_path.replace("\\", "/")
        if os.name == "nt" and ":" in ass_path_filter:
            ass_path_filter = ass_path_filter.replace(":", "\\:")
        subtitle_filter = f"ass=filename='{ass_path_filter}':original_size={res_string}"

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

        post_filters: list[str] = [f"scale={width}:{height}"]
        if subtitles:
            post_filters.append(subtitle_filter)
        lines.append(f"[vcat]{','.join(post_filters)}[v]")

        filter_script_path = os.path.join(folder, "ffmpeg_filter_complex.txt")
        with open(filter_script_path, "w", encoding="utf-8") as f:
            f.write(";\n".join(lines) + "\n")
    else:
        # Legacy global speed-up path (video+audio).
        video_speed_filter = f"setpts=PTS/{speed_up}"
        audio_speed_filter = f"atempo={speed_up}"
        v_filters: list[str] = [video_speed_filter, f"scale={width}:{height}"]
        if subtitles:
            v_filters.append(subtitle_filter)
        filter_complex = f"[0:v]{','.join(v_filters)}[v];[1:a]{audio_speed_filter}[a]"
        
    video_encoder = "h264_nvenc" if use_nvenc else "libx264"
    # 音频编码参数：上采样到 48kHz 并使用较高码率以保证音质
    audio_sample_rate = str(_VIDEO_AUDIO_SAMPLE_RATE)
    audio_bitrate = str(_VIDEO_AUDIO_BITRATE)
    
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
            "-c:v",
            video_encoder,
            "-c:a",
            "aac",
            "-ar",
            audio_sample_rate,
            "-b:a",
            audio_bitrate,
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
            "-c:v",
            video_encoder,
            "-c:a",
            "aac",
            "-ar",
            audio_sample_rate,
            "-b:a",
            audio_bitrate,
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
        if use_nvenc:
            with _nvenc_slot():
                _run_ffmpeg(ffmpeg_command)
        else:
            _run_ffmpeg(ffmpeg_command)
        sleep_with_cancel(1)
        _write_video_meta(
            folder,
            subtitles=subtitles,
            bilingual_subtitle=bilingual_subtitle,
            adaptive_segment_stretch=adaptive_segment_stretch,
            speed_up=meta_speed_up,
            fps=meta_fps,
            resolution=resolution,
            use_nvenc=use_nvenc,
        )
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
            _write_video_meta(
                folder,
                subtitles=subtitles,
                bilingual_subtitle=bilingual_subtitle,
                adaptive_segment_stretch=adaptive_segment_stretch,
                speed_up=meta_speed_up,
                fps=meta_fps,
                resolution=resolution,
                use_nvenc=False,
            )
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
    auto_upload_video: bool = False,
) -> str:
    count = 0
    for root, _dirs, files in os.walk(folder):
        check_cancelled()
        if 'download.mp4' not in files:
            continue
        # Use the same freshness logic as synthesize_video() to avoid stale reuse.
        meta_speed_up = 1.0 if adaptive_segment_stretch else float(speed_up)
        meta_fps = 0 if adaptive_segment_stretch else int(fps)
        up_to_date = _video_up_to_date(
            root,
            subtitles=subtitles,
            bilingual_subtitle=bilingual_subtitle,
            adaptive_segment_stretch=adaptive_segment_stretch,
            speed_up=meta_speed_up,
            fps=meta_fps,
            resolution=resolution,
            use_nvenc=use_nvenc,
        )
        if not up_to_date:
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

        if auto_upload_video:
            # Enqueue in background so it won't block synthesizing other videos.
            try:
                from .upload import upload_video_async  # local import to avoid hard dependency for non-upload users

                upload_video_async(root)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"加入B站后台上传队列失败（忽略）: {exc}")
    msg = f"视频合成完成: {folder}（处理 {count} 个文件）"
    logger.info(msg)
    return msg
