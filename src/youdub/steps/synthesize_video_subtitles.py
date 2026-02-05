from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from loguru import logger

from ..interrupts import check_cancelled


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


def split_text(input_data: list[dict[str, Any]], punctuations: list[str] | None = None) -> list[dict[str, Any]]:
    puncts = set(punctuations or ["，", "；", "：", "。", "？", "！", "\n", "”"])

    def is_punctuation(char: str) -> bool:
        return char in puncts

    output_data: list[dict[str, Any]] = []
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
            if i < len(text) - 1 and is_punctuation(text[i + 1]):
                continue

            sentence = text[sentence_start : i + 1]
            sentence_end = start + duration_per_char * len(sentence)

            output_data.append(
                {
                    "start": round(start, 3),
                    "end": round(sentence_end, 3),
                    "text": original_text,
                    "translation": sentence,
                    "speaker": speaker,
                }
            )

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


def _wrap_cjk(text: str, max_chars: int, *, min_tail: int = 5) -> list[str]:
    """Wrap CJK text by character count, avoiding very short tail lines.

    If the last segment would be shorter than `min_tail`, merge it with the previous line.
    """
    if max_chars <= 0:
        return [text]
    s = (text or "").strip()
    if not s:
        return []
    lines = [s[i : i + max_chars] for i in range(0, len(s), max_chars)]
    # Merge short tail into previous line to avoid orphan punctuation.
    if len(lines) >= 2 and len(lines[-1]) < min_tail:
        lines[-2] = lines[-2] + lines[-1]
        lines.pop()
    return lines


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

    # Rebalance: avoid an overly short tail line by shifting words backward.
    # This keeps line count unchanged but reduces "ragged" wrapping.
    if len(lines) >= 2 and max_chars > 0:
        min_tail = max(8, int(round(float(max_chars) * 0.55)))
        min_prev = max(8, int(round(float(max_chars) * 0.45)))
        for i in range(len(lines) - 1, 0, -1):
            while True:
                tail = lines[i].strip()
                if len(tail) >= min_tail:
                    break
                prev_words = lines[i - 1].strip().split(" ")
                if len(prev_words) <= 1:
                    break
                w = prev_words[-1]
                candidate_tail = f"{w} {tail}".strip() if tail else w
                if len(candidate_tail) > max_chars:
                    break
                candidate_prev = " ".join(prev_words[:-1]).strip()
                if len(candidate_prev) < min_prev:
                    break
                lines[i] = candidate_tail
                lines[i - 1] = candidate_prev

    return [ln for ln in lines if ln and ln.strip()]


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


def _calc_subtitle_wrap_chars(width: int, font_size: int, *, en_font_scale: float = 0.75) -> tuple[int, int]:
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
    # 0.65 was overly conservative for common English subtitles and caused excessive wrapping.
    # Use a slightly smaller average glyph width so long English sentences don't explode into 5-6 lines.
    max_chars_en = max(1, int(safe_w / (float(en_fs) * 0.58)))
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

    Scaling logic:
    - Use the shorter edge (height for landscape, width for portrait) as the base dimension.
    - Apply a fixed percentage so the subtitle maintains consistent visual proportion across resolutions.
    - 1080p landscape → ~49 (4.5% of 1080); 4K landscape → ~97; 720p landscape → ~32.
    - Portrait uses a slightly smaller ratio (4.0%) since width is limited.
    """
    w = max(1, int(width))
    h = max(1, int(height))

    # Base dimension is always the shorter edge.
    base_dim = min(w, h)

    # Landscape: 4.5% of height; Portrait: 4.0% of width (slightly smaller to fit limited width).
    if w >= h:
        font_scale = 0.045
    else:
        font_scale = 0.040

    font_size = int(round(float(base_dim) * float(font_scale)))
    font_size = max(18, min(font_size, 120))

    # Outline: ~8% of font size for strong contrast (49 -> 4, 32 -> 3).
    outline = max(1, int(round(float(font_size) * 0.08)))
    # Bottom margin: ~20% of font size, minimum 10.
    margin_v = max(10, int(round(float(font_size) * 0.20)))
    max_chars_zh, max_chars_en = _calc_subtitle_wrap_chars(int(w), int(font_size), en_font_scale=float(en_font_scale))
    return int(font_size), int(outline), int(margin_v), int(max_chars_zh), int(max_chars_en)


_TOKEN_RE = re.compile(r"[0-9A-Za-z_]+(?:['’][0-9A-Za-z_]+)?")


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _tokenize_for_count(text: str) -> list[str]:
    # Normalize apostrophes so tokenization is stable.
    s = _normalize_ws(text).replace("’", "'")
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(s)]


def _unit_weight(text: str) -> int:
    """
    Weight for balancing clause grouping.

    Prefer ASCII token count (English-like); fallback to character length for CJK/others.
    """
    toks = _tokenize_for_count(text)
    if toks:
        return int(len(toks))
    s = _normalize_ws(text)
    return int(max(1, len(s)))


def _split_words_to_count(text: str, target_count: int) -> list[str]:
    """Fallback: split by words as evenly as possible."""
    if target_count <= 0:
        return []
    s = _normalize_ws(text)
    if not s:
        return [""] * target_count
    if target_count == 1:
        return [s]

    words = s.split(" ")
    if not words:
        return [""] * target_count

    base = len(words) // target_count
    rem = len(words) % target_count
    out: list[str] = []
    idx = 0
    for i in range(target_count):
        size = base + (1 if i < rem else 0)
        if size <= 0:
            out.append("")
            continue
        out.append(" ".join(words[idx : idx + size]).strip())
        idx += size
    return out


def _group_clauses_evenly(clauses: list[str], target_count: int) -> list[str]:
    """Group clauses in order so each group has similar weight."""
    cleaned = [c.strip() for c in clauses if c and c.strip()]
    if target_count <= 1:
        return [" ".join(cleaned).strip()] if cleaned else []
    if not cleaned:
        return [""] * target_count
    if len(cleaned) < target_count:
        # Not enough clauses to make target_count groups.
        return []

    weights = [_unit_weight(c) for c in cleaned]
    total = sum(weights) or 1
    target = float(total) / float(target_count)

    out: list[str] = []
    buf: list[str] = []
    buf_cnt = 0
    remaining_groups = int(target_count)

    for i, clause in enumerate(cleaned):
        w = int(weights[i])
        remaining_clauses = len(cleaned) - i

        # If remaining clauses == remaining groups, we must allocate 1 clause per group.
        if buf and remaining_clauses == remaining_groups:
            out.append(" ".join(buf).strip())
            buf = [clause]
            buf_cnt = w
            remaining_groups -= 1
            continue

        if not buf:
            buf = [clause]
            buf_cnt = w
            continue

        # Last group: take everything remaining.
        if remaining_groups <= 1:
            buf.append(clause)
            buf_cnt += w
            continue

        # Close group when reaching target.
        if float(buf_cnt) >= target:
            out.append(" ".join(buf).strip())
            buf = [clause]
            buf_cnt = w
            remaining_groups -= 1
            continue

        # If adding this clause overshoots target a lot, close early if current buf isn't too small.
        if (float(buf_cnt + w) > target) and (buf_cnt >= max(3, int(target * 0.6))):
            out.append(" ".join(buf).strip())
            buf = [clause]
            buf_cnt = w
            remaining_groups -= 1
            continue

        buf.append(clause)
        buf_cnt += w

    if buf:
        out.append(" ".join(buf).strip())

    # Defensive: if grouping drifted, signal failure so caller can fallback.
    if len(out) != int(target_count):
        return []
    return out


def _bilingual_source_text(
    text: str,
    *,
    max_words: int = 18,
    max_chars: int = 110,
) -> str:
    """
    Source text rendering for bilingual subtitles.

    Strategy (mirrors `scripts/resplit_by_diarization.py` step 6):
    - Split by sentence terminators (.?! and Chinese equivalents).
    - If a single sentence is too long, split into clause groups as evenly as possible.
    - Unlike the legacy logic, we do NOT truncate to the first sentence.
    """
    raw = str(text or "").replace("\r", "").strip()
    if not raw:
        return ""

    s = _normalize_ws(raw)
    if not s:
        return ""

    # Lazy import to avoid pulling translation stack unless needed.
    try:
        from .translate import _split_source_text_into_sentences, _split_source_text_relaxed
    except Exception:
        return s

    sentences = [x.strip() for x in _split_source_text_into_sentences(s) if str(x).strip()]
    if not sentences:
        sentences = [s]

    rendered: list[str] = []
    for sent in sentences:
        sent = _normalize_ws(sent)
        if not sent:
            continue

        # Prefer token-based counting for English. If no tokens (e.g. CJK), fall back to chars.
        token_count = len(_tokenize_for_count(sent))
        wc = int(token_count) if token_count > 0 else int(len(sent.split()))
        if wc <= 0:
            wc = 1

        too_long = (max_words > 0 and wc > int(max_words)) or (max_chars > 0 and len(sent) > int(max_chars))
        if not too_long:
            rendered.append(sent)
            continue

        # Decide how many chunks we need.
        target_count = 2
        if max_words > 0:
            target_count = max(target_count, int(math.ceil(float(wc) / float(max_words))))
        if max_chars > 0:
            target_count = max(target_count, int(math.ceil(float(len(sent)) / float(max_chars))))

        clauses = [c.strip() for c in _split_source_text_relaxed(sent) if str(c).strip()]
        grouped = _group_clauses_evenly(clauses, target_count)
        if grouped:
            # Keep as a single paragraph; actual line wrapping is handled later by `wrap_text`.
            # Injecting newlines here often causes double-wrapping (6+ lines with very short lines).
            rendered.append(" ".join(grouped).strip())
            continue

        # Fallback: word-based even split (guarantees count).
        rendered.append(" ".join([p for p in _split_words_to_count(sent, target_count) if p.strip()]).strip())

    return " ".join([x for x in rendered if x]).strip()


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

        start = format_timestamp_ass(float(seg.get("start", 0.0) or 0.0))
        end = format_timestamp_ass(float(seg.get("end", 0.0) or 0.0))

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
        en_raw = _bilingual_source_text(str(seg.get("text") or ""))
        if not zh_raw and not en_raw:
            continue

        start = format_timestamp_ass(float(seg.get("start", 0.0) or 0.0))
        end = format_timestamp_ass(float(seg.get("end", 0.0) or 0.0))

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

    with open(srt_path, "w", encoding="utf-8") as f:
        seq = 0
        for line in translation:
            start = format_timestamp(line["start"])
            end = format_timestamp(line["end"])
            tr_text = str(line.get("translation", "") or "").strip()
            src_text = (
                _bilingual_source_text(str(line.get("text", "") or ""))
                if bilingual_subtitle
                else str(line.get("text", "") or "").strip()
            )

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
            wrapped_text = "\n".join(lines)

            f.write(f"{seq}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{wrapped_text}\n\n")

