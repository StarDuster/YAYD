from __future__ import annotations

import html
import json
import os
import re
from typing import Any

from loguru import logger

from ...interrupts import check_cancelled
from .segments import _assign_speakers_by_overlap, generate_speaker_audio, merge_segments


def _read_download_info(folder: str) -> dict[str, Any] | None:
    info_path = os.path.join(folder, "download.info.json")
    if not os.path.exists(info_path):
        return None
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _pick_preferred_manual_subtitle_lang(info: dict[str, Any]) -> str | None:
    subtitles = info.get("subtitles")
    if not isinstance(subtitles, dict) or not subtitles:
        return None

    keys = [str(k) for k in subtitles.keys() if str(k).strip()]
    if not keys:
        return None

    preferred: list[str] = []
    for k in (info.get("language"), info.get("original_language")):
        if isinstance(k, str) and k.strip():
            preferred.append(k.strip())

    # 1) Prefer the video's language / original_language when available.
    for p in preferred:
        if p in subtitles:
            return p

    # 2) Prefer English (en / en-*)
    if "en" in subtitles:
        return "en"
    for k in keys:
        if k.lower().startswith("en"):
            return k

    # 3) Fallback: first available language key (stable).
    return keys[0]


_VTT_TS_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})\.(\d{3})$")
_SRT_TS_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2}),(\d{3})$")
_SUB_TAG_RE = re.compile(r"<[^>]+>")
_SUB_WS_RE = re.compile(r"\s+")


def _parse_vtt_timestamp(ts: str) -> float | None:
    s = (ts or "").strip()
    m = _VTT_TS_RE.match(s)
    if not m:
        return None
    hh = int(m.group(1) or 0)
    mm = int(m.group(2) or 0)
    ss = int(m.group(3) or 0)
    ms = int(m.group(4) or 0)
    return float(hh * 3600 + mm * 60 + ss) + float(ms) / 1000.0


def _parse_srt_timestamp(ts: str) -> float | None:
    s = (ts or "").strip()
    m = _SRT_TS_RE.match(s)
    if not m:
        return None
    hh = int(m.group(1) or 0)
    mm = int(m.group(2) or 0)
    ss = int(m.group(3) or 0)
    ms = int(m.group(4) or 0)
    return float(hh * 3600 + mm * 60 + ss) + float(ms) / 1000.0


def _clean_subtitle_text(text: str) -> str:
    s = (text or "").replace("\r", "\n")
    # Unescape entities first so `&lt;...&gt;` can be stripped by tag regex.
    try:
        s = html.unescape(s)
    except Exception:
        pass
    # Strip basic tags (WebVTT often contains <c>, <v>, <i>, etc).
    s = _SUB_TAG_RE.sub("", s)
    # Collapse whitespace across lines.
    s = _SUB_WS_RE.sub(" ", s).strip()
    return s


def _parse_vtt_segments(content: str) -> list[dict[str, Any]]:
    lines = (content or "").splitlines()
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0].lstrip("\ufeff")

    segs: list[dict[str, Any]] = []
    i = 0
    n = len(lines)

    # Skip header (WEBVTT + optional metadata until blank line).
    if i < n and lines[i].strip().upper().startswith("WEBVTT"):
        i += 1
        while i < n and lines[i].strip() != "":
            i += 1
    while i < n and lines[i].strip() == "":
        i += 1

    while i < n:
        check_cancelled()
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # Skip NOTE/STYLE/REGION blocks.
        if line.startswith("NOTE") or line.startswith("STYLE") or line.startswith("REGION"):
            i += 1
            while i < n and lines[i].strip() != "":
                i += 1
            continue

        # Optional cue identifier line.
        if "-->" not in line and (i + 1) < n and ("-->" in lines[i + 1]):
            i += 1
            line = lines[i].strip()

        if "-->" not in line:
            i += 1
            continue

        left, right = line.split("-->", 1)
        start_ts = left.strip().split()[0] if left.strip() else ""
        end_ts = right.strip().split()[0] if right.strip() else ""
        start_s = _parse_vtt_timestamp(start_ts)
        end_s = _parse_vtt_timestamp(end_ts)
        if start_s is None or end_s is None:
            i += 1
            continue

        i += 1
        text_lines: list[str] = []
        while i < n and lines[i].strip() != "":
            text_lines.append(lines[i])
            i += 1

        text = _clean_subtitle_text("\n".join(text_lines))
        if text and end_s > start_s:
            segs.append(
                {
                    "start": round(float(start_s), 3),
                    "end": round(float(end_s), 3),
                    "text": text,
                    "speaker": "SPEAKER_00",
                }
            )

        while i < n and lines[i].strip() == "":
            i += 1

    return segs


def _parse_srt_segments(content: str) -> list[dict[str, Any]]:
    lines = (content or "").splitlines()
    if lines and lines[0].startswith("\ufeff"):
        lines[0] = lines[0].lstrip("\ufeff")

    segs: list[dict[str, Any]] = []
    i = 0
    n = len(lines)
    while i < n:
        check_cancelled()
        if not lines[i].strip():
            i += 1
            continue

        # Optional numeric index line
        if lines[i].strip().isdigit():
            i += 1
            if i >= n:
                break

        if "-->" not in lines[i]:
            i += 1
            continue

        left, right = lines[i].split("-->", 1)
        start_ts = left.strip()
        end_ts = right.strip().split()[0] if right.strip() else ""
        start_s = _parse_srt_timestamp(start_ts)
        end_s = _parse_srt_timestamp(end_ts)
        i += 1

        text_lines: list[str] = []
        while i < n and lines[i].strip() != "":
            text_lines.append(lines[i])
            i += 1

        if start_s is None or end_s is None:
            continue

        text = _clean_subtitle_text("\n".join(text_lines))
        if text and end_s > start_s:
            segs.append(
                {
                    "start": round(float(start_s), 3),
                    "end": round(float(end_s), 3),
                    "text": text,
                    "speaker": "SPEAKER_00",
                }
            )

    return segs


def _find_subtitle_file(folder: str, preferred_lang: str | None) -> tuple[str | None, str | None]:
    """
    Find a parseable subtitle file written by yt-dlp under `folder`.

    Returns: (file_path, detected_lang)
    """
    exts = (".vtt", ".srt")

    def _candidate(lang: str, ext: str) -> str:
        return os.path.join(folder, f"download.{lang}{ext}")

    if preferred_lang:
        for ext in exts:
            p = _candidate(preferred_lang, ext)
            if os.path.exists(p):
                return p, preferred_lang

    # Fallback: scan folder for any download.*.vtt/srt
    try:
        names = [
            n
            for n in os.listdir(folder)
            if n.startswith("download.")
            and n.lower().endswith(exts)
            # Ignore auto captions artifacts (yt-dlp typically uses *.auto.vtt).
            and (".auto." not in n.lower())
            and (not n.lower().endswith(".auto.vtt"))
            and (not n.lower().endswith(".auto.srt"))
        ]
    except Exception:
        names = []

    def _lang_from_name(name: str) -> str:
        low = name
        for ext in exts:
            if low.lower().endswith(ext):
                low = low[: -len(ext)]
                break
        # download.<lang>
        if low.startswith("download."):
            return low[len("download.") :]
        return ""

    candidates: list[tuple[str, str]] = []
    for n in sorted(names):
        lang = _lang_from_name(n).strip()
        candidates.append((os.path.join(folder, n), lang))

    if not candidates:
        return None, None

    if preferred_lang:
        # Case-insensitive match.
        for p, lang in candidates:
            if lang.lower() == preferred_lang.lower():
                return p, lang
        for p, lang in candidates:
            ll = lang.lower()
            pl = preferred_lang.lower()
            if ll.startswith(pl) and (len(ll) == len(pl) or ll[len(pl)] in {"-", "_"}):
                return p, lang

    # Prefer English if present.
    for p, lang in candidates:
        ll = lang.lower()
        if ll == "en" or (ll.startswith("en") and (len(ll) == 2 or ll[2] in {"-", "_"})):
            return p, lang

    return candidates[0]


def _parse_subtitle_file(path: str) -> list[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return []

    low = path.lower()
    if low.endswith(".vtt"):
        return _parse_vtt_segments(content)
    if low.endswith(".srt"):
        return _parse_srt_segments(content)
    return []


def _is_youtube_subtitle_transcript(transcript: Any, *, lang: str | None = None) -> bool:
    if not isinstance(transcript, list) or not transcript:
        return False
    # Best-effort marker: we store source/lang per-segment.
    head = transcript[0]
    if not isinstance(head, dict):
        return False
    if str(head.get("source") or "") != "youtube_subtitles":
        return False
    if lang is None:
        return True
    return str(head.get("subtitle_lang") or "") == str(lang)


def _save_youtube_subtitle_transcript(
    folder: str,
    segments: list[dict[str, Any]],
    *,
    subtitle_lang: str | None,
    turns: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    transcript: list[dict[str, Any]] = segments
    for seg in transcript:
        seg["source"] = "youtube_subtitles"
        seg["subtitle_lang"] = str(subtitle_lang or "")

    if turns:
        _assign_speakers_by_overlap(transcript, turns, default_speaker="SPEAKER_00")
    else:
        for seg in transcript:
            seg["speaker"] = str(seg.get("speaker") or "SPEAKER_00")

    check_cancelled()
    transcript = merge_segments(transcript)

    check_cancelled()
    transcript_path = os.path.join(folder, "transcript.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    logger.info(f"已保存字幕转录: {transcript_path}")

    # Best-effort: refresh speaker reference files.
    try:
        check_cancelled()
        generate_speaker_audio(folder, transcript)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(f"生成说话人参考音频失败（忽略）: {exc}")

    return transcript


def _invalidate_downstream_cache_for_new_transcript(folder: str) -> None:
    """
    When transcript.json is replaced, downstream artifacts (translation/TTS) become invalid.

    Best-effort cleanup only; failures should not block the pipeline.
    """
    # Translation artifacts
    rm_files = [
        os.path.join(folder, "translation.json"),
        os.path.join(folder, "translation_raw.json"),
        os.path.join(folder, "summary.json"),
        os.path.join(folder, "transcript_punctuated.json"),
    ]
    removed = 0
    for p in rm_files:
        try:
            if os.path.exists(p):
                os.remove(p)
                removed += 1
        except Exception:
            continue

    # TTS artifacts under wavs/
    wavs_dir = os.path.join(folder, "wavs")
    wavs_removed = 0
    try:
        if os.path.isdir(wavs_dir):
            for ent in os.scandir(wavs_dir):
                if not ent.is_file():
                    continue
                name = ent.name.lower()
                if name.endswith(".wav") or name.endswith(".json"):
                    try:
                        os.remove(ent.path)
                        wavs_removed += 1
                    except Exception:
                        continue
    except Exception:
        wavs_removed = 0

    if removed or wavs_removed:
        logger.info(f"已清理下游缓存: {folder} (translation={removed}, wavs={wavs_removed})")

