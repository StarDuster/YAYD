import json
import os
import subprocess
import time
from typing import Any
import re
import math

from loguru import logger

from ..interrupts import check_cancelled, sleep_with_cancel


def _is_cjk_char(ch: str) -> bool:
    # CJK Unified Ideographs + Extension A (covers most Chinese/Japanese Kanji).
    return ("\u4e00" <= ch <= "\u9fff") or ("\u3400" <= ch <= "\u4dbf")


def _looks_like_cjk(text: str, *, threshold: float = 0.3) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    non_space = [ch for ch in s if not ch.isspace()]
    if not non_space:
        return True
    cjk = sum(1 for ch in non_space if _is_cjk_char(ch))
    return (cjk / len(non_space)) >= threshold


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

        # push current line
        lines.append(cur)

        # next line starts with w; if a single word is too long, we must split it.
        if len(w) > max_chars:
            for i in range(0, len(w), max_chars):
                lines.append(w[i : i + max_chars])
            cur = ""
        else:
            cur = w

    if cur:
        lines.append(cur)
    return lines


def wrap_text(text: str, *, max_chars_zh: int = 55, max_chars_en: int = 100) -> str:
    """Wrap subtitle text with language-aware rules.

    - Chinese/CJK: max_chars_zh characters per line (hard wrap).
    - English/Latin: max_chars_en characters per line (wrap by word boundary).
    """
    raw = (text or "").strip()
    if not raw:
        return ""

    out_lines: list[str] = []
    for para in raw.splitlines():
        p = para.strip()
        if not p:
            continue
        if _looks_like_cjk(p):
            out_lines.extend(_wrap_cjk(p, max_chars_zh))
        else:
            out_lines.extend(_wrap_words(p, max_chars_en))

    return "\n".join(out_lines)


def _split_tokens_by_punct(text: str, puncts: set[str]) -> list[str]:
    """Split text into tokens ending with punctuation (keeps punctuation)."""
    s = (text or "").strip()
    if not s:
        return []
    out: list[str] = []
    buf: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        buf.append(ch)
        if ch in puncts:
            # Keep following whitespace with this token so we don't accidentally
            # drop spaces when re-joining English sentences.
            j = i + 1
            while j < len(s) and s[j].isspace():
                buf.append(s[j])
                j += 1
            tok = "".join(buf)
            if tok.strip():
                out.append(tok)
            buf = []
            i = j
            continue
        i += 1
    tail = "".join(buf)
    if tail:
        if tail.strip():
            out.append(tail)
    return out


def _hard_split(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0:
        return [(text or "").strip()] if (text or "").strip() else []
    s = (text or "").strip()
    if not s:
        return []
    return [s[i : i + max_chars] for i in range(0, len(s), max_chars)]


def _split_to_n_chunks(
    text: str,
    *,
    n: int,
    max_chars: int,
    puncts: set[str],
) -> list[str]:
    """Split text into exactly n chunks (best-effort) using punctuation then hard split."""
    s = (text or "").strip()
    if not s:
        return [""] * max(1, int(n))
    n = max(1, int(n))

    # First pass: punctuation-aware tokens, then merge into <= max_chars chunks.
    tokens = _split_tokens_by_punct(s, puncts)
    if not tokens:
        tokens = [s]

    chunks: list[str] = []
    cur = ""
    for tok in tokens:
        t = tok
        if not t:
            continue
        if not cur:
            if max_chars > 0 and len(t) > max_chars:
                chunks.extend(_hard_split(t, max_chars))
            else:
                cur = t
            continue

        if max_chars > 0 and (len(cur) + len(t) > max_chars):
            chunks.append(cur)
            cur = ""
            if max_chars > 0 and len(t) > max_chars:
                chunks.extend(_hard_split(t, max_chars))
            else:
                cur = t
        else:
            cur = f"{cur}{t}"

    if cur:
        chunks.append(cur)

    if not chunks:
        chunks = [s]

    # Adjust to exactly n chunks.
    while len(chunks) > n:
        # Merge tail into previous to reduce count.
        last = chunks.pop()
        chunks[-1] = (chunks[-1] + last).strip()

    def _split_longest_once(parts: list[str]) -> bool:
        if not parts:
            return False
        idx = max(range(len(parts)), key=lambda i: len(parts[i]))
        p = parts[idx].strip()
        if len(p) <= 1:
            return False
        # Try split on space near middle (helps English); fallback to mid.
        mid = len(p) // 2
        cut = -1
        for delta in range(0, min(30, mid + 1)):
            j = mid - delta
            if j > 0 and p[j].isspace():
                cut = j
                break
            j = mid + delta
            if j < len(p) and p[j].isspace():
                cut = j
                break
        if cut <= 0 or cut >= len(p) - 1:
            cut = mid
        left = p[:cut].strip()
        right = p[cut:].strip()
        if not left or not right:
            return False
        parts[idx : idx + 1] = [left, right]
        return True

    while len(chunks) < n:
        if not _split_longest_once(chunks):
            # Last resort: pad empties.
            chunks.append("")

    # Final normalization: keep list length exactly n.
    if len(chunks) != n:
        chunks = (chunks + [""] * n)[:n]
    return [c.strip() for c in chunks]


def _split_bilingual_segment_for_display(
    seg: dict[str, Any],
    *,
    max_event_chars_zh: int,
    max_event_chars_en: int,
    max_event_duration_s: float,
    min_event_duration_s: float,
) -> list[dict[str, Any]]:
    """Split a (start,end,translation,text) segment into multiple smaller display segments."""
    start_s = float(seg.get("start", 0.0) or 0.0)
    end_s = float(seg.get("end", 0.0) or 0.0)
    if end_s < start_s:
        start_s, end_s = end_s, start_s
    duration = float(max(0.0, end_s - start_s))

    zh_raw = str(seg.get("translation") or "").strip()
    en_raw = str(seg.get("text") or "").strip()
    speaker = str(seg.get("speaker") or "SPEAKER_00")

    if not zh_raw and not en_raw:
        return []

    # Decide number of chunks by length, capped by minimum on-screen duration.
    def _ceil_div(a: int, b: int) -> int:
        if b <= 0:
            return 1
        return int((a + b - 1) // b)

    n_len = max(
        1,
        _ceil_div(len(zh_raw), max(1, int(max_event_chars_zh))),
        _ceil_div(len(en_raw), max(1, int(max_event_chars_en))),
    )

    n_time = 1
    if duration > 0 and max_event_duration_s > 0:
        n_time = max(1, int(math.ceil(duration / float(max_event_duration_s))))

    n_target = max(n_len, n_time)

    if duration > 0 and min_event_duration_s > 0:
        max_n = max(1, int(duration // float(min_event_duration_s)))
        n = min(n_target, max_n)
    else:
        n = n_target

    zh_puncts = set("，。！？；：、")
    en_puncts = set(",.;:!?")
    zh_parts = _split_to_n_chunks(zh_raw, n=n, max_chars=max_event_chars_zh, puncts=zh_puncts)
    en_parts = _split_to_n_chunks(en_raw, n=n, max_chars=max_event_chars_en, puncts=en_puncts)

    weights: list[int] = []
    for z, e in zip(zh_parts, en_parts):
        w = len(z.replace(" ", "")) if z else 0
        if w <= 0:
            w = len(e.replace(" ", "")) if e else 0
        weights.append(max(1, int(w)))
    total = float(sum(weights)) or 1.0

    out: list[dict[str, Any]] = []
    cur = float(start_s)
    for i, (z, e, w) in enumerate(zip(zh_parts, en_parts, weights)):
        seg_dur = 0.0 if duration <= 0 else duration * (float(w) / total)
        seg_end = float(end_s) if i == (n - 1) else float(cur + seg_dur)
        out.append(
            {
                "start": round(cur, 3),
                "end": round(seg_end, 3),
                "translation": z,
                "text": e,
                "speaker": speaker,
            }
        )
        cur = seg_end
    return out


def _split_source_sentences(text: str) -> list[str]:
    """Best-effort split for source (often English) sentences.

    We keep this intentionally simple (no heavy NLP) to avoid mismatched bilingual lines
    when translation segments were split into multiple sentences.
    """
    s = " ".join((text or "").split()).strip()
    if not s:
        return []

    # Chinese punctuation
    s = re.sub(r"([。！？])([^”’])", r"\1\n\2", s)
    s = re.sub(r"([。！？][”’])([^，。！？])", r"\1\n\2", s)

    # English punctuation: split on .!? followed by whitespace, keep punctuation in sentence.
    s = re.sub(r"([.!?][\"”’']?)\s+(?=\S)", r"\1\n", s)

    out = [x.strip() for x in s.splitlines() if x.strip()]
    return out or ([s] if s else [])


def split_text(
    input_data: list[dict[str, Any]], 
    punctuations: list[str] | None = None
) -> list[dict[str, Any]]:
    puncts = set(punctuations or ['，', '；', '：', '。', '？', '！', '\n', '”'])

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
        segments: list[dict[str, Any]] = []

        for i, char in enumerate(text):
            if not is_punctuation(char) and i != len(text) - 1:
                continue
            if i - sentence_start < 5 and i != len(text) - 1:
                continue
            if i < len(text) - 1 and is_punctuation(text[i + 1]):
                continue

            sentence = text[sentence_start : i + 1]
            sentence_end = start + duration_per_char * len(sentence)
            segments.append(
                {
                    "start": round(start, 3),
                    "end": round(sentence_end, 3),
                    "translation": sentence,
                    "speaker": speaker,
                }
            )

            start = sentence_end
            sentence_start = i + 1

        for seg in segments:
            # Keep original source text for now; we'll remap across consecutive runs below.
            seg["text"] = original_text
            output_data.append(seg)

    return output_data


def format_timestamp(seconds: float) -> str:
    millisec = int((seconds - int(seconds)) * 1000)
    hours, seconds_int = divmod(int(seconds), 3600)
    minutes, seconds_int = divmod(seconds_int, 60)
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{millisec:03}"


def format_timestamp_ass(seconds: float) -> str:
    # ASS uses centiseconds: H:MM:SS.CS
    total_cs = int(round(seconds * 100))
    hours, rem = divmod(total_cs, 3600 * 100)
    minutes, rem = divmod(rem, 60 * 100)
    secs, cs = divmod(rem, 100)
    return f"{hours:d}:{minutes:02}:{secs:02}.{cs:02}"


def generate_srt(
    translation: list[dict[str, Any]], 
    srt_path: str, 
    speed_up: float = 1.0, 
    max_chars_zh: int = 55,
    max_chars_en: int = 100,
) -> None:
    translation = split_text(translation)
    with open(srt_path, 'w', encoding='utf-8') as f:
        idx = 1
        for line in translation:
            start = format_timestamp(line['start'] / speed_up)
            end = format_timestamp(line['end'] / speed_up)
            text = line['translation']

            if not text:
                continue

            wrapped_text = wrap_text(text, max_chars_zh=max_chars_zh, max_chars_en=max_chars_en)

            f.write(f'{idx}\n')
            f.write(f'{start} --> {end}\n')
            f.write(f'{wrapped_text}\n\n')
            idx += 1


def _ass_escape_text(text: str) -> str:
    # Escape characters that have special meaning in ASS override tags.
    # Keep it simple: prevent accidental override tag injection.
    return (
        (text or "")
        .replace("\\", "\\\\")
        .replace("{", "｛")
        .replace("}", "｝")
        .strip()
    )


def generate_bilingual_ass(
    translation: list[dict[str, Any]],
    ass_path: str,
    *,
    speed_up: float = 1.0,
    play_res_x: int = 1920,
    play_res_y: int = 1080,
    font_name: str = "Arial",
    font_size: int = 20,
    outline: int = 2,
    max_chars_zh: int = 55,
    max_chars_en: int = 100,
    english_scale: float = 0.75,
) -> None:
    """Generate bilingual subtitles in ASS format.

    Chinese (translation) first, English (original text) second with smaller font size.
    """
    # Keep transcript/translation data unchanged; only split for on-screen readability here.
    # Strategy: split by punctuation + character count, capped by minimum per-event duration.
    base_segments = [dict(x) for x in translation]
    min_event_duration_s = 1.2
    max_event_duration_s = 6.0
    max_event_chars_zh = max(1, int(max_chars_zh) * 2)
    max_event_chars_en = max(1, int(max_chars_en) * 2)

    segments: list[dict[str, Any]] = []
    for seg in base_segments:
        pieces = _split_bilingual_segment_for_display(
            seg,
            max_event_chars_zh=max_event_chars_zh,
            max_event_chars_en=max_event_chars_en,
            max_event_duration_s=max_event_duration_s,
            min_event_duration_s=min_event_duration_s,
        )
        if pieces:
            segments.extend(pieces)
        else:
            segments.append(seg)

    # Fix legacy translation.json: multiple consecutive entries may share the same long source `text`.
    # Remap those runs into per-sentence English so bilingual subtitles align.
    i = 0
    while i < len(segments):
        base = str(segments[i].get("text") or "").strip()
        if not base:
            i += 1
            continue
        j = i + 1
        while j < len(segments) and str(segments[j].get("text") or "").strip() == base:
            j += 1

        run_len = j - i
        src_sentences = _split_source_sentences(base)
        if len(src_sentences) > 1 and run_len >= 1:
            if len(src_sentences) == run_len:
                mapped = src_sentences
            elif len(src_sentences) > run_len:
                mapped = [
                    src_sentences[k] if k < run_len - 1 else " ".join(src_sentences[k:])
                    for k in range(run_len)
                ]
            else:
                mapped = [
                    src_sentences[k] if k < len(src_sentences) else src_sentences[-1]
                    for k in range(run_len)
                ]
            for k, s in enumerate(mapped):
                segments[i + k]["text"] = s

        i = j
    en_font_size = max(1, int(round(font_size * float(english_scale))))

    header = "\n".join(
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
            f"Style: Default,{font_name},{int(font_size)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{int(outline)},0,2,10,10,10,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )

    lines: list[str] = [header]
    for seg in segments:
        zh_raw = str(seg.get("translation") or "").strip()
        en_raw = str(seg.get("text") or "").strip()
        if not zh_raw and not en_raw:
            continue

        start = format_timestamp_ass(float(seg["start"]) / speed_up)
        end = format_timestamp_ass(float(seg["end"]) / speed_up)

        zh = _ass_escape_text(wrap_text(zh_raw, max_chars_zh=max_chars_zh, max_chars_en=max_chars_en)).replace("\n", r"\N")
        en = _ass_escape_text(wrap_text(en_raw, max_chars_zh=max_chars_zh, max_chars_en=max_chars_en)).replace("\n", r"\N")

        if zh and en:
            text = f"{zh}" + r"\N" + r"{\fs" + str(en_font_size) + "}" + f"{en}"
        else:
            text = zh or (r"{\fs" + str(en_font_size) + "}" + en)

        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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


def synthesize_video(
    folder: str, 
    subtitles: bool = True, 
    bilingual_subtitle: bool = False,
    speed_up: float = 1.2, 
    fps: int = 30, 
    resolution: str = '1080p',
    use_nvenc: bool = False,
) -> None:
    check_cancelled()
    output_video = os.path.join(folder, 'video.mp4')
    if os.path.exists(output_video):
        try:
            if os.path.getsize(output_video) >= 1024:
                logger.info(f"已合成视频: {folder}")
                return
            logger.warning(f"video.mp4 疑似无效(过小)，将重新生成: {output_video}")
            os.remove(output_video)
        except Exception:
            # Best-effort: proceed to re-generate
            pass
    
    translation_path = os.path.join(folder, 'translation.json')
    input_audio = os.path.join(folder, 'audio_combined.wav')
    input_video = os.path.join(folder, 'download.mp4')
    
    missing = [p for p in (translation_path, input_audio, input_video) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"缺少合成视频所需文件：{missing}")
    
    with open(translation_path, 'r', encoding='utf-8') as f:
        translation = json.load(f)

    check_cancelled()
    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    res_string = f'{width}x{height}'
    
    # Subtitle font size: make it readable across resolutions.
    # Empirically, ~4.5% of height works well (1080p -> ~48).
    font_size = int(round(height * 0.045))
    font_size = max(18, min(font_size, 120))

    outline = int(round(font_size / 12))
    outline = max(2, outline)
    
    video_speed_filter = f"setpts=PTS/{speed_up}"
    audio_speed_filter = f"atempo={speed_up}"
    
    if subtitles:
        if bilingual_subtitle:
            ass_path = os.path.join(folder, "subtitles.ass")
            generate_bilingual_ass(
                translation,
                ass_path,
                speed_up=speed_up,
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
            srt_path = os.path.join(folder, "subtitles.srt")
            generate_srt(translation, srt_path, speed_up=speed_up)

            srt_path_filter = srt_path.replace("\\", "/")
            if os.name == "nt" and ":" in srt_path_filter:
                srt_path_filter = srt_path_filter.replace(":", "\\:")

            subtitle_filter = (
                f"subtitles='{srt_path_filter}':force_style="
                f"'FontName=Arial,FontSize={font_size},PrimaryColour=&HFFFFFF,"
                f"OutlineColour=&H000000,Outline={outline},WrapStyle=2'"
            )

        filter_complex = f"[0:v]{video_speed_filter},{subtitle_filter}[v];[1:a]{audio_speed_filter}[a]"
    else:
        filter_complex = f"[0:v]{video_speed_filter}[v];[1:a]{audio_speed_filter}[a]"
        
    video_encoder = "h264_nvenc" if use_nvenc else "libx264"
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_video,
        '-i', input_audio,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-map', '[a]',
        '-r', str(fps),
        '-s', res_string,
        '-c:v', video_encoder,
        '-c:a', 'aac',
        output_video,
        '-y'
    ]
    
    def _run_ffmpeg(cmd: list[str]) -> None:
        import sys
        import threading
        
        # Capture stderr so ffmpeg progress shows in Gradio UI
        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )  # noqa: S603
        
        # Stream stderr in a background thread to avoid blocking
        def _stream_stderr() -> None:
            try:
                if proc.stderr is None:
                    return
                # Read in small chunks to catch ffmpeg's \r-based progress updates
                buf = b""
                while True:
                    chunk = proc.stderr.read(256)
                    if not chunk:
                        break
                    buf += chunk
                    # Split on \r or \n, emit complete lines
                    while b"\r" in buf or b"\n" in buf:
                        idx_r = buf.find(b"\r")
                        idx_n = buf.find(b"\n")
                        if idx_r >= 0 and (idx_n < 0 or idx_r < idx_n):
                            line = buf[:idx_r]
                            buf = buf[idx_r + 1:]
                            if line.strip():
                                sys.stderr.write("\r" + line.decode("utf-8", errors="replace"))
                                sys.stderr.flush()
                        elif idx_n >= 0:
                            line = buf[:idx_n]
                            buf = buf[idx_n + 1:]
                            if line.strip():
                                sys.stderr.write(line.decode("utf-8", errors="replace") + "\n")
                                sys.stderr.flush()
                # Emit any remaining buffer
                if buf.strip():
                    sys.stderr.write(buf.decode("utf-8", errors="replace") + "\n")
                    sys.stderr.flush()
            except Exception:
                pass
        
        stderr_thread = threading.Thread(target=_stream_stderr, daemon=True)
        stderr_thread.start()
        
        try:
            while True:
                check_cancelled()
                rc = proc.poll()
                if rc is not None:
                    stderr_thread.join(timeout=2)
                    if rc != 0:
                        raise subprocess.CalledProcessError(rc, cmd)
                    return
                time.sleep(0.2)
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
    bilingual_subtitle: bool = False,
    speed_up: float = 1.2, 
    fps: int = 30, 
    resolution: str = '1080p',
    use_nvenc: bool = False,
) -> str:
    count = 0
    for root, _dirs, files in os.walk(folder):
        check_cancelled()
        if 'download.mp4' not in files:
            continue
        video_path = os.path.join(root, 'video.mp4')
        try:
            if os.path.exists(video_path) and os.path.getsize(video_path) >= 1024:
                continue
        except Exception:
            # Treat as not present/invalid and re-synthesize.
            pass
        synthesize_video(
            root,
            subtitles=subtitles,
            bilingual_subtitle=bilingual_subtitle,
            speed_up=speed_up,
            fps=fps,
            resolution=resolution,
            use_nvenc=use_nvenc,
        )
        count += 1
    msg = f"视频合成完成: {folder}（处理 {count} 个文件）"
    logger.info(msg)
    return msg
