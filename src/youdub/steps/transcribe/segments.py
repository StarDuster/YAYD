from __future__ import annotations

import os
import re
from typing import Any

import librosa
import numpy as np
from loguru import logger

from ...interrupts import check_cancelled
from ...utils import read_speaker_ref_seconds, save_wav


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?\.])\s*")


def _split_text_with_timing(text: str, start_s: float, end_s: float) -> list[dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        return []
    duration = float(max(0.0, end_s - start_s))
    if duration <= 0:
        return [{"start": float(start_s), "end": float(start_s), "text": s, "speaker": "SPEAKER_00"}]

    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(s) if p and p.strip()]
    if not parts:
        return [{"start": float(start_s), "end": float(end_s), "text": s, "speaker": "SPEAKER_00"}]

    weights = [max(1, len(p.replace(" ", ""))) for p in parts]
    total = float(sum(weights)) or 1.0

    out: list[dict[str, Any]] = []
    cur = float(start_s)
    for i, (p, w) in enumerate(zip(parts, weights)):
        seg_dur = duration * (float(w) / total)
        seg_end = float(end_s) if i == (len(parts) - 1) else float(cur + seg_dur)
        out.append({"start": float(cur), "end": float(seg_end), "text": p, "speaker": "SPEAKER_00"})
        cur = seg_end
    return out


def merge_segments(transcript: list[dict[str, Any]], ending: str = '!"\').:;?]}~') -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    buffer: dict[str, Any] | None = None

    for segment in transcript:
        if buffer is None:
            buffer = segment
            continue

        # Never merge across speaker boundaries.
        if buffer.get("speaker") != segment.get("speaker"):
            merged.append(buffer)
            buffer = segment
            continue

        if buffer.get("text") and str(buffer["text"])[-1] in ending:
            merged.append(buffer)
            buffer = segment
            continue

        buffer["text"] = (str(buffer.get("text", "")).strip() + " " + str(segment.get("text", "")).strip()).strip()
        buffer["end"] = segment.get("end", buffer.get("end"))
        # Merge word-level timestamps only when both sides have them.
        buf_words = buffer.get("words")
        seg_words = segment.get("words")
        if buf_words is None or seg_words is None:
            buffer["words"] = None
        else:
            buffer["words"] = list(buf_words) + list(seg_words)

    if buffer is not None:
        merged.append(buffer)
    return merged


def generate_speaker_audio(folder: str, transcript: list[dict[str, Any]]) -> None:
    check_cancelled()
    wav_path = os.path.join(folder, "audio_vocals.wav")
    if not os.path.exists(wav_path):
        logger.warning(f"未找到音频文件: {wav_path}")
        return

    target_sr = 24000
    max_ref_seconds = read_speaker_ref_seconds()
    max_ref_samples = int(max_ref_seconds * float(target_sr))
    delay = 0.05
    speakers = {str(seg.get("speaker") or "SPEAKER_00") for seg in transcript}
    speaker_dict: dict[str, np.ndarray] = {}
    for spk in speakers:
        check_cancelled()
        speaker_dict[spk] = np.zeros((0,), dtype=np.float32)

    for segment in transcript:
        check_cancelled()
        start_s = float(segment.get("start", 0.0))
        end_s = float(segment.get("end", 0.0))
        speaker = str(segment.get("speaker") or "SPEAKER_00")
        if max_ref_samples > 0 and speaker_dict.get(speaker, np.zeros((0,), dtype=np.float32)).shape[0] >= max_ref_samples:
            continue

        offset = max(0.0, start_s - delay)
        duration = max(0.0, (end_s - start_s) + 2.0 * delay)
        if duration <= 0:
            continue
        try:
            check_cancelled()
            chunk, _sr = librosa.load(wav_path, sr=target_sr, mono=True, offset=offset, duration=duration)
        except Exception as exc:
            logger.warning(f"加载说话人音频块失败 (speaker={speaker}, offset={offset:.2f}秒): {exc}")
            continue
        if chunk.size <= 0:
            continue

        remaining = max_ref_samples - int(speaker_dict[speaker].shape[0]) if max_ref_samples > 0 else chunk.shape[0]
        if remaining <= 0:
            continue
        if chunk.shape[0] > remaining:
            chunk = chunk[:remaining]
        speaker_dict[speaker] = np.concatenate((speaker_dict[speaker], chunk.astype(np.float32)))

        if max_ref_samples > 0 and all(v.shape[0] >= max_ref_samples for v in speaker_dict.values()):
            break

    speaker_folder = os.path.join(folder, "SPEAKER")
    os.makedirs(speaker_folder, exist_ok=True)

    # Fallback: if a speaker has 0 samples, take the first N seconds from the file.
    for speaker in speakers:
        check_cancelled()
        if speaker_dict[speaker].size > 0:
            continue
        try:
            check_cancelled()
            chunk, _sr = librosa.load(wav_path, sr=target_sr, mono=True, offset=0.0, duration=max_ref_seconds)
            if chunk.size > 0:
                speaker_dict[speaker] = chunk.astype(np.float32)
        except Exception:
            # Best-effort: keep empty
            continue

    for speaker, audio in speaker_dict.items():
        check_cancelled()
        if audio.size <= 0:
            continue
        speaker_file_path = os.path.join(speaker_folder, f"{speaker}.wav")
        save_wav(audio, speaker_file_path, sample_rate=target_sr)
        if max_ref_samples > 0 and audio.shape[0] >= max_ref_samples:
            logger.info(f"已保存说话人参考 ({max_ref_seconds:.1f}秒): {speaker_file_path}")


def _assign_speakers_by_overlap(
    segments: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    default_speaker: str = "SPEAKER_00",
) -> None:
    if not segments or not turns:
        for seg in segments:
            check_cancelled()
            seg["speaker"] = default_speaker
        return

    turns_sorted = sorted(turns, key=lambda x: float(x["start"]))
    idx = 0

    for seg in segments:
        check_cancelled()
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        if seg_end <= seg_start:
            seg["speaker"] = default_speaker
            continue

        while idx < len(turns_sorted) and float(turns_sorted[idx]["end"]) <= seg_start:
            check_cancelled()
            idx += 1

        best_speaker = None
        best_overlap = 0.0
        j = idx
        while j < len(turns_sorted) and float(turns_sorted[j]["start"]) < seg_end:
            check_cancelled()
            t = turns_sorted[j]
            ov = max(0.0, min(seg_end, float(t["end"])) - max(seg_start, float(t["start"])))
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = str(t.get("speaker") or default_speaker)
            j += 1

        seg["speaker"] = best_speaker if best_speaker is not None else default_speaker

