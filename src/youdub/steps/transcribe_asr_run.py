from __future__ import annotations

import importlib.util
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import librosa
import numpy as np
from loguru import logger

from ..interrupts import check_cancelled
from ..utils import wav_duration_seconds
from .transcribe_segments import _split_text_with_timing


_VAD_ONNXRUNTIME_WARNED = False


def _onnxruntime_available() -> bool:
    try:
        return importlib.util.find_spec("onnxruntime") is not None
    except Exception:
        return False


def run_qwen_asr(
    wav_path: str,
    *,
    model: object,
    num_threads: int,
    vad_segment_threshold_seconds: int,
) -> tuple[list[dict[str, Any]], str | None]:
    # Qwen3-ASR: chunk the audio (seconds) and approximate timestamps per sentence.
    transcript: list[dict[str, Any]] = []

    num_threads = int(max(1, min(int(num_threads), 32)))
    chunk_s = int(max(5, int(vad_segment_threshold_seconds)))

    logger.info(f"转录中(Qwen3-ASR) {wav_path}")
    t0 = time.time()

    dur = wav_duration_seconds(wav_path)
    if dur is None:
        # Fallback: use librosa header-based duration.
        try:
            dur = float(librosa.get_duration(filename=wav_path))
        except Exception:
            dur = None

    if dur is None:
        # Last resort: treat whole file as one chunk without timestamps.
        dur = float(chunk_s)

    starts = [float(x) for x in np.arange(0.0, float(dur) + 1e-6, float(chunk_s)).tolist()]

    def _run_chunk(start_s: float) -> tuple[float, float, str, str | None]:
        check_cancelled()
        logger.info(f"Qwen3-ASR: 处理 chunk offset={start_s:.1f}s / {dur:.1f}s")
        # Load chunk as 16k mono.
        chunk, sr = librosa.load(
            wav_path,
            sr=16000,
            mono=True,
            offset=max(0.0, float(start_s)),
            duration=float(chunk_s),
        )
        if chunk.size <= 0:
            return float(start_s), float(start_s), "", None
        chunk_dur = float(chunk.shape[0]) / float(sr)
        end_s = float(start_s) + chunk_dur
        check_cancelled()
        results = model.transcribe(audio=(chunk, int(sr)), language=None)  # type: ignore[attr-defined]
        if not results:
            return float(start_s), float(end_s), "", None
        r0 = results[0]
        text = str(getattr(r0, "text", "") or "").strip()
        lang = getattr(r0, "language", None)
        lang_str = str(lang) if lang is not None else None
        return float(start_s), float(end_s), text, lang_str

    chunk_results: list[tuple[float, float, str, str | None]] = []
    if num_threads <= 1 or len(starts) <= 1:
        for st in starts:
            check_cancelled()
            chunk_results.append(_run_chunk(float(st)))
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            futs = [ex.submit(_run_chunk, float(st)) for st in starts]
            for fut in as_completed(futs):
                check_cancelled()
                chunk_results.append(fut.result())

    chunk_results.sort(key=lambda x: x[0])
    lang_seen: str | None = None
    for st, ed, txt, lang in chunk_results:
        check_cancelled()
        if lang and not lang_seen:
            lang_seen = lang
        if not txt:
            continue
        transcript.extend(_split_text_with_timing(txt, st, ed))

    logger.info(f"ASR完成(Qwen3-ASR)，耗时 {time.time() - t0:.2f}秒 (段数={len(transcript)}, 语言={lang_seen})")
    return transcript, lang_seen


def run_whisper_asr(
    wav_path: str,
    *,
    asr_model: object,
    asr_pipeline: object | None,
    batch_size: int,
) -> tuple[list[dict[str, Any]], str | None]:
    transcript: list[dict[str, Any]] = []

    logger.info(f"转录中 {wav_path}")
    t0 = time.time()

    # VAD:
    # - faster-whisper VAD requires `onnxruntime` (cpu or gpu). Our default install keeps it optional.
    # - When `onnxruntime` is missing (or VAD init fails), gracefully fall back to `vad_filter=False`.
    global _VAD_ONNXRUNTIME_WARNED  # noqa: PLW0603
    vad_filter = bool(_onnxruntime_available())
    if not vad_filter and not _VAD_ONNXRUNTIME_WARNED:
        _VAD_ONNXRUNTIME_WARNED = True
        logger.warning(
            "未安装 onnxruntime，已自动关闭 faster-whisper VAD 过滤（语音切分可能变差）。"
            "如需启用 VAD，请安装 CPU/GPU 依赖：uv sync --extra cpu 或 uv sync --extra gpu。"
        )

    def _do_transcribe(*, vad: bool):
        check_cancelled()
        if asr_pipeline is not None:
            return asr_pipeline.transcribe(  # type: ignore[attr-defined]
                wav_path,
                batch_size=batch_size,
                beam_size=5,
                vad_filter=bool(vad),
                condition_on_previous_text=False,
                word_timestamps=True,  # 开启词级时间戳
                no_speech_threshold=0.8,  # 提高阈值，防止漏句
            )
        return asr_model.transcribe(  # type: ignore[attr-defined]
            wav_path,
            beam_size=5,
            vad_filter=bool(vad),
            condition_on_previous_text=False,
            word_timestamps=True,  # 开启词级时间戳
            no_speech_threshold=0.8,  # 提高阈值，防止漏句
        )

    try:
        segments_iter, info = _do_transcribe(vad=vad_filter)
    except RuntimeError as exc:
        # Typical upstream error: "Applying the VAD filter requires the onnxruntime package"
        if vad_filter and "onnxruntime" in str(exc).lower():
            logger.warning(f"faster-whisper VAD 初始化失败，将关闭 VAD 重试: {exc}")
            segments_iter, info = _do_transcribe(vad=False)
        else:
            raise

    segments_list = []
    for seg in segments_iter:
        check_cancelled()
        segments_list.append(seg)
    logger.info(
        f"ASR完成，耗时 {time.time() - t0:.2f}秒 (段数={len(segments_list)}, 语言={getattr(info, 'language', None)})"
    )

    for seg in segments_list:
        check_cancelled()
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        words = getattr(seg, "words", None)
        words_data: list[dict[str, Any]] | None = None
        if words:
            words_data = []
            for w in words:
                if w is None:
                    continue
                words_data.append(
                    {
                        "start": float(getattr(w, "start", 0.0) or 0.0),
                        "end": float(getattr(w, "end", 0.0) or 0.0),
                        "word": str(getattr(w, "word", "") or ""),
                        "probability": (
                            float(getattr(w, "probability", 0.0) or 0.0)
                            if getattr(w, "probability", None) is not None
                            else None
                        ),
                    }
                )
        transcript.append(
            {
                "start": float(getattr(seg, "start", 0.0)),
                "end": float(getattr(seg, "end", 0.0)),
                "text": text,
                "speaker": "SPEAKER_00",
                "words": words_data,
            }
        )

    lang = getattr(info, "language", None)
    lang_str = str(lang) if lang is not None else None
    return transcript, lang_str

