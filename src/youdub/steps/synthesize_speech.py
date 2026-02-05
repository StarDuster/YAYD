"""Synthesize speech (TTS) step.

This file is intentionally a small facade to keep the public interface stable.
Backend-specific details live under `steps/tts/`:
  - `bytedance.py`
  - `gemini.py`
  - `qwen_worker.py`
  - `speaker_ref.py`
  - `prompt.py`
  - `wav_guard.py`

Compatibility shims keep the old import paths working (e.g. `youdub.steps.tts_gemini`).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import uuid
from typing import Any, Iterable

import librosa
import numpy as np
from loguru import logger

from ..config import Settings
from ..interrupts import check_cancelled, sleep_with_cancel
from ..models import ModelManager
from ..text_norm_zh import normalize_zh_nsw, should_normalize_zh_nsw
from ..utils import read_speaker_ref_seconds, save_wav

from .tts import bytedance as _tts_bytedance
from .tts import gemini as _tts_gemini
from .tts.prompt import _tts_text_for_attempt
from .tts.qwen_worker import _QwenTtsWorker
from .tts.speaker_ref import (
    _cap_all_speaker_refs,
    _clear_speaker_voice_mappings_if_refs_updated,
    _ensure_missing_speaker_refs,
    _ensure_speaker_ref_multi,
)
from .tts.wav_guard import (
    _qwen_resp_duration_seconds,
    _qwen_tts_hit_max_tokens_seconds,
    _qwen_tts_is_degenerate_hit_cap,
    _qwen_tts_max_new_tokens,
    _tts_duration_guard_params,
    _tts_wav_ok_for_segment,
    is_valid_wav,
)

# --------------------------------------------------------------------------- #
# Default settings / model manager
# --------------------------------------------------------------------------- #

_DEFAULT_SETTINGS = Settings()
_DEFAULT_MODEL_MANAGER = ModelManager(_DEFAULT_SETTINGS)


def _sync_backend_defaults(settings: Settings, model_manager: ModelManager) -> None:
    # Keep backend modules in sync (they keep their own defaults for direct import usage).
    try:
        _tts_bytedance._DEFAULT_SETTINGS = settings  # type: ignore[attr-defined]
        _tts_bytedance._DEFAULT_MODEL_MANAGER = model_manager  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        _tts_gemini._DEFAULT_SETTINGS = settings  # type: ignore[attr-defined]
    except Exception:
        pass


# Re-export backend callables so tests/monkeypatching keep working.
bytedance_tts = _tts_bytedance.bytedance_tts
gemini_tts = _tts_gemini.gemini_tts


def init_TTS(settings: Settings | None = None, model_manager: ModelManager | None = None) -> None:
    """Initialize TTS globals (best-effort warmup hook used by pipeline)."""
    global _DEFAULT_SETTINGS, _DEFAULT_MODEL_MANAGER  # noqa: PLW0603
    if settings is not None:
        _DEFAULT_SETTINGS = settings
    if model_manager is not None:
        _DEFAULT_MODEL_MANAGER = model_manager
    _sync_backend_defaults(_DEFAULT_SETTINGS, _DEFAULT_MODEL_MANAGER)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #


def preprocess_text(text: str) -> str:
    """Lightweight text cleanup for TTS backends."""
    def _env_flag(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return bool(default)
        s = str(raw).strip().lower()
        if not s:
            return bool(default)
        return s not in {"0", "false", "no", "off"}

    t = str(text or "")
    t = re.sub(r"\s+", " ", t).strip()
    if _env_flag("TTS_TEXT_ZH_NSW_NORMALIZE", True) and t:
        force = _env_flag("TTS_TEXT_ZH_NSW_NORMALIZE_FORCE", False)
        target_lang = getattr(_DEFAULT_SETTINGS, "translation_target_language", None)
        if force or should_normalize_zh_nsw(t, target_language=target_lang):
            try:
                t = normalize_zh_nsw(t)
            except Exception:
                # TTS text normalization must never break synthesis.
                pass
            t = re.sub(r"\s+", " ", t).strip()
    return t


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #


def adjust_audio_length(
    wav_path: str,
    desired_length: float,
    sample_rate: int = 24000,
    min_speed_factor: float = 0.6,
    max_speed_factor: float = 1.1,
) -> tuple[np.ndarray, float]:
    """
    Time-stretch (pitch-preserving) and then pad/trim to the exact desired length.

    Returns: (waveform, duration_seconds)
    """
    sr = int(sample_rate or 0)
    desired = float(desired_length or 0.0)
    if sr <= 0 or not (desired > 0.0):
        return np.zeros((0,), dtype=np.float32), 0.0

    try:
        y0, _ = librosa.load(wav_path, sr=sr, mono=True)
    except Exception:
        y0 = np.zeros((0,), dtype=np.float32)

    y = np.asarray(y0, dtype=np.float32).reshape(-1)
    target_samples = int(max(0, round(desired * float(sr))))
    if target_samples <= 0:
        return np.zeros((0,), dtype=np.float32), 0.0
    if y.size <= 0:
        return np.zeros((target_samples,), dtype=np.float32), float(target_samples) / float(sr)

    cur_dur = float(y.shape[0]) / float(sr)
    speed_need = float(cur_dur) / max(float(desired), 1e-6)  # >1 means speed up, <1 means slow down
    speed = float(min(max(speed_need, float(min_speed_factor)), float(max_speed_factor)))
    ratio = float(1.0 / max(speed, 1e-6))  # audiostretchy ratio: new_duration = old_duration * ratio

    tmp_out: str | None = None
    if abs(ratio - 1.0) > 1e-3:
        try:
            from audiostretchy.stretch import stretch_audio

            tag = f"youdub_tts_stretch_{os.getpid()}_{threading.get_ident()}_{uuid.uuid4().hex}"
            tmp_out = os.path.join(tempfile.gettempdir(), f"{tag}.wav")
            stretch_audio(wav_path, tmp_out, ratio=float(ratio), sample_rate=sr)
            y2, _ = librosa.load(tmp_out, sr=sr, mono=True)
            y = np.asarray(y2, dtype=np.float32).reshape(-1)
        except Exception as exc:
            logger.warning(f"TSM失败（audiostretchy），回退到librosa: {exc}")
            try:
                # librosa: new_duration = old_duration / rate  => rate = 1 / ratio
                rate = float(1.0 / max(float(ratio), 1e-6))
                y2 = librosa.effects.time_stretch(y, rate=rate)
                y = np.asarray(y2, dtype=np.float32).reshape(-1)
            except Exception as exc2:
                logger.warning(f"TSM失败（librosa），返回原音频: {exc2}")
        finally:
            if tmp_out:
                try:
                    if os.path.exists(tmp_out):
                        os.remove(tmp_out)
                except Exception:
                    pass

    if y.shape[0] < target_samples:
        y = np.pad(y, (0, target_samples - int(y.shape[0])), mode="constant")
    else:
        y = y[:target_samples]
    return y.astype(np.float32, copy=False), float(target_samples) / float(sr)


# --------------------------------------------------------------------------- #
# TTS generation
# --------------------------------------------------------------------------- #


def _translation_path(folder: str) -> str:
    return os.path.join(folder, "translation.json")


def _tts_dir(folder: str) -> str:
    return os.path.join(folder, "wavs")


def _tts_done_marker_path(folder: str) -> str:
    return os.path.join(_tts_dir(folder), ".tts_done.json")


def _segment_duration_seconds(seg: dict[str, Any]) -> float:
    try:
        s = float(seg.get("start", 0.0) or 0.0)
        e = float(seg.get("end", 0.0) or 0.0)
    except Exception:
        return 0.0
    if e < s:
        s, e = e, s
    return float(max(0.0, e - s))


def _read_translation(folder: str) -> list[dict[str, Any]]:
    path = _translation_path(folder)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"translation.json 格式错误（应为 list）: {path}")
    return [x for x in obj if isinstance(x, dict)]


def _tts_cache_ok(folder: str, translation: list[dict[str, Any]], *, tts_method: str) -> bool:
    marker = _tts_done_marker_path(folder)
    if not os.path.exists(marker):
        return False

    try:
        with open(marker, "r", encoding="utf-8") as f:
            state = json.load(f) or {}
    except Exception:
        return False
    if not (isinstance(state, dict) and str(state.get("tts_method", "")).strip().lower() == str(tts_method).lower()):
        return False

    tr_path = _translation_path(folder)
    try:
        if os.path.getmtime(marker) + 1e-6 < os.path.getmtime(tr_path):
            return False
    except Exception:
        return False

    ratio, extra, abs_cap, _retries = _tts_duration_guard_params()
    wavs_dir = _tts_dir(folder)
    for i, seg in enumerate(translation):
        p = os.path.join(wavs_dir, f"{i:04d}.wav")
        seg_dur = _segment_duration_seconds(seg)
        if not _tts_wav_ok_for_segment(p, seg_dur, ratio, extra, abs_cap):
            return False
    return True


def generate_all_wavs_under_folder(
    root_folder: str,
    *,
    tts_method: str = "bytedance",
    qwen_tts_batch_size: int | None = None,
) -> None:
    """Generate TTS wavs for root folder (or its immediate subfolders)."""
    root = str(root_folder)
    if not root:
        return

    def _iter_jobs(p: str) -> Iterable[str]:
        if os.path.exists(_translation_path(p)):
            yield p
        try:
            for name in sorted(os.listdir(p)):
                if name.startswith("."):
                    continue
                sub = os.path.join(p, name)
                if not os.path.isdir(sub):
                    continue
                if os.path.exists(_translation_path(sub)):
                    yield sub
        except Exception:
            return

    for folder in _iter_jobs(root):
        check_cancelled()
        try:
            translation = _read_translation(folder)
        except Exception:
            continue
        if not translation:
            continue
        if _tts_cache_ok(folder, translation, tts_method=tts_method):
            continue
        generate_wavs(folder, tts_method=tts_method, qwen_tts_batch_size=qwen_tts_batch_size)


def generate_wavs(
    folder: str,
    *,
    tts_method: str = "bytedance",
    qwen_tts_batch_size: int | None = None,
) -> None:
    """
    Generate per-segment WAVs under `<folder>/wavs/` and write `<folder>/wavs/.tts_done.json`.

    Contract:
    - Does NOT modify `translation.json`.
    - Uses `.tts_done.json` as the skip marker.
    """
    check_cancelled()
    init_TTS(_DEFAULT_SETTINGS, _DEFAULT_MODEL_MANAGER)

    job = str(folder)
    translation = _read_translation(job)
    if not translation:
        raise ValueError(f"翻译文件为空: {_translation_path(job)}")

    os.makedirs(_tts_dir(job), exist_ok=True)

    # Speaker reference management (best-effort).
    speakers: set[str] = set()
    for seg in translation:
        spk = str(seg.get("speaker") or "").strip()
        if spk:
            speakers.add(spk)
    if not speakers:
        speakers = {"SPEAKER_00"}

    speaker_dir = os.path.join(job, "SPEAKER")
    vocals_path = os.path.join(job, "audio_vocals.wav")
    max_ref_seconds = float(read_speaker_ref_seconds(15.0))

    # Try multi-candidate selection first (optional); fallback to simple auto-gen for missing refs.
    try:
        _ensure_speaker_ref_multi(
            segments=translation,
            speakers=set(speakers),
            speaker_dir=speaker_dir,
            vocals_path=vocals_path,
            ref_seconds=max_ref_seconds,
            sample_rate=24000,
        )
    except Exception:
        pass
    _ensure_missing_speaker_refs(
        transcript=translation,
        speakers=set(speakers),
        speaker_dir=speaker_dir,
        vocals_path=vocals_path,
        max_ref_seconds=max_ref_seconds,
        sample_rate=24000,
    )
    _cap_all_speaker_refs(set(speakers), speaker_dir=speaker_dir, max_ref_seconds=max_ref_seconds)
    _clear_speaker_voice_mappings_if_refs_updated(job, set(speakers), speaker_dir=speaker_dir)

    ratio, extra, abs_cap, retries = _tts_duration_guard_params()
    method = str(tts_method or "bytedance").strip().lower()
    wavs_dir = _tts_dir(job)

    logger.info(f"TTS({method}) 开始: {job}（段数={len(translation)}）")

    if method == "qwen":
        bs = int(qwen_tts_batch_size or 1)
        bs = int(max(1, min(bs, 64)))
        # Build work items (skip cache / silence segments early) BEFORE starting the worker.
        items_meta: dict[int, dict[str, Any]] = {}
        for i, seg in enumerate(translation):
            check_cancelled()
            out = os.path.join(wavs_dir, f"{i:04d}.wav")
            seg_dur = _segment_duration_seconds(seg)

            if _tts_wav_ok_for_segment(out, seg_dur, ratio, extra, abs_cap):
                continue

            raw_text = str(seg.get("translation") or seg.get("text") or "")
            if not raw_text.strip():
                save_wav(np.zeros((0,), dtype=np.float32), out, sample_rate=24000)
                continue

            speaker = str(seg.get("speaker") or "SPEAKER_00")
            speaker_wav = os.path.join(speaker_dir, f"{speaker}.wav")
            items_meta[int(i)] = {
                "out": out,
                "seg_dur": float(seg_dur),
                "speaker_wav": speaker_wav,
                "raw_text": raw_text,
            }

        pending = sorted(items_meta.keys())
        if not pending:
            logger.info(f"TTS(qwen) 已全部命中缓存/无需合成: {job}")
        else:
            logger.info(f"TTS(qwen) 待合成 {len(pending)} 段（batch_size={bs} retries={int(retries)}）: {job}")

            worker = _QwenTtsWorker.from_settings(_DEFAULT_SETTINGS)
            try:
                def _cleanup_invalid(p: str) -> None:
                    if os.path.exists(p) and not is_valid_wav(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

                def _mark_failed(i: int, *, why: str) -> None:
                    meta = items_meta.get(int(i)) or {}
                    outp = str(meta.get("out") or "")
                    if outp:
                        try:
                            if os.path.exists(outp):
                                os.remove(outp)
                        except Exception:
                            pass
                    logger.warning(f"Qwen TTS失败（段={i}）: {why}")

                progress_every_sec = 15.0

                if bs <= 1:
                    # Sequential path (keeps behavior simple for debugging).
                    total = int(len(pending))
                    progress_last = time.monotonic()
                    for done, i in enumerate(pending):
                        check_cancelled()
                        now = time.monotonic()
                        if done == 0 or (now - progress_last) >= float(progress_every_sec):
                            progress_last = now
                            logger.info(f"TTS(qwen) 进度 {done}/{total}: {job}")

                        meta = items_meta[int(i)]
                        out = str(meta["out"])
                        seg_dur = float(meta["seg_dur"])
                        speaker_wav = str(meta["speaker_wav"])
                        raw_text = str(meta["raw_text"])

                        ok = False
                        for attempt in range(int(retries)):
                            check_cancelled()
                            tts_text = preprocess_text(_tts_text_for_attempt(raw_text, attempt))
                            _cleanup_invalid(out)
                            try:
                                resp = worker.synthesize(
                                    tts_text, speaker_wav=speaker_wav, output_path=out, language="Auto"
                                )
                            except Exception as exc:
                                _mark_failed(int(i), why=f"attempt={attempt} {exc}")
                                sleep_with_cancel(0.2)
                                continue

                            dur = _qwen_resp_duration_seconds(resp if isinstance(resp, dict) else None)
                            if _qwen_tts_is_degenerate_hit_cap(wav_dur=dur):
                                cap_sec = _qwen_tts_hit_max_tokens_seconds()
                                _mark_failed(
                                    int(i),
                                    why=f"qwen_tts_degenerate_hit_max_new_tokens ({_qwen_tts_max_new_tokens()}, ~{cap_sec:.1f}s)",
                                )
                                sleep_with_cancel(0.2)
                                continue

                            if not _tts_wav_ok_for_segment(out, seg_dur, ratio, extra, abs_cap):
                                _mark_failed(int(i), why="raw_wav_guard_failed")
                                sleep_with_cancel(0.2)
                                continue

                            y, _dur2 = adjust_audio_length(out, seg_dur, sample_rate=24000)
                            save_wav(y, out, sample_rate=24000)
                            if _tts_wav_ok_for_segment(out, seg_dur, ratio, extra, abs_cap):
                                ok = True
                                break

                            _mark_failed(int(i), why="aligned_wav_guard_failed")
                            sleep_with_cancel(0.2)

                        if not ok:
                            raise RuntimeError(f"Qwen TTS 段落合成失败: idx={i}")
                else:
                    # Batch path: retry pending items across attempts.
                    for attempt in range(int(retries)):
                        check_cancelled()
                        if not pending:
                            break

                        attempt_total = int(len(pending))
                        total_batches = int((attempt_total + int(bs) - 1) // int(bs))
                        processed = 0
                        progress_last = time.monotonic()
                        next_pending: list[int] = []

                        logger.info(
                            f"TTS(qwen) attempt {attempt + 1}/{int(retries)} 开始: "
                            f"pending={attempt_total} batch_size={int(bs)} ({job})"
                        )

                        batch_no = 0
                        for off in range(0, len(pending), int(bs)):
                            check_cancelled()
                            batch_no += 1
                            batch_idx = pending[off : off + int(bs)]
                            batch_items: list[dict[str, Any]] = []
                            for i in batch_idx:
                                meta = items_meta[int(i)]
                                out = str(meta["out"])
                                _cleanup_invalid(out)
                                batch_items.append(
                                    {
                                        "text": preprocess_text(_tts_text_for_attempt(str(meta["raw_text"]), attempt)),
                                        "language": "Auto",
                                        "speaker_wav": str(meta["speaker_wav"]),
                                        "output_path": out,
                                    }
                                )

                            try:
                                resp = worker.synthesize_batch(batch_items, timeout_sec=300.0)
                                results = resp.get("results") if isinstance(resp, dict) else None
                            except Exception as exc:
                                results = None
                                logger.warning(f"Qwen TTS批处理失败（attempt={attempt} batch={batch_idx}）: {exc}")

                            if not isinstance(results, list) or len(results) != len(batch_items):
                                for i in batch_idx:
                                    _mark_failed(int(i), why=f"batch_invalid_results attempt={attempt}")
                                next_pending.extend(batch_idx)
                                processed += int(len(batch_idx))
                            else:
                                for i, r in zip(batch_idx, results):
                                    meta = items_meta[int(i)]
                                    out = str(meta["out"])
                                    seg_dur = float(meta["seg_dur"])

                                    ok_item = bool(isinstance(r, dict) and r.get("ok"))
                                    if ok_item:
                                        dur = _qwen_resp_duration_seconds(r)
                                        if _qwen_tts_is_degenerate_hit_cap(wav_dur=dur):
                                            cap_sec = _qwen_tts_hit_max_tokens_seconds()
                                            _mark_failed(
                                                int(i),
                                                why=(
                                                    "qwen_tts_degenerate_hit_max_new_tokens"
                                                    f" ({_qwen_tts_max_new_tokens()}, ~{cap_sec:.1f}s)"
                                                ),
                                            )
                                            next_pending.append(int(i))
                                            continue

                                    if not ok_item:
                                        _mark_failed(
                                            int(i),
                                            why=f"batch_item_failed attempt={attempt} err={getattr(r, 'get', lambda _k: None)('error')}",
                                        )
                                        next_pending.append(int(i))
                                        continue

                                    if not _tts_wav_ok_for_segment(out, seg_dur, ratio, extra, abs_cap):
                                        _mark_failed(int(i), why="raw_wav_guard_failed")
                                        next_pending.append(int(i))
                                        continue

                                    y, _dur2 = adjust_audio_length(out, seg_dur, sample_rate=24000)
                                    save_wav(y, out, sample_rate=24000)
                                    if not _tts_wav_ok_for_segment(out, seg_dur, ratio, extra, abs_cap):
                                        _mark_failed(int(i), why="aligned_wav_guard_failed")
                                        next_pending.append(int(i))

                                processed += int(len(batch_idx))

                            now = time.monotonic()
                            if batch_no >= total_batches or (now - progress_last) >= float(progress_every_sec):
                                progress_last = now
                                logger.info(
                                    f"TTS(qwen) attempt {attempt + 1}/{int(retries)} 进度 "
                                    f"{processed}/{attempt_total} (batch {batch_no}/{total_batches}), "
                                    f"待重试累计={len(next_pending)} ({job})"
                                )

                        pending = next_pending
                        if pending:
                            sleep_with_cancel(0.2)

                    if pending:
                        raise RuntimeError(f"Qwen TTS 段落合成失败（超出重试次数）: idx={pending[:10]} ...")
            finally:
                try:
                    worker.close()
                except Exception:
                    pass

    elif method == "gemini":
        for i, seg in enumerate(translation):
            check_cancelled()
            out = os.path.join(wavs_dir, f"{i:04d}.wav")
            seg_dur = _segment_duration_seconds(seg)
            if _tts_wav_ok_for_segment(out, seg_dur, ratio, extra, abs_cap):
                continue

            raw_text = str(seg.get("translation") or seg.get("text") or "")
            if not raw_text.strip():
                save_wav(np.zeros((0,), dtype=np.float32), out, sample_rate=24000)
                continue

            ok = False
            for attempt in range(int(retries)):
                check_cancelled()
                tts_text = preprocess_text(_tts_text_for_attempt(raw_text, attempt))
                try:
                    gemini_tts(tts_text, out)
                except Exception as exc:
                    logger.warning(f"Gemini TTS失败（段={i} attempt={attempt}）: {exc}")
                    try:
                        if os.path.exists(out):
                            os.remove(out)
                    except Exception:
                        pass
                    sleep_with_cancel(0.2)
                    continue

                if not _tts_wav_ok_for_segment(out, seg_dur, ratio, extra, abs_cap):
                    try:
                        if os.path.exists(out):
                            os.remove(out)
                    except Exception:
                        pass
                    sleep_with_cancel(0.2)
                    continue

                y, _dur2 = adjust_audio_length(out, seg_dur, sample_rate=24000)
                save_wav(y, out, sample_rate=24000)
                if _tts_wav_ok_for_segment(out, seg_dur, ratio, extra, abs_cap):
                    ok = True
                    break

            if not ok:
                raise RuntimeError(f"Gemini TTS 段落合成失败: idx={i}")

    else:
        # Default: ByteDance
        for i, seg in enumerate(translation):
            check_cancelled()
            out = os.path.join(wavs_dir, f"{i:04d}.wav")
            seg_dur = _segment_duration_seconds(seg)
            if _tts_wav_ok_for_segment(out, seg_dur, ratio, extra, abs_cap):
                continue

            speaker = str(seg.get("speaker") or "SPEAKER_00")
            speaker_wav = os.path.join(speaker_dir, f"{speaker}.wav")

            raw_text = str(seg.get("translation") or seg.get("text") or "")
            if not raw_text.strip():
                save_wav(np.zeros((0,), dtype=np.float32), out, sample_rate=24000)
                continue

            ok = False
            for attempt in range(int(retries)):
                check_cancelled()
                tts_text = preprocess_text(_tts_text_for_attempt(raw_text, attempt))
                try:
                    bytedance_tts(tts_text, out, speaker_wav)
                except Exception as exc:
                    logger.warning(f"ByteDance TTS失败（段={i} attempt={attempt}）: {exc}")
                    try:
                        if os.path.exists(out):
                            os.remove(out)
                    except Exception:
                        pass
                    sleep_with_cancel(0.2)
                    continue

                # Guard raw backend output before stretching (avoid "compressing a 5-minute failure into 2s").
                if not _tts_wav_ok_for_segment(out, seg_dur, ratio, extra, abs_cap):
                    logger.warning(f"ByteDance TTS输出异常过长/无效（段={i}），将重试")
                    try:
                        if os.path.exists(out):
                            os.remove(out)
                    except Exception:
                        pass
                    sleep_with_cancel(0.2)
                    continue

                y, _dur2 = adjust_audio_length(out, seg_dur, sample_rate=24000)
                save_wav(y, out, sample_rate=24000)
                if _tts_wav_ok_for_segment(out, seg_dur, ratio, extra, abs_cap):
                    ok = True
                    break

                try:
                    if os.path.exists(out):
                        os.remove(out)
                except Exception:
                    pass
                sleep_with_cancel(0.2)

            if not ok:
                raise RuntimeError(f"ByteDance TTS 段落合成失败: idx={i}")

    # Done marker (lives in wavs/).
    marker = _tts_done_marker_path(job)
    payload = {
        "tts_method": method,
        "segments": int(len(translation)),
        "generated_at": float(time.time()),
        "max_ref_seconds": float(max_ref_seconds),
        "guard": {
            "ratio": float(ratio),
            "extra": float(extra),
            "abs_cap": float(abs_cap),
            "retries": int(retries),
        },
    }
    try:
        with open(marker, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning(f"写入TTS完成标记失败（忽略）: {marker} ({exc})")
    else:
        logger.info(f"TTS({method}) 完成: {job}")

