from __future__ import annotations

import json
import os
import time
from typing import Any

import librosa
import numpy as np
from loguru import logger

from ..interrupts import check_cancelled
from ..utils import prepare_speaker_ref_audio, save_wav_norm, wav_duration_seconds
from .tts_wav_guard import _read_env_float, _read_env_int, is_valid_wav


def _ensure_wav_max_duration(path: str, max_seconds: float, sample_rate: int = 24000) -> None:
    """
    Ensure speaker reference audio is at most max_seconds long and apply anti-pop processing.

    This function:
    1. Loads and truncates audio to max_seconds
    2. Applies anti-pop processing (trim silence, soft clip, smooth transients)
    3. Saves the processed audio
    """
    if not path or not os.path.exists(path):
        return
    if max_seconds <= 0:
        return

    dur = wav_duration_seconds(path)
    needs_processing = dur is None or dur > max_seconds + 0.02

    # Also check if the audio might have harsh transients (re-process if it does)
    # This ensures existing audio files get the anti-pop treatment
    if not needs_processing:
        try:
            wav_check, _ = librosa.load(path, sr=sample_rate)
            if wav_check.size > 0:
                diff = np.abs(np.diff(wav_check))
                harsh_transients = int(np.sum(diff > 0.4))
                if harsh_transients > 50:
                    logger.info(f"检测到参考音频有 {harsh_transients} 个急剧变化点，将进行防爆音处理: {path}")
                    needs_processing = True
        except Exception:
            pass

    if not needs_processing:
        return

    try:
        wav, _sr = librosa.load(path, sr=sample_rate, duration=max_seconds)
        if wav.size <= 0:
            return

        # Apply anti-pop processing
        wav_processed = prepare_speaker_ref_audio(
            wav,
            sample_rate=sample_rate,
            trim_silence=True,
            trim_top_db=30.0,
            apply_soft_clip=True,
            clip_threshold=0.85,
            apply_smooth=True,
            smooth_max_diff=0.25,
        )

        if wav_processed.size <= 0:
            wav_processed = wav.astype(np.float32)

        # Save with normalization
        save_wav_norm(wav_processed, path, sample_rate=sample_rate)
        logger.info(f"已处理说话人参考音频(防爆音+裁剪至{max_seconds:.1f}秒): {path}")
    except Exception as exc:
        logger.warning(f"处理说话人音频失败 {path}: {exc}")


def _speaker_ref_multi_enabled() -> bool:
    """
    Enable multi-candidate speaker reference selection by default.

    Set env `TTS_SPEAKER_REF_MULTI=0/false/no` to disable.
    """
    raw = os.getenv("TTS_SPEAKER_REF_MULTI")
    if raw is None:
        return True
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _speaker_ref_multi_params() -> tuple[int, float, float, float, float]:
    """
    Returns: (n_candidates, stride_seconds, silence_threshold, max_silence_ratio, transient_threshold)
    """
    n_candidates = _read_env_int("TTS_SPEAKER_REF_CANDIDATES", 5)
    stride = _read_env_float("TTS_SPEAKER_REF_SCAN_STRIDE_SECONDS", 15.0)
    silence_th = _read_env_float("TTS_SPEAKER_REF_SILENCE_THRESHOLD", 0.02)
    max_silence_ratio = _read_env_float("TTS_SPEAKER_REF_MAX_SILENCE_RATIO", 0.75)
    transient_th = _read_env_float("TTS_SPEAKER_REF_TRANSIENT_THRESHOLD", 0.4)

    n_candidates = int(max(1, min(int(n_candidates), 20)))
    stride = float(max(1.0, float(stride)))
    silence_th = float(max(0.0, min(float(silence_th), 0.5)))
    max_silence_ratio = float(max(0.0, min(float(max_silence_ratio), 0.99)))
    transient_th = float(max(0.05, min(float(transient_th), 2.0)))
    return n_candidates, stride, silence_th, max_silence_ratio, transient_th


def _speaker_ref_quick_metrics(
    wav: np.ndarray,
    *,
    silence_threshold: float,
    transient_threshold: float,
) -> dict:
    """
    Fast, reference-free speaker audio quality heuristics.

    Returns dict with:
      - peak, rms, silence_ratio, clip_ratio, transients
    """
    y = np.asarray(wav, dtype=np.float32).reshape(-1)
    if y.size <= 0:
        return {
            "peak": 0.0,
            "rms": 0.0,
            "silence_ratio": 1.0,
            "clip_ratio": 0.0,
            "transients": 0,
        }

    peak = float(max(abs(float(np.max(y))), abs(float(np.min(y)))))
    if peak > 1e-6:
        y = (y / np.float32(peak)).astype(np.float32, copy=False)
    else:
        peak = 0.0

    rms = float(np.sqrt(float(np.mean(np.square(y), dtype=np.float64))))
    silence_ratio = float(np.mean(np.abs(y) < float(silence_threshold)))
    clip_ratio = float(np.mean(np.abs(y) >= 0.999))
    if y.size >= 2:
        diff = np.abs(np.diff(y))
        transients = int(np.sum(diff >= float(transient_threshold)))
    else:
        transients = 0

    return {
        "peak": peak,
        "rms": rms,
        "silence_ratio": silence_ratio,
        "clip_ratio": clip_ratio,
        "transients": transients,
    }


def _speaker_ref_candidate_starts(
    segments: list[dict],
    speaker: str,
    *,
    stride_seconds: float,
    total_dur: float | None,
    ref_seconds: float,
) -> list[float]:
    """
    Propose candidate start offsets for a speaker, based on translation segment timestamps.

    We keep it simple:
    - only use segments where seg["speaker"] == speaker
    - take at most one start per `stride_seconds` window
    """
    stride = float(max(0.1, float(stride_seconds)))
    starts: list[float] = []
    last_kept = -1e18
    for seg in segments:
        if str(seg.get("speaker") or "") != str(speaker):
            continue
        try:
            s = float(seg.get("start", 0.0) or 0.0)
        except Exception:
            continue
        if s < 0:
            s = 0.0
        if total_dur is not None:
            if s + float(ref_seconds) > float(total_dur) + 1e-6:
                continue
        if s - last_kept < stride:
            continue
        starts.append(float(s))
        last_kept = float(s)

    # If timestamps produce too few candidates (e.g., missing speaker tags), caller will fallback.
    return starts


def _speaker_ref_candidate_filename(speaker: str, *, rank: int, start_sec: float) -> str:
    # Keep file names short and filesystem-friendly.
    ss = int(max(0.0, float(start_sec)))
    return f"{speaker}.ref_{int(rank):02d}_{ss:05d}s.wav"


def _speaker_ref_meta_path(speaker_dir: str, speaker: str) -> str:
    return os.path.join(speaker_dir, f"{speaker}.ref_meta.json")


def _ensure_speaker_ref_multi(
    *,
    folder: str,
    segments: list[dict],
    speakers: set[str],
    speaker_dir: str,
    vocals_path: str,
    ref_seconds: float,
    sample_rate: int = 24000,
) -> None:
    """
    Save multiple reference candidates under `SPEAKER/` and auto-select the best.

    Output:
      - `SPEAKER/<speaker>.ref_XX_YYYYYs.wav` (top-N candidates)
      - `SPEAKER/<speaker>.wav` (best candidate, used by TTS/voice cloning)
      - `SPEAKER/<speaker>.ref_meta.json` (selection details)
    """
    if not _speaker_ref_multi_enabled():
        return
    if not (vocals_path and os.path.exists(vocals_path)):
        return
    if not isinstance(segments, list) or not segments:
        return

    n_candidates, stride_s, silence_th, max_silence_ratio, transient_th = _speaker_ref_multi_params()

    total_dur = wav_duration_seconds(vocals_path)
    if total_dur is None:
        try:
            total_dur = float(librosa.get_duration(filename=vocals_path))
        except Exception:
            total_dur = None

    # Cache key: if vocals didn't change and params unchanged, skip regeneration.
    try:
        src_mtime = float(os.path.getmtime(vocals_path))
    except Exception:
        src_mtime = None
    try:
        src_size = int(os.path.getsize(vocals_path))
    except Exception:
        src_size = None

    for spk in sorted(speakers):
        check_cancelled()
        speaker = str(spk)
        speaker_path = os.path.join(speaker_dir, f"{speaker}.wav")
        meta_path = _speaker_ref_meta_path(speaker_dir, speaker)

        # If meta is fresh, keep it.
        if os.path.exists(meta_path) and os.path.exists(speaker_path) and is_valid_wav(speaker_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}
                if (
                    isinstance(meta, dict)
                    and meta.get("source_path") == vocals_path
                    and meta.get("source_mtime") == src_mtime
                    and meta.get("source_size") == src_size
                    and float(meta.get("ref_seconds") or 0.0) == float(ref_seconds)
                    and float(meta.get("stride_seconds") or 0.0) == float(stride_s)
                    and int(meta.get("n_candidates") or 0) == int(n_candidates)
                    and float(meta.get("silence_threshold") or 0.0) == float(silence_th)
                    and float(meta.get("max_silence_ratio") or 0.0) == float(max_silence_ratio)
                    and float(meta.get("transient_threshold") or 0.0) == float(transient_th)
                ):
                    # Only skip when we have a valid selection record (otherwise retry).
                    sel = meta.get("selected")
                    sel_name = None
                    if isinstance(sel, dict):
                        sel_name = sel.get("path")
                    if isinstance(sel_name, str) and sel_name.endswith(".wav"):
                        sel_path = os.path.join(speaker_dir, sel_name)
                        if os.path.exists(sel_path) and is_valid_wav(sel_path):
                            continue
            except Exception:
                pass

        # Candidate starts primarily from this speaker's transcript timestamps.
        starts = _speaker_ref_candidate_starts(
            segments,
            speaker,
            stride_seconds=stride_s,
            total_dur=total_dur,
            ref_seconds=ref_seconds,
        )
        if not starts:
            # Fallback: scan whole file by stride.
            if total_dur is not None and total_dur > 0 and ref_seconds > 0:
                starts = [
                    float(x)
                    for x in np.arange(
                        0.0, max(0.0, float(total_dur) - float(ref_seconds)), float(stride_s)
                    ).tolist()
                ]
            else:
                starts = [0.0]

        # Evaluate candidates (quick metrics only; heavy processing runs on top-N only).
        evals: list[dict] = []
        for s in starts:
            check_cancelled()
            try:
                y, _sr = librosa.load(
                    vocals_path, sr=int(sample_rate), mono=True, offset=max(0.0, float(s)), duration=float(ref_seconds)
                )
            except Exception:
                continue
            if y.size <= int(sample_rate):  # require >= 1s
                continue
            m = _speaker_ref_quick_metrics(
                y,
                silence_threshold=float(silence_th),
                transient_threshold=float(transient_th),
            )
            m["start_sec"] = float(s)
            evals.append(m)

        if not evals:
            continue

        def _key(m: dict) -> tuple:
            # Prefer segments with enough speech activity (low silence), then fewer transients/clips, then higher rms.
            sil = float(m.get("silence_ratio", 1.0) or 1.0)
            rms = float(m.get("rms", 0.0) or 0.0)
            trans = int(m.get("transients", 0) or 0)
            clip = float(m.get("clip_ratio", 0.0) or 0.0)
            bad_sil = 1 if sil > float(max_silence_ratio) else 0
            bad_rms = 1 if rms < 0.03 else 0
            return (bad_sil, bad_rms, trans, clip, -rms, sil, float(m.get("start_sec", 0.0) or 0.0))

        evals.sort(key=_key)
        top = evals[: min(int(n_candidates), len(evals))]

        # Remove old candidates for this speaker to avoid confusion.
        try:
            prefix = f"{speaker}.ref_"
            for name in os.listdir(speaker_dir):
                if name.startswith(prefix) and name.endswith(".wav"):
                    try:
                        os.remove(os.path.join(speaker_dir, name))
                    except Exception:
                        pass
        except Exception:
            pass

        # Generate & save candidates with anti-pop processing.
        saved: list[dict] = []
        best_written = False
        for rank, m in enumerate(top):
            check_cancelled()
            start_sec = float(m.get("start_sec", 0.0) or 0.0)
            out_name = _speaker_ref_candidate_filename(speaker, rank=rank, start_sec=start_sec)
            out_path = os.path.join(speaker_dir, out_name)
            try:
                y, _sr = librosa.load(
                    vocals_path, sr=int(sample_rate), mono=True, offset=max(0.0, start_sec), duration=float(ref_seconds)
                )
                if y.size <= 0:
                    continue
                y_proc = prepare_speaker_ref_audio(
                    y,
                    sample_rate=int(sample_rate),
                    trim_silence=True,
                    trim_top_db=30.0,
                    apply_soft_clip=True,
                    clip_threshold=0.85,
                    apply_smooth=True,
                    smooth_max_diff=0.25,
                )
                if y_proc.size <= 0:
                    y_proc = y.astype(np.float32)
                save_wav_norm(y_proc.astype(np.float32, copy=False), out_path, sample_rate=int(sample_rate))
                saved.append(
                    {
                        "rank": int(rank),
                        "start_sec": float(start_sec),
                        "path": out_name,
                        "metrics": {
                            "rms": float(m.get("rms", 0.0) or 0.0),
                            "silence_ratio": float(m.get("silence_ratio", 1.0) or 1.0),
                            "clip_ratio": float(m.get("clip_ratio", 0.0) or 0.0),
                            "transients": int(m.get("transients", 0) or 0),
                        },
                    }
                )
                if rank == 0:
                    save_wav_norm(y_proc.astype(np.float32, copy=False), speaker_path, sample_rate=int(sample_rate))
                    best_written = True
            except Exception as exc:
                logger.warning(f"生成参考片段失败 speaker={speaker} start={start_sec:.2f}s: {exc}")
                continue

        if best_written and os.path.exists(speaker_path):
            # Ensure duration cap + anti-pop on the final selected file.
            _ensure_wav_max_duration(speaker_path, float(ref_seconds), sample_rate=int(sample_rate))

        # Write meta (even if saving failed, to avoid retry storm; user can delete it to force rebuild).
        payload = {
            "version": 1,
            "speaker": speaker,
            "source_path": vocals_path,
            "source_mtime": src_mtime,
            "source_size": src_size,
            "sample_rate": int(sample_rate),
            "ref_seconds": float(ref_seconds),
            "stride_seconds": float(stride_s),
            "n_candidates": int(n_candidates),
            "silence_threshold": float(silence_th),
            "max_silence_ratio": float(max_silence_ratio),
            "transient_threshold": float(transient_th),
            "evaluated": int(len(evals)),
            "saved": saved,
            "selected": (saved[0] if saved else None),
            "created_at": float(time.time()),
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def _ensure_missing_speaker_refs(
    *,
    transcript: list[dict[str, Any]],
    speakers: set[str],
    speaker_dir: str,
    vocals_path: str,
    max_ref_seconds: float,
    sample_rate: int = 24000,
) -> None:
    """Best-effort: generate missing/invalid `SPEAKER/<speaker>.wav` using `audio_vocals.wav`."""
    if not (vocals_path and os.path.exists(vocals_path)):
        return

    os.makedirs(speaker_dir, exist_ok=True)

    missing: list[str] = []
    for spk in speakers:
        check_cancelled()
        p = os.path.join(speaker_dir, f"{spk}.wav")
        if os.path.exists(p) and is_valid_wav(p):
            continue
        missing.append(str(spk))

    if not missing:
        return

    sr_ref = int(sample_rate)
    max_ref_samples = int(float(max_ref_seconds) * float(sr_ref))
    delay = 0.05

    for spk in missing:
        check_cancelled()
        spk_path = os.path.join(speaker_dir, f"{spk}.wav")
        try:
            y = np.zeros((0,), dtype=np.float32)
            if len(speakers) <= 1:
                wav0, _sr = librosa.load(vocals_path, sr=sr_ref, mono=True, duration=float(max_ref_seconds))
                y = wav0.astype(np.float32, copy=False)
            else:
                # Stitch together per-segment audio for this speaker (best-effort).
                for seg in transcript:
                    check_cancelled()
                    if str(seg.get("speaker") or "") != str(spk):
                        continue
                    try:
                        start_s = float(seg.get("start", 0.0) or 0.0)
                        end_s = float(seg.get("end", 0.0) or 0.0)
                    except Exception:
                        continue
                    if end_s <= start_s:
                        continue

                    offset = max(0.0, float(start_s) - float(delay))
                    duration = max(0.0, (float(end_s) - float(start_s)) + 2.0 * float(delay))
                    if duration <= 0:
                        continue
                    try:
                        chunk, _sr = librosa.load(
                            vocals_path, sr=sr_ref, mono=True, offset=float(offset), duration=float(duration)
                        )
                    except Exception:
                        continue
                    if chunk.size <= 0:
                        continue

                    remaining = (max_ref_samples - int(y.shape[0])) if max_ref_samples > 0 else int(chunk.shape[0])
                    if remaining <= 0:
                        break
                    if int(chunk.shape[0]) > int(remaining):
                        chunk = chunk[:remaining]
                    y = np.concatenate((y, chunk.astype(np.float32, copy=False)))
                    if max_ref_samples > 0 and int(y.shape[0]) >= int(max_ref_samples):
                        break

                if y.size <= 0:
                    logger.warning(f"未能从该说话人的时间戳提取参考音频，将回退到文件开头: speaker={spk}")
                    wav0, _sr = librosa.load(vocals_path, sr=sr_ref, mono=True, duration=float(max_ref_seconds))
                    y = wav0.astype(np.float32, copy=False)

            if y.size <= 0:
                continue

            # Apply anti-pop processing before saving
            wav_processed = prepare_speaker_ref_audio(
                y,
                sample_rate=sr_ref,
                trim_silence=True,
                trim_top_db=30.0,
                apply_soft_clip=True,
                clip_threshold=0.85,
                apply_smooth=True,
                smooth_max_diff=0.25,
            )
            if wav_processed.size > 0:
                save_wav_norm(wav_processed, spk_path, sample_rate=sr_ref)
            else:
                save_wav_norm(y.astype(np.float32, copy=False), spk_path, sample_rate=sr_ref)
            logger.info(f"已生成缺失的说话人参考(防爆音处理) ({max_ref_seconds:.1f}秒): {spk_path}")
        except Exception as exc:
            logger.warning(f"生成说话人参考音频失败 {spk}: {exc}")


def _cap_all_speaker_refs(speakers: set[str], *, speaker_dir: str, max_ref_seconds: float) -> None:
    for spk in speakers:
        check_cancelled()
        _ensure_wav_max_duration(os.path.join(speaker_dir, f"{spk}.wav"), max_ref_seconds, sample_rate=24000)


def _clear_speaker_voice_mappings_if_refs_updated(folder: str, speakers: set[str], *, speaker_dir: str) -> None:
    # 说话人参考音频更新后，必须清理 speaker->voice 的缓存映射，否则容易出现“台词用错音色/克隆声线”的错配。
    try:
        speaker_ref_mtime_max = 0.0
        for spk in speakers:
            p = os.path.join(speaker_dir, f"{spk}.wav")
            if not os.path.exists(p):
                continue
            try:
                m = float(os.path.getmtime(p))
            except Exception:
                m = 0.0
            if m > speaker_ref_mtime_max:
                speaker_ref_mtime_max = m
        if speaker_ref_mtime_max > 0:
            for name in ("speaker_to_voice_type.json", "speaker_to_cloned_voice.json"):
                mp = os.path.join(folder, name)
                if not os.path.exists(mp):
                    continue
                try:
                    if float(os.path.getmtime(mp)) + 1e-6 < float(speaker_ref_mtime_max):
                        os.remove(mp)
                        logger.info(f"检测到说话人参考已更新，已清理缓存映射: {mp}")
                except Exception:
                    continue
    except Exception:
        pass

