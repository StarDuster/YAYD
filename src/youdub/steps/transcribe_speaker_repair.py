from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from typing import Any

import librosa
import numpy as np
import torch
from loguru import logger

from ..config import Settings
from ..interrupts import check_cancelled
from ..models import ModelManager
from ..utils import save_wav, torch_load_weights_only_compat
from .transcribe_assets import ensure_assets
from .transcribe_compat import patch_torchaudio_backend_compat
from .transcribe_segments import _assign_speakers_by_overlap


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, None)
    if raw is None:
        return float(default)
    s = str(raw).strip()
    if not s:
        return float(default)
    try:
        x = float(s)
    except Exception:
        return float(default)
    return float(x) if math.isfinite(x) else float(default)


def _speaker_repair_enabled() -> bool:
    return _env_flag("YOUDUB_SPEAKER_REPAIR", default=False)


def _speaker_repair_params() -> dict[str, float]:
    # Conservative defaults: only fix very confident mismatches.
    min_dur = float(max(0.2, _env_float("YOUDUB_SPEAKER_REPAIR_MIN_DURATION", 1.5)))
    centroid_min_dur = float(max(min_dur, _env_float("YOUDUB_SPEAKER_REPAIR_CENTROID_MIN_DURATION", 3.0)))
    sim_th = float(max(-1.0, min(1.0, _env_float("YOUDUB_SPEAKER_REPAIR_SIM_THRESHOLD", 0.20))))
    margin_th = float(max(0.0, min(1.0, _env_float("YOUDUB_SPEAKER_REPAIR_MARGIN_THRESHOLD", 0.10))))
    clip_min = float(max(0.5, _env_float("YOUDUB_SPEAKER_REPAIR_CLIP_MIN_SECONDS", 2.0)))
    clip_max = float(max(clip_min, _env_float("YOUDUB_SPEAKER_REPAIR_CLIP_MAX_SECONDS", 6.0)))
    max_changes_ratio = float(max(0.0, min(1.0, _env_float("YOUDUB_SPEAKER_REPAIR_MAX_CHANGES_RATIO", 0.30))))
    return {
        "min_duration": min_dur,
        "centroid_min_duration": centroid_min_dur,
        "sim_threshold": sim_th,
        "margin_threshold": margin_th,
        "clip_min_seconds": clip_min,
        "clip_max_seconds": clip_max,
        "max_changes_ratio": max_changes_ratio,
    }


_EMBEDDING_MODEL = None
_EMBEDDING_INFERENCE = None
_EMBEDDING_KEY: str | None = None
_EMBEDDING_MODEL_LOAD_FAILED = False


def _load_embedding_inference(
    *,
    device: str,
    settings: Settings,
    model_manager: ModelManager,
) -> object | None:
    """
    Best-effort load pyannote/embedding for post-diarization speaker repair.

    - Works offline when cached under whisper_diarization_model_dir.
    - If unavailable, returns None (repair will be skipped).
    """
    global _EMBEDDING_MODEL, _EMBEDDING_INFERENCE, _EMBEDDING_KEY, _EMBEDDING_MODEL_LOAD_FAILED  # noqa: PLW0603

    if _EMBEDDING_MODEL_LOAD_FAILED:
        return None

    patch_torchaudio_backend_compat()

    diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
    cache_dir = str(diar_dir) if diar_dir else None
    token = settings.hf_token

    key = f"{device}|{cache_dir or ''}|pyannote/embedding"
    if _EMBEDDING_INFERENCE is not None and _EMBEDDING_KEY == key:
        return _EMBEDDING_INFERENCE

    try:
        from pyannote.audio import Inference, Model  # type: ignore
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(f"说话人修复: 缺少 pyannote.audio，跳过 embedding 修复: {exc}")
        _EMBEDDING_MODEL_LOAD_FAILED = True
        return None

    # Ensure diarization assets are ready (offline). Embedding model shares the same cache dir.
    try:
        ensure_assets(settings, model_manager, require_diarization=True)
    except Exception:
        # Best-effort; proceed anyway.
        pass

    try:
        logger.info("说话人修复: 加载 pyannote/embedding 模型...")
        with torch_load_weights_only_compat():
            try:
                model = Model.from_pretrained("pyannote/embedding", token=token, cache_dir=cache_dir)
            except TypeError:
                try:
                    model = Model.from_pretrained("pyannote/embedding", use_auth_token=token, cache_dir=cache_dir)
                except TypeError:
                    try:
                        model = Model.from_pretrained("pyannote/embedding", token=token)
                    except TypeError:
                        model = Model.from_pretrained("pyannote/embedding", use_auth_token=token)

        if model is None:
            raise RuntimeError("Model.from_pretrained returned None")

        try:
            model.to(torch.device(device))  # type: ignore[name-defined]
        except Exception:
            # Best-effort: keep default device.
            pass

        _EMBEDDING_MODEL = model
        _EMBEDDING_INFERENCE = Inference(_EMBEDDING_MODEL, window="whole")
        _EMBEDDING_KEY = key
        logger.info("说话人修复: pyannote/embedding 模型就绪")
        return _EMBEDDING_INFERENCE
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(f"说话人修复: 加载 embedding 模型失败，跳过修复: {exc}")
        _EMBEDDING_MODEL_LOAD_FAILED = True
        return None


def _speaker_repair_segment_embedding(
    wav_path: str,
    *,
    start_s: float,
    end_s: float,
    inference: object,
    clip_min_seconds: float,
    clip_max_seconds: float,
    sample_rate: int = 16000,
) -> np.ndarray | None:
    dur = float(max(0.0, float(end_s) - float(start_s)))
    if dur <= 0.0:
        return None

    clip_s = float(min(float(clip_max_seconds), max(float(clip_min_seconds), dur)))
    mid = float((float(start_s) + float(end_s)) * 0.5)
    offset = float(max(0.0, mid - 0.5 * clip_s))

    try:
        y, _sr = librosa.load(wav_path, sr=int(sample_rate), mono=True, offset=float(offset), duration=float(clip_s))
    except Exception:
        return None

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size <= int(sample_rate):
        return None

    try:
        yt, _ = librosa.effects.trim(y, top_db=30.0)
        if yt is not None and yt.size > 0:
            y = np.asarray(yt, dtype=np.float32).reshape(-1)
    except Exception:
        pass

    if y.size <= int(sample_rate):
        return None

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-6:
        y = (y / np.float32(peak)).astype(np.float32, copy=False)

    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="youdub_spkrepair_", suffix=".wav")
        os.close(fd)
        save_wav(y.astype(np.float32, copy=False), tmp_path, sample_rate=int(sample_rate))
        # pyannote Inference expects a path.
        emb = inference(tmp_path)  # type: ignore[operator]
        e = np.asarray(emb, dtype=np.float32).reshape(-1)
        n = float(np.linalg.norm(e))
        if n > 1e-9:
            e = (e / np.float32(n)).astype(np.float32, copy=False)
        return e if e.size > 0 else None
    except Exception:
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _repair_speakers_by_embedding(
    *,
    folder: str,
    transcript: list[dict[str, Any]],
    wav_path: str,
    settings: Settings,
    model_manager: ModelManager,
    device: str,
) -> dict[str, Any] | None:
    """
    Post-process speaker labels using embeddings on real audio chunks.

    This is a *repair* step: only apply changes when confidence is high.
    """
    if not transcript:
        return None

    params = _speaker_repair_params()
    inference = _load_embedding_inference(device=device, settings=settings, model_manager=model_manager)
    if inference is None:
        return None

    min_dur = float(params["min_duration"])
    centroid_min_dur = float(params["centroid_min_duration"])
    sim_th = float(params["sim_threshold"])
    margin_th = float(params["margin_threshold"])
    clip_min_s = float(params["clip_min_seconds"])
    clip_max_s = float(params["clip_max_seconds"])
    max_changes_ratio = float(params["max_changes_ratio"])

    seg_emb: dict[int, np.ndarray] = {}
    seg_dur: dict[int, float] = {}
    for i, seg in enumerate(transcript):
        check_cancelled()
        try:
            st = float(seg.get("start", 0.0) or 0.0)
            ed = float(seg.get("end", st) or st)
        except Exception:
            continue
        dur = float(max(0.0, ed - st))
        seg_dur[i] = dur
        if dur + 1e-9 < min_dur:
            continue
        e = _speaker_repair_segment_embedding(
            wav_path,
            start_s=st,
            end_s=ed,
            inference=inference,
            clip_min_seconds=clip_min_s,
            clip_max_seconds=clip_max_s,
            sample_rate=16000,
        )
        if e is not None and e.size > 0:
            seg_emb[i] = e

    if not seg_emb:
        return None

    # Build centroids from longer segments (more stable embeddings).
    by_spk: dict[str, list[np.ndarray]] = defaultdict(list)
    for i, seg in enumerate(transcript):
        e = seg_emb.get(i)
        if e is None:
            continue
        if float(seg_dur.get(i, 0.0)) + 1e-9 < centroid_min_dur:
            continue
        spk = str(seg.get("speaker") or "SPEAKER_00")
        by_spk[spk].append(e)

    centroid: dict[str, np.ndarray] = {}
    support: dict[str, int] = {}
    for spk, arr in by_spk.items():
        if len(arr) < 3:
            continue
        m = np.mean(np.stack(arr, axis=0), axis=0)
        n = float(np.linalg.norm(m))
        if n > 1e-9:
            m = (m / np.float32(n)).astype(np.float32, copy=False)
        centroid[spk] = np.asarray(m, dtype=np.float32).reshape(-1)
        support[spk] = int(len(arr))

    if len(centroid) < 2:
        # Not enough speakers/centroids to repair.
        return None

    spk_list = sorted(centroid.keys())
    changes: list[dict[str, Any]] = []
    confusion: Counter[tuple[str, str]] = Counter()
    for i, seg in enumerate(transcript):
        check_cancelled()
        old = str(seg.get("speaker") or "SPEAKER_00")
        e = seg_emb.get(i)
        if e is None:
            confusion[(old, old)] += 1
            continue

        best_spk = None
        best_sim = -999.0
        second_sim = -999.0
        for spk in spk_list:
            sim = float(np.dot(e, centroid[spk]))
            if sim > best_sim:
                second_sim = best_sim
                best_sim = sim
                best_spk = spk
            elif sim > second_sim:
                second_sim = sim

        if best_spk is None:
            confusion[(old, old)] += 1
            continue

        margin = float(best_sim - second_sim) if math.isfinite(second_sim) else float("inf")
        new = old
        if best_spk != old and best_sim >= sim_th and margin >= margin_th:
            new = best_spk
            changes.append(
                {
                    "index": int(i),
                    "start": float(seg.get("start", 0.0) or 0.0),
                    "end": float(seg.get("end", 0.0) or 0.0),
                    "duration": float(seg_dur.get(i, 0.0)),
                    "old": old,
                    "new": new,
                    "best_sim": float(best_sim),
                    "margin": float(margin),
                    "text": str(seg.get("text") or "")[:160],
                }
            )
        confusion[(old, new)] += 1

    # Guardrail: if too many changes, centroids are likely unreliable (heavy mixing / overlap).
    applied = False
    if changes:
        ratio = float(len(changes)) / float(max(1, len(transcript)))
        if ratio > max_changes_ratio + 1e-9:
            logger.warning(
                f"说话人修复: 变更比例过高 {ratio:.1%} ({len(changes)}/{len(transcript)})，"
                f"超过阈值 {max_changes_ratio:.1%}；将只输出报告，不自动应用。"
            )
        else:
            for ch in changes:
                idx = int(ch["index"])
                if 0 <= idx < len(transcript):
                    transcript[idx]["speaker"] = str(ch["new"])
            applied = True

    report = {
        "version": 1,
        "applied": bool(applied),
        "params": params,
        "device": str(device),
        "wav_path": str(wav_path),
        "segments_total": int(len(transcript)),
        "segments_with_embedding": int(len(seg_emb)),
        "centroids": {k: int(support.get(k, 0)) for k in spk_list},
        "changes": changes,
        "confusion": {f"{a}->{b}": int(n) for (a, b), n in confusion.items()},
    }

    try:
        out_path = os.path.join(folder, "speaker_repair_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"说话人修复: {'已应用' if applied else '未应用'} {len(changes)} 条变更，报告: {out_path}")
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(f"说话人修复: 写报告失败（忽略）: {exc}")

    return report


def _update_translation_speakers_from_transcript(
    folder: str,
    transcript: list[dict[str, Any]],
) -> bool:
    """Best-effort: patch translation.json speaker labels by time overlap with transcript."""
    tr_path = os.path.join(folder, "translation.json")
    if not os.path.exists(tr_path):
        return False
    try:
        with open(tr_path, "r", encoding="utf-8") as f:
            items = json.load(f) or []
        if not isinstance(items, list) or not items:
            return False
    except Exception:
        return False

    turns: list[dict[str, Any]] = []
    for seg in transcript:
        try:
            s0 = float(seg.get("start", 0.0) or 0.0)
            s1 = float(seg.get("end", s0) or s0)
        except Exception:
            continue
        if s1 <= s0:
            continue
        turns.append({"start": float(s0), "end": float(s1), "speaker": str(seg.get("speaker") or "SPEAKER_00")})

    if not turns:
        return False

    before = [str(it.get("speaker") or "SPEAKER_00") for it in items]
    _assign_speakers_by_overlap(items, turns, default_speaker="SPEAKER_00")
    after = [str(it.get("speaker") or "SPEAKER_00") for it in items]
    if before == after:
        return False

    try:
        with open(tr_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        logger.info(f"说话人修复: 已更新 translation.json 的 speaker 标签: {tr_path}")
        return True
    except Exception:
        return False

