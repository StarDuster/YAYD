from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import librosa
import numpy as np
from loguru import logger

from ..interrupts import check_cancelled
from ..speech_rate import apply_scaling_ratio, compute_en_speech_rate, compute_scaling_ratio, compute_zh_speech_rate


def _read_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    s = str(raw).strip().lower()
    if not s:
        return bool(default)
    return s not in {"0", "false", "no", "off"}


def _read_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    s = str(raw).strip()
    if not s:
        return float(default)
    try:
        return float(s)
    except Exception:
        return float(default)


def _mtime(path: str) -> float | None:
    try:
        return float(os.path.getmtime(path))
    except Exception:
        return None


def _is_up_to_date(target: str, deps: list[str]) -> bool:
    t = _mtime(target)
    if t is None:
        return False
    for p in deps:
        mt = _mtime(p)
        if mt is None:
            continue
        if mt > t:
            return False
    return True


def prepare_adaptive_alignment(folder: str, sample_rate: int = 24000) -> None:
    """
    Pipeline step: generate `translation_adaptive.json` and `adaptive_plan.json`.

    Inputs:
    - translation.json (ASR timeline)
    - wavs/*.wav (TTS segments)

    Outputs:
    - translation_adaptive.json (speech-only cues on the *new* timeline)
    - adaptive_plan.json (speech + pause segments for video/instruments alignment)
    """
    check_cancelled()

    sample_rate = int(sample_rate) if int(sample_rate) > 0 else 24000

    wavs_folder = os.path.join(folder, "wavs")
    translation_path = os.path.join(folder, "translation.json")
    translation_adaptive_path = os.path.join(folder, "translation_adaptive.json")
    adaptive_plan_path = os.path.join(folder, "adaptive_plan.json")
    marker_path = os.path.join(folder, "wavs", ".tts_done.json")

    if not os.path.exists(translation_path):
        raise FileNotFoundError(f"缺少翻译文件: {translation_path}")
    if not os.path.exists(wavs_folder):
        raise FileNotFoundError(f"缺少 wavs 目录: {wavs_folder}")

    deps = [translation_path]
    if os.path.exists(marker_path):
        deps.append(marker_path)

    # Idempotent: if both outputs are newer than key inputs, skip.
    if _is_up_to_date(translation_adaptive_path, deps) and _is_up_to_date(adaptive_plan_path, deps):
        return

    with open(translation_path, "r", encoding="utf-8") as f:
        translation = json.load(f)
    if not isinstance(translation, list) or not translation:
        raise ValueError(f"翻译文件为空或格式不正确: {translation_path}")

    # Get all wav files (prefer numeric names).
    wav_files = sorted([f for f in os.listdir(wavs_folder) if re.fullmatch(r"\d+\.wav", f)])
    if not wav_files:
        wav_files = sorted([f for f in os.listdir(wavs_folder) if f.endswith(".wav")])
    if not wav_files:
        raise ValueError(f"wavs 目录为空: {wavs_folder}")

    expected_count = len(translation)
    if len(wav_files) < expected_count:
        raise FileNotFoundError(
            f"wav文件数量({len(wav_files)})少于翻译段数({expected_count})，请先确保TTS已完整生成"
        )

    trim_top_db = 35.0
    gap_default_s = 0.12
    gap_clause_s = 0.18
    gap_sentence_s = 0.25

    # Optional: speech-rate based TTS time-scale modification (TSM) to match EN pacing.
    align_enabled = _read_env_bool("SPEECH_RATE_ALIGN_ENABLED", True)
    align_mode = str(os.getenv("SPEECH_RATE_ALIGN_MODE", "single") or "single").strip().lower()
    voice_min = _read_env_float("SPEECH_RATE_VOICE_MIN_RATIO", 0.7)
    # Default: only speed up, never slow down (avoid “拖音/橡皮筋”感)
    voice_max = _read_env_float("SPEECH_RATE_VOICE_MAX_RATIO", 1.0)
    silence_min = _read_env_float("SPEECH_RATE_SILENCE_MIN_RATIO", 0.3)
    silence_max = _read_env_float("SPEECH_RATE_SILENCE_MAX_RATIO", 3.0)
    overall_min = _read_env_float("SPEECH_RATE_OVERALL_MIN_RATIO", 0.5)
    overall_max = _read_env_float("SPEECH_RATE_OVERALL_MAX_RATIO", 2.0)
    align_threshold = _read_env_float("SPEECH_RATE_ALIGN_THRESHOLD", 0.05)
    # Stabilize per-segment scaling to avoid audible jitter between adjacent sentences.
    # Max allowed delta of voice_ratio between consecutive speech segments.
    ratio_max_jump = _read_env_float("SPEECH_RATE_MAX_RATIO_JUMP", 0.15)
    ratio_max_jump = float(max(0.0, min(ratio_max_jump, 0.5)))
    # When TTS is shorter than the original segment, pad up to this many seconds of
    # silence so the video doesn't have to be sped up as aggressively.
    tail_pad_max = _read_env_float("SPEECH_RATE_TAIL_PAD_MAX_SEC", 1.0)
    tail_pad_max = float(max(0.0, tail_pad_max))
    en_vad_top_db = _read_env_float("SPEECH_RATE_EN_VAD_TOP_DB", 30.0)
    zh_vad_top_db = _read_env_float("SPEECH_RATE_ZH_VAD_TOP_DB", 30.0)
    audio_vocals_path = os.path.join(folder, "audio_vocals.wav")

    # Global bias to avoid overall pacing drift.
    align_global_bias = 1.0
    if align_enabled:
        try:
            total_en = 0.0
            for _seg in translation:
                check_cancelled()
                try:
                    s0 = float(_seg.get("start", 0.0) or 0.0)
                    s1 = float(_seg.get("end", 0.0) or 0.0)
                    if s1 < s0:
                        s0, s1 = s1, s0
                    total_en += float(max(0.0, s1 - s0))
                except Exception:
                    continue

            total_zh = 0.0
            for _i in range(expected_count):
                check_cancelled()
                wav_path = os.path.join(wavs_folder, wav_files[_i])
                if not os.path.exists(wav_path):
                    continue
                try:
                    y, _sr = librosa.load(wav_path, sr=sample_rate, mono=True)
                    y = y.astype(np.float32, copy=False)
                    if y.size > 0:
                        try:
                            trimmed, _idx = librosa.effects.trim(y, top_db=float(trim_top_db))
                            if trimmed is not None and trimmed.size > 0:
                                y = trimmed.astype(np.float32, copy=False)
                        except Exception:
                            pass
                    total_zh += float(y.shape[0]) / float(sample_rate) if y.size > 0 else 0.0
                except Exception:
                    continue

            if total_en > 0.0 and total_zh > 0.0:
                align_global_bias = float(total_en / total_zh)
                # We only use this to speed up overall pacing (cap at 1.0).
                align_global_bias = float(max(0.3, min(align_global_bias, 1.0)))
                logger.info(f"语速对齐: 全局bias={align_global_bias:.3f} (en={total_en:.1f}s, zh={total_zh:.1f}s)")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"语速对齐: 计算全局bias失败，将回退为 1.0: {exc}")
            align_global_bias = 1.0

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

    adaptive_translation: list[dict[str, Any]] = []
    adaptive_plan: list[dict[str, Any]] = []

    t_cursor_samples = 0  # output timeline cursor (samples)
    prev_voice_ratio = 1.0

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

        ratio_info: dict[str, Any] | None = None
        en_stats: dict[str, Any] | None = None
        zh_stats: dict[str, Any] | None = None

        # Optional: time-scale modify TTS to match EN pacing (speech rate) for this segment.
        if align_enabled and tts_audio.size > 0:
            try:
                en_text = str(seg.get("text") or "")
                en_total_duration = float(max(0.0, orig_end - orig_start))

                # Estimate EN voiced duration from the original vocal stem (VAD).
                en_voiced_duration = float(en_total_duration)
                if en_total_duration > 0 and os.path.exists(audio_vocals_path):
                    try:
                        en_audio, en_sr = librosa.load(
                            audio_vocals_path,
                            sr=None,
                            mono=True,
                            offset=max(0.0, float(orig_start)),
                            duration=float(en_total_duration),
                        )
                        en_audio = np.asarray(en_audio, dtype=np.float32).reshape(-1)
                        if en_audio.size > 0 and int(en_sr or 0) > 0:
                            try:
                                intervals = librosa.effects.split(en_audio, top_db=float(en_vad_top_db))
                            except Exception:
                                intervals = np.zeros((0, 2), dtype=np.int64)
                            voiced_samples = 0
                            for st, ed in intervals:
                                st_i = int(st)
                                ed_i = int(ed)
                                if ed_i > st_i:
                                    voiced_samples += (ed_i - st_i)
                            en_voiced_duration = float(voiced_samples) / float(en_sr) if voiced_samples > 0 else 0.0
                    except Exception:
                        en_voiced_duration = float(en_total_duration)

                en_voiced_duration = float(max(0.0, min(en_voiced_duration, en_total_duration)))
                en_silence_duration = float(max(0.0, en_total_duration - en_voiced_duration))
                en_pause_ratio = float(en_silence_duration / en_total_duration) if en_total_duration > 0 else 0.0

                en_stats = dict(compute_en_speech_rate(en_text, en_voiced_duration))
                en_stats.update(
                    {
                        "total_duration": float(en_total_duration),
                        "voiced_duration": float(en_voiced_duration),
                        "silence_duration": float(en_silence_duration),
                        "pause_ratio": float(en_pause_ratio),
                    }
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(f"段落 {i}: 计算英文语速失败，将跳过语速对齐: {exc}")
                en_stats = None

            if en_stats:
                try:
                    zh_text = str(seg.get("translation") or "")
                    zh_total_duration = float(tts_audio.shape[0]) / float(sample_rate) if tts_audio.shape[0] > 0 else 0.0

                    # ZH voiced duration from TTS itself (VAD).
                    zh_voiced_duration = float(zh_total_duration)
                    if tts_audio.size > 0 and sample_rate > 0:
                        try:
                            intervals = librosa.effects.split(tts_audio.astype(np.float32, copy=False), top_db=float(zh_vad_top_db))
                        except Exception:
                            intervals = np.zeros((0, 2), dtype=np.int64)
                        voiced_samples = 0
                        for st, ed in intervals:
                            st_i = int(st)
                            ed_i = int(ed)
                            if ed_i > st_i:
                                voiced_samples += (ed_i - st_i)
                        zh_voiced_duration = float(voiced_samples) / float(sample_rate) if voiced_samples > 0 else 0.0

                    zh_voiced_duration = float(max(0.0, min(zh_voiced_duration, zh_total_duration)))
                    zh_silence_duration = float(max(0.0, zh_total_duration - zh_voiced_duration))
                    zh_pause_ratio = float(zh_silence_duration / zh_total_duration) if zh_total_duration > 0 else 0.0

                    zh_stats = dict(compute_zh_speech_rate(zh_text, zh_voiced_duration))
                    zh_stats.update(
                        {
                            "total_duration": float(zh_total_duration),
                            "voiced_duration": float(zh_voiced_duration),
                            "silence_duration": float(zh_silence_duration),
                            "pause_ratio": float(zh_pause_ratio),
                        }
                    )

                    # Apply global bias by effectively increasing EN reference rate.
                    en_stats_used = dict(en_stats)
                    if float(align_global_bias) > 1e-6 and abs(float(align_global_bias) - 1.0) > 1e-6:
                        en_stats_used["syllable_rate"] = float(en_stats_used.get("syllable_rate", 0.0)) / float(
                            align_global_bias
                        )

                    ratio_info = compute_scaling_ratio(
                        en_stats_used,
                        zh_stats,
                        mode=align_mode,
                        voice_min=voice_min,
                        voice_max=voice_max,
                        silence_min=silence_min,
                        silence_max=silence_max,
                        overall_min=overall_min,
                        overall_max=overall_max,
                    )

                    # Per-segment smoothing: limit ratio jump vs previous segment.
                    if ratio_info and float(ratio_max_jump) > 1e-9:
                        try:
                            vr0 = float(ratio_info.get("voice_ratio", 1.0) or 1.0)
                            lo = float(prev_voice_ratio) - float(ratio_max_jump)
                            hi = float(prev_voice_ratio) + float(ratio_max_jump)
                            vr1 = float(max(lo, min(hi, vr0)))
                            vr1 = float(max(float(voice_min), min(float(voice_max), vr1)))
                            if abs(vr1 - vr0) > 1e-9:
                                ratio_info["voice_ratio"] = float(vr1)
                                # Keep silence ratio consistent in single mode (or when it's effectively the same).
                                m = str(ratio_info.get("mode") or align_mode).strip().lower()
                                sr0 = float(ratio_info.get("silence_ratio", vr0) or vr0)
                                if m == "single" or abs(float(sr0) - float(vr0)) <= 1e-6:
                                    ratio_info["silence_ratio"] = float(vr1)
                            prev_voice_ratio = float(ratio_info.get("voice_ratio", vr1) or vr1)
                        except Exception:
                            # Best-effort: never fail alignment due to smoothing.
                            prev_voice_ratio = float(ratio_info.get("voice_ratio", prev_voice_ratio) or prev_voice_ratio)
                    elif ratio_info:
                        try:
                            prev_voice_ratio = float(ratio_info.get("voice_ratio", prev_voice_ratio) or prev_voice_ratio)
                        except Exception:
                            pass
                    if abs(float(ratio_info.get("voice_ratio", 1.0)) - 1.0) > float(align_threshold) or abs(
                        float(ratio_info.get("silence_ratio", 1.0)) - 1.0
                    ) > float(align_threshold):
                        tts_audio, _scale_info = apply_scaling_ratio(
                            tts_audio, sample_rate, ratio_info, mode=str(ratio_info.get("mode", align_mode))
                        )
                        if bool(ratio_info.get("clamped")):
                            logger.warning(
                                f"段落 {i}: 语速比例已触发 clamp "
                                f"(voice={ratio_info.get('voice_ratio_raw', 1.0):.3f}->{ratio_info.get('voice_ratio', 1.0):.3f}, "
                                f"silence={ratio_info.get('silence_ratio_raw', 1.0):.3f}->{ratio_info.get('silence_ratio', 1.0):.3f})"
                            )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(f"段落 {i}: 语速对齐失败，将使用原始TTS: {exc}")
                    ratio_info = None
                    en_stats = None
                    zh_stats = None

        # If alignment wasn't applied, reset the smoothing anchor.
        if ratio_info is None:
            prev_voice_ratio = 1.0

        target_samples = int(tts_audio.shape[0])
        target_duration = float(target_samples) / float(sample_rate) if target_samples > 0 else 0.0

        # Tail padding: if speech is shorter than the original segment, pad silence
        # so the corresponding video segment doesn't need to be fast-forwarded as much.
        orig_seg_duration = float(max(0.0, orig_end - orig_start))
        if target_duration > 0 and target_duration < orig_seg_duration and float(tail_pad_max) > 0:
            shortfall = float(orig_seg_duration - target_duration)
            pad_sec = float(min(shortfall, float(tail_pad_max)))
            if pad_sec > 0.01:
                pad_samples = int(round(pad_sec * float(sample_rate)))
                target_samples += pad_samples
                target_duration = float(target_samples) / float(sample_rate)

        # Record adaptive translation (speech segments only).
        start_s = float(t_cursor_samples) / float(sample_rate)
        end_s = float(t_cursor_samples + target_samples) / float(sample_rate)
        out_seg = dict(seg)
        out_seg["start"] = round(start_s, 3)
        out_seg["end"] = round(end_s, 3)
        adaptive_translation.append(out_seg)

        adaptive_plan.append(
            {
                "kind": "speech",
                "index": i,
                "src_start": round(orig_start, 6),
                "src_end": round(orig_end, 6),
                "target_duration": round(target_duration, 6),
                "target_samples": int(target_samples),
                "speech_rate_mode": (ratio_info.get("mode") if ratio_info else None),
                "voice_ratio": (round(float(ratio_info.get("voice_ratio", 1.0)), 6) if ratio_info else None),
                "silence_ratio": (round(float(ratio_info.get("silence_ratio", 1.0)), 6) if ratio_info else None),
                "voice_ratio_raw": (round(float(ratio_info.get("voice_ratio_raw", 1.0)), 6) if ratio_info else None),
                "voice_ratio_rate_raw": (round(float(ratio_info.get("voice_ratio_rate_raw", 1.0)), 6) if ratio_info else None),
                "voice_ratio_budget_raw": (
                    round(float(ratio_info.get("voice_ratio_budget_raw", 1.0)), 6) if ratio_info else None
                ),
                "speech_rate_budget_weight": (
                    round(float(ratio_info.get("speech_rate_budget_weight", 0.0)), 6) if ratio_info else None
                ),
                "silence_ratio_raw": (round(float(ratio_info.get("silence_ratio_raw", 1.0)), 6) if ratio_info else None),
                "speech_rate_global_bias": (round(float(align_global_bias), 6) if ratio_info else None),
                "speech_rate_en": (round(float(en_stats.get("syllable_rate", 0.0)), 6) if en_stats else None),
                "speech_rate_zh": (round(float(zh_stats.get("syllable_rate", 0.0)), 6) if zh_stats else None),
                "speech_syllables_en": (int(en_stats.get("syllable_count", 0)) if en_stats else None),
                "speech_syllables_zh": (int(zh_stats.get("syllable_count", 0)) if zh_stats else None),
                "en_total_duration": (round(float(en_stats.get("total_duration", 0.0)), 6) if en_stats else None),
                "en_voiced_duration": (round(float(en_stats.get("voiced_duration", 0.0)), 6) if en_stats else None),
                "en_silence_duration": (round(float(en_stats.get("silence_duration", 0.0)), 6) if en_stats else None),
                "en_pause_ratio": (round(float(en_stats.get("pause_ratio", 0.0)), 6) if en_stats else None),
                "zh_total_duration": (round(float(zh_stats.get("total_duration", 0.0)), 6) if zh_stats else None),
                "zh_voiced_duration": (round(float(zh_stats.get("voiced_duration", 0.0)), 6) if zh_stats else None),
                "zh_silence_duration": (round(float(zh_stats.get("silence_duration", 0.0)), 6) if zh_stats else None),
                "zh_pause_ratio": (round(float(zh_stats.get("pause_ratio", 0.0)), 6) if zh_stats else None),
                "speech_rate_clamped": (bool(ratio_info.get("clamped")) if ratio_info else None),
            }
        )

        t_cursor_samples += int(max(0, target_samples))

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

            # Compute pause samples using the exact same rounding as the audio cursor.
            n_pause = int(round(float(pause_duration) * float(sample_rate)))
            t_cursor_samples += int(max(0, n_pause))
            adaptive_plan.append(
                {
                    "kind": "pause",
                    "src_start": round(float(pause_src_start), 6),
                    "src_end": round(float(pause_src_end), 6),
                    "target_duration": round(float(pause_duration), 6),
                    "target_samples": int(max(0, n_pause)),
                }
            )

    if t_cursor_samples <= 0:
        raise ValueError("没有有效的TTS音频片段")

    # Save translation_adaptive.json (speech-only cues).
    with open(translation_adaptive_path, "w", encoding="utf-8") as f:
        json.dump(adaptive_translation, f, ensure_ascii=False, indent=2)
    logger.info(f"已生成 translation_adaptive.json: {translation_adaptive_path}")

    # Save plan for video composition (speech + pause segments).
    plan_payload: dict[str, Any] = {
        "version": 1,
        "sample_rate": int(sample_rate),
        "trim_top_db": float(trim_top_db),
        "align_enabled": bool(align_enabled),
        "created_at": float(time.time()),
        "segments": adaptive_plan,
    }
    with open(adaptive_plan_path, "w", encoding="utf-8") as f:
        json.dump(plan_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"已生成 adaptive_plan.json: {adaptive_plan_path}")


def prepare_all_adaptive_alignment_under_folder(folder: str, sample_rate: int = 24000) -> str:
    check_cancelled()
    count = 0
    skipped = 0
    for root, _dirs, files in os.walk(folder):
        check_cancelled()
        if "translation.json" not in files:
            continue
        wavs_dir = os.path.join(root, "wavs")
        if not os.path.exists(wavs_dir):
            continue
        try:
            prepare_adaptive_alignment(root, sample_rate=int(sample_rate))
            count += 1
        except Exception as exc:  # pylint: disable=broad-except
            skipped += 1
            logger.warning(f"自适应对轴准备失败（忽略）: {root} ({exc})")

    parts = []
    if count:
        parts.append(f"处理 {count} 个")
    if skipped:
        parts.append(f"失败 {skipped} 个")
    msg = f"自适应对轴准备完成: {folder}" + (f"（{', '.join(parts)}）" if parts else "")
    logger.info(msg)
    return msg

