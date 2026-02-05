from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from typing import Any

import librosa
import numpy as np
from loguru import logger

from ...interrupts import check_cancelled
from ...speech_rate import apply_scaling_ratio
from ...utils import save_wav, save_wav_norm, valid_file

from .fs import _is_stale


_AUDIO_COMBINED_META_NAME = ".audio_combined.json"
# Bump when the mixing/output semantics change.
# v2: audio_combined = audio_tts + instruments (NO original vocals mixed in)
# v3: match TTS loudness to original vocals peak + normalize output;
#     adaptive mode no longer phase-vocoder stretches TTS (avoid “回音/空洞”感)
# v4: restore original mix balance by estimating stem scales against audio.wav;
#     match TTS loudness to original vocals (RMS on active samples) instead of peak.
# v5: estimate voice/BGM levels on multiple windows and pick the best-balanced ones;
#     log original voice/BGM loudness + ratio to make debugging easy.
# v6: improve window search by probing stems first (avoid missing music-heavy moments).
# v7: adaptive mode aligns instruments per adaptive_plan (fix BGM drift vs stretched video).
# v8: adaptive mode optionally applies speech-rate based TTS time-scale modification (TSM) per segment.
# v9: refine speech-rate alignment metadata and clamp behavior.
# v10: refine EN stats by using segment start/end bounds (affects speech-rate alignment output).
# v11: switch to subtitle syllable counting for speech rate alignment.
# v12: normalize EN syllable counting for numbers/initialisms (affects alignment ratios).
# v13: use VAD-based voiced duration + global bias for alignment (avoid overall pacing drift).
# v14: blend speech-rate ratio with time-budget ratio (stabilize per-segment pacing).
_AUDIO_COMBINED_MIX_VERSION = 14


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


def _audio_combined_needs_rebuild(
    folder: str,
    *,
    adaptive_segment_stretch: bool,
    sample_rate: int = 24000,
) -> bool:
    audio_combined_path = os.path.join(folder, "audio_combined.wav")
    if not valid_file(audio_combined_path, min_bytes=44):
        logger.debug("audio_combined 需要重建: 文件不存在或过小")
        return True

    meta = _read_audio_combined_meta(folder)
    if not meta:
        logger.debug("audio_combined 需要重建: 元数据文件不存在")
        return True
    if int(meta.get("mix_version") or 0) != int(_AUDIO_COMBINED_MIX_VERSION):
        logger.debug(
            f"audio_combined 需要重建: mix_version 不匹配 ({meta.get('mix_version')} != {_AUDIO_COMBINED_MIX_VERSION})"
        )
        return True
    if bool(meta.get("adaptive_segment_stretch")) != bool(adaptive_segment_stretch):
        logger.debug("audio_combined 需要重建: adaptive_segment_stretch 参数不匹配")
        return True
    if int(meta.get("sample_rate") or 0) != int(sample_rate):
        logger.debug(f"audio_combined 需要重建: sample_rate 不匹配 ({meta.get('sample_rate')} != {sample_rate})")
        return True

    # audio_combined depends on: TTS wavs, translation timeline/text (pauses), instruments, and vocals (for loudness match)
    deps = [
        os.path.join(folder, "translation.json"),
        os.path.join(folder, "wavs", ".tts_done.json"),
        os.path.join(folder, "audio.wav"),
        os.path.join(folder, "audio_instruments.wav"),
        os.path.join(folder, "audio_vocals.wav"),
    ]
    if _is_stale(audio_combined_path, deps):
        logger.debug("audio_combined 需要重建: 依赖文件比 audio_combined.wav 更新")
        return True
    return False


def _atempo_chain(tempo: float) -> list[str]:
    """
    Build an FFmpeg `atempo` filter chain for an arbitrary tempo factor.

    FFmpeg限制：单个 atempo 仅支持 [0.5, 2.0]，超出范围需链式组合。
    """
    t = float(tempo)
    if not (t > 0.0):
        t = 1.0
    # Guard against insane values to avoid infinite loops.
    t = float(max(0.01, min(t, 100.0)))

    parts: list[float] = []
    while t < 0.5:
        parts.append(0.5)
        t /= 0.5
    while t > 2.0:
        parts.append(2.0)
        t /= 2.0
    parts.append(t)

    out: list[str] = []
    for p in parts:
        if not (p > 0.0):
            continue
        out.append(f"atempo={float(p):.9f}")
    return out or ["atempo=1.0"]


def _run_process_with_cancel(cmd: list[str]) -> None:
    """
    Run a subprocess while keeping cancellation responsive.

    This is a lightweight variant used for short-lived FFmpeg helpers.
    """
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        while True:
            check_cancelled()
            rc = proc.poll()
            if rc is not None:
                break
            time.sleep(0.1)
        stdout = ""
        stderr = ""
        try:
            stdout, stderr = proc.communicate(timeout=1)
        except Exception:
            pass
        if proc.returncode not in (0, None):
            raise subprocess.CalledProcessError(proc.returncode or 1, cmd, output=stdout, stderr=stderr)
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


def _render_adaptive_instruments(
    folder: str,
    *,
    instruments_path: str,
    adaptive_plan: list[dict[str, Any]],
    sample_rate: int,
) -> np.ndarray:
    """
    Render an "adaptive-aligned" instruments track following adaptive_plan.

    Goal: In adaptive_segment_stretch mode, the video is built by per-segment trim+setpts+concat.
    The instruments track must follow the SAME segment boundaries; otherwise BGM will drift.
    """
    if not adaptive_plan:
        return np.zeros((0,), dtype=np.float32)
    if not instruments_path or not os.path.exists(instruments_path):
        return np.zeros((0,), dtype=np.float32)

    # Use hidden temp files inside the job folder (safe under multi-job runs).
    tag = f"{int(time.time() * 1000)}_{threading.get_ident()}"
    script_path = os.path.join(folder, f".ffmpeg_instruments_adaptive_{tag}.txt")
    out_path = os.path.join(folder, f".instruments_adaptive_{tag}.wav")

    # Build filter_complex_script for audio:
    # - atrim by src_start/src_end
    # - aresample to target SR
    # - atempo to match target duration
    # - apad + atrim(end_sample) to guarantee exact samples per segment (keeps segment boundaries aligned)
    lines: list[str] = []
    a_labels: list[str] = []
    for idx, seg in enumerate(adaptive_plan):
        check_cancelled()
        try:
            s0 = float(seg.get("src_start", 0.0) or 0.0)
            s1 = float(seg.get("src_end", 0.0) or 0.0)
        except Exception:
            s0, s1 = 0.0, 0.0
        if s1 < s0:
            s0, s1 = s1, s0
        s0 = max(0.0, float(s0))
        s1 = max(0.0, float(s1))
        src_dur = max(0.001, float(s1 - s0))

        target_samples = None
        try:
            ts = seg.get("target_samples")
            if ts is not None:
                target_samples = int(ts)
        except Exception:
            target_samples = None
        if target_samples is None:
            try:
                out_dur = float(seg.get("target_duration", 0.0) or 0.0)
            except Exception:
                out_dur = 0.0
            if not (out_dur > 0.0):
                out_dur = src_dur
            target_samples = int(round(out_dur * float(sample_rate)))

        # Avoid zero-length segments: they break concat and will desync boundaries.
        if target_samples <= 0:
            target_samples = int(round(0.001 * float(sample_rate)))

        out_dur_exact = float(target_samples) / float(sample_rate)
        tempo = float(src_dur) / float(out_dur_exact) if out_dur_exact > 0 else 1.0
        chain = _atempo_chain(tempo)

        a = f"a{idx}"
        a_labels.append(f"[{a}]")
        filters: list[str] = [
            f"atrim=start={s0:.6f}:end={s1:.6f}",
            "asetpts=PTS-STARTPTS",
            f"aresample={int(sample_rate)}",
            "aformat=channel_layouts=mono",
            *chain,
            f"apad=pad_len={int(target_samples)}",
            f"atrim=end_sample={int(target_samples)}",
            "asetpts=PTS-STARTPTS",
        ]
        lines.append(f"[0:a]{','.join(filters)}[{a}]")

    concat_in = "".join(a_labels)
    lines.append(f"{concat_in}concat=n={len(a_labels)}:v=0:a=1[ainst]")

    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(";\n".join(lines) + "\n")
    except Exception as exc:
        raise RuntimeError(f"写入 instruments 自适应滤镜脚本失败: {exc}") from exc

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        instruments_path,
        "-filter_complex_script",
        script_path,
        "-map",
        "[ainst]",
        "-c:a",
        "pcm_s16le",
        "-ar",
        str(int(sample_rate)),
        "-ac",
        "1",
        out_path,
    ]

    try:
        _run_process_with_cancel(cmd)
        y, _ = librosa.load(out_path, sr=int(sample_rate), mono=True)
        return y.astype(np.float32, copy=False)
    finally:
        # Best-effort cleanup of temp files.
        for p in (out_path, script_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


def _ensure_audio_combined(
    folder: str,
    adaptive_segment_stretch: bool = False,
    sample_rate: int = 24000,
) -> None:
    """
    生成 audio_combined.wav 和 audio_tts.wav

    如果 adaptive_segment_stretch=True:
        - 不使用相位声码器 time_stretch（避免“回音/空洞”感）；可选使用 audiostretchy 做 TSM 语速对齐
        - 逐段拼接（可裁剪首尾静音）并插入短停顿，生成 translation_adaptive.json
        - 同时生成 adaptive_plan.json，用于逐段拉伸/裁剪原视频并 concat
    否则:
        - 按顺序拼接TTS音频片段
    """
    check_cancelled()

    audio_combined_path = os.path.join(folder, "audio_combined.wav")
    audio_tts_path = os.path.join(folder, "audio_tts.wav")
    wavs_folder = os.path.join(folder, "wavs")
    translation_path = os.path.join(folder, "translation.json")
    adaptive_plan_path = os.path.join(folder, "adaptive_plan.json")
    audio_instruments_path = os.path.join(folder, "audio_instruments.wav")

    # 检查必要文件
    if not os.path.exists(wavs_folder):
        raise FileNotFoundError(f"缺少 wavs 目录: {wavs_folder}")

    # 自适应模式的输入时间轴必须来自 translation.json（原始 ASR 时间戳）。
    # translation_adaptive.json 是本函数在自适应模式下生成的输出时间轴。
    if not os.path.exists(translation_path):
        raise FileNotFoundError(f"缺少翻译文件: {translation_path}")

    with open(translation_path, "r", encoding="utf-8") as f:
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
        # 自适应对轴：生成 translation_adaptive.json + adaptive_plan.json（若缺失或过期）。
        from .. import adaptive_align

        check_cancelled()
        adaptive_align.prepare_adaptive_alignment(folder, sample_rate=int(sample_rate))

        # 按 adaptive_plan.json 重建“新时间轴”的 TTS 音轨（speech + pause），保证后续混音/视频合成一致。
        try:
            with open(adaptive_plan_path, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            plan_segments = list(payload.get("segments") or [])
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"读取 adaptive_plan.json 失败: {adaptive_plan_path} ({exc})") from exc

        if not plan_segments:
            raise RuntimeError(f"adaptive_plan.json 缺失或为空: {adaptive_plan_path}")

        trim_top_db = 35.0
        audio_segments: list[np.ndarray] = []

        for pseg in plan_segments:
            check_cancelled()
            kind = str(pseg.get("kind") or "").strip().lower()
            if kind == "pause":
                try:
                    n = int(pseg.get("target_samples") or 0)
                except Exception:
                    n = 0
                if n > 0:
                    audio_segments.append(np.zeros((n,), dtype=np.float32))
                continue

            if kind != "speech":
                continue

            try:
                idx = int(pseg.get("index") or 0)
            except Exception:
                idx = 0
            if idx < 0 or idx >= len(wav_files):
                raise IndexError(f"adaptive_plan.json 段落 index 越界: {idx} (len(wavs)={len(wav_files)})")

            wav_path = os.path.join(wavs_folder, wav_files[idx])
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"TTS音频文件不存在: {wav_path}")

            tts_audio, _ = librosa.load(wav_path, sr=sample_rate, mono=True)
            tts_audio = np.asarray(tts_audio, dtype=np.float32).reshape(-1)

            # Trim leading/trailing silence (must match plan generation).
            if tts_audio.size > 0:
                try:
                    trimmed, _idx = librosa.effects.trim(tts_audio, top_db=float(trim_top_db))
                    if trimmed is not None and trimmed.size > 0:
                        tts_audio = np.asarray(trimmed, dtype=np.float32).reshape(-1)
                except Exception:
                    pass

            try:
                target_samples = int(pseg.get("target_samples") or int(tts_audio.shape[0]))
            except Exception:
                target_samples = int(tts_audio.shape[0])

            # Re-apply speech-rate scaling only when it actually affected output length.
            vr = pseg.get("voice_ratio", None)
            if (
                vr is not None
                and target_samples > 0
                and tts_audio.size > 0
                and int(tts_audio.shape[0]) != int(target_samples)
            ):
                try:
                    ratio_info = {"voice_ratio": float(vr)}
                    tts_audio, _scale_info = apply_scaling_ratio(tts_audio, sample_rate, ratio_info)
                    tts_audio = np.asarray(tts_audio, dtype=np.float32).reshape(-1)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(f"按计划应用语速对齐失败，将回退到原始TTS: {exc}")

            if target_samples <= 0:
                continue

            if tts_audio.shape[0] < target_samples:
                tts_audio = np.pad(tts_audio, (0, target_samples - int(tts_audio.shape[0])), mode="constant")
            else:
                tts_audio = tts_audio[:target_samples]

            audio_segments.append(tts_audio.astype(np.float32, copy=False))

        if not audio_segments:
            raise ValueError("没有有效的TTS音频片段")

        audio_tts = np.concatenate(audio_segments).astype(np.float32, copy=False)

    else:
        # 非自适应模式：按顺序拼接
        audio_segments: list[np.ndarray] = []
        for _i, wav_file in enumerate(wav_files[: len(translation)]):
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

    # 混音校准：
    # - audio_vocals.wav / audio_instruments.wav 是 Demucs 分离产物（历史上会被各自归一化到峰值）
    # - 直接 1:1 相加会破坏原视频的“人声/伴奏相对音量”（常见表现：背景音乐过低或过高）
    # 因此这里尝试用 audio.wav（原混合音轨）对两个 stem 做线性拟合，恢复更接近原始的比例，
    # 同时用“有效样本 RMS”匹配 TTS 音量到原人声（比 peak 更贴近听感）。
    audio_vocals_path = os.path.join(folder, "audio_vocals.wav")
    audio_mix_path = os.path.join(folder, "audio.wav")

    def _active_rms(y: np.ndarray, *, rel_th: float = 0.02) -> float:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size <= 0:
            return 0.0
        peak = float(max(abs(float(np.max(y))), abs(float(np.min(y)))))
        if not (peak > 1e-6):
            return 0.0
        th = float(peak) * float(rel_th)
        mask = np.abs(y) > th
        if bool(np.any(mask)):
            yy = y[mask].astype(np.float32, copy=False)
            return float(np.sqrt(float(np.mean(np.square(yy), dtype=np.float64))))
        return float(np.sqrt(float(np.mean(np.square(y), dtype=np.float64))))

    def _estimate_stem_scales(mix: np.ndarray, voc: np.ndarray, inst: np.ndarray) -> tuple[float, float]:
        # Solve least squares for: mix ~= a*voc + b*inst  (2x2 normal equations, constant memory)
        n = int(min(mix.shape[0], voc.shape[0], inst.shape[0]))
        if n <= 0:
            return 1.0, 1.0
        m = mix[:n].astype(np.float64, copy=False)
        v = voc[:n].astype(np.float64, copy=False)
        i = inst[:n].astype(np.float64, copy=False)

        vv = float(np.dot(v, v))
        ii = float(np.dot(i, i))
        vi = float(np.dot(v, i))
        vm = float(np.dot(v, m))
        im = float(np.dot(i, m))

        det = vv * ii - vi * vi
        if not (det > 1e-8):
            return 1.0, 1.0

        a = (vm * ii - im * vi) / det
        b = (im * vv - vm * vi) / det
        # Clamp to sane positive range; negative values indicate bad fit region.
        a = float(max(0.0, min(a, 5.0)))
        b = float(max(0.0, min(b, 5.0)))
        if not (a > 0.0):
            a = 1.0
        if not (b > 0.0):
            b = 1.0
        return a, b

    stem_vocal_scale = 1.0
    stem_inst_scale = 1.0
    tts_scale = 1.0
    orig_voice_rms: float | None = None
    orig_bgm_rms: float | None = None
    orig_ratio_db: float | None = None

    def _safe_db_ratio(num: float, den: float) -> float | None:
        n = float(num)
        d = float(den)
        if not (n > 0.0) or not (d > 0.0):
            return None
        return float(20.0 * np.log10(n / d))

    def _duration_seconds(path: str) -> float | None:
        try:
            d = float(librosa.get_duration(path=path))
            return d if d > 0 else None
        except Exception:
            return None

    if len(audio_tts) > 0 and os.path.exists(audio_vocals_path) and os.path.exists(audio_instruments_path) and os.path.exists(audio_mix_path):
        try:
            check_cancelled()
            # 校准策略（关键点）：
            # 单一窗口很容易落在“纯人声/纯音乐/片头片尾”导致估计偏差，从而把 BGM 拉得几乎听不见。
            # 这里改为采样多个窗口，按“人声与BGM都显著”的窗口打分，选 Top-K 后取中位数作为全局比例。
            total_dur = (
                _duration_seconds(audio_mix_path)
                or _duration_seconds(audio_vocals_path)
                or _duration_seconds(audio_instruments_path)
            )

            try:
                calib_window = float(os.getenv("YOUDUB_MIX_CALIB_WINDOW_SECONDS", "30") or "30")
            except Exception:
                calib_window = 30.0
            try:
                calib_windows = int(os.getenv("YOUDUB_MIX_CALIB_MAX_WINDOWS", "10") or "10")
            except Exception:
                calib_windows = 10

            calib_window = float(max(8.0, min(calib_window, 120.0)))
            calib_windows = int(max(1, min(calib_windows, 25)))
            topk = int(max(1, min(5, calib_windows)))

            # Stage 1: coarse scan using only stems (fast) to find windows where BOTH vocals & BGM are present.
            if total_dur is None:
                probe_starts = [0.0]
            else:
                start_min = 60.0 if (float(total_dur) > 60.0 + float(calib_window) + 1e-6) else 0.0
                start_max = max(float(start_min), float(total_dur) - float(calib_window))
                if start_max <= start_min + 1e-6:
                    probe_starts = [float(start_min)]
                else:
                    # Probe stride: trade accuracy for speed; keep it small enough to not miss "music comes in" moments.
                    try:
                        probe_stride = float(os.getenv("YOUDUB_MIX_CALIB_PROBE_STRIDE_SECONDS", "15") or "15")
                    except Exception:
                        probe_stride = 15.0
                    probe_stride = float(max(5.0, min(probe_stride, 120.0)))
                    probe_starts = [
                        float(x) for x in np.arange(start_min, start_max + 1e-6, probe_stride).tolist()
                    ]

            probe_scores: list[tuple[float, float]] = []
            for s in probe_starts:
                check_cancelled()
                try:
                    voc_p, _ = librosa.load(
                        audio_vocals_path,
                        sr=sample_rate,
                        mono=True,
                        offset=max(0.0, float(s)),
                        duration=float(calib_window),
                    )
                    inst_p, _ = librosa.load(
                        audio_instruments_path,
                        sr=sample_rate,
                        mono=True,
                        offset=max(0.0, float(s)),
                        duration=float(calib_window),
                    )
                except Exception:
                    continue
                if voc_p.size <= 0 or inst_p.size <= 0:
                    continue
                score = float(min(_active_rms(voc_p), _active_rms(inst_p)))
                probe_scores.append((float(score), float(s)))

            # Pick the best windows for full evaluation (with mix + scale fitting).
            if probe_scores:
                probe_scores.sort(key=lambda x: x[0], reverse=True)
                # Keep only top-N unique-ish starts (avoid duplicates in very flat audio).
                starts = [s for _score, s in probe_scores[: int(calib_windows)]]
            else:
                starts = [0.0]

            samples: list[dict[str, float]] = []
            for s in starts:
                check_cancelled()
                try:
                    voc, _ = librosa.load(
                        audio_vocals_path,
                        sr=sample_rate,
                        mono=True,
                        offset=max(0.0, float(s)),
                        duration=float(calib_window),
                    )
                    inst, _ = librosa.load(
                        audio_instruments_path,
                        sr=sample_rate,
                        mono=True,
                        offset=max(0.0, float(s)),
                        duration=float(calib_window),
                    )
                    mix, _ = librosa.load(
                        audio_mix_path,
                        sr=sample_rate,
                        mono=True,
                        offset=max(0.0, float(s)),
                        duration=float(calib_window),
                    )
                except Exception:
                    continue

                if voc.size <= 0 or inst.size <= 0 or mix.size <= 0:
                    continue

                a, b = _estimate_stem_scales(
                    mix.astype(np.float32, copy=False),
                    voc.astype(np.float32, copy=False),
                    inst.astype(np.float32, copy=False),
                )
                voice_rms = _active_rms((voc.astype(np.float32, copy=False) * np.float32(a)))
                bgm_rms = _active_rms((inst.astype(np.float32, copy=False) * np.float32(b)))

                both_score = float(min(voice_rms, bgm_rms))
                samples.append(
                    {
                        "start": float(s),
                        "a": float(a),
                        "b": float(b),
                        "voice_rms": float(voice_rms),
                        "bgm_rms": float(bgm_rms),
                        "both": float(both_score),
                    }
                )

            if samples:
                samples.sort(key=lambda x: float(x.get("both", 0.0)), reverse=True)
                picked = samples[: min(int(topk), len(samples))]

                stem_vocal_scale = float(np.median([x["a"] for x in picked]))
                stem_inst_scale = float(np.median([x["b"] for x in picked]))
                orig_voice_rms = float(np.median([x["voice_rms"] for x in picked]))
                orig_bgm_rms = float(np.median([x["bgm_rms"] for x in picked]))
                orig_ratio_db = _safe_db_ratio(float(orig_bgm_rms), float(orig_voice_rms))

                if orig_ratio_db is not None:
                    logger.info(
                        "原视频响度估计(Top窗口中位数): "
                        f"voice_active_rms={orig_voice_rms:.4f}, bgm_active_rms={orig_bgm_rms:.4f}, "
                        f"bgm_vs_voice={orig_ratio_db:.1f}dB, "
                        f"stem_vocal_scale={stem_vocal_scale:.3f}, stem_instruments_scale={stem_inst_scale:.3f}, "
                        f"windows_used={len(picked)}/{len(samples)}"
                    )
                else:
                    logger.info(
                        "原视频响度估计(Top窗口中位数): "
                        f"voice_active_rms={orig_voice_rms:.4f}, bgm_active_rms={orig_bgm_rms:.4f}, "
                        f"stem_vocal_scale={stem_vocal_scale:.3f}, stem_instruments_scale={stem_inst_scale:.3f}, "
                        f"windows_used={len(picked)}/{len(samples)}"
                    )

            # Match TTS loudness to original voice (active RMS).
            tts_r = _active_rms(audio_tts)
            if orig_voice_rms is not None and orig_voice_rms > 0.0 and tts_r > 0.0:
                tts_scale = float(float(orig_voice_rms) / float(tts_r))
                tts_scale = float(max(0.2, min(tts_scale, 3.0)))
                audio_tts = (audio_tts * np.float32(tts_scale)).astype(np.float32, copy=False)
                if orig_ratio_db is not None:
                    logger.info(
                        f"TTS 音量已匹配到原人声(active RMS): scale={tts_scale:.3f} (原BGM/人声={orig_ratio_db:.1f}dB)"
                    )
                else:
                    logger.info(f"TTS 音量已匹配到原人声(active RMS): scale={tts_scale:.3f}")
        except Exception as e:
            logger.warning(f"混音校准/匹配音量失败（将回退默认混音）: {e}")

    # Avoid int16 overflow when saving debug audio_tts.wav (numpy cast may wrap).
    if len(audio_tts) > 0:
        try:
            peak = max(abs(float(np.max(audio_tts))), abs(float(np.min(audio_tts))))
            if peak > 0.99:
                audio_tts = (audio_tts * np.float32(0.99 / peak)).astype(np.float32, copy=False)
        except Exception:
            pass

    # 保存音量匹配后的音轨，便于排查/试听
    check_cancelled()
    save_wav(audio_tts.astype(np.float32, copy=False), audio_tts_path, sample_rate=sample_rate)
    logger.info(f"已生成 audio_tts.wav: {audio_tts_path}")

    # 混合 TTS 音频和背景伴奏（不混入原人声，避免回声/双声）
    check_cancelled()
    if os.path.exists(audio_instruments_path):
        try:
            # 加载/渲染伴奏：
            # - 非自适应：直接用原 instruments（再裁剪/补零到与 TTS 一致）
            # - 自适应：必须按 adaptive_plan 逐段对齐到“拉伸后的视频时间轴”，否则背景音会累计跑轴。
            instruments: np.ndarray
            if adaptive_segment_stretch:
                try:
                    with open(adaptive_plan_path, "r", encoding="utf-8") as f:
                        payload = json.load(f) or {}
                    segs = list(payload.get("segments") or [])
                except Exception:
                    segs = []
                if not segs:
                    raise RuntimeError("adaptive_plan.json 缺失或为空，无法对齐伴奏")
                instruments = _render_adaptive_instruments(
                    folder,
                    instruments_path=audio_instruments_path,
                    adaptive_plan=segs,
                    sample_rate=int(sample_rate),
                )
            else:
                instruments, _ = librosa.load(audio_instruments_path, sr=sample_rate, mono=True)
                instruments = instruments.astype(np.float32, copy=False)
            # Apply stem scale to restore original mix balance when available.
            if float(stem_inst_scale) != 1.0:
                instruments = (instruments * np.float32(stem_inst_scale)).astype(np.float32, copy=False)

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

