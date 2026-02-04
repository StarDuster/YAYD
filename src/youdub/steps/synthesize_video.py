import json
import math
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any

import librosa
import numpy as np
from loguru import logger

from ..interrupts import check_cancelled, sleep_with_cancel
from ..speech_rate import apply_scaling_ratio
from ..utils import save_wav, save_wav_norm, valid_file

from .synthesize_video_fs import _is_stale, _mtime
from .synthesize_video_subtitles import (
    _ass_escape_text,
    _best_text_by_overlap,
    _bilingual_source_text,
    _calc_subtitle_style_params,
    _calc_subtitle_wrap_chars,
    _ensure_bilingual_source_text,
    _is_cjk_char,
    _read_transcript_segments,
    _wrap_cjk,
    _wrap_words,
    format_timestamp,
    format_timestamp_ass,
    generate_bilingual_ass,
    generate_monolingual_ass,
    generate_srt,
    split_text,
    wrap_text,
)


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

_VIDEO_META_NAME = ".video_synth.json"
# Bump when the video output semantics/config keys change.
# v3: subtitles style/scale tweaks (font size, wrap, original_size for libass)
# v4: make wrap heuristic more conservative; shrink default font size further
# v5: shrink non-bilingual subtitle font size further (1080p -> ~26)
# v6: shrink non-bilingual subtitle size further (1080p -> ~19) and reduce outline
# v7: use ASS for monolingual subtitles too (ensures font size is respected via PlayRes)
# v8: restore subtitle font scaling (1080p -> ~36) for readability
# v9: separate portrait/landscape subtitle scaling (landscape slightly larger)
# v10: keep 1080p landscape at ~36; portrait slightly smaller
# v11: increase 1080p landscape subtitle size (~49) to match expected readability
# v12: bilingual subtitles no longer truncate source text to first sentence
# v13: remove legacy global speed-up (drop speed_up from metadata)
_VIDEO_META_VERSION = 13

# Video output audio encoding (keep high enough to avoid AAC artifacts).
_VIDEO_AUDIO_SAMPLE_RATE = 48000
_VIDEO_AUDIO_BITRATE = "128k"

# NVENC 并发限制：全自动多视频时避免同时起太多 h264_nvenc 实例导致失败/性能抖动
_NVENC_MAX_CONCURRENCY = 8
_NVENC_SEMAPHORE = threading.BoundedSemaphore(_NVENC_MAX_CONCURRENCY)


@contextmanager
def _nvenc_slot():
    acquired = False
    try:
        # Use short timeouts so cancellation remains responsive while waiting.
        while not acquired:
            check_cancelled()
            acquired = _NVENC_SEMAPHORE.acquire(timeout=0.2)
        yield
    finally:
        if acquired:
            try:
                _NVENC_SEMAPHORE.release()
            except Exception:
                # Best-effort; should never fail with BoundedSemaphore.
                pass


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


def _read_video_meta(folder: str) -> dict[str, Any] | None:
    meta_path = os.path.join(folder, _VIDEO_META_NAME)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else None
    except Exception:
        return None


def _write_video_meta(
    folder: str,
    *,
    subtitles: bool,
    bilingual_subtitle: bool,
    adaptive_segment_stretch: bool,
    fps: int,
    resolution: str,
    use_nvenc: bool,
) -> None:
    meta_path = os.path.join(folder, _VIDEO_META_NAME)
    payload: dict[str, Any] = {
        "version": int(_VIDEO_META_VERSION),
        "subtitles": bool(subtitles),
        "bilingual_subtitle": bool(bilingual_subtitle),
        "adaptive_segment_stretch": bool(adaptive_segment_stretch),
        "fps": int(fps),
        "resolution": str(resolution),
        "use_nvenc": bool(use_nvenc),
        "audio_sample_rate": int(_VIDEO_AUDIO_SAMPLE_RATE),
        "audio_bitrate": str(_VIDEO_AUDIO_BITRATE),
        "created_at": float(time.time()),
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"写入 video 合成元数据失败: {e}")


def _audio_combined_needs_rebuild(
    folder: str,
    *,
    adaptive_segment_stretch: bool,
    sample_rate: int = 24000,
) -> bool:
    audio_combined_path = os.path.join(folder, "audio_combined.wav")
    if not valid_file(audio_combined_path, min_bytes=44):
        logger.debug(f"audio_combined 需要重建: 文件不存在或过小")
        return True

    meta = _read_audio_combined_meta(folder)
    if not meta:
        logger.debug(f"audio_combined 需要重建: 元数据文件不存在")
        return True
    if int(meta.get("mix_version") or 0) != int(_AUDIO_COMBINED_MIX_VERSION):
        logger.debug(f"audio_combined 需要重建: mix_version 不匹配 ({meta.get('mix_version')} != {_AUDIO_COMBINED_MIX_VERSION})")
        return True
    if bool(meta.get("adaptive_segment_stretch")) != bool(adaptive_segment_stretch):
        logger.debug(f"audio_combined 需要重建: adaptive_segment_stretch 参数不匹配")
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
        logger.debug(f"audio_combined 需要重建: 依赖文件比 audio_combined.wav 更新")
        return True
    return False


def _video_up_to_date(
    folder: str,
    *,
    subtitles: bool,
    bilingual_subtitle: bool,
    adaptive_segment_stretch: bool,
    fps: int,
    resolution: str,
    use_nvenc: bool,
    sample_rate: int = 24000,
) -> bool:
    output_video = os.path.join(folder, "video.mp4")
    if not valid_file(output_video, min_bytes=1024):
        logger.debug(f"video 过期原因: video.mp4 不存在或过小")
        return False

    # If audio needs rebuild (e.g. old mixing semantics), video is also stale.
    if _audio_combined_needs_rebuild(
        folder, adaptive_segment_stretch=adaptive_segment_stretch, sample_rate=sample_rate
    ):
        logger.debug(f"video 过期原因: audio_combined 需要重建")
        return False

    meta = _read_video_meta(folder)
    if not meta:
        logger.debug(f"video 过期原因: 元数据文件不存在")
        return False
    if int(meta.get("version") or 0) != int(_VIDEO_META_VERSION):
        logger.debug(f"video 过期原因: 元数据版本不匹配 ({meta.get('version')} != {_VIDEO_META_VERSION})")
        return False
    if bool(meta.get("subtitles")) != bool(subtitles):
        logger.debug(f"video 过期原因: subtitles 参数不匹配")
        return False
    if bool(meta.get("bilingual_subtitle")) != bool(bilingual_subtitle):
        logger.debug(f"video 过期原因: bilingual_subtitle 参数不匹配")
        return False
    if bool(meta.get("adaptive_segment_stretch")) != bool(adaptive_segment_stretch):
        logger.debug(f"video 过期原因: adaptive_segment_stretch 参数不匹配")
        return False
    if int(meta.get("fps") or 0) != int(fps):
        logger.debug(f"video 过期原因: fps 参数不匹配 ({meta.get('fps')} != {fps})")
        return False
    if str(meta.get("resolution") or "") != str(resolution):
        logger.debug(f"video 过期原因: resolution 参数不匹配 ({meta.get('resolution')} != {resolution})")
        return False
    if bool(meta.get("use_nvenc")) != bool(use_nvenc):
        logger.debug(f"video 过期原因: use_nvenc 参数不匹配")
        return False
    if int(meta.get("audio_sample_rate") or 0) != int(_VIDEO_AUDIO_SAMPLE_RATE):
        logger.debug(f"video 过期原因: audio_sample_rate 不匹配 ({meta.get('audio_sample_rate')} != {_VIDEO_AUDIO_SAMPLE_RATE})")
        return False
    if str(meta.get("audio_bitrate") or "") != str(_VIDEO_AUDIO_BITRATE):
        logger.debug(f"video 过期原因: audio_bitrate 不匹配 ({meta.get('audio_bitrate')} != {_VIDEO_AUDIO_BITRATE})")
        return False

    deps = [
        os.path.join(folder, "download.mp4"),
        os.path.join(folder, "audio_combined.wav"),
    ]
    # NOTE:
    # Do NOT include _VIDEO_META_NAME in deps: we write it after video.mp4, which would
    # make the video look "stale" forever due to mtime ordering on modern filesystems.
    if adaptive_segment_stretch:
        deps.append(os.path.join(folder, "adaptive_plan.json"))
    if subtitles:
        tr = "translation_adaptive.json" if adaptive_segment_stretch else "translation.json"
        deps.append(os.path.join(folder, tr))
    if _is_stale(output_video, deps):
        logger.debug(f"video 过期原因: 依赖文件比 video.mp4 更新")
        return False
    return True


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


def _get_video_duration(video_path: str) -> float:
    """获取视频时长（秒）"""
    check_cancelled()
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=duration', '-of', 'json', video_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data.get('streams', [{}])[0].get('duration', 0))
        if duration > 0:
            return duration
    except Exception as e:
        logger.warning(f"获取视频时长失败: {e}")
    
    # 回退：尝试从音频流获取
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'a:0',
        '-show_entries', 'stream=duration', '-of', 'json', video_path
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data.get('streams', [{}])[0].get('duration', 0))
        if duration > 0:
            return duration
    except Exception:
        pass
    
    return 0.0


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
    
    audio_combined_path = os.path.join(folder, 'audio_combined.wav')
    audio_tts_path = os.path.join(folder, 'audio_tts.wav')
    wavs_folder = os.path.join(folder, 'wavs')
    translation_path = os.path.join(folder, 'translation.json')
    translation_adaptive_path = os.path.join(folder, 'translation_adaptive.json')
    adaptive_plan_path = os.path.join(folder, "adaptive_plan.json")
    audio_instruments_path = os.path.join(folder, 'audio_instruments.wav')
    
    # 检查必要文件
    if not os.path.exists(wavs_folder):
        raise FileNotFoundError(f"缺少 wavs 目录: {wavs_folder}")
    
    # 自适应模式的输入时间轴必须来自 translation.json（原始 ASR 时间戳）。
    # translation_adaptive.json 是本函数在自适应模式下生成的输出时间轴。
    if not os.path.exists(translation_path):
        raise FileNotFoundError(f"缺少翻译文件: {translation_path}")
    
    with open(translation_path, 'r', encoding='utf-8') as f:
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
        from . import adaptive_align

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
            sr_mode = pseg.get("speech_rate_mode", None)
            sil = pseg.get("silence_ratio", None)
            if vr is not None and target_samples > 0 and tts_audio.size > 0 and int(tts_audio.shape[0]) != int(target_samples):
                try:
                    voice_ratio = float(vr)
                    silence_ratio = float(sil) if sil is not None else float(voice_ratio)
                    ratio_info = {"voice_ratio": float(voice_ratio), "silence_ratio": float(silence_ratio)}
                    tts_audio, _scale_info = apply_scaling_ratio(
                        tts_audio, sample_rate, ratio_info, mode=str(sr_mode or "single")
                    )
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
        for _i, wav_file in enumerate(wav_files[:len(translation)]):
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

    if (
        len(audio_tts) > 0
        and os.path.exists(audio_vocals_path)
        and os.path.exists(audio_instruments_path)
        and os.path.exists(audio_mix_path)
    ):
        try:
            check_cancelled()
            # 校准策略（关键点）：
            # 单一窗口很容易落在“纯人声/纯音乐/片头片尾”导致估计偏差，从而把 BGM 拉得几乎听不见。
            # 这里改为采样多个窗口，按“人声与BGM都显著”的窗口打分，选 Top-K 后取中位数作为全局比例。
            total_dur = _duration_seconds(audio_mix_path) or _duration_seconds(audio_vocals_path) or _duration_seconds(audio_instruments_path)

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
                    probe_starts = [float(x) for x in np.arange(start_min, start_max + 1e-6, probe_stride).tolist()]

            probe_scores: list[tuple[float, float]] = []
            for s in probe_starts:
                check_cancelled()
                try:
                    voc_p, _ = librosa.load(
                        audio_vocals_path, sr=sample_rate, mono=True, offset=max(0.0, float(s)), duration=float(calib_window)
                    )
                    inst_p, _ = librosa.load(
                        audio_instruments_path, sr=sample_rate, mono=True, offset=max(0.0, float(s)), duration=float(calib_window)
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

                a, b = _estimate_stem_scales(mix.astype(np.float32, copy=False), voc.astype(np.float32, copy=False), inst.astype(np.float32, copy=False))
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
                        f"TTS 音量已匹配到原人声(active RMS): scale={tts_scale:.3f} "
                        f"(原BGM/人声={orig_ratio_db:.1f}dB)"
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


def synthesize_video(
    folder: str, 
    subtitles: bool = True, 
    fps: int = 30, 
    resolution: str = '1080p',
    use_nvenc: bool = False,
    adaptive_segment_stretch: bool = False,
    bilingual_subtitle: bool = False,
) -> None:
    check_cancelled()
    output_video = os.path.join(folder, "video.mp4")

    # 缓存元数据中的“有效参数”（自适应模式下 fps 不参与输出）
    meta_fps = 0 if adaptive_segment_stretch else int(fps)

    # NOTE:
    # 以前只要 video.mp4 存在就直接跳过，这会导致：
    # - 混音逻辑修复后仍复用旧的 audio_combined/video（用户听感依旧“回声/双声”）
    # - translation/音频更新后视频也不重建
    # 因此这里改为基于 mtime + 元数据判断是否需要重建。
    if _video_up_to_date(
        folder,
        subtitles=subtitles,
        bilingual_subtitle=bilingual_subtitle,
        adaptive_segment_stretch=adaptive_segment_stretch,
        fps=meta_fps,
        resolution=resolution,
        use_nvenc=use_nvenc,
    ):
        logger.info(f"已合成视频: {folder}")
        return
    if os.path.exists(output_video):
        try:
            if os.path.getsize(output_video) < 1024:
                logger.warning(f"video.mp4 疑似无效(过小)，将重新生成: {output_video}")
            else:
                logger.info(f"video.mp4 已存在但已过期，将重新生成: {output_video}")
            os.remove(output_video)
        except Exception:
            # Best-effort: proceed to re-generate
            pass
    
    translation_path = os.path.join(folder, 'translation.json')
    translation_adaptive_path = os.path.join(folder, 'translation_adaptive.json')
    adaptive_plan_path = os.path.join(folder, "adaptive_plan.json")
    input_audio = os.path.join(folder, 'audio_combined.wav')
    input_video = os.path.join(folder, 'download.mp4')
    
    # 自适应模式下，除了 audio_combined.wav，还需要 translation_adaptive.json / adaptive_plan.json。
    if adaptive_segment_stretch:
        if _audio_combined_needs_rebuild(folder, adaptive_segment_stretch=True) or not (
            os.path.exists(input_audio) and os.path.exists(translation_adaptive_path)
        ):
            check_cancelled()
            logger.info("自适应模式所需文件不存在，正在生成 audio/translation_adaptive/plan ...")
            _ensure_audio_combined(folder, adaptive_segment_stretch=True)
        # adaptive_plan.json 缺失时，可由 translation.json + translation_adaptive.json 重建（无需重跑TTS）。
        if not os.path.exists(adaptive_plan_path) and os.path.exists(translation_path) and os.path.exists(
            translation_adaptive_path
        ):
            try:
                with open(translation_path, "r", encoding="utf-8") as f:
                    _orig = json.load(f) or []
                with open(translation_adaptive_path, "r", encoding="utf-8") as f:
                    _adapt = json.load(f) or []
                segs: list[dict[str, Any]] = []
                for i in range(min(len(_orig), len(_adapt))):
                    o0 = float(_orig[i].get("start", 0.0) or 0.0)
                    o1 = float(_orig[i].get("end", 0.0) or 0.0)
                    if o1 < o0:
                        o0, o1 = o1, o0
                    a0 = float(_adapt[i].get("start", 0.0) or 0.0)
                    a1 = float(_adapt[i].get("end", 0.0) or 0.0)
                    if a1 < a0:
                        a0, a1 = a1, a0
                    segs.append(
                        {
                            "kind": "speech",
                            "index": i,
                            "src_start": round(o0, 6),
                            "src_end": round(o1, 6),
                            "target_duration": round(max(0.001, a1 - a0), 6),
                        }
                    )
                    if i < (min(len(_orig), len(_adapt)) - 1):
                        # Pause duration in output timeline comes from adaptive timestamps.
                        next_a0 = float(_adapt[i + 1].get("start", a1) or a1)
                        pause_dur = max(0.0, next_a0 - a1)
                        # Pause visuals from original gap if possible; else take a small tail slice.
                        next_o0 = float(_orig[i + 1].get("start", o1) or o1)
                        gap = float(next_o0 - o1)
                        if gap > 0:
                            p0, p1 = o1, next_o0
                        else:
                            p1 = o1
                            p0 = max(0.0, p1 - 0.08)
                        if pause_dur > 1e-6:
                            segs.append(
                                {
                                    "kind": "pause",
                                    "src_start": round(p0, 6),
                                    "src_end": round(p1, 6),
                                    "target_duration": round(pause_dur, 6),
                                }
                            )
                with open(adaptive_plan_path, "w", encoding="utf-8") as f:
                    json.dump({"segments": segs}, f, ensure_ascii=False, indent=2)
                logger.info(f"已重建 adaptive_plan.json: {adaptive_plan_path}")
            except Exception as e:
                logger.warning(f"重建 adaptive_plan.json 失败，将继续尝试直接合成视频: {e}")
    else:
        # 如果 audio_combined.wav 不存在，先生成它
        if _audio_combined_needs_rebuild(folder, adaptive_segment_stretch=False) or not os.path.exists(input_audio):
            check_cancelled()
            logger.info("audio_combined.wav 不存在/已过期，正在生成...")
            _ensure_audio_combined(folder, adaptive_segment_stretch=False)
    
    # 字幕在自适应模式下应跟随 translation_adaptive.json。
    translation_path_to_use = translation_adaptive_path if adaptive_segment_stretch else translation_path

    missing = [p for p in (translation_path_to_use, input_audio, input_video) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"缺少合成视频所需文件：{missing}")
    
    with open(translation_path_to_use, 'r', encoding='utf-8') as f:
        translation = json.load(f)

    if subtitles and bilingual_subtitle:
        translation = _ensure_bilingual_source_text(
            folder, translation, adaptive_segment_stretch=adaptive_segment_stretch
        )

    check_cancelled()
    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    res_string = f'{width}x{height}'
    
    font_size, outline, margin_v, max_chars_zh, max_chars_en = _calc_subtitle_style_params(
        width, height, en_font_scale=0.75
    )

    srt_path = os.path.join(folder, 'subtitles.srt')
    if subtitles:
        generate_srt(
            translation,
            srt_path,
            bilingual_subtitle=bilingual_subtitle,
            max_chars_zh=max_chars_zh,
            max_chars_en=max_chars_en,
        )

    srt_path_filter = srt_path.replace('\\', '/')
    
    if os.name == 'nt' and ':' in srt_path_filter:
        srt_path_filter = srt_path_filter.replace(':', '\\:')

    subtitle_filter = ""
    if subtitles:
        ass_path = os.path.join(folder, "subtitles.ass")
        if bilingual_subtitle:
            generate_bilingual_ass(
                translation,
                ass_path,
                play_res_x=width,
                play_res_y=height,
                font_name="Arial",
                font_size=font_size,
                outline=outline,
                margin_v=margin_v,
                max_chars_zh=max_chars_zh,
                max_chars_en=max_chars_en,
            )
        else:
            # Use ASS for monolingual too (ensures font size is respected via PlayRes).
            generate_monolingual_ass(
                translation,
                ass_path,
                play_res_x=width,
                play_res_y=height,
                font_name="Arial",
                font_size=font_size,
                outline=outline,
                margin_v=margin_v,
                max_chars_zh=max_chars_zh,
            )
        ass_path_filter = ass_path.replace("\\", "/")
        if os.name == "nt" and ":" in ass_path_filter:
            ass_path_filter = ass_path_filter.replace(":", "\\:")
        subtitle_filter = f"ass=filename='{ass_path_filter}':original_size={res_string}"

    # Build ffmpeg filtergraph.
    filter_complex: str | None = None
    filter_script_path: str | None = None

    if adaptive_segment_stretch:
        # Per-segment trim + setpts + concat (align each utterance start frame).
        segments: list[dict[str, Any]] = []
        try:
            if os.path.exists(adaptive_plan_path):
                with open(adaptive_plan_path, "r", encoding="utf-8") as f:
                    payload = json.load(f) or {}
                segments = list(payload.get("segments") or [])
        except Exception:
            segments = []

        if not segments:
            raise FileNotFoundError(f"自适应模式缺少有效的 adaptive_plan.json: {adaptive_plan_path}")

        video_dur = _get_video_duration(input_video)
        if video_dur <= 0:
            video_dur = None

        # Write filter_complex_script to avoid command length limits.
        lines: list[str] = []
        v_labels: list[str] = []
        for idx, seg in enumerate(segments):
            check_cancelled()
            s0 = float(seg.get("src_start", 0.0) or 0.0)
            s1 = float(seg.get("src_end", 0.0) or 0.0)
            if s1 < s0:
                s0, s1 = s1, s0
            if video_dur is not None:
                s0 = max(0.0, min(s0, float(video_dur)))
                s1 = max(0.0, min(s1, float(video_dur)))
            src_dur = max(0.001, s1 - s0)
            out_dur = float(seg.get("target_duration", src_dur) or src_dur)
            if out_dur <= 0:
                out_dur = src_dur
            factor = out_dur / src_dur
            v = f"v{idx}"
            v_labels.append(f"[{v}]")
            lines.append(
                f"[0:v]trim=start={s0:.6f}:end={s1:.6f},setpts=(PTS-STARTPTS)*{factor:.9f}[{v}]"
            )

        concat_in = "".join(v_labels)
        lines.append(f"{concat_in}concat=n={len(v_labels)}:v=1:a=0[vcat]")

        post_filters: list[str] = [f"scale={width}:{height}"]
        if subtitles:
            post_filters.append(subtitle_filter)
        lines.append(f"[vcat]{','.join(post_filters)}[v]")

        filter_script_path = os.path.join(folder, "ffmpeg_filter_complex.txt")
        with open(filter_script_path, "w", encoding="utf-8") as f:
            f.write(";\n".join(lines) + "\n")
    else:
        # Non-adaptive path: no global speed-up, only scale + optional subtitles.
        v_filters: list[str] = [f"scale={width}:{height}"]
        if subtitles:
            v_filters.append(subtitle_filter)
        filter_complex = f"[0:v]{','.join(v_filters)}[v]"
        
    video_encoder = "h264_nvenc" if use_nvenc else "libx264"
    # 音频编码参数：上采样到 48kHz 并使用较高码率以保证音质
    audio_sample_rate = str(_VIDEO_AUDIO_SAMPLE_RATE)
    audio_bitrate = str(_VIDEO_AUDIO_BITRATE)
    
    if adaptive_segment_stretch:
        if not filter_script_path:
            raise RuntimeError("adaptive_segment_stretch=True 但未生成滤镜脚本")
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            input_video,
            "-i",
            input_audio,
            "-filter_complex_script",
            filter_script_path,
            "-map",
            "[v]",
            "-map",
            "1:a",
            # Adaptive mode uses per-segment setpts, so we preserve timestamps and
            # avoid enforcing a fixed FPS here (which can introduce dup/drop drift).
            "-vsync",
            "vfr",
            "-c:v",
            video_encoder,
            "-c:a",
            "aac",
            "-ar",
            audio_sample_rate,
            "-b:a",
            audio_bitrate,
            "-shortest",
            output_video,
            "-y",
        ]
    else:
        if filter_complex is None:
            raise RuntimeError("未生成 filter_complex")
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            input_video,
            "-i",
            input_audio,
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "1:a",
            "-r",
            str(fps),
            "-c:v",
            video_encoder,
            "-c:a",
            "aac",
            "-ar",
            audio_sample_rate,
            "-b:a",
            audio_bitrate,
            output_video,
            "-y",
        ]
    
    def _run_ffmpeg(cmd: list[str]) -> None:
        import sys
        import select
        import io

        # Capture ffmpeg output and forward to sys.stderr so Gradio can display it
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )  # noqa: S603
        try:
            # Non-blocking read from stderr (ffmpeg progress goes there)
            if proc.stderr is not None:
                import os
                fd = proc.stderr.fileno()
                try:
                    import fcntl
                    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                except (ImportError, OSError):
                    pass  # Windows or other platforms without fcntl

            buf = b""
            while True:
                check_cancelled()
                rc = proc.poll()

                # Read available stderr output
                if proc.stderr is not None:
                    try:
                        chunk = proc.stderr.read(4096)
                        if chunk:
                            buf += chunk
                            # Process complete lines
                            while b"\n" in buf or b"\r" in buf:
                                # Find the first line terminator
                                idx_n = buf.find(b"\n")
                                idx_r = buf.find(b"\r")
                                if idx_n == -1:
                                    idx = idx_r
                                elif idx_r == -1:
                                    idx = idx_n
                                else:
                                    idx = min(idx_n, idx_r)
                                line = buf[:idx]
                                buf = buf[idx + 1:]
                                if line.strip():
                                    try:
                                        text = line.decode("utf-8", errors="replace").rstrip()
                                        # Use \r prefix for progress lines (ffmpeg uses carriage return)
                                        if idx_r != -1 and (idx_n == -1 or idx_r < idx_n):
                                            sys.stderr.write("\r" + text + "\n")
                                        else:
                                            sys.stderr.write(text + "\n")
                                        sys.stderr.flush()
                                    except Exception:
                                        pass
                    except (BlockingIOError, IOError):
                        pass  # No data available yet

                if rc is not None:
                    # Process remaining output
                    if proc.stderr is not None:
                        try:
                            remaining = proc.stderr.read()
                            if remaining:
                                for line in remaining.decode("utf-8", errors="replace").splitlines():
                                    if line.strip():
                                        sys.stderr.write(line + "\n")
                                sys.stderr.flush()
                        except Exception:
                            pass
                    if rc != 0:
                        raise subprocess.CalledProcessError(rc, cmd)
                    return
                time.sleep(0.1)
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
        if use_nvenc:
            with _nvenc_slot():
                _run_ffmpeg(ffmpeg_command)
        else:
            _run_ffmpeg(ffmpeg_command)
        sleep_with_cancel(1)
        _write_video_meta(
            folder,
            subtitles=subtitles,
            bilingual_subtitle=bilingual_subtitle,
            adaptive_segment_stretch=adaptive_segment_stretch,
            fps=meta_fps,
            resolution=resolution,
            use_nvenc=use_nvenc,
        )
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
            _write_video_meta(
                folder,
                subtitles=subtitles,
                bilingual_subtitle=bilingual_subtitle,
                adaptive_segment_stretch=adaptive_segment_stretch,
                fps=meta_fps,
                resolution=resolution,
                use_nvenc=False,
            )
            logger.info(f"视频已生成(回退libx264): {output_video}")
            return

        logger.error(f"FFmpeg 失败: {e}")
        raise


def synthesize_all_video_under_folder(
    folder: str, 
    subtitles: bool = True, 
    fps: int = 30, 
    resolution: str = '1080p',
    use_nvenc: bool = False,
    adaptive_segment_stretch: bool = False,
    bilingual_subtitle: bool = False,
    auto_upload_video: bool = False,
    max_workers: int | None = None,
) -> str:
    # Collect targets first so we can optionally run them concurrently.
    targets: list[str] = []
    for root, _dirs, files in os.walk(folder):
        check_cancelled()
        if "download.mp4" not in files:
            continue
        # Use the same freshness logic as synthesize_video() to avoid stale reuse.
        meta_fps = 0 if adaptive_segment_stretch else int(fps)
        up_to_date = _video_up_to_date(
            root,
            subtitles=subtitles,
            bilingual_subtitle=bilingual_subtitle,
            adaptive_segment_stretch=adaptive_segment_stretch,
            fps=meta_fps,
            resolution=resolution,
            use_nvenc=use_nvenc,
        )
        if not up_to_date:
            targets.append(root)

    if not targets:
        msg = f"视频合成完成: {folder}（处理 0 个文件）"
        logger.info(msg)
        return msg

    def _maybe_enqueue_upload(root: str) -> None:
        if not auto_upload_video:
            return
        # Enqueue in background so it won't block synthesizing other videos.
        try:
            from .upload import upload_video_async  # local import to avoid hard dependency for non-upload users

            upload_video_async(root)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"加入B站后台上传队列失败（忽略）: {exc}")

    ok = 0
    failed: list[str] = []

    # Only parallelize when NVENC is enabled and there are multiple videos.
    # Rationale:
    # - NVENC has a dedicated concurrency guard (_NVENC_SEMAPHORE).
    # - libx264 parallelism can easily saturate CPU and degrade overall throughput.
    parallel = bool(use_nvenc) and len(targets) > 1
    if max_workers is None:
        effective_workers = min(_NVENC_MAX_CONCURRENCY, len(targets)) if parallel else 1
    else:
        effective_workers = min(max(1, max_workers), _NVENC_MAX_CONCURRENCY, len(targets)) if parallel else 1

    if parallel:
        logger.info(
            f"检测到多个视频且启用 NVENC，将并发合成视频: max_workers={effective_workers}（NVENC并发上限={_NVENC_MAX_CONCURRENCY}）"
        )

    def _run_one(root: str) -> None:
        synthesize_video(
            root,
            subtitles=subtitles,
            bilingual_subtitle=bilingual_subtitle,
            fps=fps,
            resolution=resolution,
            use_nvenc=use_nvenc,
            adaptive_segment_stretch=adaptive_segment_stretch,
        )

    if effective_workers <= 1:
        for root in targets:
            check_cancelled()
            try:
                _run_one(root)
                ok += 1
                _maybe_enqueue_upload(root)
            except Exception as exc:  # noqa: BLE001
                failed.append(root)
                logger.exception(f"视频合成失败（已跳过）: {root} ({exc})")
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            future_to_root = {executor.submit(_run_one, root): root for root in targets}
            for future in as_completed(future_to_root):
                check_cancelled()
                root = future_to_root[future]
                try:
                    future.result()
                    ok += 1
                    _maybe_enqueue_upload(root)
                except Exception as exc:  # noqa: BLE001
                    failed.append(root)
                    logger.exception(f"视频合成失败（已跳过）: {root} ({exc})")

    msg = f"视频合成完成: {folder}（处理 {len(targets)} 个文件，成功 {ok}，失败 {len(failed)}）"
    if failed:
        # Keep it short to avoid spamming logs for huge batches.
        preview = failed[:8]
        more = f" ...（另有 {len(failed) - len(preview)} 个）" if len(failed) > len(preview) else ""
        logger.warning(f"以下目录视频合成失败，请查看日志定位原因: {preview}{more}")
    logger.info(msg)
    return msg
