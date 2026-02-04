import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any

from loguru import logger

from ..interrupts import check_cancelled, sleep_with_cancel
from ..utils import valid_file

from .synthesize_video_audio import (
    _AUDIO_COMBINED_META_NAME,
    _AUDIO_COMBINED_MIX_VERSION,
    _audio_combined_needs_rebuild,
    _ensure_audio_combined,
    _read_audio_combined_meta,
    _write_audio_combined_meta,
)
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
