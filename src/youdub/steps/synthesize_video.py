import json
import os
import subprocess
import time
from typing import Any

import librosa
import numpy as np
from loguru import logger

from ..interrupts import check_cancelled, sleep_with_cancel
from ..utils import save_wav


def split_text(
    input_data: list[dict[str, Any]], 
    punctuations: list[str] | None = None
) -> list[dict[str, Any]]:
    puncts = set(punctuations or ['，', '；', '：', '。', '？', '！', '\n', '”'])

    def is_punctuation(char: str) -> bool:
        return char in puncts

    output_data = []
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
        
        for i, char in enumerate(text):
            if not is_punctuation(char) and i != len(text) - 1:
                continue
            if i - sentence_start < 5 and i != len(text) - 1:
                continue
            if i < len(text) - 1 and is_punctuation(text[i+1]):
                continue
                
            sentence = text[sentence_start:i+1]
            sentence_end = start + duration_per_char * len(sentence)

            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": original_text,
                "translation": sentence,
                "speaker": speaker
            })

            start = sentence_end
            sentence_start = i + 1

    return output_data


def format_timestamp(seconds: float) -> str:
    millisec = int((seconds - int(seconds)) * 1000)
    hours, seconds_int = divmod(int(seconds), 3600)
    minutes, seconds_int = divmod(seconds_int, 60)
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{millisec:03}"


def generate_srt(
    translation: list[dict[str, Any]], 
    srt_path: str, 
    speed_up: float = 1.0, 
    max_line_char: int = 30
) -> None:
    translation = split_text(translation)
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(translation):
            start = format_timestamp(line['start'] / speed_up)
            end = format_timestamp(line['end'] / speed_up)
            text = line['translation']

            if not text:
                continue
                
            lines_count = len(text) // (max_line_char + 1) + 1
            avg_chars = min(round(len(text) / lines_count), max_line_char)
            
            wrapped_text = '\n'.join([
                text[j * avg_chars : (j + 1) * avg_chars]
                for j in range(lines_count)
            ])
            
            f.write(f'{i+1}\n')
            f.write(f'{start} --> {end}\n')
            f.write(f'{wrapped_text}\n\n')


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


def _ensure_audio_combined(
    folder: str,
    adaptive_segment_stretch: bool = False,
    speed_up: float = 1.0,
    sample_rate: int = 24000,
) -> None:
    """
    生成 audio_combined.wav 和 audio_tts.wav
    
    如果 adaptive_segment_stretch=True:
        - 按原视频时间轴做全局缩放，每一段都放到它在缩放后应在的位置
        - 如果这段TTS更短，就在段内补静音把时长补齐
    否则:
        - 按顺序拼接TTS音频片段
    """
    check_cancelled()
    
    audio_combined_path = os.path.join(folder, 'audio_combined.wav')
    audio_tts_path = os.path.join(folder, 'audio_tts.wav')
    wavs_folder = os.path.join(folder, 'wavs')
    translation_path = os.path.join(folder, 'translation.json')
    translation_adaptive_path = os.path.join(folder, 'translation_adaptive.json')
    video_path = os.path.join(folder, 'download.mp4')
    audio_vocals_path = os.path.join(folder, 'audio_vocals.wav')
    audio_instruments_path = os.path.join(folder, 'audio_instruments.wav')
    
    # 检查必要文件
    if not os.path.exists(wavs_folder):
        raise FileNotFoundError(f"缺少 wavs 目录: {wavs_folder}")
    
    # 优先使用 translation_adaptive.json（如果存在），否则使用 translation.json
    if os.path.exists(translation_adaptive_path):
        translation_path_to_use = translation_adaptive_path
    elif os.path.exists(translation_path):
        translation_path_to_use = translation_path
    else:
        raise FileNotFoundError(f"缺少翻译文件: {translation_path}")
    
    with open(translation_path_to_use, 'r', encoding='utf-8') as f:
        translation = json.load(f)
    
    if not translation:
        raise ValueError(f"翻译文件为空: {translation_path_to_use}")
    
    # 获取所有wav文件
    wav_files = sorted([f for f in os.listdir(wavs_folder) if f.endswith('.wav')])
    if not wav_files:
        raise ValueError(f"wavs 目录为空: {wavs_folder}")
    
    # 确保wav文件数量与translation段数一致
    expected_count = len(translation)
    if len(wav_files) < expected_count:
        logger.warning(f"wav文件数量({len(wav_files)})少于翻译段数({expected_count})")
    
    if adaptive_segment_stretch:
        # 自适应模式：按原视频时间轴做全局缩放
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"自适应模式需要视频文件: {video_path}")
        
        video_duration = _get_video_duration(video_path)
        if video_duration <= 0:
            raise ValueError(f"无法获取视频时长: {video_path}")
        
        # 计算缩放后的总时长
        scaled_duration = video_duration / speed_up
        
        # 计算缩放后的时间轴
        scaled_segments = []
        for seg in translation:
            scaled_start = seg['start'] / speed_up
            scaled_end = seg['end'] / speed_up
            scaled_segments.append({
                'start': scaled_start,
                'end': scaled_end,
                'duration': scaled_end - scaled_start,
            })
        
        # 创建输出音频数组（缩放后的总时长）
        total_samples = int(scaled_duration * sample_rate)
        audio_tts = np.zeros(total_samples, dtype=np.float32)
        
        # 将每段TTS放到它在缩放后应在的位置
        for i, seg in enumerate(scaled_segments):
            check_cancelled()
            if i >= len(wav_files):
                break
            
            wav_file = os.path.join(wavs_folder, wav_files[i])
            if not os.path.exists(wav_file):
                logger.warning(f"TTS音频文件不存在: {wav_file}")
                continue
            
            try:
                # 加载TTS音频
                tts_audio, _ = librosa.load(wav_file, sr=sample_rate)
                tts_duration = len(tts_audio) / sample_rate
                
                # 计算这段在缩放后应在的位置
                target_start = seg['start']
                target_end = seg['end']
                target_duration = target_end - target_start
                
                # 计算在输出数组中的位置
                start_sample = int(target_start * sample_rate)
                end_sample = int(target_end * sample_rate)
                
                # 确保不越界
                start_sample = max(0, min(start_sample, total_samples - 1))
                end_sample = max(start_sample + 1, min(end_sample, total_samples))
                
                # 如果TTS更短，在段内补静音
                if tts_duration < target_duration:
                    # TTS音频放在段开始，后面补静音
                    tts_samples = len(tts_audio)
                    available_samples = end_sample - start_sample
                    
                    if tts_samples <= available_samples:
                        audio_tts[start_sample:start_sample + tts_samples] = tts_audio
                        # 后面已经是0（静音），不需要额外填充
                    else:
                        # TTS太长，裁剪到目标长度
                        audio_tts[start_sample:end_sample] = tts_audio[:available_samples]
                else:
                    # TTS更长或相等，裁剪到目标长度
                    available_samples = end_sample - start_sample
                    audio_tts[start_sample:end_sample] = tts_audio[:available_samples]
                    
            except Exception as e:
                logger.warning(f"处理TTS音频段 {i} 失败: {e}")
                continue
        
        # 保存 audio_tts.wav
        save_wav(audio_tts, audio_tts_path, sample_rate=sample_rate)
        logger.info(f"已生成 audio_tts.wav (自适应模式): {audio_tts_path}")
        
    else:
        # 非自适应模式：按顺序拼接
        audio_segments = []
        for i, wav_file in enumerate(wav_files[:len(translation)]):
            check_cancelled()
            wav_path = os.path.join(wavs_folder, wav_file)
            if not os.path.exists(wav_path):
                logger.warning(f"TTS音频文件不存在: {wav_path}")
                continue
            
            try:
                audio, _ = librosa.load(wav_path, sr=sample_rate)
                audio_segments.append(audio)
            except Exception as e:
                logger.warning(f"加载TTS音频失败 {wav_path}: {e}")
                continue
        
        if not audio_segments:
            raise ValueError("没有有效的TTS音频片段")
        
        audio_tts = np.concatenate(audio_segments)
        save_wav(audio_tts, audio_tts_path, sample_rate=sample_rate)
        logger.info(f"已生成 audio_tts.wav: {audio_tts_path}")
    
    # 混合TTS音频和背景音乐
    check_cancelled()
    if os.path.exists(audio_vocals_path) and os.path.exists(audio_instruments_path):
        try:
            # 加载背景音乐
            vocals, _ = librosa.load(audio_vocals_path, sr=sample_rate)
            instruments, _ = librosa.load(audio_instruments_path, sr=sample_rate)
            
            # 对齐长度（以TTS音频为准）
            tts_len = len(audio_tts)
            vocals_len = len(vocals)
            instruments_len = len(instruments)
            
            # 如果背景音乐更长，裁剪；如果更短，循环填充（或补零）
            if vocals_len < tts_len:
                # 循环填充
                repeat_count = (tts_len + vocals_len - 1) // vocals_len
                vocals = np.tile(vocals, repeat_count)[:tts_len]
            else:
                vocals = vocals[:tts_len]
            
            if instruments_len < tts_len:
                repeat_count = (tts_len + instruments_len - 1) // instruments_len
                instruments = np.tile(instruments, repeat_count)[:tts_len]
            else:
                instruments = instruments[:tts_len]
            
            # 混合：TTS + 背景音乐（降低背景音量）
            audio_combined = audio_tts + 0.3 * vocals + 0.2 * instruments
            
            # 归一化防止削波
            max_val = np.abs(audio_combined).max()
            if max_val > 1.0:
                audio_combined = audio_combined / max_val * 0.95
            
            save_wav(audio_combined.astype(np.float32), audio_combined_path, sample_rate=sample_rate)
            logger.info(f"已生成 audio_combined.wav: {audio_combined_path}")
        except Exception as e:
            logger.warning(f"混合背景音乐失败，仅使用TTS音频: {e}")
            save_wav(audio_tts, audio_combined_path, sample_rate=sample_rate)
    else:
        # 没有背景音乐，直接使用TTS音频
        save_wav(audio_tts, audio_combined_path, sample_rate=sample_rate)
        logger.info(f"已生成 audio_combined.wav (无背景音乐): {audio_combined_path}")


def synthesize_video(
    folder: str, 
    subtitles: bool = True, 
    speed_up: float = 1.2, 
    fps: int = 30, 
    resolution: str = '1080p',
    use_nvenc: bool = False,
    adaptive_segment_stretch: bool = False,
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
    
    # 如果 audio_combined.wav 不存在，先生成它
    if not os.path.exists(input_audio):
        check_cancelled()
        logger.info(f"audio_combined.wav 不存在，正在生成...")
        _ensure_audio_combined(
            folder,
            adaptive_segment_stretch=adaptive_segment_stretch,
            speed_up=speed_up,
        )
    
    missing = [p for p in (translation_path, input_audio, input_video) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"缺少合成视频所需文件：{missing}")
    
    with open(translation_path, 'r', encoding='utf-8') as f:
        translation = json.load(f)
        
    srt_path = os.path.join(folder, 'subtitles.srt')
    
    generate_srt(translation, srt_path, speed_up)

    check_cancelled()
    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    res_string = f'{width}x{height}'
    
    font_size = int(width / 128)
    
    outline = int(round(font_size / 8))
    
    video_speed_filter = f"setpts=PTS/{speed_up}"
    audio_speed_filter = f"atempo={speed_up}"
    
    srt_path_filter = srt_path.replace('\\', '/')
    
    if os.name == 'nt' and ':' in srt_path_filter:
        srt_path_filter = srt_path_filter.replace(':', '\\:')
        
    subtitle_filter = (
        f"subtitles='{srt_path_filter}':force_style="
        f"'FontName=Arial,FontSize={font_size},PrimaryColour=&HFFFFFF,"
        f"OutlineColour=&H000000,Outline={outline},WrapStyle=2'"
    )
    
    if subtitles:
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
        proc = subprocess.Popen(cmd, start_new_session=True)  # noqa: S603
        try:
            while True:
                check_cancelled()
                rc = proc.poll()
                if rc is not None:
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
    speed_up: float = 1.2, 
    fps: int = 30, 
    resolution: str = '1080p',
    use_nvenc: bool = False,
    adaptive_segment_stretch: bool = False,
    bilingual_subtitle: bool = False,
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
            speed_up=speed_up,
            fps=fps,
            resolution=resolution,
            use_nvenc=use_nvenc,
            adaptive_segment_stretch=adaptive_segment_stretch,
        )
        count += 1
    msg = f"视频合成完成: {folder}（处理 {count} 个文件）"
    logger.info(msg)
    return msg
