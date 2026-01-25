import base64
import json
import os
import re
import subprocess
import threading
import time
import uuid
import wave
from collections import deque
from pathlib import Path

import librosa
import numpy as np
import requests
from audiostretchy.stretch import stretch_audio
from loguru import logger
from scipy.spatial.distance import cosine

from ..config import Settings
from ..models import ModelCheckError, ModelManager
from ..interrupts import CancelledByUser, check_cancelled, sleep_with_cancel
from ..cn_tx import TextNorm
from ..utils import ensure_torchaudio_backend_compat, save_wav, save_wav_norm

_EMBEDDING_MODEL = None
_EMBEDDING_INFERENCE = None
_EMBEDDING_MODEL_LOAD_FAILED = False

_DEFAULT_SETTINGS = Settings()
_DEFAULT_MODEL_MANAGER = ModelManager(_DEFAULT_SETTINGS)

_BYTEDANCE_HOST = "openspeech.bytedance.com"
_BYTEDANCE_API_URL = f"https://{_BYTEDANCE_HOST}/api/v1/tts"

_GEMINI_CLIENT = None

def _get_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        try:
            from google import genai
        except ImportError:
            logger.error("未安装google-genai。请运行 'uv sync'。")
            return None
            
        api_key = _DEFAULT_SETTINGS.gemini_api_key
        if not api_key:
            logger.warning("未在config/env中设置GEMINI_API_KEY")
            return None
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


def is_valid_wav(path: str) -> bool:
    """Check if a file is a valid WAV file."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 44:  # Minimal WAV header size
        return False
    try:
        # NOTE:
        # We intentionally validate via stdlib `wave` here (instead of libsndfile/soundfile).
        # Reason: downstream time-stretch (`audiostretchy`) also uses `wave` and will crash
        # on files that libsndfile can decode (e.g. FLAC/OGG/MP3 content saved as *.wav).
        with wave.open(path, "rb") as wf:
            if int(wf.getframerate() or 0) <= 0:
                return False
            if int(wf.getnchannels() or 0) <= 0:
                return False
            if int(wf.getsampwidth() or 0) <= 0:
                return False
            # Allow zero-length wav (treated as valid but will be handled later).
            _ = int(wf.getnframes() or 0)
        return True
    except Exception:
        return False


def _read_speaker_ref_seconds(default: float = 15.0) -> float:
    """
    Speaker reference audio duration (seconds) for voice cloning.

    Official recommendation is usually 10-20s; we default to 15s.
    Clamp to [3, 60] seconds to avoid pathological inputs.
    """
    raw = os.getenv("TTS_SPEAKER_REF_SECONDS")
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    if not (v > 0):
        return default
    return float(max(3.0, min(v, 60.0)))


def _wav_duration_seconds(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as wf:
            rate = int(wf.getframerate() or 0)
            if rate <= 0:
                return None
            frames = int(wf.getnframes() or 0)
            if frames <= 0:
                return 0.0
            return frames / float(rate)
    except Exception:
        return None


def _ensure_wav_max_duration(path: str, max_seconds: float, sample_rate: int = 24000) -> None:
    if not path or not os.path.exists(path):
        return
    if max_seconds <= 0:
        return

    dur = _wav_duration_seconds(path)
    if dur is not None and dur <= max_seconds + 0.02:
        return

    try:
        wav, _sr = librosa.load(path, sr=sample_rate, duration=max_seconds)
        if wav.size <= 0:
            return
        save_wav(wav.astype(np.float32), path, sample_rate=sample_rate)
        logger.info(f"已裁剪说话人参考音频至 {max_seconds:.1f}秒: {path}")
    except Exception as exc:
        logger.warning(f"裁剪说话人音频失败 {path}: {exc}")


def preprocess_text(text: str) -> str:
    text = text.replace('AI', '人工智能')
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    normalizer = TextNorm()
    text = normalizer(text)
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return text


def adjust_audio_length(
    wav_path: str, 
    desired_length: float, 
    sample_rate: int = 24000, 
    min_speed_factor: float = 0.6, 
    max_speed_factor: float = 1.1
) -> tuple[np.ndarray, float]:
    wav, _ = librosa.load(wav_path, sr=sample_rate)
    current_length = len(wav) / sample_rate
    
    if current_length == 0:
        return wav, 0.0

    speed_factor = max(
        min(desired_length / current_length, max_speed_factor), min_speed_factor)
    
    new_desired_length = current_length * speed_factor
    
    target_path = wav_path.replace(".wav", "_adjusted.wav")
    wav_stretched: np.ndarray | None = None
    try:
        stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
        wav_stretched, _ = librosa.load(target_path, sr=sample_rate)
    except Exception as exc:
        # Don't crash the whole pipeline due to one bad/cached segment.
        # Fallback: in-memory time-stretch via librosa; if that also fails, return original (trim/pad).
        logger.warning(f"音频拉伸失败 {wav_path}: {exc} (回退到librosa)")
        try:
            # librosa's `rate` is inverse of duration ratio:
            # new_duration = old_duration / rate  => rate = 1 / speed_factor
            rate = float(1.0 / max(float(speed_factor), 1e-6))
            wav_stretched = librosa.effects.time_stretch(wav.astype(np.float32, copy=False), rate=rate)
        except Exception as exc2:
            logger.warning(f"librosa时间拉伸失败 {wav_path}: {exc2} (回退到裁剪/填充)")
            wav_stretched = wav
    finally:
        # 临时文件仅用于中间 stretch，避免长期堆积占用磁盘。
        try:
            if os.path.exists(target_path):
                os.remove(target_path)
        except Exception:
            pass

    if wav_stretched is None:
        wav_stretched = wav

    target_samples = int(new_desired_length * sample_rate)
    if target_samples <= 0:
        return np.zeros((0,), dtype=np.float32), 0.0

    wav_stretched = wav_stretched.astype(np.float32, copy=False)
    if wav_stretched.shape[0] < target_samples:
        wav_stretched = np.pad(wav_stretched, (0, target_samples - wav_stretched.shape[0]), mode="constant")
    else:
        wav_stretched = wav_stretched[:target_samples]

    # Match reported length to actual returned samples to keep timeline consistent.
    actual_len = float(target_samples) / float(sample_rate)
    return wav_stretched, actual_len


def load_embedding_model() -> None:
    global _EMBEDDING_MODEL, _EMBEDDING_INFERENCE, _EMBEDDING_MODEL_LOAD_FAILED
    
    if _EMBEDDING_MODEL_LOAD_FAILED:
        raise RuntimeError("Embedding model previously failed to load. Skipping retry.")

    if _EMBEDDING_MODEL is not None and _EMBEDDING_INFERENCE is not None:
        return

    try:
        try:
            ensure_torchaudio_backend_compat()
            from pyannote.audio import Inference, Model  # type: ignore
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                "缺少依赖 pyannote.audio，无法进行说话人 embedding/音色匹配。"
                "如不需要该功能，可忽略；否则请安装 pyannote.audio 并准备离线模型缓存。"
            ) from exc

        logger.info("加载pyannote/embedding模型...")
        token = _DEFAULT_SETTINGS.hf_token
        # pyannote.audio v4 uses `token=...`; older versions use `use_auth_token=...`.
        try:
            _EMBEDDING_MODEL = Model.from_pretrained("pyannote/embedding", token=token)
        except TypeError:
            _EMBEDDING_MODEL = Model.from_pretrained("pyannote/embedding", use_auth_token=token)
        
        if _EMBEDDING_MODEL is None:
            logger.error("加载pyannote/embedding失败。请检查HF_TOKEN。")
            raise ValueError("Model.from_pretrained returned None")
            
        _EMBEDDING_INFERENCE = Inference(_EMBEDDING_MODEL, window="whole")
        logger.info("pyannote/embedding模型加载成功")
    except Exception as e:
        _EMBEDDING_MODEL_LOAD_FAILED = True
        logger.error(f"加载pyannote/embedding模型出错: {e}")
        raise


def generate_embedding(wav_path: str) -> np.ndarray:
    load_embedding_model()
    if _EMBEDDING_INFERENCE is None:
        raise RuntimeError("Embedding model is not available.")
    embedding = _EMBEDDING_INFERENCE(wav_path)
    return embedding


_BYTEDANCE_V3_SUBMIT_URL = f"https://{_BYTEDANCE_HOST}/api/v3/tts/submit"
_BYTEDANCE_V3_QUERY_URL = f"https://{_BYTEDANCE_HOST}/api/v3/tts/query"


def bytedance_tts_v3_api(
    text: str,
    speaker: str = 'BV001_streaming',
    use_cloned_voice: bool = False,
) -> bytes | None:
    """
    Use the V3 async long-text API for TTS synthesis.
    This API properly supports ICL 2.0 voice cloning.
    
    Args:
        text: Text to synthesize
        speaker: Speaker/voice ID (e.g., 'BV001_streaming' or 'S_xxx' for cloned)
        use_cloned_voice: Whether the speaker is a cloned voice (ICL 2.0)
    
    Returns:
        Audio bytes (WAV format) or None on failure
    """
    appid = _DEFAULT_SETTINGS.bytedance_appid
    access_token = _DEFAULT_SETTINGS.bytedance_access_token
    
    if not appid or not access_token:
        logger.warning("未设置ByteDance APPID或ACCESS_TOKEN")
        return None

    if use_cloned_voice:
        resource_id = "seed-icl-2.0"
    else:
        resource_id = "volc.service_type.10029"  # Standard TTS
    
    headers = {
        "X-Api-App-Id": appid,
        "X-Api-Access-Key": access_token,
        "X-Api-Resource-Id": resource_id,
        "Content-Type": "application/json",
    }
    
    unique_id = str(uuid.uuid4())
    
    submit_payload = {
        "user": {"uid": "youdub-webui"},
        "unique_id": unique_id,
        "namespace": "BidirectionalTTS",
        "req_params": {
            "text": text,
            "speaker": speaker,
            "audio_params": {
                "format": "wav",
                "sample_rate": 24000,
            }
        }
    }
    
    try:
        check_cancelled()
        logger.debug(f"V3 TTS提交: speaker={speaker}, text={text[:30]}...")
        resp = requests.post(_BYTEDANCE_V3_SUBMIT_URL, json=submit_payload, headers=headers, timeout=30)
        resp_json = resp.json()
        
        if resp_json.get("code") != 20000000:
            logger.warning(f"V3 TTS提交失败: {resp_json}")
            return None
        
        task_id = resp_json.get("data", {}).get("task_id")
        if not task_id:
            logger.warning(f"V3 TTS提交未返回task_id: {resp_json}")
            return None
        
        logger.debug(f"V3 TTS任务已提交: {task_id}")
        
    except Exception as e:
        logger.error(f"V3 TTS提交异常: {e}")
        return None
    
    query_payload = {"task_id": task_id}
    max_polls = 60  # Max 60 seconds
    poll_interval = 1.0
    
    for _ in range(max_polls):
        try:
            check_cancelled()
            sleep_with_cancel(poll_interval)
            check_cancelled()
            resp = requests.post(_BYTEDANCE_V3_QUERY_URL, json=query_payload, headers=headers, timeout=30)
            resp_json = resp.json()
            
            if resp_json.get("code") != 20000000:
                logger.warning(f"V3 TTS查询错误: {resp_json}")
                return None
            
            task_status = resp_json.get("data", {}).get("task_status")
            
            if task_status == 1:  # Running
                continue
            elif task_status == 2:  # Success
                audio_url = resp_json.get("data", {}).get("audio_url")
                if audio_url:
                    audio_resp = requests.get(audio_url, timeout=60)
                    if audio_resp.status_code == 200:
                        logger.info(f"V3 TTS成功: {len(audio_resp.content)} 字节")
                        return audio_resp.content
                    else:
                        logger.warning(f"V3 TTS音频下载失败: {audio_resp.status_code}")
                        return None
                else:
                    logger.warning("V3 TTS成功但无audio_url")
                    return None
            elif task_status == 3:  # Failure
                logger.warning(f"V3 TTS任务失败: {resp_json}")
                return None
            else:
                logger.warning(f"V3 TTS未知状态: {task_status}")
                continue
                
        except Exception as e:
            logger.error(f"V3 TTS查询异常: {e}")
            return None
    
    logger.warning("V3 TTS轮询超时")
    return None


def bytedance_tts_api(
    text: str, 
    voice_type: str = 'BV001_streaming',
    use_cloned_voice: bool = False,
) -> bytes | None:
    appid = _DEFAULT_SETTINGS.bytedance_appid
    access_token = _DEFAULT_SETTINGS.bytedance_access_token
    
    if not appid or not access_token:
        logger.warning("未设置ByteDance APPID或ACCESS_TOKEN")
        return None

    cluster = 'volcano_icl' if use_cloned_voice else 'volcano_tts'
    
    header = {"Authorization": f"Bearer;{access_token}"}
    if use_cloned_voice:
        header["X-Api-Resource-Id"] = "seed-icl-2.0"
    request_json = {
        "app": {
            "appid": appid,
            "token": access_token,
            "cluster": cluster
        },
        "user": {
            "uid": "https://github.com/liuzhao1225/YouDub-webui"
        },
        "audio": {
            "voice_type": voice_type,
            "encoding": "wav",
            "speed_ratio": 1.0,
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0,
        },
        "request": {
            "reqid": str(uuid.uuid4()),
            "text": text,
            "text_type": "plain",
            "operation": "query",
            "with_frontend": 1,
            "frontend_type": "unitTson"
        }
    }
    
    try:
        resp = requests.post(_BYTEDANCE_API_URL, json.dumps(request_json), headers=header, timeout=60)
        if "data" in resp.json():
            data = resp.json()["data"]
            return base64.b64decode(data)
        else:
             logger.warning(f"ByteDance API错误: {resp.json()}")
    except Exception as e:
        logger.warning(f"ByteDance TTS请求失败: {e}")
        
    return None


def init_bytedance_reference_voices() -> None:
    """Ensure reference voice files exist for speaker matching."""
    voice_type_dir = 'voice_type'
    if not os.path.exists(voice_type_dir):
        os.makedirs(voice_type_dir)
        
    voice_types = [
        'BV001_streaming', 'BV002_streaming', 'BV005_streaming', 'BV007_streaming', 
        'BV033_streaming', 'BV034_streaming', 'BV056_streaming', 'BV102_streaming', 
        'BV113_streaming', 'BV115_streaming', 'BV119_streaming', 'BV700_streaming', 
        'BV701_streaming'
    ]
    
    try:
        load_embedding_model()
    except Exception:
        logger.warning("由于embedding模型失败，跳过ByteDance参考音色初始化")
        return

    sample_text = 'YouDub 是一个创新的开源工具，专注于将 YouTube 等平台的优质视频翻译和配音为中文版本。'
    
    for voice_type in voice_types:
        wav_path = os.path.join(voice_type_dir, f'{voice_type}.wav')
        npy_path = wav_path.replace('.wav', '.npy')
        
        if os.path.exists(wav_path) and os.path.exists(npy_path) and is_valid_wav(wav_path):
            continue
            
        logger.info(f"生成参考音频 {voice_type}...")
        audio_data = bytedance_tts_api(sample_text, voice_type=voice_type)
        if audio_data:
            with open(wav_path, "wb") as f:
                f.write(audio_data)
            
            try:
                embedding = generate_embedding(wav_path)
                np.save(npy_path, embedding)
            except Exception as e:
                logger.error(f"生成embedding失败 {voice_type}: {e}")
                if os.path.exists(wav_path):
                    os.remove(wav_path)  # Cleanup


def generate_speaker_to_voice_type(folder: str) -> dict[str, str]:
    speaker_to_voice_type_path = os.path.join(folder, 'speaker_to_voice_type.json')
    if os.path.exists(speaker_to_voice_type_path):
        with open(speaker_to_voice_type_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    init_bytedance_reference_voices()
    
    speaker_to_voice_type = {}
    speaker_folder = os.path.join(folder, 'SPEAKER')
    voice_types = {}
    voice_type_dir = 'voice_type'
    
    if not os.path.exists(voice_type_dir):
         logger.warning("未找到voice_type目录，无法进行说话人匹配")
         return {}

    for file in os.listdir(voice_type_dir):
        if file.endswith('.npy'):
            voice_type = file.replace('.npy', '')
            voice_types[voice_type] = np.load(os.path.join(voice_type_dir, file))
            
    if not os.path.exists(speaker_folder):
         return {}
         
    try:
        load_embedding_model()
    except Exception:
        logger.warning("Embedding模型不可用，回退到所有说话人的默认音色")
        for file in os.listdir(speaker_folder):
            if file.endswith('.wav'):
                speaker = file.replace('.wav', '')
                speaker_to_voice_type[speaker] = 'BV001_streaming'
        
        with open(speaker_to_voice_type_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_to_voice_type, f, indent=2, ensure_ascii=False)
        return speaker_to_voice_type

    for file in os.listdir(speaker_folder):
        if not file.endswith('.wav'):
            continue
        speaker = file.replace('.wav', '')
        wav_path = os.path.join(speaker_folder, file)
        
        try:
            embedding = generate_embedding(wav_path)
            np.save(wav_path.replace('.wav', '.npy'), embedding)
            
            best_match = sorted(voice_types.keys(), key=lambda x: 1 - cosine(voice_types[x], embedding))[0]
            speaker_to_voice_type[speaker] = best_match
            logger.info(f'匹配 {speaker} 到 {best_match}')
        except Exception as e:
            logger.error(f"匹配说话人失败 {speaker}: {e}")
            speaker_to_voice_type[speaker] = 'BV001_streaming' # Default fallback

    with open(speaker_to_voice_type_path, 'w', encoding='utf-8') as f:
        json.dump(speaker_to_voice_type, f, indent=2, ensure_ascii=False)
        
    return speaker_to_voice_type


def _upload_audio_for_cloning(audio_path: str, appid: str, token: str, speaker_id: str) -> bool:
    """Uploads audio to Volcano Engine to register a voice for cloning."""
    url = f"https://{_BYTEDANCE_HOST}/api/v1/mega_tts/audio/upload"
    header = {
        "Authorization": f"Bearer;{token}",
        "Resource-Id": "seed-icl-2.0", 
    }
    
    try:
        # Load audio and truncate to first 15 seconds (API limit is 10MB, 15s@24kHz ≈ 720KB)
        MAX_DURATION_SECONDS = 15
        SAMPLE_RATE = 24000
        
        wav_data, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_DURATION_SECONDS)
        
        if len(wav_data) < SAMPLE_RATE:  # Less than 1 second
            logger.warning(f"音频 {audio_path} 过短无法克隆 (< 1秒)")
            return False
        
        import io
        from scipy.io import wavfile
        
        wav_int16 = (wav_data * 32767).astype(np.int16)
        
        buffer = io.BytesIO()
        wavfile.write(buffer, SAMPLE_RATE, wav_int16)
        audio_data = buffer.getvalue()
        
        logger.info(f"上传 {len(audio_data) / 1024:.1f}KB 音频用于声音克隆 ({len(wav_data) / SAMPLE_RATE:.1f}秒)")

        payload = {
            "appid": appid,
            "speaker_id": speaker_id,
            "audios": [{
                "audio_bytes": base64.b64encode(audio_data).decode('utf-8'),
                "audio_format": "wav",
            }],
            "source": 2, # 2 for user uploaded
            "language": 0, # 0 for auto (CN default), 1 for EN. Ideally should be detected.
            "model_type": 4, # 4: ICL2.0 (Voice Cloning 2.0)
        }
        
        logger.info("--- 声音克隆上传请求 ---")
        logger.info(f"地址: {url}")
        logger.info(f"请求头: {header}")
        debug_payload = payload.copy()
        if debug_payload.get('audios'):
             debug_payload['audios'] = [{'audio_bytes': '<hidden>', 'audio_format': a['audio_format']} for a in debug_payload['audios']]
        logger.info(f"请求体: {json.dumps(debug_payload, ensure_ascii=False)}")
        logger.info("------------------------------------")
        
        resp = requests.post(url, json=payload, headers=header, timeout=120)
        
        try:
            resp_json = resp.json()
        except json.JSONDecodeError:
            logger.error(f"声音克隆上传响应不是JSON: status={resp.status_code} text={resp.text[:200]}")
            return False
            
        # Check response status (API can return PascalCase or snake_case)
        base_resp = resp_json.get("BaseResp") or resp_json.get("base_resp", {})
        status_code = base_resp.get("StatusCode") if "StatusCode" in base_resp else base_resp.get("status_code")
        
        if status_code == 0:
            logger.info(f"成功上传音频用于克隆: {speaker_id}")
            return True
        else:
            logger.warning(f"上传音频用于克隆失败: {resp_json}")
            return False
            
    except Exception as e:
        logger.error(f"声音克隆上传异常: {e}")
        return False


def get_or_create_cloned_voice(folder: str, speaker: str, speaker_wav: str) -> str | None:
    """
    Gets existing cloned voice ID or creates a new one using pre-allocated IDs.
    
    ICL 2.0 requires using speaker IDs that are already allocated in the Volcano console.
    Set VOLCANO_CLONE_SPEAKER_IDS in .env as a comma-separated list of available IDs.
    Example: VOLCANO_CLONE_SPEAKER_IDS=S_PoN0a1CN1,S_OoN0a1CN1,S_NoN0a1CN1
    """
    
    mapping_path = os.path.join(folder, 'speaker_to_cloned_voice.json')
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            
    if speaker in mapping:
        cached_value = mapping[speaker]
        if cached_value == "FAILED":
            return None
        return cached_value

    volcano_speaker_ids_str = os.getenv('VOLCANO_CLONE_SPEAKER_IDS', '')
    if not volcano_speaker_ids_str:
        logger.warning("未在.env中设置VOLCANO_CLONE_SPEAKER_IDS。将跳过声音克隆。")
        logger.warning("请将VOLCANO_CLONE_SPEAKER_IDS设置为Volcano控制台中的逗号分隔ID列表。")
        return None
    
    available_ids = [s.strip() for s in volcano_speaker_ids_str.split(',') if s.strip()]
    if not available_ids:
        logger.warning("VOLCANO_CLONE_SPEAKER_IDS为空。将跳过声音克隆。")
        return None
    
    used_ids = set(v for v in mapping.values() if v != "FAILED")
    unused_ids = [id for id in available_ids if id not in used_ids]
    
    if not unused_ids:
        logger.warning(f"所有 {len(available_ids)} 个预分配说话人ID已用完。{speaker} 无可用ID。")
        return None
    
    speaker_id = unused_ids[0]  # Use first available
    logger.info(f"分配预分配说话人ID {speaker_id} 给 {speaker}")
    
    appid = _DEFAULT_SETTINGS.bytedance_appid
    access_token = _DEFAULT_SETTINGS.bytedance_access_token
    
    if not appid or not access_token:
        return None
        
    if _upload_audio_for_cloning(speaker_wav, appid, access_token, speaker_id):
        mapping[speaker] = speaker_id
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        return speaker_id
    
    mapping[speaker] = "FAILED"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    logger.warning(f"声音克隆失败 {speaker}，将使用所有段的默认音色。")
    
    return None


def bytedance_tts(
    text: str, 
    output_path: str, 
    speaker_wav: str, 
    voice_type: str | None = None
) -> None:
    if os.path.exists(output_path) and is_valid_wav(output_path):
        logger.info(f'ByteDance TTS {text[:20]}... 已存在（已验证）')
        return
    elif os.path.exists(output_path):
        logger.warning(f"删除无效缓存文件: {output_path}")
        os.remove(output_path)

    folder = os.path.dirname(os.path.dirname(output_path))
    speaker = os.path.basename(speaker_wav).replace('.wav', '')
    
    if voice_type is None:
        cloned_voice_id = get_or_create_cloned_voice(folder, speaker, speaker_wav)
        if cloned_voice_id:
            voice_type = cloned_voice_id
            logger.info(f"使用克隆音色 {speaker}: {voice_type}")
        else:
            speaker_to_voice_type = generate_speaker_to_voice_type(folder)
            voice_type = speaker_to_voice_type.get(speaker, 'BV001_streaming')
            logger.info(f"使用匹配音色 {speaker}: {voice_type}")

    is_cloned = voice_type.startswith('S_') if voice_type else False
    
    for _ in range(3):
        check_cancelled()
        try:
            # ICL 2.0 当前使用 V1 接口更稳定
            audio_data = bytedance_tts_api(text, voice_type=voice_type, use_cloned_voice=is_cloned)
            if audio_data:
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                
                if is_valid_wav(output_path):
                    logger.info(f'ByteDance TTS已保存: {output_path}')
                    sleep_with_cancel(0.1)
                    break
                else:
                    logger.warning("保存的wav文件似乎已损坏或无效")
                    if os.path.exists(output_path):
                        os.remove(output_path)
            
            sleep_with_cancel(0.5)
        except Exception as e:
            logger.error(f"ByteDance TTS循环失败: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            sleep_with_cancel(0.5)


_QWEN_WORKER_READY = "__READY__"
_QWEN_WORKER_STUB_ENV = "YOUDUB_QWEN_WORKER_STUB"
_QWEN_TTS_ICL_ENV = "QWEN_TTS_ICL"


def _env_flag(name: str) -> bool:
    return (os.getenv(name, "") or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_qwen_worker_script_path() -> Path:
    # Project layout (repo checkout):
    # - <repo>/scripts/qwen_tts_worker.py
    # - <repo>/src/youdub/steps/synthesize_speech.py
    #
    # Resolve robustly by walking upwards so refactors won't break the path.
    here = Path(__file__).resolve()
    for p in here.parents:
        cand = p / "scripts" / "qwen_tts_worker.py"
        if cand.exists():
            return cand
    # Best-effort fallback: assume current working directory is repo root.
    return Path.cwd() / "scripts" / "qwen_tts_worker.py"


class _QwenTtsWorker:
    def __init__(self, python_exe: str, model_path: str, stub: bool = False):
        script = _get_qwen_worker_script_path()
        if not script.exists():
            raise ModelCheckError(f"未找到 Qwen3-TTS worker 脚本: {script}")

        cmd = [python_exe, "-u", str(script), "--model-path", model_path]
        if stub:
            cmd.append("--stub")

        logger.info(f"启动Qwen3-TTS工作进程 (stub={stub})")
        logger.debug(f"Qwen3-TTS工作进程命令: {' '.join(cmd)}")

        try:
            self._proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            raise ModelCheckError(f"启动 Qwen3-TTS worker 失败: {exc}") from exc

        logger.info(f"Qwen3-TTS工作进程 pid={self._proc.pid}，等待就绪...")

        self._stderr_tail: deque[str] = deque(maxlen=200)

        def _drain_stderr() -> None:
            stream = self._proc.stderr
            if stream is None:
                return
            try:
                for line in stream:
                    s = line.rstrip("\n")
                    if s:
                        self._stderr_tail.append(s)
            except Exception:
                # Best-effort: avoid crashing main flow due to stderr reader issues.
                return

        threading.Thread(target=_drain_stderr, daemon=True).start()

        startup_begin = time.monotonic()
        startup_done = threading.Event()
        startup_timed_out = threading.Event()
        try:
            startup_timeout_sec = float(os.getenv("YOUDUB_QWEN_WORKER_STARTUP_TIMEOUT_SEC", "1800") or "1800")
        except Exception:
            startup_timeout_sec = 1800.0

        def _startup_heartbeat() -> None:
            # 避免“正在加载但完全没日志输出”的错觉
            while not startup_done.wait(10.0):
                elapsed = time.monotonic() - startup_begin
                logger.info(f"Qwen3-TTS工作进程加载中... ({elapsed:.1f}秒)")

        def _startup_watchdog() -> None:
            if startup_timeout_sec <= 0:
                return
            if startup_done.wait(startup_timeout_sec):
                return
            startup_timed_out.set()
            logger.error(
                f"Qwen3-TTS工作进程启动超时 {startup_timeout_sec:.0f}秒，正在终止。"
            )
            try:
                self._proc.terminate()
            except Exception:
                return

        threading.Thread(target=_startup_heartbeat, daemon=True).start()
        threading.Thread(target=_startup_watchdog, daemon=True).start()

        assert self._proc.stdout is not None
        startup_stdout_tail: list[str] = []
        ready_ok = False
        try:
            while True:
                check_cancelled()
                line = self._read_stdout_line(timeout_sec=0.2)
                if line is None:
                    if self._proc.poll() is not None:
                        break
                    continue
                if not line:
                    break
                s = line.strip()
                if not s:
                    continue
                if s == _QWEN_WORKER_READY:
                    ready_ok = True
                    break
                startup_stdout_tail.append(s)
                if len(startup_stdout_tail) > 50:
                    startup_stdout_tail = startup_stdout_tail[-50:]
        except CancelledByUser:
            self.close()
            raise
        finally:
            startup_done.set()

        if ready_ok:
            elapsed = time.monotonic() - startup_begin
            logger.info(f"Qwen3-TTS工作进程就绪，耗时 {elapsed:.1f}秒")

        if not ready_ok:
            rc = self._proc.poll()
            stdout_tail = "\n".join(startup_stdout_tail).strip()
            stderr_tail = "\n".join(list(self._stderr_tail)).strip()
            self.close()

            details: list[str] = []
            if startup_timed_out.is_set():
                details.append(f"startup_timeout_sec={startup_timeout_sec:.0f}")
            if rc is not None:
                details.append(f"exit_code={rc}")
                if rc == -9:
                    details.append("看起来像是被 SIGKILL 杀掉（常见原因：内存不足/OOM 或 WSL2 内存限制）")
            if stdout_tail:
                details.append("stdout:\n" + stdout_tail)
            if stderr_tail:
                details.append("stderr:\n" + stderr_tail)

            extra = ("\n" + "\n".join(details)) if details else ""
            raise ModelCheckError(f"Qwen3-TTS worker 启动失败。{extra}")

    @classmethod
    def from_settings(cls, settings: Settings) -> "_QwenTtsWorker":
        py = settings.resolve_path(settings.qwen_tts_python_path)
        if not py or not py.exists():
            raise ModelCheckError(
                f"未找到 Qwen3-TTS Python: {py}。请设置 QWEN_TTS_PYTHON 为有效的 python 可执行文件（该环境需安装 qwen-tts）。"
            )

        model_dir = settings.resolve_path(settings.qwen_tts_model_path)
        if not model_dir or not model_dir.exists():
            raise ModelCheckError(
                f"未找到 Qwen3-TTS 模型目录: {model_dir}。请先下载模型权重并设置 QWEN_TTS_MODEL_PATH。"
            )

        stub = os.getenv(_QWEN_WORKER_STUB_ENV, "").strip() in {"1", "true", "TRUE", "yes", "YES"}
        return cls(python_exe=str(py), model_path=str(model_dir), stub=stub)

    def _read_stdout_line(self, timeout_sec: float = 0.2) -> str | None:
        """Read one line from worker stdout with a small timeout.

        On Linux, use select() so Ctrl+C cancellation can be observed promptly.
        On Windows, fall back to blocking readline(); Ctrl+C will typically also stop the child.
        """
        assert self._proc.stdout is not None
        if os.name == "nt":
            return self._proc.stdout.readline()

        # IMPORTANT:
        # subprocess.Popen(..., text=True) wraps stdout in TextIOWrapper with its own buffer.
        # If a previous readline() pulled multiple lines into the internal buffer, the OS-level fd
        # may have no pending bytes, causing select(fd) to return empty while data is still available
        # in the Python buffer. This can deadlock handshake (e.g. "__READY__" already buffered).
        try:
            buf = getattr(self._proc.stdout, "buffer", None)
            if buf is not None and hasattr(buf, "peek"):
                # peek() returns bytes already buffered (may be empty)
                if buf.peek(1):
                    return self._proc.stdout.readline()
        except Exception:
            pass

        try:
            import select
            fd = self._proc.stdout.fileno()
            r, _w, _x = select.select([fd], [], [], max(0.0, float(timeout_sec)))
            if not r:
                return None
        except Exception:
            # Fallback: best-effort blocking read.
            return self._proc.stdout.readline()
        return self._proc.stdout.readline()

    def synthesize(
        self,
        text: str,
        speaker_wav: str,
        output_path: str,
        language: str = "Auto",
        *,
        speaker_anchor_wav: str | None = None,
        icl_audio: str | None = None,
        icl_text: str | None = None,
        x_vector_only_mode: bool | None = None,
    ) -> dict:
        if self._proc.poll() is not None:
            raise RuntimeError("Qwen3-TTS worker 已退出")

        req = {
            "cmd": "synthesize",
            "text": text,
            "language": language,
            "speaker_wav": speaker_wav,
            "output_path": output_path,
        }
        if speaker_anchor_wav:
            req["speaker_anchor_wav"] = speaker_anchor_wav
        if icl_audio:
            req["icl_audio"] = icl_audio
        if icl_text:
            req["icl_text"] = icl_text
        if x_vector_only_mode is not None:
            req["x_vector_only_mode"] = bool(x_vector_only_mode)

        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

        skipped: list[str] = []
        max_skip = 50
        while True:
            check_cancelled()
            line = self._read_stdout_line(timeout_sec=0.2)
            if line is None:
                if self._proc.poll() is not None:
                    stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                    extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                    raise RuntimeError("Qwen3-TTS worker 已退出" + extra)
                continue
            if not line:
                stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                raise RuntimeError("Qwen3-TTS worker 无输出" + extra)

            s = line.strip()
            if not s:
                continue

            try:
                resp = json.loads(s)
                break
            except json.JSONDecodeError:
                skipped.append(s)
                if len(skipped) >= max_skip:
                    stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                    extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                    noise = "\n".join(skipped[-10:])
                    raise RuntimeError(
                        "Qwen3-TTS worker 输出无法解析为 JSON（协议被日志污染）。"
                        f"\nstdout_tail:\n{noise}{extra}"
                    )
                continue

        if not resp.get("ok"):
            err = str(resp.get("error", "unknown error"))
            trace = resp.get("trace")
            if trace:
                raise RuntimeError(err + "\n" + str(trace))
            raise RuntimeError(err)
        return resp

    def synthesize_batch(self, items: list[dict]) -> dict:
        if self._proc.poll() is not None:
            raise RuntimeError("Qwen3-TTS worker 已退出")

        req = {
            "cmd": "synthesize_batch",
            "items": items,
        }

        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

        skipped: list[str] = []
        max_skip = 50
        while True:
            check_cancelled()
            line = self._read_stdout_line(timeout_sec=0.2)
            if line is None:
                if self._proc.poll() is not None:
                    stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                    extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                    raise RuntimeError("Qwen3-TTS worker 已退出" + extra)
                continue
            if not line:
                stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                raise RuntimeError("Qwen3-TTS worker 无输出" + extra)

            s = line.strip()
            if not s:
                continue

            try:
                resp = json.loads(s)
                break
            except json.JSONDecodeError:
                skipped.append(s)
                if len(skipped) >= max_skip:
                    stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                    extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                    noise = "\n".join(skipped[-10:])
                    raise RuntimeError(
                        "Qwen3-TTS worker 输出无法解析为 JSON（协议被日志污染）。"
                        f"\nstdout_tail:\n{noise}{extra}"
                    )
                continue

        if not resp.get("ok"):
            err = str(resp.get("error", "unknown error"))
            trace = resp.get("trace")
            if trace:
                raise RuntimeError(err + "\n" + str(trace))
            raise RuntimeError(err)
        return resp

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if not proc:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        # Close stdin early so the worker can exit its read loop.
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        try:
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass


def gemini_tts(
    text: str,
    output_path: str,
    voice_name: str | None = None,
) -> None:
    """Generate speech using Gemini TTS and save to file using Best Practices."""
    
    if os.path.exists(output_path) and is_valid_wav(output_path):
        logger.info(f'Gemini TTS {text[:20]}... 已存在（已验证）')
        return
    elif os.path.exists(output_path):
        logger.warning(f"删除无效缓存文件: {output_path}")
        os.remove(output_path)
    
    try:
        from google.genai import types
        import wave
    except ImportError:
        logger.error("未安装google-genai。请安装它。")
        return

    client = _get_gemini_client()
    if not client:
        return
    
    model_name = _DEFAULT_SETTINGS.gemini_tts_model or "gemini-2.5-flash-preview-tts"
    voice_name = voice_name or _DEFAULT_SETTINGS.gemini_tts_voice or 'Kore'

    RATE = 24000
    SAMPLE_WIDTH = 2 # 16-bit
    CHANNELS = 1

    max_retries = 10  # Increase retries for rate limits
    retry_count = 0
    
    while retry_count < max_retries:
        check_cancelled()
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                )
            )
            
            if (response.candidates and 
                response.candidates[0].content and 
                response.candidates[0].content.parts and 
                response.candidates[0].content.parts[0].inline_data):
                
                pcm_data = response.candidates[0].content.parts[0].inline_data.data
                
                with wave.open(output_path, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(SAMPLE_WIDTH)
                    wf.setframerate(RATE)
                    wf.writeframes(pcm_data)
                
                if is_valid_wav(output_path):
                    logger.info(f'Gemini TTS已保存: {output_path}')
                    sleep_with_cancel(0.1)
                    return # Success
                else:
                    logger.warning("Gemini TTS写入无效文件，重试中...")
                    if os.path.exists(output_path):
                        os.remove(output_path)
            else:
                logger.warning(f"Gemini TTS响应结构异常。Candidates: {response.candidates}, PromptFeedback: {getattr(response, 'prompt_feedback', 'N/A')}")
                try:
                    logger.warning(f"完整响应转储: {response}")
                except Exception:
                    pass
                retry_count += 1
                sleep_with_cancel(1)
                continue

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                import re
                match = re.search(r'retry in (\d+\.?\d*)s', error_str)
                delay = float(match.group(1)) if match else 20.0
                delay += 1.0
                
                logger.warning(f"Gemini TTS速率限制(429)。{delay}秒后重试... (尝试 {retry_count+1}/{max_retries})")
                sleep_with_cancel(delay)
                retry_count += 1
                continue
            else:
                logger.error(f"Gemini TTS生成失败: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                retry_count += 1
                sleep_with_cancel(1)
        


def init_TTS(settings: Settings | None = None, model_manager: ModelManager | None = None) -> None:
    """Pre-initialize models if needed (mainly for warming up)."""
    return


def generate_wavs(
    folder: str,
    tts_method: str = "bytedance",
    qwen_tts_batch_size: int = 1,
    adaptive_segment_stretch: bool = False,
) -> None:
    check_cancelled()
    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    if not os.path.exists(transcript_path):
        logger.warning(f"未找到翻译文件: {transcript_path}")
        return

    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
        
    speakers = set(line["speaker"] for line in transcript)
    num_speakers = len(speakers)
    total_segments = len(transcript)
    logger.info(
        f"TTS({tts_method}) 开始: segments={total_segments}, speakers={num_speakers}, folder={folder}"
    )

    # 防止历史数据/旧逻辑生成超长 SPEAKER/*.wav 导致 voice cloning 显存/时间爆炸
    max_ref_seconds = _read_speaker_ref_seconds()
    allow_silence_fallback = (os.getenv("TTS_ALLOW_SILENCE_FALLBACK", "").strip().lower() in {"1", "true", "yes", "y"})
    failed_segments: list[int] = []
    speaker_dir = os.path.join(folder, "SPEAKER")
    os.makedirs(speaker_dir, exist_ok=True)

    # 如果 SPEAKER/*.wav 丢失（常见于旧任务目录或中途清理），从 audio_vocals.wav 兜底生成
    vocals_path = os.path.join(folder, "audio_vocals.wav")
    if os.path.exists(vocals_path):
        for spk in speakers:
            check_cancelled()
            spk_path = os.path.join(speaker_dir, f"{spk}.wav")
            if os.path.exists(spk_path):
                continue
            try:
                check_cancelled()
                wav, _sr = librosa.load(vocals_path, sr=24000, mono=True, duration=max_ref_seconds)
                if wav.size > 0:
                    save_wav(wav.astype(np.float32), spk_path, sample_rate=24000)
                    logger.info(f"已生成缺失的说话人参考 ({max_ref_seconds:.1f}秒): {spk_path}")
            except Exception as exc:
                logger.warning(f"生成说话人参考音频失败 {spk}: {exc}")

    for spk in speakers:
        check_cancelled()
        _ensure_wav_max_duration(os.path.join(speaker_dir, f"{spk}.wav"), max_ref_seconds, sample_rate=24000)

    # --- Qwen ICL（语气/节奏）支持 ---
    # 启用方式：设置环境变量 QWEN_TTS_ICL=1。
    #
    # 设计：
    # - 音色锚点：每个视频取 audio_vocals.wav 前 max_ref_seconds 秒
    # - 语气锚点：按“同 speaker + 同 text 且时间连续”的片段组，提取该组对应的音频区间，并用该组的英文 text 做 ref_text
    #
    # NOTE: translation.json 可能经过 split_sentences()（只拆译文不拆原文），导致单条记录的 text 与其时间窗不严格对齐。
    # 为了保证 ICL 的 (ref_audio, ref_text) 对齐，我们在这里以“组”为单位做 ICL，而不是对每个拆分后的子句强行切音频。
    qwen_use_icl = (tts_method == "qwen") and _env_flag(_QWEN_TTS_ICL_ENV)
    qwen_speaker_anchor_wav: str | None = None
    icl_group_for_index: dict[int, int] = {}
    icl_groups: dict[int, dict[str, object]] = {}

    if qwen_use_icl:
        if not os.path.exists(vocals_path):
            logger.warning("Qwen ICL 已启用但未找到 audio_vocals.wav，将回退到 x-vector only。")
            qwen_use_icl = False
        else:
            qwen_speaker_anchor_wav = os.path.join(folder, ".qwen_speaker_anchor.wav")
            if not (os.path.exists(qwen_speaker_anchor_wav) and is_valid_wav(qwen_speaker_anchor_wav)):
                try:
                    wav, _sr = librosa.load(vocals_path, sr=24000, mono=True, duration=max_ref_seconds)
                    if wav.size > 0:
                        save_wav(wav.astype(np.float32), qwen_speaker_anchor_wav, sample_rate=24000)
                except Exception as exc:
                    logger.warning(f"生成 Qwen 音色锚点失败，将回退到 x-vector only: {exc}")
                    qwen_use_icl = False

            if qwen_use_icl and qwen_speaker_anchor_wav and is_valid_wav(qwen_speaker_anchor_wav):
                icl_ref_dir = os.path.join(folder, ".qwen_icl_ref")
                os.makedirs(icl_ref_dir, exist_ok=True)

                # Build groups.
                tol = 0.02  # translation.json 的时间戳通常 round 到 1e-3；这里留一点容差
                gid = 0
                i = 0
                while i < total_segments:
                    seg = transcript[i]
                    spk = str(seg.get("speaker", ""))
                    txt = str(seg.get("text", "") or "")
                    try:
                        g_start = float(seg.get("start", 0.0) or 0.0)
                        g_end = float(seg.get("end", g_start) or g_start)
                    except Exception:
                        g_start, g_end = 0.0, 0.0

                    j = i
                    while (j + 1) < total_segments:
                        nxt = transcript[j + 1]
                        if str(nxt.get("speaker", "")) != spk:
                            break
                        if str(nxt.get("text", "") or "") != txt:
                            break
                        try:
                            nxt_start = float(nxt.get("start", 0.0) or 0.0)
                            nxt_end = float(nxt.get("end", nxt_start) or nxt_start)
                        except Exception:
                            break
                        if abs(nxt_start - g_end) > tol:
                            break
                        g_end = nxt_end
                        j += 1

                    for k in range(i, j + 1):
                        icl_group_for_index[k] = gid
                    icl_groups[gid] = {
                        "speaker": spk,
                        "text": txt,
                        "start": float(g_start),
                        "end": float(g_end),
                        "wav_path": os.path.join(icl_ref_dir, f"{gid:04d}.wav"),
                    }
                    gid += 1
                    i = j + 1

                logger.info(
                    f"Qwen ICL已启用: anchor={Path(qwen_speaker_anchor_wav).name}, groups={len(icl_groups)}"
                )
            else:
                logger.warning("Qwen ICL 已启用但音色锚点无效，将回退到 x-vector only。")
                qwen_use_icl = False

    def _ensure_qwen_icl_ref_for_index(idx: int) -> tuple[str, str] | None:
        """Return (icl_audio_path, icl_text) if available; otherwise None."""
        if not qwen_use_icl:
            return None
        gid = icl_group_for_index.get(int(idx))
        if gid is None:
            return None
        meta = icl_groups.get(gid)
        if not meta:
            return None
        txt = str(meta.get("text", "") or "").strip()
        if not txt:
            return None

        wav_path = str(meta.get("wav_path", "") or "")
        if not wav_path:
            return None
        if os.path.exists(wav_path) and is_valid_wav(wav_path):
            return wav_path, txt

        # Extract group audio from vocals.
        try:
            start_s = float(meta.get("start", 0.0) or 0.0)
            end_s = float(meta.get("end", start_s) or start_s)
        except Exception:
            return None
        dur = float(end_s - start_s)
        if dur <= 0.05:
            return None
        try:
            wav, _sr = librosa.load(
                vocals_path,
                sr=24000,
                mono=True,
                offset=max(0.0, float(start_s)),
                duration=max(0.0, dur),
            )
            if wav.size <= 0:
                return None
            save_wav(wav.astype(np.float32), wav_path, sample_rate=24000)
            if is_valid_wav(wav_path):
                return wav_path, txt
        except Exception as exc:
            logger.warning(f"生成 Qwen ICL 参考音频失败 (idx={idx}, gid={gid}): {exc}")
            return None
        return None
    
    # 使用分块累积，避免对 full_wav 反复 np.concatenate 造成 O(n^2) 内存峰值；
    # 同时确保全程 float32，避免默认 np.zeros(float64) 导致整体上浮到 float64 占用翻倍。
    chunks: list[np.ndarray] = []
    full_samples = 0  # 已累积的总采样点数（24kHz）

    qwen_worker: _QwenTtsWorker | None = None
    if tts_method == "qwen":
        qwen_worker = _QwenTtsWorker.from_settings(_DEFAULT_SETTINGS)

    try:
        qwen_batch_size = 1
        if tts_method == "qwen":
            try:
                qwen_batch_size = int(qwen_tts_batch_size or 1)
            except Exception:
                qwen_batch_size = 1
            qwen_batch_size = max(1, min(qwen_batch_size, 64))

        qwen_resp_by_index: dict[int, dict] = {}
        qwen_cached_before: set[int] = set()
        if tts_method == "qwen" and qwen_worker is not None and qwen_batch_size > 1:
            logger.info(f"Qwen3-TTS批处理已启用: batch_size={qwen_batch_size}")
            for i in range(total_segments):
                check_cancelled()
                out = os.path.join(output_folder, f"{str(i).zfill(4)}.wav")
                if os.path.exists(out) and is_valid_wav(out):
                    qwen_cached_before.add(i)

            def _looks_like_oom(err: str) -> bool:
                s = (err or "").lower()
                return ("out of memory" in s) or ("cuda" in s and "memory" in s) or ("oom" in s)

            def _run_qwen_batch(batch_indices: list[int]) -> None:
                if not batch_indices:
                    return
                check_cancelled()

                batch_items: list[dict] = []
                for j in batch_indices:
                    check_cancelled()
                    seg = transcript[j]
                    speaker = seg["speaker"]
                    text = preprocess_text(seg["translation"])
                    out = os.path.join(output_folder, f"{str(j).zfill(4)}.wav")
                    spk_wav = os.path.join(folder, "SPEAKER", f"{speaker}.wav")
                    speaker_wav_for_req = qwen_speaker_anchor_wav if qwen_speaker_anchor_wav else spk_wav
                    # If invalid/corrupted file exists, remove so worker can rewrite cleanly.
                    if os.path.exists(out) and not is_valid_wav(out):
                        try:
                            os.remove(out)
                        except Exception:
                            pass
                    item: dict = {
                        "text": text,
                        "language": "Auto",
                        "speaker_wav": speaker_wav_for_req,
                        "output_path": out,
                    }
                    if qwen_use_icl and qwen_speaker_anchor_wav:
                        icl = _ensure_qwen_icl_ref_for_index(j)
                        if icl is not None:
                            icl_audio, icl_text = icl
                            item.update(
                                {
                                    "speaker_anchor_wav": qwen_speaker_anchor_wav,
                                    "icl_audio": icl_audio,
                                    "icl_text": icl_text,
                                    "x_vector_only_mode": False,
                                }
                            )
                        else:
                            # Fallback to embedding-only while keeping the anchor timbre.
                            item.update(
                                {
                                    "speaker_anchor_wav": qwen_speaker_anchor_wav,
                                    "x_vector_only_mode": True,
                                }
                            )
                    batch_items.append(item)

                try:
                    check_cancelled()
                    resp = qwen_worker.synthesize_batch(batch_items)
                except Exception as exc:
                    if len(batch_indices) > 1 and _looks_like_oom(str(exc)):
                        mid = len(batch_indices) // 2
                        _run_qwen_batch(batch_indices[:mid])
                        _run_qwen_batch(batch_indices[mid:])
                        return
                    logger.warning(f"Qwen3-TTS批处理失败，段 {batch_indices}: {exc}")
                    for j in batch_indices:
                        qwen_resp_by_index[j] = {"ok": False, "error": str(exc)}
                    return

                results = resp.get("results")
                if not isinstance(results, list) or len(results) != len(batch_items):
                    logger.warning(
                        f"Qwen3-TTS批处理返回意外结果: {type(results)} len={getattr(results, '__len__', lambda: -1)()}"
                    )
                    for j in batch_indices:
                        qwen_resp_by_index[j] = {"ok": False, "error": "invalid batch results"}
                    return

                for j, r in zip(batch_indices, results):
                    if isinstance(r, dict):
                        qwen_resp_by_index[j] = r
                        if not r.get("ok"):
                            logger.warning(f"Qwen3-TTS批处理项失败 (段={j}): {r.get('error')}")
                    else:
                        qwen_resp_by_index[j] = {"ok": False, "error": "invalid item result"}

                # If the worker reports OOM for some items, retry those with smaller batches.
                oom_failed: list[int] = []
                for j in batch_indices:
                    rj = qwen_resp_by_index.get(j) or {}
                    if rj.get("ok") is False and _looks_like_oom(str(rj.get("error", ""))):
                        oom_failed.append(j)
                if len(oom_failed) > 1:
                    mid = len(oom_failed) // 2
                    _run_qwen_batch(oom_failed[:mid])
                    _run_qwen_batch(oom_failed[mid:])

        for i, line in enumerate(transcript):
            check_cancelled()
            speaker = line["speaker"]
            text = preprocess_text(line["translation"])
            output_path = os.path.join(output_folder, f"{str(i).zfill(4)}.wav")
            speaker_wav = os.path.join(folder, "SPEAKER", f"{speaker}.wav")
            seg_no = i + 1
            if tts_method == "qwen" and qwen_worker is not None and qwen_batch_size > 1:
                was_cached = i in qwen_cached_before
            else:
                was_cached = os.path.exists(output_path) and is_valid_wav(output_path)
            cache_tag = " [cached]" if was_cached else ""
            logger.info(
                f"TTS({tts_method}) {seg_no}/{total_segments}: {Path(output_path).name}{cache_tag} speaker={speaker}"
            )

            start = line["start"]
            original_length = line["end"] - start

            tts_elapsed = 0.0
            qwen_resp: dict | None = qwen_resp_by_index.get(i)
            needs_tts = not (os.path.exists(output_path) and is_valid_wav(output_path))
            if needs_tts:
                tts_begin = time.monotonic()
                try:
                    check_cancelled()
                    if tts_method == "qwen" and qwen_worker is not None:
                        if qwen_batch_size > 1:
                            # Generate current + upcoming missing segments in one worker call.
                            batch_indices: list[int] = []
                            j = i
                            while j < total_segments and len(batch_indices) < qwen_batch_size:
                                check_cancelled()
                                out = os.path.join(output_folder, f"{str(j).zfill(4)}.wav")
                                if not (os.path.exists(out) and is_valid_wav(out)):
                                    batch_indices.append(j)
                                j += 1
                            _run_qwen_batch(batch_indices)
                            qwen_resp = qwen_resp_by_index.get(i)
                        else:
                            if qwen_use_icl and qwen_speaker_anchor_wav:
                                icl = _ensure_qwen_icl_ref_for_index(i)
                                if icl is not None:
                                    icl_audio, icl_text = icl
                                    qwen_resp = qwen_worker.synthesize(
                                        text,
                                        speaker_wav=qwen_speaker_anchor_wav,
                                        output_path=output_path,
                                        language="Auto",
                                        speaker_anchor_wav=qwen_speaker_anchor_wav,
                                        icl_audio=icl_audio,
                                        icl_text=icl_text,
                                        x_vector_only_mode=False,
                                    )
                                else:
                                    qwen_resp = qwen_worker.synthesize(
                                        text,
                                        speaker_wav=qwen_speaker_anchor_wav,
                                        output_path=output_path,
                                        language="Auto",
                                        speaker_anchor_wav=qwen_speaker_anchor_wav,
                                        x_vector_only_mode=True,
                                    )
                            else:
                                qwen_resp = qwen_worker.synthesize(
                                    text, speaker_wav, output_path, language="Auto"
                                )
                    elif tts_method == "gemini":
                        gemini_tts(text, output_path)
                    else:
                        bytedance_tts(text, output_path, speaker_wav)
                except Exception as exc:
                    logger.warning(f"TTS({tts_method})段 {i} 失败: {exc}")
                tts_elapsed = time.monotonic() - tts_begin
            
            current_full_end = full_samples / 24000.0
            
            if start > current_full_end:
                silence_dur = start - current_full_end
                silence_samples = int(silence_dur * 24000)
                if silence_samples > 0:
                    chunks.append(np.zeros((silence_samples,), dtype=np.float32))
                    full_samples += silence_samples
                
            new_start = full_samples / 24000.0
            line['start'] = new_start
            
            if i < len(transcript) - 1:
                next_line = transcript[i+1]
                target_end = min(new_start + original_length, next_line['end'])
            else:
                target_end = new_start + original_length
                
            desired_len = target_end - new_start
            
            adjust_elapsed = 0.0
            valid_wav = os.path.exists(output_path) and is_valid_wav(output_path)
            if valid_wav:
                adjust_begin = time.monotonic()
                check_cancelled()
                # 默认限制“放慢(拉长)”幅度，避免明显失真；但有些语言对（如英->中）
                # 可能需要更大的拉伸才能消除无声段，因此提供可选的更激进模式。
                max_stretch = 1.35 if adaptive_segment_stretch else 1.1
                wav_chunk, actual_len = adjust_audio_length(
                    output_path,
                    desired_len,
                    max_speed_factor=max_stretch,
                )
                adjust_elapsed = time.monotonic() - adjust_begin
            else:
                logger.warning(f"TTS生成失败 {tts_method} 段 {i}，使用静音回退。")
                failed_segments.append(i)
                target_samples = int(desired_len * 24000)
                wav_chunk = np.zeros(target_samples, dtype=np.float32)
                actual_len = desired_len
            
            wav_chunk = wav_chunk.astype(np.float32, copy=False)
            chunks.append(wav_chunk)
            full_samples += int(wav_chunk.shape[0])
            line['end'] = new_start + actual_len

            # 段落完成日志：便于在终端观察“是否卡死/进度/耗时”
            qwen_extra = ""
            if qwen_resp:
                sr = qwen_resp.get("sr")
                n_samples = qwen_resp.get("n_samples")
                if sr is not None and n_samples is not None:
                    qwen_extra = f", sr={sr}, n_samples={n_samples}"
            source = "cached" if was_cached else ("tts" if valid_wav else "silence")
            logger.info(
                f"TTS({tts_method}) {seg_no}/{total_segments} 完成: source={source}, "
                f"tts={tts_elapsed:.2f}s, adjust={adjust_elapsed:.2f}s, desired={desired_len:.2f}s, actual={actual_len:.2f}s"
                f"{qwen_extra}"
            )
    finally:
        if qwen_worker is not None:
            qwen_worker.close()

    if failed_segments and not allow_silence_fallback:
        head = ", ".join(str(i) for i in failed_segments[:10])
        more = "..." if len(failed_segments) > 10 else ""
        raise RuntimeError(
            f"TTS({tts_method}) 失败 {len(failed_segments)}/{total_segments} 段（indexes: {head}{more}）。"
            "为避免产出大量静音，已中止；如确实要用静音补齐请设置 TTS_ALLOW_SILENCE_FALLBACK=1。"
        )

    check_cancelled()
    if chunks:
        full_wav = np.concatenate(chunks).astype(np.float32, copy=False)
    else:
        full_wav = np.zeros((0,), dtype=np.float32)
    # 释放分块引用，降低峰值占用
    del chunks

    vocal_wav_path = os.path.join(folder, 'audio_vocals.wav')
    if os.path.exists(vocal_wav_path):
        check_cancelled()
        vocal_wav, _ = librosa.load(vocal_wav_path, sr=24000)
        if full_wav.size > 0:
            # 避免 np.abs(full_wav) 产生同等大小的临时数组（长音频会非常吃内存）
            max_val = float(np.max(full_wav))
            min_val = float(np.min(full_wav))
            full_peak = max(abs(max_val), abs(min_val))
            if full_peak > 0.0:
                v_max = float(np.max(vocal_wav))
                v_min = float(np.min(vocal_wav))
                vocal_peak = max(abs(v_max), abs(v_min))
                if vocal_peak > 0.0:
                    scale = np.float32(vocal_peak / full_peak)
                    full_wav *= scale
        # 释放大数组引用
        del vocal_wav
            
    check_cancelled()
    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'), sample_rate=24000)
    logger.info(f'已生成 {os.path.join(folder, "audio_tts.wav")}')
    
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    instruments_path = os.path.join(folder, 'audio_instruments.wav')
    if os.path.exists(instruments_path):
        check_cancelled()
        instruments_wav, _ = librosa.load(instruments_path, sr=24000)
        instruments_wav = instruments_wav.astype(np.float32, copy=False)
        
        len_full = len(full_wav)
        len_inst = len(instruments_wav)
        
        if len_full > len_inst:
            instruments_wav = np.pad(instruments_wav, (0, len_full - len_inst), mode='constant')
        elif len_inst > len_full:
            full_wav = np.pad(full_wav, (0, len_inst - len_full), mode='constant')
            
        # audio_tts.wav 已经写入，这里允许原地相加以避免额外分配
        full_wav += instruments_wav
        del instruments_wav
        check_cancelled()
        save_wav_norm(full_wav, os.path.join(folder, 'audio_combined.wav'), sample_rate=24000)
        logger.info(f'已生成 {os.path.join(folder, "audio_combined.wav")}')
    else:
        logger.warning("未找到乐器音频，仅保存TTS作为combined。")
        check_cancelled()
        save_wav_norm(full_wav, os.path.join(folder, 'audio_combined.wav'), sample_rate=24000)

    # Write a small marker so we can distinguish "valid result" from stale/partial artifacts.
    try:
        state_path = os.path.join(folder, ".tts_done.json")
        state = {
            "tts_method": tts_method,
            "speaker_ref_seconds": max_ref_seconds,
            "created_at": time.time(),
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception:
        # Best-effort marker only.
        pass


def generate_all_wavs_under_folder(
    root_folder: str,
    tts_method: str = "bytedance",
    qwen_tts_batch_size: int = 1,
    adaptive_segment_stretch: bool = False,
) -> str:
    count = 0
    for root, _dirs, files in os.walk(root_folder):
        check_cancelled()
        if 'translation.json' not in files:
            continue
        combined_path = os.path.join(root, 'audio_combined.wav')
        done_path = os.path.join(root, ".tts_done.json")
        if os.path.exists(combined_path) and is_valid_wav(combined_path) and os.path.exists(done_path):
            try:
                with open(done_path, "r", encoding="utf-8") as f:
                    st = json.load(f)
                if st.get("tts_method") == tts_method:
                    # If translation.json is newer than marker, re-run to reflect changes.
                    tr_mtime = os.path.getmtime(os.path.join(root, "translation.json"))
                    done_mtime = os.path.getmtime(done_path)
                    if done_mtime >= tr_mtime:
                        continue
            except Exception:
                # Treat as not done and re-generate.
                pass
        generate_wavs(
            root,
            tts_method,
            qwen_tts_batch_size=qwen_tts_batch_size,
            adaptive_segment_stretch=adaptive_segment_stretch,
        )
        count += 1
    msg = f"语音合成完成: {root_folder}（处理 {count} 个文件）"
    logger.info(msg)
    return msg
