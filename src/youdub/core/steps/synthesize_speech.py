import base64
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import requests
import torch
from audiostretchy.stretch import stretch_audio
from loguru import logger
from pyannote.audio import Inference, Model
from scipy.spatial.distance import cosine

from ...config import Settings
from ...models import ModelCheckError, ModelManager
from ..cn_tx import TextNorm
from ..utils import save_wav, save_wav_norm

# --- Global Caches ---
_XTTS_MODEL = None
_EMBEDDING_MODEL = None
_EMBEDDING_INFERENCE = None

_DEFAULT_SETTINGS = Settings()
_DEFAULT_MODEL_MANAGER = ModelManager(_DEFAULT_SETTINGS)

# --- ByteDance TTS Constants ---
_BYTEDANCE_HOST = "openspeech.bytedance.com"
_BYTEDANCE_API_URL = f"https://{_BYTEDANCE_HOST}/api/v1/tts"


# --- Helper Functions ---

def preprocess_text(text: str) -> str:
    text = text.replace('AI', '人工智能')
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    normalizer = TextNorm()
    text = normalizer(text)
    # Insert space between letters and numbers
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
    
    target_path = wav_path.replace('.wav', '_adjusted.wav')
    stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
    
    wav_stretched, _ = librosa.load(target_path, sr=sample_rate)
    
    # Trim to exactly new_desired_length in samples
    target_samples = int(new_desired_length * sample_rate)
    return wav_stretched[:target_samples], new_desired_length


# --- Embedding & Speaker Matching Logic ---

def load_embedding_model() -> None:
    global _EMBEDDING_MODEL, _EMBEDDING_INFERENCE
    if _EMBEDDING_MODEL is not None and _EMBEDDING_INFERENCE is not None:
        return

    try:
        logger.info("Loading pyannote/embedding model...")
        token = _DEFAULT_SETTINGS.hf_token
        _EMBEDDING_MODEL = Model.from_pretrained("pyannote/embedding", use_auth_token=token)
        
        if _EMBEDDING_MODEL is None:
            logger.error("Failed to load pyannote/embedding. Please check HF_TOKEN.")
            raise ValueError("Model.from_pretrained returned None")
            
        _EMBEDDING_INFERENCE = Inference(_EMBEDDING_MODEL, window="whole")
        logger.info("pyannote/embedding model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading pyannote/embedding model: {e}")
        raise


def generate_embedding(wav_path: str) -> np.ndarray:
    load_embedding_model()
    if _EMBEDDING_INFERENCE is None:
        raise RuntimeError("Embedding model is not available.")
    embedding = _EMBEDDING_INFERENCE(wav_path)
    return embedding


def bytedance_tts_api(
    text: str, 
    voice_type: str = 'BV001_streaming'
) -> bytes | None:
    appid = _DEFAULT_SETTINGS.bytedance_appid
    access_token = _DEFAULT_SETTINGS.bytedance_access_token
    
    if not appid or not access_token:
        logger.warning("ByteDance APPID or ACCESS_TOKEN not set.")
        return None

    header = {"Authorization": f"Bearer;{access_token}"}
    request_json = {
        "app": {
            "appid": appid,
            "token": access_token,
            "cluster": 'volcano_tts'
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
             logger.warning(f"ByteDance API error: {resp.json()}")
    except Exception as e:
        logger.warning(f"ByteDance TTS request failed: {e}")
        
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
    
    sample_text = 'YouDub 是一个创新的开源工具，专注于将 YouTube 等平台的优质视频翻译和配音为中文版本。'
    
    for voice_type in voice_types:
        wav_path = os.path.join(voice_type_dir, f'{voice_type}.wav')
        npy_path = wav_path.replace('.wav', '.npy')
        
        if os.path.exists(wav_path) and os.path.exists(npy_path):
            continue
            
        logger.info(f"Generating reference audio for {voice_type}...")
        audio_data = bytedance_tts_api(sample_text, voice_type=voice_type)
        if audio_data:
            with open(wav_path, "wb") as f:
                f.write(audio_data)
            
            try:
                embedding = generate_embedding(wav_path)
                np.save(npy_path, embedding)
            except Exception as e:
                logger.error(f"Failed to generate embedding for {voice_type}: {e}")


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
         logger.warning("No voice_type directory found, cannot perform speaker matching.")
         return {}

    for file in os.listdir(voice_type_dir):
        if file.endswith('.npy'):
            voice_type = file.replace('.npy', '')
            voice_types[voice_type] = np.load(os.path.join(voice_type_dir, file))
            
    if not os.path.exists(speaker_folder):
         return {}

    for file in os.listdir(speaker_folder):
        if not file.endswith('.wav'):
            continue
        speaker = file.replace('.wav', '')
        wav_path = os.path.join(speaker_folder, file)
        
        try:
            embedding = generate_embedding(wav_path)
            np.save(wav_path.replace('.wav', '.npy'), embedding)
            
            # Find closest voice type
            best_match = sorted(voice_types.keys(), key=lambda x: 1 - cosine(voice_types[x], embedding))[0]
            speaker_to_voice_type[speaker] = best_match
            logger.info(f'Matched {speaker} to {best_match}')
        except Exception as e:
            logger.error(f"Error matching speaker {speaker}: {e}")
            speaker_to_voice_type[speaker] = 'BV001_streaming' # Default fallback

    with open(speaker_to_voice_type_path, 'w', encoding='utf-8') as f:
        json.dump(speaker_to_voice_type, f, indent=2, ensure_ascii=False)
        
    return speaker_to_voice_type


def bytedance_tts(
    text: str, 
    output_path: str, 
    speaker_wav: str, 
    voice_type: str | None = None
) -> None:
    if os.path.exists(output_path):
        logger.info(f'ByteDance TTS {text[:20]}... exists')
        return

    folder = os.path.dirname(os.path.dirname(output_path))
    
    if voice_type is None:
        speaker_to_voice_type = generate_speaker_to_voice_type(folder)
        speaker = os.path.basename(speaker_wav).replace('.wav', '')
        voice_type = speaker_to_voice_type.get(speaker, 'BV001_streaming')

    for _ in range(3):
        audio_data = bytedance_tts_api(text, voice_type=voice_type) # type: ignore
        if audio_data:
            with open(output_path, "wb") as f:
                f.write(audio_data)
            
            # Validation read
            try:
                librosa.load(output_path, sr=24000)
                logger.info(f'ByteDance TTS saved: {output_path}')
                time.sleep(0.1)
                break
            except Exception:
                logger.warning("Saved wav file seems corrupted, retrying...")
        
        time.sleep(0.5)


# --- XTTS Logic ---

def init_xtts(settings: Settings | None = None, model_manager: ModelManager | None = None) -> None:
    load_xtts_model(settings=settings, model_manager=model_manager)


def load_xtts_model(
    model_path: str | None = None,
    device: str = 'auto',
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> None:
    global _XTTS_MODEL
    if _XTTS_MODEL is not None:
        return
        
    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER

    try:
        from TTS.api import TTS
    except ImportError:
        logger.error("TTS module not found. Please install 'TTS' or use ByteDance TTS.")
        raise

    model_manager.enforce_offline()

    if model_path is None:
        resolved = settings.resolve_path(settings.xtts_model_path)
        model_path = str(resolved) if resolved else ""

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise ModelCheckError(
            f"XTTS 模型目录不存在：{model_path_obj}。请提前下载 XTTS v2 到本地并设置 XTTS_MODEL_PATH。"
        )

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    logger.info(f'Loading XTTS model from {model_path}')
    t_start = time.time()
    _XTTS_MODEL = TTS(model_path).to(device)
    t_end = time.time()
    logger.info(f'XTTS model loaded in {t_end - t_start:.2f}s')


def xtts_tts(
    text: str,
    output_path: str,
    speaker_wav: str,
    model_name: str | None = None,
    device: str = 'auto',
    language: str = 'zh-cn',
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> None:
    global _XTTS_MODEL
    
    if os.path.exists(output_path):
        logger.info(f'XTTS {text[:20]}... exists')
        return
    
    if _XTTS_MODEL is None:
        load_xtts_model(model_name, device, settings=settings, model_manager=model_manager)
    
    for _ in range(3):
        try:
            wav = _XTTS_MODEL.tts(text, speaker_wav=speaker_wav, language=language) # type: ignore
            wav = np.array(wav)
            save_wav(wav, output_path)
            logger.info(f'XTTS Generated: {text[:20]}...')
            break
        except Exception as e:
            logger.warning(f'XTTS failed: {e}')


# --- Init for Pipeline ---

def init_TTS(settings: Settings | None = None, model_manager: ModelManager | None = None) -> None:
    """Pre-initialize models if needed (mainly for warming up)."""
    pass


# --- Main Generation Logic ---

def generate_wavs(folder: str, force_bytedance: bool = False) -> None:
    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    if not os.path.exists(transcript_path):
        logger.warning(f"Translation file not found: {transcript_path}")
        return

    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
        
    speakers = set(line['speaker'] for line in transcript)
    num_speakers = len(speakers)
    logger.info(f'Found {num_speakers} speakers')
    
    full_wav = np.zeros((0, ))
    
    for i, line in enumerate(transcript):
        speaker = line['speaker']
        text = preprocess_text(line['translation'])
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
        
        # Determine TTS Method
        if num_speakers == 1:
             bytedance_tts(text, output_path, speaker_wav, voice_type='BV701_streaming')
        elif force_bytedance:
            bytedance_tts(text, output_path, speaker_wav)
        else:
            xtts_tts(text, output_path, speaker_wav)
            
        # Audio stitching and timing adjustment
        start = line['start']
        original_length = line['end'] - start
        
        current_full_end = len(full_wav) / 24000.0
        
        # If there is a gap between current end and next start, fill with silence
        if start > current_full_end:
            silence_dur = start - current_full_end
            full_wav = np.concatenate((full_wav, np.zeros((int(silence_dur * 24000), ))))
            
        new_start = len(full_wav) / 24000.0
        line['start'] = new_start
        
        if i < len(transcript) - 1:
            next_line = transcript[i+1]
            target_end = min(new_start + original_length, next_line['end'])
        else:
            target_end = new_start + original_length
            
        desired_len = target_end - new_start
        
        # Adjust generated wav to fit desired length
        wav_chunk, actual_len = adjust_audio_length(output_path, desired_len)
        
        full_wav = np.concatenate((full_wav, wav_chunk))
        line['end'] = new_start + actual_len

    # Mixing with background music
    vocal_wav_path = os.path.join(folder, 'audio_vocals.wav')
    if os.path.exists(vocal_wav_path):
        vocal_wav, _ = librosa.load(vocal_wav_path, sr=24000)
        # Normalize volume of TTS to match original vocals peak
        if len(full_wav) > 0 and np.max(np.abs(full_wav)) > 0:
            full_wav = full_wav / np.max(np.abs(full_wav)) * np.max(np.abs(vocal_wav))
            
    save_wav(full_wav, os.path.join(folder, 'audio_tts.wav'), sample_rate=24000)
    
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    # Combine with instruments
    instruments_path = os.path.join(folder, 'audio_instruments.wav')
    if os.path.exists(instruments_path):
        instruments_wav, _ = librosa.load(instruments_path, sr=24000)
        
        len_full = len(full_wav)
        len_inst = len(instruments_wav)
        
        if len_full > len_inst:
            instruments_wav = np.pad(instruments_wav, (0, len_full - len_inst), mode='constant')
        elif len_inst > len_full:
            full_wav = np.pad(full_wav, (0, len_inst - len_full), mode='constant')
            
        combined_wav = full_wav + instruments_wav
        save_wav_norm(combined_wav, os.path.join(folder, 'audio_combined.wav'), sample_rate=24000)
        logger.info(f'Generated {os.path.join(folder, "audio_combined.wav")}')
    else:
        logger.warning("No instruments audio found, saving TTS only as combined.")
        save_wav_norm(full_wav, os.path.join(folder, 'audio_combined.wav'), sample_rate=24000)


def generate_all_wavs_under_folder(root_folder: str, force_bytedance: bool = False) -> str:
    count = 0
    for root, dirs, files in os.walk(root_folder):
        if 'translation.json' in files and 'audio_combined.wav' not in files:
            generate_wavs(root, force_bytedance)
            count += 1
    msg = f'Generated all wavs under {root_folder} (processed {count} files)'
    logger.info(msg)
    return msg
