import base64
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import requests
from audiostretchy.stretch import stretch_audio
from loguru import logger
from scipy.spatial.distance import cosine

from ...config import Settings
from ...models import ModelCheckError, ModelManager
from ..cn_tx import TextNorm
from ..utils import save_wav, save_wav_norm

# --- Global Caches ---
_EMBEDDING_MODEL = None
_EMBEDDING_INFERENCE = None
_EMBEDDING_MODEL_LOAD_FAILED = False

_DEFAULT_SETTINGS = Settings()
_DEFAULT_MODEL_MANAGER = ModelManager(_DEFAULT_SETTINGS)

# --- ByteDance TTS Constants ---
_BYTEDANCE_HOST = "openspeech.bytedance.com"
_BYTEDANCE_API_URL = f"https://{_BYTEDANCE_HOST}/api/v1/tts"

# --- Gemini TTS Client Cache ---
_GEMINI_CLIENT = None

def _get_gemini_client():
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        try:
            from google import genai
        except ImportError:
            logger.error("google-genai not installed. Please run 'uv sync'.")
            return None
            
        api_key = _DEFAULT_SETTINGS.gemini_api_key
        if not api_key:
            logger.warning("GEMINI_API_KEY not set in config/env.")
            return None
        # Explicitly pass the API key from settings as requested
        _GEMINI_CLIENT = genai.Client(api_key=api_key)
    return _GEMINI_CLIENT


# --- Helper Functions ---

def is_valid_wav(path: str) -> bool:
    """Check if a file is a valid WAV file."""
    if not os.path.exists(path):
        return False
    if os.path.getsize(path) < 44:  # Minimal WAV header size
        return False
    try:
        # Quick check with soundfile which is faster and strict on headers
        import soundfile as sf
        with sf.SoundFile(path) as f:
            return True
    except Exception:
        return False


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
    global _EMBEDDING_MODEL, _EMBEDDING_INFERENCE, _EMBEDDING_MODEL_LOAD_FAILED
    
    if _EMBEDDING_MODEL_LOAD_FAILED:
        raise RuntimeError("Embedding model previously failed to load. Skipping retry.")

    if _EMBEDDING_MODEL is not None and _EMBEDDING_INFERENCE is not None:
        return

    try:
        try:
            from pyannote.audio import Inference, Model  # type: ignore
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                "缺少依赖 pyannote.audio，无法进行说话人 embedding/音色匹配。"
                "如不需要该功能，可忽略；否则请安装 pyannote.audio 并准备离线模型缓存。"
            ) from exc

        logger.info("Loading pyannote/embedding model...")
        token = _DEFAULT_SETTINGS.hf_token
        _EMBEDDING_MODEL = Model.from_pretrained("pyannote/embedding", use_auth_token=token)
        
        if _EMBEDDING_MODEL is None:
            logger.error("Failed to load pyannote/embedding. Please check HF_TOKEN.")
            raise ValueError("Model.from_pretrained returned None")
            
        _EMBEDDING_INFERENCE = Inference(_EMBEDDING_MODEL, window="whole")
        logger.info("pyannote/embedding model loaded successfully.")
    except Exception as e:
        _EMBEDDING_MODEL_LOAD_FAILED = True
        logger.error(f"Error loading pyannote/embedding model: {e}")
        raise


def generate_embedding(wav_path: str) -> np.ndarray:
    load_embedding_model()
    if _EMBEDDING_INFERENCE is None:
        raise RuntimeError("Embedding model is not available.")
    embedding = _EMBEDDING_INFERENCE(wav_path)
    return embedding


# --- V3 Async TTS API (supports ICL 2.0 voice cloning) ---
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
        logger.warning("ByteDance APPID or ACCESS_TOKEN not set.")
        return None

    # Determine resource ID based on voice type
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
    
    # Step 1: Submit task
    try:
        logger.debug(f"V3 TTS Submit: speaker={speaker}, text={text[:30]}...")
        resp = requests.post(_BYTEDANCE_V3_SUBMIT_URL, json=submit_payload, headers=headers, timeout=30)
        resp_json = resp.json()
        
        if resp_json.get("code") != 20000000:
            logger.warning(f"V3 TTS submit failed: {resp_json}")
            return None
        
        task_id = resp_json.get("data", {}).get("task_id")
        if not task_id:
            logger.warning(f"V3 TTS submit returned no task_id: {resp_json}")
            return None
        
        logger.debug(f"V3 TTS task submitted: {task_id}")
        
    except Exception as e:
        logger.error(f"V3 TTS submit exception: {e}")
        return None
    
    # Step 2: Poll for result
    query_payload = {"task_id": task_id}
    max_polls = 60  # Max 60 seconds
    poll_interval = 1.0
    
    for _ in range(max_polls):
        try:
            time.sleep(poll_interval)
            resp = requests.post(_BYTEDANCE_V3_QUERY_URL, json=query_payload, headers=headers, timeout=30)
            resp_json = resp.json()
            
            if resp_json.get("code") != 20000000:
                logger.warning(f"V3 TTS query error: {resp_json}")
                return None
            
            task_status = resp_json.get("data", {}).get("task_status")
            
            if task_status == 1:  # Running
                continue
            elif task_status == 2:  # Success
                audio_url = resp_json.get("data", {}).get("audio_url")
                if audio_url:
                    # Download the audio
                    audio_resp = requests.get(audio_url, timeout=60)
                    if audio_resp.status_code == 200:
                        logger.info(f"V3 TTS success: {len(audio_resp.content)} bytes")
                        return audio_resp.content
                    else:
                        logger.warning(f"V3 TTS audio download failed: {audio_resp.status_code}")
                        return None
                else:
                    logger.warning("V3 TTS success but no audio_url")
                    return None
            elif task_status == 3:  # Failure
                logger.warning(f"V3 TTS task failed: {resp_json}")
                return None
            else:
                logger.warning(f"V3 TTS unknown status: {task_status}")
                continue
                
        except Exception as e:
            logger.error(f"V3 TTS query exception: {e}")
            return None
    
    logger.warning("V3 TTS polling timeout")
    return None


def bytedance_tts_api(
    text: str, 
    voice_type: str = 'BV001_streaming',
    use_cloned_voice: bool = False,
) -> bytes | None:
    appid = _DEFAULT_SETTINGS.bytedance_appid
    access_token = _DEFAULT_SETTINGS.bytedance_access_token
    
    if not appid or not access_token:
        logger.warning("ByteDance APPID or ACCESS_TOKEN not set.")
        return None

    # Use volcano_icl cluster for cloned voices (ICL 1.0/2.0)
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
    
    # Try to load model first. If it fails, there's no point in generating audio that we can't embed.
    try:
        load_embedding_model()
    except Exception:
        logger.warning("Skipping initialization of Bytedance reference voices due to embedding model failure.")
        return

    sample_text = 'YouDub 是一个创新的开源工具，专注于将 YouTube 等平台的优质视频翻译和配音为中文版本。'
    
    for voice_type in voice_types:
        wav_path = os.path.join(voice_type_dir, f'{voice_type}.wav')
        npy_path = wav_path.replace('.wav', '.npy')
        
        if os.path.exists(wav_path) and os.path.exists(npy_path) and is_valid_wav(wav_path):
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
         logger.warning("No voice_type directory found, cannot perform speaker matching.")
         return {}

    for file in os.listdir(voice_type_dir):
        if file.endswith('.npy'):
            voice_type = file.replace('.npy', '')
            voice_types[voice_type] = np.load(os.path.join(voice_type_dir, file))
            
    if not os.path.exists(speaker_folder):
         return {}
         
    # Check model before loop to avoid spamming errors if model is broken
    try:
        load_embedding_model()
    except Exception:
        logger.warning("Embedding model unavailable, falling back to default voice for all speakers.")
        # Fallback all to default
        for file in os.listdir(speaker_folder):
            if file.endswith('.wav'):
                speaker = file.replace('.wav', '')
                speaker_to_voice_type[speaker] = 'BV001_streaming'
        
        # Save fallback mapping
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


def _upload_audio_for_cloning(audio_path: str, appid: str, token: str, speaker_id: str) -> bool:
    """Uploads audio to Volcano Engine to register a voice for cloning."""
    url = f"https://{_BYTEDANCE_HOST}/api/v1/mega_tts/audio/upload"
    # Header requires Resource-Id for ICL 2.0 (seed-icl-2.0)
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
            logger.warning(f"Audio {audio_path} too short for cloning (< 1 second).")
            return False
        
        # Save truncated audio to a temporary buffer
        import io
        from scipy.io import wavfile
        
        # Normalize to int16
        wav_int16 = (wav_data * 32767).astype(np.int16)
        
        buffer = io.BytesIO()
        wavfile.write(buffer, SAMPLE_RATE, wav_int16)
        audio_data = buffer.getvalue()
        
        logger.info(f"Uploading {len(audio_data) / 1024:.1f}KB audio for voice cloning ({len(wav_data) / SAMPLE_RATE:.1f}s)")

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
        
        logger.info(f"--- Voice Cloning Upload Request ---")
        logger.info(f"URL: {url}")
        logger.info(f"Headers: {header}")
        # Log payload but truncate audio data
        debug_payload = payload.copy()
        if debug_payload.get('audios'):
             debug_payload['audios'] = [{'audio_bytes': '<hidden>', 'audio_format': a['audio_format']} for a in debug_payload['audios']]
        logger.info(f"Payload: {json.dumps(debug_payload, ensure_ascii=False)}")
        logger.info(f"------------------------------------")
        
        resp = requests.post(url, json=payload, headers=header, timeout=120)
        
        try:
            resp_json = resp.json()
        except json.JSONDecodeError:
            logger.error(f"Voice cloning upload response is not JSON: status={resp.status_code} text={resp.text[:200]}")
            return False
            
        # Check response status (API can return PascalCase or snake_case)
        base_resp = resp_json.get("BaseResp") or resp_json.get("base_resp", {})
        status_code = base_resp.get("StatusCode") if "StatusCode" in base_resp else base_resp.get("status_code")
        
        if status_code == 0:
            logger.info(f"Successfully uploaded audio for cloning: {speaker_id}")
            return True
        else:
            logger.warning(f"Failed to upload audio for cloning: {resp_json}")
            return False
            
    except Exception as e:
        logger.error(f"Exception during voice cloning upload: {e}")
        return False


def get_or_create_cloned_voice(folder: str, speaker: str, speaker_wav: str) -> str | None:
    """
    Gets existing cloned voice ID or creates a new one using pre-allocated IDs.
    
    ICL 2.0 requires using speaker IDs that are already allocated in the Volcano console.
    Set VOLCANO_CLONE_SPEAKER_IDS in .env as a comma-separated list of available IDs.
    Example: VOLCANO_CLONE_SPEAKER_IDS=S_PoN0a1CN1,S_OoN0a1CN1,S_NoN0a1CN1
    """
    
    # 1. Check local cache
    mapping_path = os.path.join(folder, 'speaker_to_cloned_voice.json')
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
            
    if speaker in mapping:
        cached_value = mapping[speaker]
        # Return None if previously marked as failed
        if cached_value == "FAILED":
            return None
        return cached_value

    # 2. Get pre-allocated speaker IDs from environment
    volcano_speaker_ids_str = os.getenv('VOLCANO_CLONE_SPEAKER_IDS', '')
    if not volcano_speaker_ids_str:
        logger.warning("VOLCANO_CLONE_SPEAKER_IDS not set in .env. Voice cloning will be skipped.")
        logger.warning("Please set VOLCANO_CLONE_SPEAKER_IDS to a comma-separated list of IDs from your Volcano console.")
        return None
    
    available_ids = [s.strip() for s in volcano_speaker_ids_str.split(',') if s.strip()]
    if not available_ids:
        logger.warning("VOLCANO_CLONE_SPEAKER_IDS is empty. Voice cloning will be skipped.")
        return None
    
    # 3. Find an unused ID
    used_ids = set(v for v in mapping.values() if v != "FAILED")
    unused_ids = [id for id in available_ids if id not in used_ids]
    
    if not unused_ids:
        logger.warning(f"All {len(available_ids)} pre-allocated speaker IDs are used. No ID available for {speaker}.")
        return None
    
    speaker_id = unused_ids[0]  # Use first available
    logger.info(f"Assigning pre-allocated speaker ID {speaker_id} to {speaker}")
    
    # 4. Prepare credentials
    appid = _DEFAULT_SETTINGS.bytedance_appid
    access_token = _DEFAULT_SETTINGS.bytedance_access_token
    
    if not appid or not access_token:
        return None
        
    # 5. Upload audio to this pre-allocated ID
    if _upload_audio_for_cloning(speaker_wav, appid, access_token, speaker_id):
        mapping[speaker] = speaker_id
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        return speaker_id
    
    # Mark as failed to avoid retrying
    mapping[speaker] = "FAILED"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    logger.warning(f"Voice cloning failed for {speaker}, will use default voice for all segments.")
    
    return None


def bytedance_tts(
    text: str, 
    output_path: str, 
    speaker_wav: str, 
    voice_type: str | None = None
) -> None:
    if os.path.exists(output_path) and is_valid_wav(output_path):
        logger.info(f'ByteDance TTS {text[:20]}... exists (verified)')
        return
    elif os.path.exists(output_path):
        logger.warning(f"Removing invalid cached file: {output_path}")
        os.remove(output_path)

    folder = os.path.dirname(os.path.dirname(output_path))
    speaker = os.path.basename(speaker_wav).replace('.wav', '')
    
    # Strategy:
    # 1. If voice_type provided explicitly (e.g. from UI override), use it.
    # 2. If not, try to use CLONED voice (Voice Cloning).
    # 3. If cloning fails/not available, fall back to MATCHED preset voice (Voice Matching).
    
    if voice_type is None:
        # Try cloning first
        cloned_voice_id = get_or_create_cloned_voice(folder, speaker, speaker_wav)
        if cloned_voice_id:
            voice_type = cloned_voice_id
            logger.info(f"Using CLONED voice for {speaker}: {voice_type}")
        else:
            # Fallback to matching
            speaker_to_voice_type = generate_speaker_to_voice_type(folder)
            voice_type = speaker_to_voice_type.get(speaker, 'BV001_streaming')
            logger.info(f"Using MATCHED voice for {speaker}: {voice_type}")

    # Detect if we're using a cloned voice (cloned voice IDs start with 'S_')
    is_cloned = voice_type.startswith('S_') if voice_type else False
    
    for _ in range(3):
        try:
            # Use V3 API for cloned voices (ICL 2.0), V1 API for preset voices
            # Use V1 API for both, as V3 gives 55000000 error for ICL currently 
            audio_data = bytedance_tts_api(text, voice_type=voice_type, use_cloned_voice=is_cloned)
            if audio_data:
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                
                # Validation
                if is_valid_wav(output_path):
                    logger.info(f'ByteDance TTS saved: {output_path}')
                    time.sleep(0.1)
                    break
                else:
                    logger.warning("Saved wav file seems corrupted or invalid.")
                    if os.path.exists(output_path):
                        os.remove(output_path)
            
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"ByteDance TTS loop failed: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            time.sleep(0.5)


# --- Qwen3-TTS (Worker) Logic ---

_QWEN_WORKER_READY = "__READY__"
_QWEN_WORKER_STUB_ENV = "YOUDUB_QWEN_WORKER_STUB"


def _get_qwen_worker_script_path() -> Path:
    # Repository layout:
    #   repo/scripts/qwen_tts_worker.py
    #   repo/src/youdub/core/steps/synthesize_speech.py  (this file)
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "scripts" / "qwen_tts_worker.py"


class _QwenTtsWorker:
    def __init__(self, python_exe: str, model_path: str, stub: bool = False):
        script = _get_qwen_worker_script_path()
        if not script.exists():
            raise ModelCheckError(f"找不到 Qwen3-TTS worker 脚本：{script}")

        cmd = [python_exe, "-u", str(script), "--model-path", model_path]
        if stub:
            cmd.append("--stub")

        try:
            self._proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            raise ModelCheckError(f"无法启动 Qwen3-TTS worker：{exc}") from exc

        assert self._proc.stdout is not None
        ready = self._proc.stdout.readline().strip()
        if ready != _QWEN_WORKER_READY:
            self.close()
            raise ModelCheckError(f"Qwen3-TTS worker 启动失败：{ready}")

    @classmethod
    def from_settings(cls, settings: Settings) -> "_QwenTtsWorker":
        py = settings.resolve_path(settings.qwen_tts_python_path)
        if not py or not py.exists():
            raise ModelCheckError(
                f"Qwen3-TTS Python 不存在：{py}。请创建独立环境安装 qwen-tts，并设置 QWEN_TTS_PYTHON。"
            )

        model_dir = settings.resolve_path(settings.qwen_tts_model_path)
        if not model_dir or not model_dir.exists():
            raise ModelCheckError(
                f"Qwen3-TTS 模型目录不存在：{model_dir}。请先下载模型权重到本地并设置 QWEN_TTS_MODEL_PATH。"
            )

        stub = os.getenv(_QWEN_WORKER_STUB_ENV, "").strip() in {"1", "true", "TRUE", "yes", "YES"}
        return cls(python_exe=str(py), model_path=str(model_dir), stub=stub)

    def synthesize(self, text: str, speaker_wav: str, output_path: str, language: str = "Auto") -> None:
        if self._proc.poll() is not None:
            raise RuntimeError("Qwen3-TTS worker 已退出")

        req = {
            "cmd": "synthesize",
            "text": text,
            "language": language,
            "speaker_wav": speaker_wav,
            "output_path": output_path,
        }

        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Qwen3-TTS worker 无输出")

        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(str(resp.get("error", "unknown error")))

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
        try:
            proc.terminate()
        except Exception:
            pass


# --- Gemini TTS Logic ---

def gemini_tts(
    text: str,
    output_path: str,
    voice_name: str | None = None,
) -> None:
    """Generate speech using Gemini TTS and save to file using Best Practices."""
    
    if os.path.exists(output_path) and is_valid_wav(output_path):
        logger.info(f'Gemini TTS {text[:20]}... exists (verified)')
        return
    elif os.path.exists(output_path):
        logger.warning(f"Removing invalid cached file: {output_path}")
        os.remove(output_path)
    
    # Lazy import to avoid hard dependency
    try:
        from google.genai import types
        import wave
    except ImportError:
        logger.error("google-genai not installed. Please install it.")
        return

    client = _get_gemini_client()
    if not client:
        return
    
    model_name = _DEFAULT_SETTINGS.gemini_tts_model or "gemini-2.5-flash-preview-tts"
    voice_name = voice_name or _DEFAULT_SETTINGS.gemini_tts_voice or 'Kore'

    # Audio params
    RATE = 24000
    SAMPLE_WIDTH = 2 # 16-bit
    CHANNELS = 1

    max_retries = 10  # Increase retries for rate limits
    retry_count = 0
    
    while retry_count < max_retries:
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
            
            # Extract PCM data
            if (response.candidates and 
                response.candidates[0].content and 
                response.candidates[0].content.parts and 
                response.candidates[0].content.parts[0].inline_data):
                
                pcm_data = response.candidates[0].content.parts[0].inline_data.data
                
                # Write WAV using standard wave module for robustness
                with wave.open(output_path, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(SAMPLE_WIDTH)
                    wf.setframerate(RATE)
                    wf.writeframes(pcm_data)
                
                # Verify
                if is_valid_wav(output_path):
                    logger.info(f'Gemini TTS saved: {output_path}')
                    time.sleep(0.1)
                    return # Success
                else:
                    logger.warning("Gemini TTS wrote invalid file, retrying...")
                    if os.path.exists(output_path):
                        os.remove(output_path)
            else:
                logger.warning(f"Gemini TTS response structure unexpected. Candidates: {response.candidates}, PromptFeedback: {getattr(response, 'prompt_feedback', 'N/A')}")
                try:
                    logger.warning(f"Full response dump: {response}")
                except Exception:
                    pass
                # Non-retriable unless it's a transient server error, but structure errors are usually permanent for the same input.
                # Just incase, we increment retry but don't sleep huge amounts.
                retry_count += 1
                time.sleep(1)
                continue

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                import re
                # Try to find retry delay in seconds from error message
                match = re.search(r'retry in (\d+\.?\d*)s', error_str)
                delay = float(match.group(1)) if match else 20.0
                # Add a little buffer
                delay += 1.0
                
                logger.warning(f"Gemini TTS Rate Limit (429). Retrying in {delay}s... (Attempt {retry_count+1}/{max_retries})")
                time.sleep(delay)
                retry_count += 1
                continue
            else:
                logger.error(f"Gemini TTS generation failed: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                # Non-429 errors might not be worth retrying 10 times, but we'll stick to the loop for robustness
                retry_count += 1
                time.sleep(1)
        


# --- Init for Pipeline ---

def init_TTS(settings: Settings | None = None, model_manager: ModelManager | None = None) -> None:
    """Pre-initialize models if needed (mainly for warming up)."""
    # Qwen3-TTS runs in an external worker process (separate venv) to avoid dependency conflicts.
    # Warm-up is intentionally skipped here to avoid heavy model load at pipeline startup.
    return


# --- Main Generation Logic ---

def generate_wavs(folder: str, tts_method: str = 'bytedance') -> None:
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

    qwen_worker: _QwenTtsWorker | None = None
    if tts_method == "qwen":
        qwen_worker = _QwenTtsWorker.from_settings(_DEFAULT_SETTINGS)

    try:
        for i, line in enumerate(transcript):
            speaker = line['speaker']
            text = preprocess_text(line['translation'])
            output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
            speaker_wav = os.path.join(folder, 'SPEAKER', f'{speaker}.wav')
            if tts_method == 'qwen' and qwen_worker is not None:
                try:
                    qwen_worker.synthesize(text, speaker_wav, output_path, language="Auto")
                except Exception as exc:
                    logger.warning(f"Qwen3-TTS worker failed for segment {i}: {exc}")
            elif tts_method == 'gemini':
                gemini_tts(text, output_path)
            else:
                bytedance_tts(text, output_path, speaker_wav)
            
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
            if os.path.exists(output_path) and is_valid_wav(output_path):
                wav_chunk, actual_len = adjust_audio_length(output_path, desired_len)
            else:
                logger.warning(f"TTS generation failed via {tts_method} for segment {i}, using silence fallback.")
                # Fallback to silence
                target_samples = int(desired_len * 24000)
                wav_chunk = np.zeros(target_samples, dtype=np.float32)
                actual_len = desired_len
            
            full_wav = np.concatenate((full_wav, wav_chunk))
            line['end'] = new_start + actual_len
    finally:
        if qwen_worker is not None:
            qwen_worker.close()

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


def generate_all_wavs_under_folder(root_folder: str, tts_method: str = 'bytedance') -> str:
    count = 0
    for root, dirs, files in os.walk(root_folder):
        if 'translation.json' in files and 'audio_combined.wav' not in files:
            generate_wavs(root, tts_method)
            count += 1
    msg = f'Generated all wavs under {root_folder} (processed {count} files)'
    logger.info(msg)
    return msg
