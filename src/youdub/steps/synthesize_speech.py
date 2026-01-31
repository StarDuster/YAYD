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
from ..utils import (
    ensure_torchaudio_backend_compat,
    read_speaker_ref_seconds,
    save_wav,
    save_wav_norm,
    wav_duration_seconds,
)

_EMBEDDING_MODEL = None
_EMBEDDING_INFERENCE = None
_EMBEDDING_MODEL_LOAD_FAILED = False

_DEFAULT_SETTINGS = Settings()
_DEFAULT_MODEL_MANAGER = ModelManager(_DEFAULT_SETTINGS)

_BYTEDANCE_HOST = "openspeech.bytedance.com"
_BYTEDANCE_API_URL = f"https://{_BYTEDANCE_HOST}/api/v1/tts"

_GEMINI_CLIENT = None


# Qwen3-TTS (Tokenizer-12Hz) output guard:
# If the model keeps generating until `max_new_tokens`, the audio duration will be ~max_new_tokens/12 seconds.
# This often indicates a degenerate loop / failure-to-stop. We treat it as a failure and retry.
_QWEN_TTS_TOKEN_HZ = 12.0
_QWEN_TTS_MAX_NEW_TOKENS_DEFAULT = 2048


def _qwen_tts_max_new_tokens() -> int:
    v = _read_env_int("YOUDUB_QWEN_TTS_MAX_NEW_TOKENS", _QWEN_TTS_MAX_NEW_TOKENS_DEFAULT)
    if v <= 0:
        return int(_QWEN_TTS_MAX_NEW_TOKENS_DEFAULT)
    return int(v)


def _qwen_tts_hit_max_tokens_seconds() -> float:
    return float(_qwen_tts_max_new_tokens()) / float(_QWEN_TTS_TOKEN_HZ)


def _qwen_tts_is_degenerate_hit_cap(*, wav_dur: float | None) -> bool:
    """
    Detect "keeps generating until max_new_tokens" by duration.

    We intentionally use a small tolerance because wav header rounding and model decoding
    may not land on the exact boundary.
    """
    if wav_dur is None:
        return False
    tol = _read_env_float("YOUDUB_QWEN_TTS_HIT_CAP_TOL_SEC", 2.0)
    if tol < 0:
        tol = 2.0
    cap = _qwen_tts_hit_max_tokens_seconds()
    return bool(float(wav_dur) >= float(cap) - float(tol))


def _qwen_resp_duration_seconds(resp: dict | None) -> float | None:
    if not isinstance(resp, dict):
        return None
    try:
        sr = resp.get("sr")
        n_samples = resp.get("n_samples")
        if sr is None or n_samples is None:
            return None
        sr_f = float(sr)
        ns_f = float(n_samples)
        if not (sr_f > 0) or not (ns_f >= 0):
            return None
        return ns_f / sr_f
    except Exception:
        return None


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


def _read_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    raw = raw.strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    raw = raw.strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _tts_duration_guard_params() -> tuple[float, float, float, int]:
    """
    Guardrail for "TTS segment should not be significantly longer than source segment".

    Defaults are intentionally permissive (to tolerate language expansion) but will catch
    pathological outputs like ~10-minute clips for a ~10s segment.
    """
    ratio = _read_env_float("YOUDUB_TTS_MAX_SEGMENT_DURATION_RATIO", 3.0)
    extra = _read_env_float("YOUDUB_TTS_MAX_SEGMENT_DURATION_EXTRA_SEC", 8.0)
    abs_cap = _read_env_float("YOUDUB_TTS_MAX_SEGMENT_DURATION_ABS_SEC", 180.0)
    retries = _read_env_int("YOUDUB_TTS_SEGMENT_MAX_RETRIES", 3)

    if not (ratio >= 1.0):
        ratio = 3.0
    if not (extra >= 0.0):
        extra = 8.0
    # abs_cap <= 0 means "disable absolute cap"
    if not (retries >= 1):
        retries = 3
    retries = int(max(1, min(retries, 10)))
    return float(ratio), float(extra), float(abs_cap), int(retries)


def _tts_segment_allowed_max_seconds(seg_dur: float, ratio: float, extra: float, abs_cap: float) -> float | None:
    """
    Compute allowed max duration for a TTS segment.

    - seg_dur <= 0: only apply abs_cap (if configured), otherwise no guard.
    - otherwise: max(seg_dur * ratio, seg_dur + extra).
    """
    sd = float(seg_dur or 0.0)
    if sd <= 0.0:
        return float(abs_cap) if (abs_cap and abs_cap > 0) else None
    allowed = max(sd * float(ratio), sd + float(extra))
    return float(allowed)


def _tts_wav_ok_for_segment(path: str, seg_dur: float, ratio: float, extra: float, abs_cap: float) -> bool:
    if not (os.path.exists(path) and is_valid_wav(path)):
        return False
    dur = wav_duration_seconds(path)
    if dur is None:
        return False
    allowed = _tts_segment_allowed_max_seconds(seg_dur, ratio, extra, abs_cap)
    if allowed is None:
        return True
    # tolerate tiny header rounding errors
    return bool(dur <= allowed + 0.02)


_TTS_PY_DOT_CHAIN_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[A-Za-z_][A-Za-z0-9_]*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
    r"(?![A-Za-z0-9_])"
)
_TTS_PY_UNDERSCORE_IDENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[A-Za-z_][A-Za-z0-9_]*"
    r"_[A-Za-z0-9_]+"
    r"(?![A-Za-z0-9_])"
)


def _tts_prompt_normalize_python_tokens(text: str) -> str:
    """
    Make Python-ish code tokens more TTS friendly.

    Examples:
    - matplotlib.pyplot -> matplotlib dot pyplot
    - os.path.join -> os dot path dot join
    - base_dir -> base dir

    Notes:
    - We only touch identifier-like tokens to avoid breaking decimals like 3.14.
    - This function is used ONLY for TTS prompt text (subtitles are unchanged).
    """
    t = str(text or "")

    def _dot_chain(m: re.Match) -> str:
        s = m.group(0)
        # Normalize snake_case inside each segment too.
        s = s.replace("_", " ")
        s = s.replace(".", " dot ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _underscore_ident(m: re.Match) -> str:
        s = m.group(0)
        s = s.replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Convert dotted chains first (also handles underscores inside them).
    t = _TTS_PY_DOT_CHAIN_RE.sub(_dot_chain, t)
    # Convert remaining snake_case identifiers.
    t = _TTS_PY_UNDERSCORE_IDENT_RE.sub(_underscore_ident, t)
    return t


def _tts_text_for_attempt(raw_text: str, attempt: int) -> str:
    """
    Text variants for retries:
    - attempt 0: keep original (normalized, but make Python-ish tokens TTS-friendly)
    - attempt 1+: strip markdown-ish code markers and make common code punctuation more TTS-friendly
    - attempt 2+: more aggressive cleanup of uncommon symbols
    """
    t = str(raw_text or "")
    if attempt > 0:
        # Remove fenced blocks and inline backticks.
        t = re.sub(r"```[\s\S]*?```", " ", t)
        t = t.replace("`", "")

    # Always make Python-ish tokens more readable for TTS.
    t = _tts_prompt_normalize_python_tokens(t)

    if attempt >= 2:
        # Keep only: CJK, ASCII alnum, whitespace, and basic punctuation.
        t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s，。！？,.!?:：；;（）()\-\+*/=]", " ", t)

    t = re.sub(r"\s+", " ", t).strip()
    return preprocess_text(t)


def _ensure_wav_max_duration(path: str, max_seconds: float, sample_rate: int = 24000) -> None:
    if not path or not os.path.exists(path):
        return
    if max_seconds <= 0:
        return

    dur = wav_duration_seconds(path)
    if dur is not None and dur <= max_seconds + 0.02:
        return

    try:
        wav, _sr = librosa.load(path, sr=sample_rate, duration=max_seconds)
        if wav.size <= 0:
            return
        save_wav_norm(wav.astype(np.float32), path, sample_rate=sample_rate)
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
                # NOTE:
                # BufferedReader.peek(n>0) may block while trying to fill from the raw pipe.
                # We only want to check *already buffered* bytes, so use peek(0) here.
                if buf.peek(0):
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
        timeout_sec: float = 180.0,
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

        assert self._proc.stdin is not None
        assert self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

        skipped: list[str] = []
        max_skip = 50
        start_time = time.monotonic()
        last_heartbeat = start_time
        while True:
            check_cancelled()
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_sec:
                stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                # Worker may be stuck in GPU inference; try hard to terminate it.
                try:
                    self.close()
                except Exception:
                    pass
                raise RuntimeError(
                    f"Qwen3-TTS worker 超时 ({timeout_sec:.0f}秒)，"
                    f"text={text[:30]}...，已等待 {elapsed:.1f}秒。{extra}"
                )
            now = time.monotonic()
            if now - last_heartbeat >= 30.0:
                last_heartbeat = now
                logger.info(f"Qwen3-TTS 等待中... ({elapsed:.1f}s / {timeout_sec:.0f}s)")
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

    def synthesize_batch(self, items: list[dict], timeout_sec: float = 300.0) -> dict:
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
        start_time = time.monotonic()
        last_heartbeat = start_time
        while True:
            check_cancelled()
            elapsed = time.monotonic() - start_time
            if elapsed > timeout_sec:
                stderr_tail = "\n".join(list(self._stderr_tail)).strip()
                extra = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
                # Worker may be stuck in GPU inference; try hard to terminate it.
                try:
                    self.close()
                except Exception:
                    pass
                raise RuntimeError(
                    f"Qwen3-TTS worker 超时 ({timeout_sec:.0f}秒)，"
                    f"batch_size={len(items)}，已等待 {elapsed:.1f}秒。"
                    f"可能是 GPU 推理卡住或显存不足。{extra}"
                )
            # Heartbeat log every 30s to show we're still waiting
            now = time.monotonic()
            if now - last_heartbeat >= 30.0:
                last_heartbeat = now
                logger.info(
                    f"Qwen3-TTS 批处理等待中... ({elapsed:.1f}s / {timeout_sec:.0f}s) batch_size={len(items)}"
                )
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


def _segment_wav_path(folder: str, idx: int) -> str:
    # Keep consistent naming with generate_wavs(): str(i).zfill(4)
    return os.path.join(folder, "wavs", f"{int(idx):04d}.wav")


def _tts_segment_wavs_complete(folder: str, expected_segments: int) -> bool:
    """
    Whether we have a complete, valid set of per-segment wav files for stitching.

    This guards against stale/partial artifacts when `.tts_done.json` exists but
    `wavs/` was truncated (manual cleanup / interrupted runs / disk issues).
    """
    n = int(expected_segments)
    if n <= 0:
        return False
    # Quick checks: first/last segment must exist and be valid.
    first = _segment_wav_path(folder, 0)
    last = _segment_wav_path(folder, n - 1)
    if not (os.path.exists(first) and is_valid_wav(first)):
        return False
    if not (os.path.exists(last) and is_valid_wav(last)):
        return False

    # Full check (cheap header parse): ensure no holes/corrupted files.
    tr_path = os.path.join(folder, "translation.json")
    transcript: list[dict] | None = None
    try:
        with open(tr_path, "r", encoding="utf-8") as f:
            tr = json.load(f)
        if isinstance(tr, list):
            transcript = tr  # type: ignore[assignment]
    except Exception:
        transcript = None

    dur_ratio, dur_extra, dur_abs_cap, _dur_retries = _tts_duration_guard_params()
    for i in range(n):
        p = _segment_wav_path(folder, i)
        if not (os.path.exists(p) and is_valid_wav(p)):
            return False
        seg_dur = 0.0
        if transcript is not None and i < len(transcript):
            it = transcript[i]
            try:
                seg_dur = float(max(0.0, float(it.get("end", 0.0)) - float(it.get("start", 0.0))))
            except Exception:
                seg_dur = 0.0
        if not _tts_wav_ok_for_segment(p, seg_dur, dur_ratio, dur_extra, dur_abs_cap):
            return False
    return True


def generate_wavs(
    folder: str,
    tts_method: str = "bytedance",
    qwen_tts_batch_size: int = 1,
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

    dur_ratio, dur_extra, dur_abs_cap, max_seg_retries = _tts_duration_guard_params()
        
    speakers = set(line["speaker"] for line in transcript)
    num_speakers = len(speakers)
    total_segments = len(transcript)
    logger.info(
        f"TTS({tts_method}) 开始: segments={total_segments}, speakers={num_speakers}, folder={folder}"
    )

    # 防止历史数据/旧逻辑生成超长 SPEAKER/*.wav 导致 voice cloning 显存/时间爆炸
    max_ref_seconds = read_speaker_ref_seconds()
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
                    save_wav_norm(wav.astype(np.float32), spk_path, sample_rate=24000)
                    logger.info(f"已生成缺失的说话人参考 ({max_ref_seconds:.1f}秒): {spk_path}")
            except Exception as exc:
                logger.warning(f"生成说话人参考音频失败 {spk}: {exc}")

    for spk in speakers:
        check_cancelled()
        _ensure_wav_max_duration(os.path.join(speaker_dir, f"{spk}.wav"), max_ref_seconds, sample_rate=24000)
    
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

        qwen_timeout_sec = 180.0
        qwen_batch_timeout_sec = 300.0
        if tts_method == "qwen":
            try:
                qwen_timeout_sec = float(os.getenv("YOUDUB_QWEN_WORKER_REQUEST_TIMEOUT_SEC", "120") or "120")
            except Exception:
                qwen_timeout_sec = 120.0
            try:
                qwen_batch_timeout_sec = float(os.getenv("YOUDUB_QWEN_WORKER_BATCH_TIMEOUT_SEC", "180") or "180")
            except Exception:
                qwen_batch_timeout_sec = 180.0
            if not (qwen_timeout_sec > 0):
                qwen_timeout_sec = 120.0
            if not (qwen_batch_timeout_sec > 0):
                qwen_batch_timeout_sec = 180.0

        def _should_restart_qwen_worker(err: str) -> bool:
            s = (err or "").lower()
            return (
                ("已退出" in err)
                or ("无输出" in err)
                or ("超时" in err)
                or ("broken pipe" in s)
                or ("eof" in s)
            )

        def _restart_qwen_worker(reason: str) -> None:
            nonlocal qwen_worker
            if qwen_worker is not None:
                try:
                    qwen_worker.close()
                except Exception:
                    pass
            qwen_worker = _QwenTtsWorker.from_settings(_DEFAULT_SETTINGS)
            logger.warning(f"Qwen3-TTS worker 已重启: {reason}")

        qwen_resp_by_index: dict[int, dict] = {}
        qwen_cached_before: set[int] = set()
        if tts_method == "qwen" and qwen_worker is not None and qwen_batch_size > 1:
            logger.info(f"Qwen3-TTS批处理已启用: batch_size={qwen_batch_size}")
            for i in range(total_segments):
                check_cancelled()
                out = os.path.join(output_folder, f"{str(i).zfill(4)}.wav")
                if os.path.exists(out) and is_valid_wav(out):
                    try:
                        seg = transcript[i]
                        seg_dur = float(max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))))
                    except Exception:
                        seg_dur = 0.0
                    if _tts_wav_ok_for_segment(out, seg_dur, dur_ratio, dur_extra, dur_abs_cap):
                        # Also guard against "hit max_new_tokens cap" outputs from old runs.
                        if not _qwen_tts_is_degenerate_hit_cap(wav_dur=wav_duration_seconds(out)):
                            qwen_cached_before.add(i)
                        else:
                            try:
                                os.remove(out)
                                logger.warning(f"删除疑似退化(命中max_new_tokens)的TTS缓存: {out}")
                            except Exception:
                                pass
                    else:
                        # Too long vs expected segment duration -> treat as bad cache and regenerate.
                        try:
                            os.remove(out)
                            logger.warning(f"删除异常超长TTS缓存: {out}")
                        except Exception:
                            pass

            def _looks_like_oom(err: str) -> bool:
                s = (err or "").lower()
                return ("out of memory" in s) or ("cuda" in s and "memory" in s) or ("oom" in s)

            def _run_qwen_batch(batch_indices: list[int], attempt: int = 0) -> None:
                if not batch_indices:
                    return
                check_cancelled()

                batch_items: list[dict] = []
                for j in batch_indices:
                    check_cancelled()
                    seg = transcript[j]
                    speaker = seg["speaker"]
                    text = _tts_text_for_attempt(str(seg.get("translation", "")), 0)
                    out = os.path.join(output_folder, f"{str(j).zfill(4)}.wav")
                    spk_wav = os.path.join(folder, "SPEAKER", f"{speaker}.wav")
                    speaker_wav_for_req = spk_wav
                    # If invalid/corrupted/too-long file exists, remove so worker can rewrite cleanly.
                    if os.path.exists(out):
                        try:
                            try:
                                seg_dur = float(max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0))))
                            except Exception:
                                seg_dur = 0.0
                            if (
                                (not _tts_wav_ok_for_segment(out, seg_dur, dur_ratio, dur_extra, dur_abs_cap))
                                or _qwen_tts_is_degenerate_hit_cap(wav_dur=wav_duration_seconds(out))
                            ):
                                os.remove(out)
                        except Exception:
                            pass
                    item: dict = {
                        "text": text,
                        "language": "Auto",
                        "speaker_wav": speaker_wav_for_req,
                        "output_path": out,
                    }
                    batch_items.append(item)

                try:
                    check_cancelled()
                    if qwen_worker is None:
                        _restart_qwen_worker("worker is None")
                    assert qwen_worker is not None
                    resp = qwen_worker.synthesize_batch(batch_items, timeout_sec=qwen_batch_timeout_sec)
                except CancelledByUser:
                    raise
                except Exception as exc:
                    err = str(exc)
                    if attempt == 0 and _should_restart_qwen_worker(err):
                        logger.warning(f"Qwen3-TTS worker 异常，将重启并重试一次: {err}")
                        try:
                            _restart_qwen_worker(err)
                            _run_qwen_batch(batch_indices, attempt=attempt + 1)
                        except Exception as exc2:
                            logger.warning(f"Qwen3-TTS worker 重启/重试失败，段 {batch_indices}: {exc2}")
                            for j in batch_indices:
                                qwen_resp_by_index[j] = {"ok": False, "error": str(exc2)}
                        return

                    if len(batch_indices) > 1 and _looks_like_oom(err):
                        mid = len(batch_indices) // 2
                        _run_qwen_batch(batch_indices[:mid], attempt=attempt)
                        _run_qwen_batch(batch_indices[mid:], attempt=attempt)
                        return

                    # Non-OOM failures: fall back to per-item synthesis to avoid "batch卡死/无输出"。
                    if len(batch_indices) > 1:
                        logger.warning(f"Qwen3-TTS批处理失败，降级为逐条合成: 段 {batch_indices} (原因: {err})")
                        for j, it in zip(batch_indices, batch_items):
                            check_cancelled()
                            try:
                                if qwen_worker is None:
                                    _restart_qwen_worker("worker missing before single fallback")
                                assert qwen_worker is not None
                                qwen_resp_by_index[j] = qwen_worker.synthesize(
                                    text=str(it.get("text", "")),
                                    speaker_wav=str(it.get("speaker_wav", "")),
                                    output_path=str(it.get("output_path", "")),
                                    language=str(it.get("language", "Auto") or "Auto"),
                                    timeout_sec=qwen_timeout_sec,
                                )
                            except CancelledByUser:
                                raise
                            except Exception as exc_one:
                                err_one = str(exc_one)
                                if _should_restart_qwen_worker(err_one):
                                    logger.warning(f"Qwen3-TTS worker 异常，重启后再试一次(单条 idx={j}): {err_one}")
                                    _restart_qwen_worker(err_one)
                                    assert qwen_worker is not None
                                    try:
                                        qwen_resp_by_index[j] = qwen_worker.synthesize(
                                            text=str(it.get("text", "")),
                                            speaker_wav=str(it.get("speaker_wav", "")),
                                            output_path=str(it.get("output_path", "")),
                                            language=str(it.get("language", "Auto") or "Auto"),
                                            timeout_sec=qwen_timeout_sec,
                                        )
                                    except Exception as exc_one2:
                                        qwen_resp_by_index[j] = {"ok": False, "error": str(exc_one2)}
                                else:
                                    qwen_resp_by_index[j] = {"ok": False, "error": err_one}
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

                # Detect degenerate "hit max_new_tokens" outputs and mark them as failures for retry.
                cap_sec = _qwen_tts_hit_max_tokens_seconds()
                for k, j in enumerate(batch_indices):
                    rj = qwen_resp_by_index.get(j) or {}
                    if not (isinstance(rj, dict) and rj.get("ok")):
                        continue
                    out = None
                    try:
                        out = str(batch_items[k].get("output_path", ""))
                    except Exception:
                        out = None
                    dur = _qwen_resp_duration_seconds(rj)
                    if _qwen_tts_is_degenerate_hit_cap(wav_dur=dur):
                        if out and os.path.exists(out):
                            try:
                                os.remove(out)
                            except Exception:
                                pass
                        qwen_resp_by_index[j] = {
                            "ok": False,
                            "error": f"qwen_tts_degenerate_hit_max_new_tokens ({_qwen_tts_max_new_tokens()}, ~{cap_sec:.1f}s)",
                        }

                # If the worker reports OOM for some items, retry those with smaller batches.
                oom_failed: list[int] = []
                for j in batch_indices:
                    rj = qwen_resp_by_index.get(j) or {}
                    if rj.get("ok") is False and _looks_like_oom(str(rj.get("error", ""))):
                        oom_failed.append(j)
                if len(oom_failed) > 1:
                    mid = len(oom_failed) // 2
                    _run_qwen_batch(oom_failed[:mid], attempt=attempt)
                    _run_qwen_batch(oom_failed[mid:], attempt=attempt)

        for i, line in enumerate(transcript):
            check_cancelled()
            speaker = line["speaker"]
            raw_translation = str(line.get("translation", ""))
            output_path = os.path.join(output_folder, f"{str(i).zfill(4)}.wav")
            speaker_wav = os.path.join(folder, "SPEAKER", f"{speaker}.wav")
            seg_no = i + 1
            # Validate cache: must be a valid wav AND not significantly longer than the source segment.
            try:
                seg_start = float(line.get("start", 0.0) or 0.0)
                seg_end = float(line.get("end", seg_start) or seg_start)
            except Exception:
                seg_start, seg_end = 0.0, 0.0
            seg_dur = float(max(0.0, seg_end - seg_start))

            wav_ok = False
            if os.path.exists(output_path):
                try:
                    if _tts_wav_ok_for_segment(output_path, seg_dur, dur_ratio, dur_extra, dur_abs_cap) and (
                        (tts_method != "qwen") or (not _qwen_tts_is_degenerate_hit_cap(wav_dur=wav_duration_seconds(output_path)))
                    ):
                        wav_ok = True
                    else:
                        # Bad cache (too long / invalid) -> delete so we can regenerate.
                        os.remove(output_path)
                        logger.warning(f"删除异常TTS片段缓存(将重试生成): {output_path}")
                except Exception:
                    # If we can't validate, remove to avoid downstream time-stretch crashes.
                    try:
                        os.remove(output_path)
                    except Exception:
                        pass

            if tts_method == "qwen" and qwen_worker is not None and qwen_batch_size > 1:
                was_cached = bool(wav_ok and (i in qwen_cached_before))
            else:
                was_cached = bool(wav_ok)
            cache_tag = " [cached]" if was_cached else ""
            logger.info(
                f"TTS({tts_method}) {seg_no}/{total_segments}: {Path(output_path).name}{cache_tag} speaker={speaker}"
            )

            tts_elapsed = 0.0
            qwen_resp: dict | None = qwen_resp_by_index.get(i)
            needs_tts = not wav_ok
            if needs_tts:
                last_err: str | None = None
                qwen_restarted_for_this_segment = False
                for attempt in range(max_seg_retries):
                    check_cancelled()
                    if attempt > 0:
                        logger.warning(f"TTS({tts_method})段 {i} 将重试(第 {attempt+1}/{max_seg_retries} 次)")

                    text = _tts_text_for_attempt(raw_translation, attempt)
                    tts_begin = time.monotonic()
                    try:
                        check_cancelled()
                        if tts_method == "qwen" and qwen_worker is not None:
                            if attempt == 0 and qwen_batch_size > 1:
                                # Generate current + upcoming missing segments in one worker call.
                                batch_indices = []
                                j = i
                                while j < total_segments and len(batch_indices) < qwen_batch_size:
                                    check_cancelled()
                                    out = os.path.join(output_folder, f"{str(j).zfill(4)}.wav")
                                    # Validate existing cache as well (including duration guard).
                                    seg_j = transcript[j]
                                    try:
                                        seg_dur_j = float(
                                            max(0.0, float(seg_j.get("end", 0.0)) - float(seg_j.get("start", 0.0)))
                                        )
                                    except Exception:
                                        seg_dur_j = 0.0
                                    if not _tts_wav_ok_for_segment(out, seg_dur_j, dur_ratio, dur_extra, dur_abs_cap):
                                        batch_indices.append(j)
                                    j += 1
                                _run_qwen_batch(batch_indices)
                                qwen_resp = qwen_resp_by_index.get(i)
                            else:
                                try:
                                    qwen_resp = qwen_worker.synthesize(
                                        text,
                                        speaker_wav,
                                        output_path,
                                        language="Auto",
                                        timeout_sec=qwen_timeout_sec,
                                    )
                                except Exception as exc_qwen:
                                    if _should_restart_qwen_worker(str(exc_qwen)):
                                        logger.warning(f"Qwen3-TTS worker 异常，将重启并重试一次: {exc_qwen}")
                                        _restart_qwen_worker(str(exc_qwen))
                                        assert qwen_worker is not None
                                        qwen_resp = qwen_worker.synthesize(
                                            text,
                                            speaker_wav,
                                            output_path,
                                            language="Auto",
                                            timeout_sec=qwen_timeout_sec,
                                        )
                                    else:
                                        raise
                        elif tts_method == "gemini":
                            gemini_tts(text, output_path)
                        else:
                            bytedance_tts(text, output_path, speaker_wav)
                        last_err = None
                    except Exception as exc:
                        last_err = str(exc)
                        logger.warning(f"TTS({tts_method})段 {i} 失败: {exc}")
                    finally:
                        tts_elapsed += max(0.0, time.monotonic() - tts_begin)

                    # Qwen退化检测：若一直生成到 max_new_tokens（约 170s），视为失败并重试。
                    if tts_method == "qwen":
                        qwen_dur = _qwen_resp_duration_seconds(qwen_resp) if qwen_resp else None
                        if qwen_dur is None and os.path.exists(output_path):
                            qwen_dur = wav_duration_seconds(output_path)
                        if _qwen_tts_is_degenerate_hit_cap(wav_dur=qwen_dur):
                            last_err = f"qwen_tts_degenerate_hit_max_new_tokens ({_qwen_tts_max_new_tokens()})"
                            logger.warning(
                                f"TTS(qwen)段 {i} 疑似退化：输出接近token上限 "
                                f"(wav_dur={qwen_dur:.2f}s ~= {_qwen_tts_hit_max_tokens_seconds():.2f}s)，将删除并重试"
                            )
                            try:
                                if os.path.exists(output_path):
                                    os.remove(output_path)
                            except Exception:
                                pass

                    # Validate the generated wav (including duration guard).
                    if _tts_wav_ok_for_segment(output_path, seg_dur, dur_ratio, dur_extra, dur_abs_cap):
                        wav_ok = True
                        break

                    wav_dur = wav_duration_seconds(output_path) if os.path.exists(output_path) else None
                    allowed = _tts_segment_allowed_max_seconds(seg_dur, dur_ratio, dur_extra, dur_abs_cap)
                    if wav_dur is not None and allowed is not None and wav_dur > allowed + 0.02:
                        logger.warning(
                            f"TTS({tts_method})段 {i} 输出异常过长: wav_dur={wav_dur:.2f}s > allowed={allowed:.2f}s "
                            f"(seg_dur={seg_dur:.2f}s)"
                        )

                    # Cleanup before next retry.
                    try:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                    except Exception:
                        pass
                    if (
                        tts_method == "qwen"
                        and qwen_worker is not None
                        and (attempt + 1) < max_seg_retries  # only if we still have retries left
                    ):
                        # 用户期望：不要“一次失败就重启 worker”。
                        # 策略：先在同一个 worker 上再试一次；若仍失败，再重启一次用于后续重试。
                        if attempt >= 1 and not qwen_restarted_for_this_segment:
                            _restart_qwen_worker(f"segment {i} invalid/too-long twice; restart worker")
                            qwen_restarted_for_this_segment = True

                if not wav_ok and last_err:
                    logger.warning(f"TTS({tts_method})段 {i} 重试耗尽仍失败: {last_err}")

            valid_wav = bool(wav_ok and os.path.exists(output_path) and is_valid_wav(output_path))
            if valid_wav:
                source = "cached" if was_cached else "tts"
            else:
                logger.warning(f"TTS生成失败 {tts_method} 段 {i}，使用静音回退。")
                failed_segments.append(i)
                source = "silence"
                if allow_silence_fallback:
                    try:
                        target_samples = int(seg_dur * 24000)
                        silence = np.zeros((max(0, target_samples),), dtype=np.float32)
                        save_wav(silence, output_path, sample_rate=24000)
                        valid_wav = is_valid_wav(output_path)
                    except Exception as exc_silence:
                        logger.warning(f"静音回退写入失败 {output_path}: {exc_silence}")

            # 段落完成日志：便于在终端观察“是否卡死/进度/耗时”
            qwen_extra = ""
            if qwen_resp:
                sr = qwen_resp.get("sr")
                n_samples = qwen_resp.get("n_samples")
                if sr is not None and n_samples is not None:
                    qwen_extra = f", sr={sr}, n_samples={n_samples}"
            logger.info(
                f"TTS({tts_method}) {seg_no}/{total_segments} 完成: source={source}, "
                f"tts={tts_elapsed:.2f}s, segment_dur={seg_dur:.2f}s, valid_wav={bool(valid_wav)}"
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

    # 清理历史遗留/多余的 wavs（例如之前 split_sentences 数量不同导致的尾部残留）
    try:
        expected = int(total_segments)
        for name in os.listdir(output_folder):
            if not name.endswith(".wav"):
                continue
            stem = name[:-4]
            # Keep segment wavs like 0000.wav; remove other temp artifacts.
            try:
                idx = int(stem)
            except ValueError:
                # e.g. *_adjusted.wav or other unknown wavs
                os.remove(os.path.join(output_folder, name))
                continue
            if idx < 0 or idx >= expected:
                os.remove(os.path.join(output_folder, name))
    except Exception:
        # Best-effort cleanup only.
        pass

    # Write a small marker so we can distinguish "valid result" from stale/partial artifacts.
    # NOTE: marker lives inside wavs/ so deleting that folder also removes the marker.
    try:
        state_path = os.path.join(folder, "wavs", ".tts_done.json")
        state = {
            "tts_method": tts_method,
            "speaker_ref_seconds": max_ref_seconds,
            "segments": int(total_segments),
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
) -> str:
    count = 0
    for root, _dirs, files in os.walk(root_folder):
        check_cancelled()
        if 'translation.json' not in files:
            continue
        done_path = os.path.join(root, "wavs", ".tts_done.json")
        if os.path.exists(done_path):
            try:
                with open(done_path, "r", encoding="utf-8") as f:
                    st = json.load(f)
                if st.get("tts_method") == tts_method:
                    # If translation.json is newer than marker, re-run to reflect changes.
                    tr_path = os.path.join(root, "translation.json")
                    tr_mtime = os.path.getmtime(tr_path)
                    done_mtime = os.path.getmtime(done_path)
                    if done_mtime >= tr_mtime:
                        expected_segments: int | None = None
                        try:
                            with open(tr_path, "r", encoding="utf-8") as f:
                                tr = json.load(f)
                            if isinstance(tr, list):
                                expected_segments = int(len(tr))
                        except Exception:
                            expected_segments = None

                        if expected_segments is not None and _tts_segment_wavs_complete(root, expected_segments):
                            continue

                        # Marker exists but wavs are incomplete/corrupted -> resume generation.
                        logger.info(
                            f"TTS产物不完整，将继续生成: {root} "
                            f"(expected_segments={expected_segments if expected_segments is not None else 'unknown'})"
                        )
            except Exception:
                # Treat as not done and re-generate.
                pass
        generate_wavs(
            root,
            tts_method,
            qwen_tts_batch_size=qwen_tts_batch_size,
        )
        count += 1
    msg = f"语音合成完成: {root_folder}（处理 {count} 个文件）"
    logger.info(msg)
    return msg
