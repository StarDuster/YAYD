from __future__ import annotations

import base64
import json
import os
import uuid

import librosa
import numpy as np
import requests
from loguru import logger
from scipy.spatial.distance import cosine

from ..config import Settings
from ..interrupts import check_cancelled, sleep_with_cancel
from ..models import ModelManager
from ..utils import prepare_speaker_ref_audio, torch_load_weights_only_compat
from .tts_wav_guard import is_valid_wav


_DEFAULT_SETTINGS = Settings()
_DEFAULT_MODEL_MANAGER = ModelManager(_DEFAULT_SETTINGS)

_BYTEDANCE_HOST = "openspeech.bytedance.com"
_BYTEDANCE_API_URL = f"https://{_BYTEDANCE_HOST}/api/v1/tts"


_EMBEDDING_MODEL = None
_EMBEDDING_INFERENCE = None
_EMBEDDING_MODEL_LOAD_FAILED = False


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

        logger.info("加载pyannote/embedding模型...")
        token = _DEFAULT_SETTINGS.hf_token
        diar_dir = _DEFAULT_SETTINGS.resolve_path(_DEFAULT_SETTINGS.whisper_diarization_model_dir)
        cache_dir = str(diar_dir) if diar_dir else None

        # PyTorch 2.6+ 默认 weights_only=True，会导致 pyannote 模型加载失败。
        with torch_load_weights_only_compat():
            # pyannote.audio v4 uses `token=...`; older versions use `use_auth_token=...`.
            try:
                _EMBEDDING_MODEL = Model.from_pretrained("pyannote/embedding", token=token, cache_dir=cache_dir)
            except TypeError:
                try:
                    _EMBEDDING_MODEL = Model.from_pretrained(
                        "pyannote/embedding", use_auth_token=token, cache_dir=cache_dir
                    )
                except TypeError:
                    # Older pyannote versions may not accept auth/cache kwargs.
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


def bytedance_tts_api(
    text: str,
    voice_type: str = "BV001_streaming",
    use_cloned_voice: bool = False,
) -> bytes | None:
    appid = _DEFAULT_SETTINGS.bytedance_appid
    access_token = _DEFAULT_SETTINGS.bytedance_access_token

    if not appid or not access_token:
        logger.warning("未设置ByteDance APPID或ACCESS_TOKEN")
        return None

    cluster = "volcano_icl" if use_cloned_voice else "volcano_tts"

    header = {"Authorization": f"Bearer;{access_token}"}
    if use_cloned_voice:
        header["X-Api-Resource-Id"] = "seed-icl-2.0"
    request_json = {
        "app": {
            "appid": appid,
            "token": access_token,
            "cluster": cluster,
        },
        "user": {
            "uid": "https://github.com/liuzhao1225/YouDub-webui",
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
            "frontend_type": "unitTson",
        },
    }

    try:
        resp = requests.post(_BYTEDANCE_API_URL, json.dumps(request_json), headers=header, timeout=60)
        if "data" in resp.json():
            data = resp.json()["data"]
            return base64.b64decode(data)
        logger.warning(f"ByteDance API错误: {resp.json()}")
    except Exception as e:
        logger.warning(f"ByteDance TTS请求失败: {e}")

    return None


def init_bytedance_reference_voices() -> None:
    """Ensure reference voice files exist for speaker matching."""
    voice_type_dir = "voice_type"
    if not os.path.exists(voice_type_dir):
        os.makedirs(voice_type_dir)

    voice_types = [
        "BV001_streaming",
        "BV002_streaming",
        "BV005_streaming",
        "BV007_streaming",
        "BV033_streaming",
        "BV034_streaming",
        "BV056_streaming",
        "BV102_streaming",
        "BV113_streaming",
        "BV115_streaming",
        "BV119_streaming",
        "BV700_streaming",
        "BV701_streaming",
    ]

    try:
        load_embedding_model()
    except Exception:
        logger.warning("由于embedding模型失败，跳过ByteDance参考音色初始化")
        return

    sample_text = "YouDub 是一个创新的开源工具，专注于将 YouTube 等平台的优质视频翻译和配音为中文版本。"

    for voice_type in voice_types:
        wav_path = os.path.join(voice_type_dir, f"{voice_type}.wav")
        npy_path = wav_path.replace(".wav", ".npy")

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
    speaker_to_voice_type_path = os.path.join(folder, "speaker_to_voice_type.json")
    if os.path.exists(speaker_to_voice_type_path):
        with open(speaker_to_voice_type_path, "r", encoding="utf-8") as f:
            return json.load(f)

    init_bytedance_reference_voices()

    speaker_to_voice_type: dict[str, str] = {}
    speaker_folder = os.path.join(folder, "SPEAKER")
    voice_types: dict[str, np.ndarray] = {}
    voice_type_dir = "voice_type"

    if not os.path.exists(voice_type_dir):
        logger.warning("未找到voice_type目录，无法进行说话人匹配")
        return {}

    for file in os.listdir(voice_type_dir):
        if file.endswith(".npy"):
            voice_type = file.replace(".npy", "")
            voice_types[voice_type] = np.load(os.path.join(voice_type_dir, file))

    if not os.path.exists(speaker_folder):
        return {}

    try:
        load_embedding_model()
    except Exception:
        logger.warning("Embedding模型不可用，回退到所有说话人的默认音色")
        for file in os.listdir(speaker_folder):
            if file.endswith(".wav"):
                speaker = file.replace(".wav", "")
                speaker_to_voice_type[speaker] = "BV001_streaming"

        with open(speaker_to_voice_type_path, "w", encoding="utf-8") as f:
            json.dump(speaker_to_voice_type, f, indent=2, ensure_ascii=False)
        return speaker_to_voice_type

    for file in os.listdir(speaker_folder):
        if not file.endswith(".wav"):
            continue
        speaker = file.replace(".wav", "")
        wav_path = os.path.join(speaker_folder, file)

        try:
            embedding = generate_embedding(wav_path)
            np.save(wav_path.replace(".wav", ".npy"), embedding)

            best_match = sorted(voice_types.keys(), key=lambda x: 1 - cosine(voice_types[x], embedding))[0]
            speaker_to_voice_type[speaker] = best_match
            logger.info(f"匹配 {speaker} 到 {best_match}")
        except Exception as e:
            logger.error(f"匹配说话人失败 {speaker}: {e}")
            speaker_to_voice_type[speaker] = "BV001_streaming"  # Default fallback

    with open(speaker_to_voice_type_path, "w", encoding="utf-8") as f:
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
        max_duration_seconds = 15
        sample_rate = 24000

        wav_data, _ = librosa.load(audio_path, sr=sample_rate, duration=max_duration_seconds)

        if len(wav_data) < sample_rate:  # Less than 1 second
            logger.warning(f"音频 {audio_path} 过短无法克隆 (< 1秒)")
            return False

        # Apply anti-pop processing before uploading
        wav_data = prepare_speaker_ref_audio(
            wav_data,
            trim_silence=True,
            trim_top_db=30.0,
            apply_soft_clip=True,
            clip_threshold=0.85,
            apply_smooth=True,
            smooth_max_diff=0.25,
        )

        if len(wav_data) < sample_rate:  # Check again after trimming
            logger.warning(f"音频 {audio_path} 处理后过短无法克隆 (< 1秒)")
            return False

        import io
        from scipy.io import wavfile

        # Normalize to safe level for int16 conversion
        peak = max(abs(float(np.max(wav_data))), abs(float(np.min(wav_data))))
        if peak > 1e-6:
            wav_data = wav_data * (0.95 / peak)

        wav_int16 = (wav_data * 32767).astype(np.int16)

        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, wav_int16)
        audio_data = buffer.getvalue()

        logger.info(f"上传 {len(audio_data) / 1024:.1f}KB 音频用于声音克隆 ({len(wav_data) / sample_rate:.1f}秒)")

        payload = {
            "appid": appid,
            "speaker_id": speaker_id,
            "audios": [
                {
                    "audio_bytes": base64.b64encode(audio_data).decode("utf-8"),
                    "audio_format": "wav",
                }
            ],
            "source": 2,  # 2 for user uploaded
            "language": 0,  # 0 for auto (CN default), 1 for EN. Ideally should be detected.
            "model_type": 4,  # 4: ICL2.0 (Voice Cloning 2.0)
        }

        logger.info("--- 声音克隆上传请求 ---")
        logger.info(f"地址: {url}")
        logger.info(f"请求头: {header}")
        debug_payload = payload.copy()
        if debug_payload.get("audios"):
            debug_payload["audios"] = [
                {"audio_bytes": "<hidden>", "audio_format": a["audio_format"]} for a in debug_payload["audios"]
            ]
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
    mapping_path = os.path.join(folder, "speaker_to_cloned_voice.json")
    mapping: dict[str, str] = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

    if speaker in mapping:
        cached_value = mapping[speaker]
        if cached_value == "FAILED":
            return None
        return cached_value

    volcano_speaker_ids_str = os.getenv("VOLCANO_CLONE_SPEAKER_IDS", "")
    if not volcano_speaker_ids_str:
        logger.warning("未在.env中设置VOLCANO_CLONE_SPEAKER_IDS。将跳过声音克隆。")
        logger.warning("请将VOLCANO_CLONE_SPEAKER_IDS设置为Volcano控制台中的逗号分隔ID列表。")
        return None

    available_ids = [s.strip() for s in volcano_speaker_ids_str.split(",") if s.strip()]
    if not available_ids:
        logger.warning("VOLCANO_CLONE_SPEAKER_IDS为空。将跳过声音克隆。")
        return None

    used_ids = {v for v in mapping.values() if v != "FAILED"}
    unused_ids = [id_ for id_ in available_ids if id_ not in used_ids]

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
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        return speaker_id

    mapping[speaker] = "FAILED"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    logger.warning(f"声音克隆失败 {speaker}，将使用所有段的默认音色。")

    return None


def bytedance_tts(
    text: str,
    output_path: str,
    speaker_wav: str,
    voice_type: str | None = None,
) -> None:
    if os.path.exists(output_path) and is_valid_wav(output_path):
        logger.info(f"ByteDance TTS {text[:20]}... 已存在（已验证）")
        return
    if os.path.exists(output_path):
        logger.warning(f"删除无效缓存文件: {output_path}")
        os.remove(output_path)

    folder = os.path.dirname(os.path.dirname(output_path))
    speaker = os.path.basename(speaker_wav).replace(".wav", "")

    if voice_type is None:
        cloned_voice_id = get_or_create_cloned_voice(folder, speaker, speaker_wav)
        if cloned_voice_id:
            voice_type = cloned_voice_id
            logger.info(f"使用克隆音色 {speaker}: {voice_type}")
        else:
            speaker_to_voice_type = generate_speaker_to_voice_type(folder)
            voice_type = speaker_to_voice_type.get(speaker, "BV001_streaming")
            logger.info(f"使用匹配音色 {speaker}: {voice_type}")

    is_cloned = voice_type.startswith("S_") if voice_type else False

    for _ in range(3):
        check_cancelled()
        try:
            # ICL 2.0 当前使用 V1 接口更稳定
            audio_data = bytedance_tts_api(text, voice_type=voice_type, use_cloned_voice=is_cloned)
            if audio_data:
                with open(output_path, "wb") as f:
                    f.write(audio_data)

                if is_valid_wav(output_path):
                    logger.info(f"ByteDance TTS已保存: {output_path}")
                    sleep_with_cancel(0.1)
                    break

                logger.warning("保存的wav文件似乎已损坏或无效")
                if os.path.exists(output_path):
                    os.remove(output_path)

            sleep_with_cancel(0.5)
        except Exception as e:
            logger.error(f"ByteDance TTS循环失败: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            sleep_with_cancel(0.5)

