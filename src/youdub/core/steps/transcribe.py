import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
from loguru import logger
from packaging import version

from ...config import Settings
from ...models import ModelCheckError, ModelManager
from ..utils import save_wav


def _import_whisperx():
    try:
        import whisperx  # type: ignore
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(
            "导入 WhisperX 失败，语音识别功能不可用。\n"
            "常见原因：`numpy/transformers` 版本不兼容或安装不完整。\n"
            "建议在当前虚拟环境中执行：\n"
            "`pip install -U 'numpy<2'` 或按 `pyproject.toml` 重装依赖。\n"
            f"原始错误：{exc}"
        ) from exc
    return whisperx

# Global model instances for caching
_WHISPER_MODEL = None
_DIARIZE_MODEL = None
_ALIGN_MODEL = None
_LANGUAGE_CODE = None
_ALIGN_METADATA = None

_DEFAULT_SETTINGS = Settings()
_DEFAULT_MODEL_MANAGER = ModelManager(_DEFAULT_SETTINGS)


def unload_all_models() -> None:
    global _WHISPER_MODEL, _DIARIZE_MODEL, _ALIGN_MODEL
    
    cleared = False
    if _WHISPER_MODEL is not None:
        del _WHISPER_MODEL
        _WHISPER_MODEL = None
        cleared = True
    if _DIARIZE_MODEL is not None:
        del _DIARIZE_MODEL
        _DIARIZE_MODEL = None
        cleared = True
    if _ALIGN_MODEL is not None:
        del _ALIGN_MODEL
        _ALIGN_MODEL = None
        cleared = True
        
    if cleared:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info("WhisperX models unloaded")



def _ensure_offline_mode() -> None:
    """Prevent unexpected online downloads."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WHISPERX_LOCAL_FILES_ONLY", "1")


def _ensure_whisper_assets(
    settings: Settings,
    model_manager: ModelManager,
    require_diarization: bool = True,
) -> None:
    model_manager.enforce_offline()
    # Only check ASR/align/diarization related assets
    names = [
        model_manager._whisper_requirement().name,
        model_manager._whisper_align_requirement().name,
    ]
    if require_diarization:
        names.append(model_manager._whisper_diarization_requirement().name)
    model_manager.ensure_ready(names=names)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _default_compute_type(device: str) -> str:
    # CTranslate2 default float16 can be unsupported/slow on CPU; int8 is a safer default.
    return "float16" if device == "cuda" else "int8"


def _looks_like_transformers_model_dir(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    has_config = os.path.isfile(os.path.join(path, "config.json"))
    has_weights = any(
        os.path.isfile(os.path.join(path, fname))
        for fname in ("pytorch_model.bin", "model.safetensors")
    )
    return has_config and has_weights


def _ensure_ct2_model(transformers_model_dir: str, output_dir: str, quantization: str) -> str:
    model_bin = os.path.join(output_dir, "model.bin")
    if os.path.isfile(model_bin):
        return output_dir

    os.makedirs(output_dir, exist_ok=True)
    converter = shutil.which("ct2-transformers-converter")
    if not converter:
        # Check in the same dir as the python executable
        candidate = os.path.join(os.path.dirname(sys.executable), "ct2-transformers-converter")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            converter = candidate
            
    if not converter:
        raise RuntimeError(
            "ct2-transformers-converter not found in PATH; please install ctranslate2."
        )

    logger.info(
        f"Converting Transformers Whisper model to CTranslate2 (quantization={quantization})..."
    )
    subprocess.run(
        [
            converter,
            "--model",
            transformers_model_dir,
            "--output_dir",
            output_dir,
            "--quantization",
            quantization,
            "--force",
        ],
        check=True,
    )
    if not os.path.isfile(model_bin):
        raise RuntimeError(f"CTranslate2 conversion finished but missing {model_bin}")
    return output_dir


def init_whisperx(settings: Settings | None = None, model_manager: ModelManager | None = None) -> None:
    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER
    
    _ensure_whisper_assets(settings, model_manager)
    load_whisper_model(
        model_name=str(settings.whisper_model_path),
        download_root=str(settings.whisper_download_root),
        device="auto",
        settings=settings,
        model_manager=model_manager,
    )
    load_align_model(settings=settings, model_manager=model_manager)
    load_diarize_model(settings=settings, model_manager=model_manager)


def load_whisper_model(
    model_name: str = "large-v3",
    download_root: str = "models/ASR/whisper",
    device: str = "auto",
    compute_type: str | None = None,
    local_files_only: bool | None = None,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
    require_diarization: bool = True,
) -> None:
    global _WHISPER_MODEL
    
    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER
    
    _ensure_whisper_assets(settings, model_manager, require_diarization=require_diarization)
    _ensure_offline_mode()

    if model_name == "large":
        model_name = "large-v3"
        
    if _WHISPER_MODEL is not None:
        return

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    # --- EARLY ENVIRONMENT SETUP ---
    # We must set HF_HOME/TORCH_HOME before ANY call to whisperx (which might import pyannote)
    # to ensure the library picks up the correct cache paths.
    if require_diarization and settings.whisper_diarization_model_dir:
        diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
        if diar_dir and diar_dir.exists():
            logger.info(f"Using Diarization dir as unified HF_HOME: {diar_dir}")
            os.environ["TRANSFORMERS_CACHE"] = str(diar_dir)
            os.environ["HF_HOME"] = str(diar_dir)
            os.environ["TORCH_HOME"] = str(Path.home() / ".cache" / "torch")
        else:
             logger.warning(f"Diarization is required but path {diar_dir} is invalid.")
             
    elif settings.whisper_align_model_dir:
        # Fallback to align dir if no diarization
        align_dir = settings.resolve_path(settings.whisper_align_model_dir)
        if align_dir and align_dir.exists():
             logger.info(f"Using Align dir as unified HF_HOME: {align_dir}")
             os.environ["TRANSFORMERS_CACHE"] = str(align_dir)
             os.environ["HF_HOME"] = str(align_dir)
             
    if compute_type is None:
        compute_type = os.getenv("WHISPERX_COMPUTE_TYPE") or _default_compute_type(device)
        
    if local_files_only is None:
        local_files_only = True

    model_path = Path(model_name)
    if not model_path.exists():
        raise ModelCheckError(
            f"WhisperX 模型未找到：{model_path}。请先手动下载/转换为 CTranslate2 模型并设置 WHISPERX_MODEL_PATH。"
        )
        
    download_root_path = Path(download_root)
    if not download_root_path.exists():
        raise ModelCheckError(
            f"WhisperX 模型根目录不存在：{download_root_path}。请检查 WHISPERX_DOWNLOAD_ROOT。"
        )

    logger.info(f"Loading WhisperX model from {model_path}")
    t_start = time.time()

    enable_ct2_convert = _env_bool("WHISPERX_ENABLE_CT2_CONVERT", default=False)
    # Basic torch version check for safety with .bin files
    try:
        torch_ver = version.parse(torch.__version__.split("+", 1)[0])
        torch_ok_for_pt = torch_ver >= version.parse("2.6")
    except Exception:
        torch_ok_for_pt = False

    def maybe_convert(transformers_model_dir: str) -> str | None:
        if not _looks_like_transformers_model_dir(transformers_model_dir):
            return None
            
        if not enable_ct2_convert:
            logger.warning(
                f"Found Transformers Whisper weights at {transformers_model_dir}, but WhisperX expects a CTranslate2 "
                "model (model.bin). Set WHISPERX_ENABLE_CT2_CONVERT=1 (and torch>=2.6) to convert offline, or set "
                "WHISPERX_MODEL_PATH to an existing CTranslate2 model directory."
            )
            if local_files_only:
                raise ModelCheckError(
                    "WHISPERX_LOCAL_FILES_ONLY 已开启，但提供的是 Transformers 权重且未启用转换。"
                    "请先本地转换或关闭本步骤。"
                )
            return None
            
        if not torch_ok_for_pt:
            raise RuntimeError(
                "Transformers .bin weights require torch>=2.6 to load (Transformers safety restriction). "
                "Either upgrade torch, re-download weights in safetensors format, or use the pre-converted "
                "faster-whisper model (CTranslate2 model.bin)."
            )
            
        quantization = os.getenv("WHISPERX_QUANTIZATION") or compute_type
        ct2_out_dir = os.getenv("WHISPERX_CT2_MODEL_PATH") or os.path.join(
            transformers_model_dir, f"ctranslate2-{quantization}"
        )
        return _ensure_ct2_model(transformers_model_dir, ct2_out_dir, quantization)

    converted = maybe_convert(str(model_path))
    if converted:
        model_name = converted
    else:
        model_name = str(model_path)

    whisperx = _import_whisperx()
    _WHISPER_MODEL = whisperx.load_model(
        model_name,
        download_root=str(download_root_path),
        device=device,
        compute_type=compute_type,
    )
    t_end = time.time()
    logger.info(f"Loaded WhisperX model: {model_name} in {t_end - t_start:.2f}s")


def load_align_model(
    language: str = "en",
    device: str = "auto",
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> None:
    global _ALIGN_MODEL, _LANGUAGE_CODE, _ALIGN_METADATA
    
    if _ALIGN_MODEL is not None and _LANGUAGE_CODE == language:
        return
        
    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER
    
    _ensure_whisper_assets(settings, model_manager, require_diarization=False)
    _ensure_offline_mode()
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    _LANGUAGE_CODE = language
    align_dir = settings.resolve_path(settings.whisper_align_model_dir)
    
    if align_dir and not align_dir.exists():
        raise ModelCheckError(f"WhisperX 对齐模型目录不存在：{align_dir}")
        
    if settings.whisper_align_model_dir:
        # We assume align models are linked into diarization dir if diarization is enabled,
        # or we set HF_HOME to align dir if diarization is disabled?
        # To be safe, if we are in this function, we probably want a single consistent cache.
        # But if we use diarization dir as main cache, we are good.
        pass

    t_start = time.time()
    whisperx = _import_whisperx()
    _ALIGN_MODEL, _ALIGN_METADATA = whisperx.load_align_model(language_code=_LANGUAGE_CODE, device=device)
    t_end = time.time()
    logger.info(f"Loaded alignment model: {_LANGUAGE_CODE} in {t_end - t_start:.2f}s")


def load_diarize_model(
    device: str = "auto",
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> None:
    global _DIARIZE_MODEL
    
    if _DIARIZE_MODEL is not None:
        return
        
    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER
    
    _ensure_whisper_assets(settings, model_manager)
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    diar_dir = settings.resolve_path(settings.whisper_diarization_model_dir)
    if diar_dir and not diar_dir.exists():
        raise ModelCheckError(f"WhisperX 说话人分离模型目录不存在：{diar_dir}")
    
    # Environment variables should have been set in load_whisper_model or globally
    logger.info(f"Loading diarization model with HF_HOME={os.environ.get('HF_HOME')}, TORCH_HOME={os.environ.get('TORCH_HOME')}")
    
    # Helper to find config.yaml
    def find_pipeline_config(root_dir: Path) -> Path | None:
        # Search for any config.yaml under snapshots
        # Structure: models--pyannote... / snapshots / <hash> / config.yaml
        if not root_dir.exists():
            return None
        for path in root_dir.rglob("config.yaml"):
            # Check if it looks like a snapshot config
            if "snapshots" in path.parts:
                return path
        return None

    pipeline_config = find_pipeline_config(diar_dir) if diar_dir else None
    
    t_start = time.time()
    
    try:
        from pyannote.audio import Pipeline
        import torch
        
        pipeline = None
        
        # Strategy 1: Load from local config.yaml directly if found
        if pipeline_config and pipeline_config.exists():
            logger.info(f"Found local pipeline config: {pipeline_config}")
            try:
                # pyannote.audio.Pipeline.from_pretrained can take a path to config.yaml
                pipeline = Pipeline.from_pretrained(str(pipeline_config))
                logger.info("Successfully loaded pipeline from local config.")
            except Exception as e:
                logger.warning(f"Failed to load from local config {pipeline_config}: {e}")
        
        # Strategy 2: Fallback to WhisperX / HF Hub ID loading
        if pipeline is None:
            logger.info("Falling back to WhisperX/HF Hub ID loading...")
            whisperx = _import_whisperx()
            # whisperx.DiarizationPipeline is just a wrapper around Pipeline.from_pretrained
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=settings.hf_token
            )

        if pipeline is None:
             raise RuntimeError("Pipeline.from_pretrained returned None")

        pipeline.to(torch.device(device))
        _DIARIZE_MODEL = pipeline
        
    except Exception as e:
        logger.error(f"Failed to load DiarizationPipeline: {e}")
        _DIARIZE_MODEL = None
        
    if _DIARIZE_MODEL is None:
        logger.error("DiarizationPipeline returned None. Check if models are correctly downloaded and cached in HF_HOME.")
        raise RuntimeError("WhisperX Diarization 模型加载失败。请检查日志中的路径配置和模型完整性。")

    t_end = time.time()
    logger.info(f"Loaded diarization model in {t_end - t_start:.2f}s")


def merge_segments(transcript: list[dict[str, Any]], ending: str = '!"\').:;?]}~') -> list[dict[str, Any]]:
    merged_transcription = []
    buffer_segment = None

    for segment in transcript:
        if buffer_segment is None:
            buffer_segment = segment
        else:
            # Check if the last character of the 'text' field is a punctuation mark
            if buffer_segment['text'] and buffer_segment['text'][-1] in ending:
                # If it is, add the buffered segment to the merged transcription
                merged_transcription.append(buffer_segment)
                buffer_segment = segment
            else:
                # If it's not, merge this segment with the buffered segment
                buffer_segment['text'] += ' ' + segment['text']
                buffer_segment['end'] = segment['end']

    # Don't forget to add the last buffered segment
    if buffer_segment is not None:
        merged_transcription.append(buffer_segment)

    return merged_transcription


def generate_speaker_audio(folder: str, transcript: list[dict[str, Any]]) -> None:
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(wav_path):
        logger.warning(f"Audio file not found: {wav_path}")
        return

    # Load with 24000 sr (TTS usually expects this)
    audio_data, samplerate = librosa.load(wav_path, sr=24000)
    speaker_dict = dict()
    length = len(audio_data)
    delay = 0.05
    
    for segment in transcript:
        start = max(0, int((segment['start'] - delay) * samplerate))
        end = min(int((segment['end'] + delay) * samplerate), length)
        
        if start >= end:
            continue
            
        speaker_segment_audio = audio_data[start:end]
        speaker = segment.get('speaker', 'SPEAKER_00')
        
        if speaker not in speaker_dict:
            speaker_dict[speaker] = np.zeros((0, ))
            
        speaker_dict[speaker] = np.concatenate((speaker_dict[speaker], speaker_segment_audio))

    speaker_folder = os.path.join(folder, 'SPEAKER')
    os.makedirs(speaker_folder, exist_ok=True)
    
    for speaker, audio in speaker_dict.items():
        speaker_file_path = os.path.join(speaker_folder, f"{speaker}.wav")
        save_wav(audio, speaker_file_path, sample_rate=24000)


def transcribe_audio(
    folder: str,
    model_name: str = "large-v3",
    download_root: str | None = "models/ASR/whisper",
    device: str = "auto",
    batch_size: int = 32,
    diarization: bool = True,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> bool:
    if os.path.exists(os.path.join(folder, 'transcript.json')):
        logger.info(f'Transcript already exists in {folder}')
        return True
    
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(wav_path):
        return False
    
    logger.info(f'Transcribing {wav_path}')
    
    settings = settings or _DEFAULT_SETTINGS
    model_manager = model_manager or _DEFAULT_MODEL_MANAGER
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    load_whisper_model(
        model_name,
        download_root,
        device,
        settings=settings,
        model_manager=model_manager,
        require_diarization=diarization,
    )
    
    # _WHISPER_MODEL is set by load_whisper_model
    rec_result = _WHISPER_MODEL.transcribe(wav_path, batch_size=batch_size)
    
    if rec_result['language'] == 'nn':
        logger.warning(f'No language detected in {wav_path}')
        return False
    
    load_align_model(rec_result['language'], settings=settings, model_manager=model_manager)
    whisperx = _import_whisperx()
    rec_result = whisperx.align(
        rec_result['segments'], 
        _ALIGN_MODEL, 
        _ALIGN_METADATA,
        wav_path, 
        device, 
        return_char_alignments=False
    )
    
    if diarization:
        load_diarize_model(device, settings=settings, model_manager=model_manager)
        logger.info(f"Starting diarization for {wav_path} (this may take a while)...")
        
        # Preload audio into memory to avoid repeated disk I/O during diarization
        import torchaudio
        waveform, sample_rate = torchaudio.load(wav_path)
        audio_dict = {
            "audio": wav_path,          # Required: original audio path
            "waveform": waveform,       # Optional: preloaded waveform
            "sample_rate": sample_rate, # Required when waveform is provided
            "uri": wav_path             # Optional: for annotation labeling
        }
        logger.info(f"Audio preloaded into memory ({waveform.shape[1] / sample_rate:.1f}s, {sample_rate}Hz)")
        
        
        diarize_annotation = _DIARIZE_MODEL(audio_dict, min_speakers=min_speakers, max_speakers=max_speakers)
        logger.info(f"Diarization completed, converting to DataFrame format...")
        
        # Convert Annotation to DataFrame format expected by WhisperX
        import pandas as pd
        diarize_df = pd.DataFrame(
            diarize_annotation.itertracks(yield_label=True),
            columns=['segment', 'label', 'speaker']
        )
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        
        rec_result = whisperx.assign_word_speakers(diarize_df, rec_result)
        
    transcript = [
        {
            'start': segment['start'], 
            'end': segment['end'], 
            'text': segment['text'].strip(), 
            'speaker': segment.get('speaker', 'SPEAKER_00')
        } 
        for segment in rec_result['segments']
    ]
    transcript = merge_segments(transcript)
    
    with open(os.path.join(folder, 'transcript.json'), 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=4, ensure_ascii=False)
        
    logger.info(f'Transcribed {wav_path} successfully, and saved to {os.path.join(folder, "transcript.json")}')
    generate_speaker_audio(folder, transcript)
    return True


def transcribe_all_audio_under_folder(
    folder: str,
    model_name: str = "large-v3",
    download_root: str | None = "models/ASR/whisper",
    device: str = "auto",
    batch_size: int = 32,
    diarization: bool = True,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    settings: Settings | None = None,
    model_manager: ModelManager | None = None,
) -> str:
    count = 0
    for root, dirs, files in os.walk(folder):
        if 'audio_vocals.wav' in files and 'transcript.json' not in files:
            success = transcribe_audio(
                root, 
                model_name,
                download_root, 
                device, 
                batch_size, 
                diarization, 
                min_speakers, 
                max_speakers, 
                settings=settings, 
                model_manager=model_manager
            )
            if success:
                count += 1
                
    msg = f'Transcribed all audio under {folder} (processed {count} files)'
    logger.info(msg)
    return msg
