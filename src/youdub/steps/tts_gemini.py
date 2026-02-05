from __future__ import annotations

import os

from loguru import logger

from ..config import Settings
from ..interrupts import check_cancelled, sleep_with_cancel
from .tts_wav_guard import is_valid_wav


_DEFAULT_SETTINGS = Settings()

_GEMINI_CLIENT = None


def _get_gemini_client():
    global _GEMINI_CLIENT  # noqa: PLW0603
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


def gemini_tts(
    text: str,
    output_path: str,
    voice_name: str | None = None,
) -> None:
    """Generate speech using Gemini TTS and save to file using Best Practices."""
    if os.path.exists(output_path) and is_valid_wav(output_path):
        logger.info(f"Gemini TTS {text[:20]}... 已存在（已验证）")
        return
    if os.path.exists(output_path):
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
    voice_name = voice_name or _DEFAULT_SETTINGS.gemini_tts_voice or "Kore"

    rate = 24000
    sample_width = 2  # 16-bit
    channels = 1

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
                ),
            )

            if (
                response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
                and response.candidates[0].content.parts[0].inline_data
            ):
                pcm_data = response.candidates[0].content.parts[0].inline_data.data

                with wave.open(output_path, "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(rate)
                    wf.writeframes(pcm_data)

                if is_valid_wav(output_path):
                    logger.info(f"Gemini TTS已保存: {output_path}")
                    sleep_with_cancel(0.1)
                    return  # Success

                logger.warning("Gemini TTS写入无效文件，重试中...")
                if os.path.exists(output_path):
                    os.remove(output_path)
            else:
                logger.warning(
                    "Gemini TTS响应结构异常。"
                    f"Candidates: {response.candidates}, PromptFeedback: {getattr(response, 'prompt_feedback', 'N/A')}"
                )
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

                match = re.search(r"retry in (\d+\.?\d*)s", error_str)
                delay = float(match.group(1)) if match else 20.0
                delay += 1.0

                logger.warning(f"Gemini TTS速率限制(429)。{delay}秒后重试... (尝试 {retry_count + 1}/{max_retries})")
                sleep_with_cancel(delay)
                retry_count += 1
                continue

            logger.error(f"Gemini TTS生成失败: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            retry_count += 1
            sleep_with_cancel(1)

