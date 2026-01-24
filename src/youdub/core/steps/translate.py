from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, cast

from loguru import logger
from openai import (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from ...config import Settings

@dataclass(frozen=True)
class _ChatBackend:
    client: OpenAI
    model: str
    timeout_s: float


class _TranslationValidationError(ValueError):
    pass


_DEFAULT_SETTINGS = Settings()

def get_necessary_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        'title': info.get('title'),
        'uploader': info.get('uploader'),
        'description': info.get('description'),
        'upload_date': info.get('upload_date'),
        'categories': info.get('categories'),
        'tags': info.get('tags'),
    }


def ensure_transcript_length(transcript: str, max_length: int = 4000) -> str:
    if len(transcript) <= max_length:
        return transcript
        
    mid = len(transcript) // 2
    length = max_length // 2
    before = transcript[:mid]
    after = transcript[mid:]
    return before[:length] + after[-length:]


def _build_chat_backend(settings: Settings) -> _ChatBackend:
    timeout_s = 240.0
    api_key = settings.openai_api_key
    if not api_key:
        raise RuntimeError("缺少 OPENAI_API_KEY：请在 .env 中配置 OpenAI 兼容 API Key。")
    base_url = settings.openai_api_base or "https://api.openai.com/v1"
    client = OpenAI(base_url=base_url, api_key=api_key)
    return _ChatBackend(client=client, model=settings.model_name, timeout_s=timeout_s)


def _extract_first_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return cast(dict[str, Any], obj)
    raise ValueError("No JSON object found in model response.")


def _chat_completion_text(backend: _ChatBackend, messages: list[dict[str, str]]) -> str:
    response = backend.client.chat.completions.create(
        model=backend.model,
        messages=messages,
        timeout=backend.timeout_s,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def _handle_sdk_exception(exc: Exception, attempt: int) -> float | None:
    """Return sleep seconds for retry, or None to stop retrying."""
    # --- OpenAI compatible SDK exceptions ---
    if isinstance(exc, (AuthenticationError,)):
        logger.error(f"LLM 鉴权失败：{exc}")
        return None
    if isinstance(exc, (BadRequestError,)):
        logger.error(f"LLM 请求参数错误：{exc}")
        return None
    if isinstance(exc, (RateLimitError,)):
        delay = min(2 ** attempt, 30)
        logger.warning(f"LLM 限流，{delay}s 后重试：{exc}")
        return float(delay)
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        delay = min(2 ** attempt, 20)
        logger.warning(f"LLM 连接/超时，{delay}s 后重试：{exc}")
        return float(delay)
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status in {500, 502, 503, 504}:
            delay = min(2 ** attempt, 20)
            logger.warning(f"LLM 服务端错误({status})，{delay}s 后重试：{exc}")
            return float(delay)
        logger.error(f"LLM 请求失败({status})：{exc}")
        return None

    return None


def summarize(
    info: dict[str, Any],
    transcript: list[dict[str, Any]],
    target_language: str = "简体中文",
    settings: Settings | None = None,
) -> dict[str, Any]:
    cfg = settings or _DEFAULT_SETTINGS
    backend = _build_chat_backend(cfg)

    transcript_text = " ".join(cast(str, line.get("text", "")) for line in transcript)
    transcript_text = ensure_transcript_length(transcript_text, max_length=2000)
    info_message = f'Title: "{info.get("title", "")}" Author: "{info.get("uploader", "")}". '

    full_description = (
        f"The following is the full content of the video:\n{info_message}\n{transcript_text}\n{info_message}\n"
        "Summarize the video content as JSON only.\n"
        'Return JSON in the format: {"title": "...", "summary": "..."}'
    )

    summary_data: dict[str, str] | None = None
    for attempt in range(5):
        try:
            messages = [
                {
                    "role": "system",
                    "content": 'You are an expert video analyst. Respond with JSON only: {"title": "...", "summary": "..."}',
                },
                {"role": "user", "content": full_description},
            ]
            content = _chat_completion_text(backend, messages).replace("\n", " ")
            logger.info(content)

            summary_json = _extract_first_json_object(content)
            title = str(summary_json.get("title", "")).replace("title:", "").strip()
            summary_text = str(summary_json.get("summary", "")).replace("summary:", "").strip()
            if not title or not summary_text:
                raise ValueError("Missing title/summary fields in JSON.")
            if "title" in title.lower():
                raise ValueError("Invalid title field.")
            summary_data = {"title": title, "summary": summary_text}
            break
        except (ValueError, JSONDecodeError) as exc:
            logger.warning(f"总结解析失败（attempt={attempt + 1}/5）：{exc}")
            time.sleep(1)
        except Exception as exc:  # SDK/network errors handled explicitly below
            delay = _handle_sdk_exception(exc, attempt)
            if delay is None:
                raise
            time.sleep(delay)

    if not summary_data:
        raise RuntimeError("总结失败：无法从模型输出解析 JSON。")

    title = summary_data["title"]
    summary_text = summary_data["summary"]
    tags = info.get("tags", [])

    translate_messages = [
        {
            "role": "system",
            "content": (
                f'You are a native speaker of {target_language}. Translate title/summary/tags into {target_language}. '
                'Respond with JSON only: {"title": "...", "summary": "...", "tags": ["..."]}'
            ),
        },
        {
            "role": "user",
            "content": (
                f'Title: "{title}"\nSummary: "{summary_text}"\nTags: {tags}\n'
                f"Translate into {target_language} and return JSON only."
            ),
        },
    ]

    for attempt in range(5):
        try:
            content = _chat_completion_text(backend, translate_messages).replace("\n", " ")
            logger.info(content)
            translated_json = _extract_first_json_object(content)
            translated_title = str(translated_json.get("title", "")).strip()
            translated_summary = str(translated_json.get("summary", "")).strip()
            translated_tags = translated_json.get("tags", [])
            if not translated_title or not translated_summary:
                raise ValueError("Missing title/summary fields in translated JSON.")
            if target_language in translated_title or target_language in translated_summary:
                raise ValueError("Model echoed language name instead of translation.")

            for quote in ['"', "“", "‘", "'", "《"]:
                if translated_title.startswith(quote):
                    translated_title = translated_title.strip(quote + "”’》")

            return {
                "title": translated_title,
                "author": info.get("uploader", ""),
                "summary": translated_summary,
                "tags": translated_tags if isinstance(translated_tags, list) else [],
                "language": target_language,
            }
        except (ValueError, JSONDecodeError) as exc:
            logger.warning(f"总结翻译解析失败（attempt={attempt + 1}/5）：{exc}")
            time.sleep(1)
        except Exception as exc:
            delay = _handle_sdk_exception(exc, attempt)
            if delay is None:
                raise
            time.sleep(delay)

    raise RuntimeError("总结翻译失败：无法从模型输出解析 JSON。")


def translation_postprocess(result: str) -> str:
    result = re.sub(r'\（[^)]*\）', '', result)
    result = result.replace('...', '，')
    result = re.sub(r'(?<=\d),(?=\d)', '', result)
    result = result.replace('²', '的平方').replace(
        '————', '：').replace('——', '：').replace('°', '度')
    result = result.replace("AI", '人工智能')
    result = result.replace('变压器', "Transformer")
    return result


def valid_translation(text: str, translation: str) -> tuple[bool, str]:
    if (translation.startswith('```') and translation.endswith('```')):
        translation = translation[3:-3]
        return True, translation_postprocess(translation)

    if (translation.startswith('“') and translation.endswith('”')) or (translation.startswith('"') and translation.endswith('"')):
        translation = translation[1:-1]
        return True, translation_postprocess(translation)

    # Heuristics to remove prefixes like "翻译：“..."
    if '翻译' in translation and '：“' in translation and '”' in translation:
        translation = translation.split('：“')[-1].split('”')[0]
        return True, translation_postprocess(translation)

    if '翻译' in translation and '："' in translation and '"' in translation:
        translation = translation.split('："')[-1].split('"')[0]
        return True, translation_postprocess(translation)

    if '翻译' in translation and ':"' in translation and '"' in translation:
        translation = translation.split('："')[-1].split('"')[0]
        return True, translation_postprocess(translation)

    if len(text) <= 10:
        if len(translation) > 15:
            return False, 'Only translate the following sentence and give me the result.'
    elif len(translation) > len(text) * 0.75:
        # Check if translation is suspiciously long compared to source
        # Note: Original logic: len(translation) > len(text)*0.75 means translation is > 75% of text length? 
        # Actually usually Chinese is shorter than English. If translation is > 75% of text, it might be okay?
        # Original logic seems to forbid translation if it is NOT significantly shorter?
        # Wait: "len(translation) > len(text)*0.75" -> if translation is longer than 0.75 * text.
        # This seems aggressive for short texts.
        # But I will keep original logic for parity.
        return False, 'The translation is too long. Only translate the following sentence and give me the result.'

    forbidden = ['翻译', '这句', '\n', '简体中文', '中文', 'translate', 'Translate', 'translation', 'Translation']
    translation = translation.strip()
    for word in forbidden:
        if word in translation:
            return False, f"Don't include `{word}` in the translation. Only translate the following sentence and give me the result."

    return True, translation_postprocess(translation)


def split_text_into_sentences(para: str) -> list[str]:
    para = re.sub(r'([。！？\?])([^，。！？\?”’》])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub(r'(\.{6})([^，。！？\?”’》])', r"\1\n\2", para)  # 英文省略号
    para = re.sub(r'(\…{2})([^，。！？\?”’》])', r"\1\n\2", para)  # 中文省略号
    para = re.sub(r'([。！？\?][”’])([^，。！？\?”’》])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


def split_sentences(translation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_data = []
    for item in translation:
        start = item['start']
        text = item['text']
        speaker = item['speaker']
        translation_text = item.get('translation', '')
        sentences = split_text_into_sentences(translation_text)
        
        if not translation_text:
             # Handle empty translation
             duration_per_char = 0
        else:
             duration_per_char = (item['end'] - item['start']) / len(translation_text)
             
        sentence_start_idx = 0
        for sentence in sentences:
            sentence_len = len(sentence)
            sentence_end = start + duration_per_char * sentence_len

            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": text,
                "speaker": speaker,
                "translation": sentence
            })

            start = sentence_end
            sentence_start_idx += sentence_len
            
    return output_data


def _translate_content(
    summary: dict[str, Any],
    transcript: list[dict[str, Any]],
    target_language: str = "简体中文",
    settings: Settings | None = None,
) -> list[str]:
    cfg = settings or _DEFAULT_SETTINGS
    backend = _build_chat_backend(cfg)

    info = f'This is a video called "{summary.get("title", "")}". {summary.get("summary", "")}.'
    full_translation = []
    
    fixed_message = [
        {'role': 'system', 'content': f'You are a expert in the field of this video.\n{info}\nTranslate the sentence into {target_language}.下面我让你来充当翻译家，你的目标是把任何语言翻译成中文，请翻译时不要带翻译腔，而是要翻译得自然、流畅和地道，使用优美和高雅的表达方式。请将人工智能的“agent”翻译为“智能体”，强化学习中是`Q-Learning`而不是`Queue Learning`。数学公式写成plain text，不要使用latex。确保翻译正确和简洁。注意信达雅。'},
        {'role': 'user', 'content': '使用地道的中文Translate:"Knowledge is power."'},
        {'role': 'assistant', 'content': '翻译：“知识就是力量。”'},
        {'role': 'user', 'content': '使用地道的中文Translate:"To be or not to be, that is the question."'},
        {'role': 'assistant', 'content': '翻译：“生存还是毁灭，这是一个值得考虑的问题。”'},
    ]
    
    history: list[dict[str, str]] = []
    for line in transcript:
        text = cast(str, line.get("text", ""))
        translation = ""
        
        for attempt in range(30):
            # Keep history short to avoid token limit
            current_history = history[-30:]
            messages = fixed_message + current_history + [
                {'role': 'user', 'content': f'使用地道的中文Translate:"{text}"'}
            ]
            
            try:
                translation = _chat_completion_text(backend, messages).replace("\n", "")
                logger.info(f'原文：{text}')
                logger.info(f'译文：{translation}')
                
                is_valid, processed = valid_translation(text, translation)
                if not is_valid:
                    raise _TranslationValidationError(processed)
                
                translation = processed
                break
            except _TranslationValidationError as exc:
                logger.warning(f"译文校验失败（attempt={attempt + 1}/30）：{exc}")
                time.sleep(0.5)
            except Exception as exc:
                delay = _handle_sdk_exception(exc, attempt)
                if delay is None:
                    raise
                time.sleep(delay)
        
        full_translation.append(translation)
        
        history.append({'role': 'user', 'content': f'Translate:"{text}"'})
        history.append({'role': 'assistant', 'content': f'翻译：“{translation}”'})
        # Avoid rate limits
        time.sleep(0.1)

    return full_translation


def translate_folder(folder: str, target_language: str = '简体中文', settings: Settings | None = None) -> bool:
    if os.path.exists(os.path.join(folder, 'translation.json')):
        logger.info(f'Translation already exists in {folder}')
        return True
    
    info_path = os.path.join(folder, 'download.info.json')
    if not os.path.exists(info_path):
        return False
        
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    info = get_necessary_info(info)
    
    transcript_path = os.path.join(folder, 'transcript.json')
    if not os.path.exists(transcript_path):
        return False
        
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    
    summary_path = os.path.join(folder, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
    else:
        summary = summarize(info, transcript, target_language, settings=settings)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    translation_path = os.path.join(folder, 'translation.json')
    
    # Perform translation
    translations = _translate_content(summary, transcript, target_language, settings=settings)
    
    for i, line in enumerate(transcript):
        line['translation'] = translations[i] if i < len(translations) else ""
        
    transcript = split_sentences(transcript)
    
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
        
    return True


def translate_all_transcript_under_folder(folder: str, target_language: str, settings: Settings | None = None) -> str:
    count = 0
    for root, dirs, files in os.walk(folder):
        if 'transcript.json' in files and 'translation.json' not in files:
            if translate_folder(root, target_language, settings=settings):
                count += 1
    msg = f'Translated all videos under {folder} (processed {count} files)'
    logger.info(msg)
    return msg
