from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


_THREAD_LOCAL = threading.local()


def _get_thread_backend(settings: Settings) -> _ChatBackend:
    """Get a per-thread OpenAI client to avoid thread-safety issues."""
    backend = getattr(_THREAD_LOCAL, "backend", None)
    if isinstance(backend, _ChatBackend):
        return backend
    backend = _build_chat_backend(settings)
    _THREAD_LOCAL.backend = backend
    return backend


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
            
    return output_data


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _normalize_translation_strategy(raw: str | None) -> str:
    s = (raw or "history").strip().lower()
    if s in {"history", "serial", "chain", "seq"}:
        return "history"
    if s in {"guide_parallel", "parallel", "parallel_guide", "guide", "p2", "2"}:
        return "guide_parallel"
    return "history"


def _build_translation_guide(
    summary: dict[str, Any],
    transcript: list[dict[str, Any]],
    target_language: str,
    settings: Settings,
) -> dict[str, Any]:
    max_chars = max(800, _read_int_env("TRANSLATION_GUIDE_MAX_CHARS", 2500))

    transcript_text = " ".join(cast(str, line.get("text", "")) for line in transcript).strip()
    transcript_text = ensure_transcript_length(transcript_text, max_length=max_chars)

    title = str(summary.get("title", "")).strip()
    summary_text = str(summary.get("summary", "")).strip()

    system = (
        f"你是专业译者，目标语言：{target_language}。\n"
        "请基于给定的视频信息与字幕样本，生成一份“翻译指南”，用于后续并发分块翻译。\n"
        "必须只输出 JSON（不要任何额外文本）。\n"
        'JSON 格式：{"style": ["..."], "glossary": {"term": "translation"}, "dont_translate": ["..."], "notes": "..."}\n'
        "- style：5~12 条风格/语气/术语偏好（list of strings）\n"
        "- glossary：只收录重要专有名词/术语/人名/机构/产品名等（object mapping string->string）\n"
        "- dont_translate：必须原样保留的 token（如代码/公式/缩写/产品名）\n"
        "- notes：额外注意事项\n"
        "硬性要求：\n"
        "- 将人工智能的“agent”翻译为“智能体”\n"
        "- 强化学习中写作 `Q-Learning`（不要写 Queue Learning）\n"
        "- 变压器指模型时保留 `Transformer`\n"
        "- 数学公式写成 plain text，不要使用 LaTeX\n"
    )
    user = (
        f'Title: "{title}"\n'
        f"Summary: {summary_text}\n"
        f"Transcript sample:\n{transcript_text}\n"
    )

    backend = _build_chat_backend(settings)
    for attempt in range(5):
        try:
            content = _chat_completion_text(backend, [{"role": "system", "content": system}, {"role": "user", "content": user}])
            guide_raw = _extract_first_json_object(content)

            style_raw = guide_raw.get("style", [])
            if isinstance(style_raw, str):
                style = [style_raw.strip()] if style_raw.strip() else []
            elif isinstance(style_raw, list):
                style = [str(x).strip() for x in style_raw if str(x).strip()]
            else:
                style = []

            glossary_raw = guide_raw.get("glossary", {})
            glossary: dict[str, str] = {}
            if isinstance(glossary_raw, dict):
                for k, v in glossary_raw.items():
                    ks = str(k).strip()
                    vs = str(v).strip()
                    if ks and vs:
                        glossary[ks] = vs

            dont_raw = guide_raw.get("dont_translate", [])
            if isinstance(dont_raw, str):
                dont_translate = [dont_raw.strip()] if dont_raw.strip() else []
            elif isinstance(dont_raw, list):
                dont_translate = [str(x).strip() for x in dont_raw if str(x).strip()]
            else:
                dont_translate = []

            notes = str(guide_raw.get("notes", "")).strip()

            # Always enforce core term preferences even if the guide omitted them.
            glossary.setdefault("agent", "智能体")
            glossary.setdefault("Agent", "智能体")
            dont_translate = list(dict.fromkeys(dont_translate + ["Q-Learning", "Transformer"]))

            return {
                "style": style,
                "glossary": glossary,
                "dont_translate": dont_translate,
                "notes": notes,
            }
        except (ValueError, JSONDecodeError) as exc:
            logger.warning(f"翻译指南解析失败（attempt={attempt + 1}/5）：{exc}")
            time.sleep(1)
        except Exception as exc:
            delay = _handle_sdk_exception(exc, attempt)
            if delay is None:
                raise
            time.sleep(delay)

    logger.warning("翻译指南生成失败，降级为空指南")
    return {"style": [], "glossary": {"agent": "智能体", "Agent": "智能体"}, "dont_translate": ["Q-Learning", "Transformer"], "notes": ""}


def _translate_single_with_guide(
    settings: Settings,
    info: str,
    guide: dict[str, Any],
    text: str,
    target_language: str,
) -> str:
    src = (text or "").strip()
    if not src:
        return ""

    guide_json = json.dumps(guide, ensure_ascii=False)
    system = (
        f"你是专业字幕翻译，目标语言：{target_language}。\n"
        f"{info}\n"
        f"翻译指南（JSON）：{guide_json}\n"
        "请只输出译文本身：\n"
        "- 不要包含“翻译”二字\n"
        "- 不要加引号\n"
        "- 不要换行\n"
        "- 不要解释\n"
        "硬性要求：agent -> 智能体；强化学习中写 `Q-Learning`（不要写 Queue Learning）；"
        "变压器指模型时保留 `Transformer`；数学公式用 plain text，不要 LaTeX。\n"
    )
    user = src

    backend = _get_thread_backend(settings)
    for attempt in range(30):
        try:
            cand = _chat_completion_text(backend, [{"role": "system", "content": system}, {"role": "user", "content": user}])
            cand = cand.replace("\n", "").strip()
            ok, processed = valid_translation(src, cand)
            if not ok:
                raise _TranslationValidationError(processed)
            return processed
        except _TranslationValidationError as exc:
            logger.warning(f"译文校验失败（attempt={attempt + 1}/30）：{exc}")
            time.sleep(0.5)
        except Exception as exc:
            delay = _handle_sdk_exception(exc, attempt)
            if delay is None:
                raise
            time.sleep(delay)

    raise RuntimeError("翻译失败：重试耗尽")


def _translate_chunk_with_guide(
    settings: Settings,
    info: str,
    guide: dict[str, Any],
    transcript: list[dict[str, Any]],
    indexes: list[int],
    target_language: str,
) -> dict[int, str]:
    payload: dict[str, str] = {}
    for i in indexes:
        payload[str(i)] = cast(str, transcript[i].get("text", "")).strip()

    # Skip empty-only chunks quickly.
    if not any(payload.values()):
        return {i: "" for i in indexes}

    guide_json = json.dumps(guide, ensure_ascii=False)
    system = (
        f"你是专业字幕翻译，目标语言：{target_language}。\n"
        f"{info}\n"
        f"翻译指南（JSON）：{guide_json}\n"
        "你会收到一个 JSON 对象：key 是句子编号（字符串），value 是原文。\n"
        "请只返回 JSON 对象，保持相同 key，value 只包含译文本身：\n"
        "- 不要包含“翻译”二字\n"
        "- 不要加引号\n"
        "- 不要换行\n"
        "- 不要解释\n"
        "硬性要求：agent -> 智能体；强化学习中写 `Q-Learning`（不要写 Queue Learning）；"
        "变压器指模型时保留 `Transformer`；数学公式用 plain text，不要 LaTeX。\n"
    )
    user = json.dumps(payload, ensure_ascii=False)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    backend = _get_thread_backend(settings)
    for attempt in range(10):
        try:
            content = _chat_completion_text(backend, messages)
            obj = _extract_first_json_object(content)
            if isinstance(obj.get("translations"), dict):
                obj = cast(dict[str, Any], obj["translations"])

            out: dict[int, str] = {}
            for i in indexes:
                src = payload.get(str(i), "")
                if not src:
                    out[i] = ""
                    continue
                v = obj.get(str(i), obj.get(i))
                if v is None:
                    continue
                cand = str(v).replace("\n", "").strip()
                ok, processed = valid_translation(src, cand)
                if not ok:
                    raise _TranslationValidationError(processed)
                out[i] = processed

            missing = [i for i in indexes if payload.get(str(i), "") and not out.get(i)]
            if missing:
                raise ValueError(f"missing translations: {missing[:10]}")
            return out
        except _TranslationValidationError as exc:
            logger.warning(f"分块译文校验失败（attempt={attempt + 1}/10）：{exc}")
            time.sleep(0.8)
        except (ValueError, JSONDecodeError) as exc:
            logger.warning(f"分块翻译解析失败（attempt={attempt + 1}/10）：{exc}")
            time.sleep(0.8)
        except Exception as exc:
            delay = _handle_sdk_exception(exc, attempt)
            if delay is None:
                raise
            time.sleep(delay)

    # Fallback: translate each line in this chunk sequentially (still no cross-chunk history).
    logger.warning(f"分块翻译失败，降级逐句翻译：indexes={indexes[:5]}.. (len={len(indexes)})")
    out: dict[int, str] = {}
    for i in indexes:
        src = payload.get(str(i), "")
        if not src:
            out[i] = ""
            continue
        out[i] = _translate_single_with_guide(settings, info, guide, src, target_language)
    return out


def _translate_content(
    summary: dict[str, Any],
    transcript: list[dict[str, Any]],
    target_language: str = "简体中文",
    settings: Settings | None = None,
) -> list[str]:
    cfg = settings or _DEFAULT_SETTINGS
    strategy = _normalize_translation_strategy(os.getenv("TRANSLATION_STRATEGY"))

    info = f'This is a video called "{summary.get("title", "")}". {summary.get("summary", "")}.'

    if strategy == "guide_parallel":
        max_workers = max(1, min(_read_int_env("TRANSLATION_MAX_CONCURRENCY", 4), 32))
        chunk_size = max(1, min(_read_int_env("TRANSLATION_CHUNK_SIZE", 8), 64))

        guide = _build_translation_guide(summary, transcript, target_language, settings=cfg)

        results: list[str] = [""] * len(transcript)
        chunks = [list(range(i, min(i + chunk_size, len(transcript)))) for i in range(0, len(transcript), chunk_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_translate_chunk_with_guide, cfg, info, guide, transcript, chunk, target_language): chunk
                for chunk in chunks
            }
            try:
                for future in as_completed(futures):
                    mapping = future.result()
                    for idx, tr in mapping.items():
                        if 0 <= idx < len(results):
                            results[idx] = tr
            except Exception:
                for f in futures:
                    f.cancel()
                raise

        return results

    backend = _build_chat_backend(cfg)
    full_translation: list[str] = []

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
    translation_path = os.path.join(folder, 'translation.json')
    summary_path = os.path.join(folder, 'summary.json')

    translation_ok = False
    if os.path.exists(translation_path):
        try:
            with open(translation_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            if not isinstance(existing, list) or not existing:
                raise ValueError("translation.json is not a non-empty list")
            for item in existing[:50]:
                if not isinstance(item, dict) or "translation" not in item:
                    raise ValueError("translation.json missing 'translation' field")
            translation_ok = True
        except Exception as exc:
            logger.warning(f"Invalid translation file, will regenerate: {translation_path} ({exc})")
            try:
                os.remove(translation_path)
            except Exception:
                pass
            translation_ok = False

    # If both translation + summary exist, we consider this step ready.
    if translation_ok and os.path.exists(summary_path):
        logger.info(f'Translation already exists in {folder}')
        return True

    info_path = os.path.join(folder, 'download.info.json')
    if not os.path.exists(info_path):
        logger.warning(f"Info file not found: {info_path}")
        return False
        
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    info = get_necessary_info(info)
    
    transcript_path = os.path.join(folder, 'transcript.json')
    if not os.path.exists(transcript_path):
        logger.warning(f"Transcript file not found: {transcript_path}")
        return False
        
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        except Exception:
            summary = summarize(info, transcript, target_language, settings=settings)
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
    else:
        summary = summarize(info, transcript, target_language, settings=settings)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # If we only needed to backfill summary.json, stop here.
    if translation_ok:
        return True
    
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
        if 'transcript.json' not in files:
            continue
        need = ('translation.json' not in files) or ('summary.json' not in files)
        if translate_folder(root, target_language, settings=settings):
            if need:
                count += 1
    msg = f'Translated all videos under {folder} (processed {count} files)'
    logger.info(msg)
    return msg
