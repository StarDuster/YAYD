from __future__ import annotations

import json
import os
import re
import threading
import time  # noqa: F401 - tests monkeypatch translate.time.sleep
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

from ..config import Settings
from ..interrupts import check_cancelled, sleep_with_cancel

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
        raise RuntimeError("缺少 OPENAI_API_KEY：请在 .env 中配置 OpenAI 兼容的 API Key。")
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
    raise ValueError("No JSON object found (模型输出中未找到 JSON 对象)。")


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
        logger.error(f"LLM认证失败: {exc}")
        return None
    if isinstance(exc, (BadRequestError,)):
        logger.error(f"LLM请求参数错误: {exc}")
        return None
    if isinstance(exc, (RateLimitError,)):
        delay = min(2 ** attempt, 30)
        logger.warning(f"LLM速率限制，{delay}秒后重试: {exc}")
        return float(delay)
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        delay = min(2 ** attempt, 20)
        logger.warning(f"LLM连接/超时，{delay}秒后重试: {exc}")
        return float(delay)
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status in {500, 502, 503, 504}:
            delay = min(2 ** attempt, 20)
            logger.warning(f"LLM服务器错误 ({status})，{delay}秒后重试: {exc}")
            return float(delay)
        logger.error(f"LLM请求失败 ({status}): {exc}")
        return None

    return None


def summarize(
    info: dict[str, Any],
    transcript: list[dict[str, Any]],
    target_language: str = "简体中文",
    settings: Settings | None = None,
) -> dict[str, Any]:
    check_cancelled()
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
        check_cancelled()
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
            logger.warning(f"摘要解析失败 (尝试={attempt + 1}/5): {exc}")
            sleep_with_cancel(1)
        except Exception as exc:  # SDK/network errors handled explicitly below
            delay = _handle_sdk_exception(exc, attempt)
            if delay is None:
                raise
            sleep_with_cancel(delay)

    if not summary_data:
        raise RuntimeError("Summary generation failed: Unable to parse JSON from model output.")

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
        check_cancelled()
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
            logger.warning(f"摘要翻译解析失败 (尝试={attempt + 1}/5): {exc}")
            sleep_with_cancel(1)
        except Exception as exc:
            delay = _handle_sdk_exception(exc, attempt)
            if delay is None:
                raise
            sleep_with_cancel(delay)

    raise RuntimeError("Summary translation failed: Unable to parse JSON from model output.")


def translation_postprocess(result: str) -> str:
    result = re.sub(r'\（[^）]*\）', '', result)
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

    if '翻译' in translation and ':"' in translation and '"' in translation:
        translation = translation.split(':"')[-1].split('"')[0]
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
    
    # Check if translation is too short (content lost during translation)
    # For longer texts (>100 chars), translation should be at least 15% of original length
    # Chinese is typically shorter than English, but <15% indicates significant content loss
    if len(text) > 100 and len(translation) < len(text) * 0.15:
        return False, 'The translation is too short, content may be lost. Please translate the complete sentence.'

    translation = translation.strip()
    
    # Newline is always forbidden in translation output
    if '\n' in translation:
        return False, "Don't include newlines in the translation. Only translate the following sentence and give me the result."
    
    # Check for explanation patterns that indicate LLM is explaining rather than translating
    # Use precise patterns instead of simple substring matching to avoid false positives
    # when the source text discusses translation, Chinese language, etc.
    
    explanation_patterns = [
        # "翻译" patterns - reject when used as meta-explanation, allow when part of content
        # e.g. reject "翻译：你好" or "翻译结果是你好", allow "机器翻译技术"
        (r'^翻译[：:]\s*', '翻译：'),
        (r'翻译(结果|如下|为|成)[：:]?\s*', '翻译结果/如下/为'),
        (r'(以下|下面)是?.{0,2}翻译', '以下是翻译'),
        
        # "中文/简体中文" patterns - reject meta-explanation, allow content discussion
        # e.g. reject "中文：你好" or "简体中文翻译：", allow "学习中文" or "中文版本"
        (r'^(简体)?中文[：:]\s*', '中文：'),
        (r'(简体)?中文翻译[：:]?\s*', '中文翻译：'),
        (r'翻译成?(简体)?中文[：:]?\s*', '翻译成中文：'),
        
        # "这句" patterns - reject explanation, allow normal translation of "this statement"
        (r'这句.{0,3}(的翻译|的意思|翻译成|意思是)', '这句的翻译/意思'),
        
        # English patterns - in Chinese translation, these usually indicate LLM explaining
        # e.g. "Translation: ..." or "Translate to Chinese: ..."
        (r'[Tt]ranslat(e|ion)[：:]\s*', 'Translation:'),
        (r'[Tt]ranslat(e|ion)\s+(to|into)\s+', 'Translate to'),
    ]
    
    for pattern, desc in explanation_patterns:
        if re.search(pattern, translation):
            return False, f"Don't include explanation patterns ({desc}) in the translation. Only give the translated result."

    return True, translation_postprocess(translation)


# --------------------------------------------------------------------------- #
# Whisper punctuation fix (before translation)
# --------------------------------------------------------------------------- #

_PUNCT_FIX_TRANSCRIPT_FILE = "transcript_punctuated.json"

# NOTE:
# Only allow changing these punctuation marks.
# Do NOT include "-" or "_" here: they are often part of tokens (e.g. Q-Learning, file_name).
_PUNCT_FIX_ALLOWED_CHARS = set(
    "，。！？；：、"
    ",.!?;:"
    "…"
    "“”‘’\"'"
    "（）()"
    "【】[]"
    "《》"
    "—–"
)


def _read_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    s = raw.strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _strip_punct_for_validation(s: str) -> str:
    if not s:
        return ""
    return "".join(ch for ch in s if ch not in _PUNCT_FIX_ALLOWED_CHARS)


def _lower_ascii_for_validation(s: str) -> str:
    """Lowercase ASCII letters only (keep everything else intact).

    Whisper transcripts for English often have inconsistent capitalization; allow the
    punctuation-fix step to correct ASCII casing without being rejected by validation.
    """
    if not s:
        return ""
    out: list[str] = []
    for ch in str(s):
        if "A" <= ch <= "Z":
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)


def _punct_fix_is_valid(src: str, out: str) -> bool:
    # Hard guarantee: only punctuation changes are allowed.
    # Compare the exact sequence of all non-punctuation characters (including spaces/newlines).
    # NOTE: we allow ASCII casing fixes (A-Z) for better readability on EN transcripts.
    a = _lower_ascii_for_validation(_strip_punct_for_validation(str(src)))
    b = _lower_ascii_for_validation(_strip_punct_for_validation(str(out)))
    return a == b


def _punct_fix_chunk(
    settings: Settings,
    payload: dict[str, str],
    *,
    attempt_limit: int = 5,
) -> tuple[dict[str, str], bool]:
    """
    Punctuate a JSON map {key: text} via LLM. Returns (mapping, ok).
    - ok=True: mapping validated for all keys (only punctuation changes)
    - ok=False: caller should treat as "best-effort failed" (mapping == original payload)
    """
    # Skip empty-only chunks quickly.
    if not any(str(v or "").strip() for v in payload.values()):
        return dict(payload), True

    # We fill results progressively; any key we can't validate falls back to the original.
    result: dict[str, str] = {}
    remaining: dict[str, str] = {}
    for k, src in payload.items():
        if not str(src or "").strip():
            # Preserve empties exactly.
            result[k] = str(src or "")
        else:
            remaining[k] = str(src)

    system = (
        "你是“转录文本修复专家”（Whisper 转写标点修复器）。你的任务：只通过【插入/删除/替换标点符号】来改善断句与可读性，"
        "避免一句话里逗号过多导致句子过长。\n"
        "\n"
        "硬性规则（必须遵守）：\n"
        "1) 除了标点符号以外，任何字符都不得改变：不得增删改任何汉字/字母/数字/大小写/空格/换行；词序必须完全保持。\n"
        "2) 允许修改的只有标点符号（中英文逗号句号问号叹号分号冒号顿号引号括号等）。\n"
        "3) 禁止：翻译、纠错、改写、润色、补词、删词、同义替换、调整措辞；禁止改变原有换行结构。\n"
        "4) 不要改变 URL / 邮箱 / 版本号 / 文件名 / 代码 token 内部的字符（例如 3.14、foo_bar、Q-Learning、https://...）。\n"
        "5) 断句目标：让句子更短更清晰。遇到“逗号串联很长”的情况，优先把部分逗号改为句号或分号；必要时插入句号/问号/叹号来断句。\n"
        "6) 自检（输出前必须做）：对每一条 value，将输入与输出的所有标点符号删除后必须完全一致；若无法满足，直接原样返回该条输入。\n"
        "\n"
        "输入是一个 JSON 对象：key 是字符串编号，value 是待处理文本。\n"
        "输出要求：只输出 JSON 对象本身，不要任何解释；必须包含与输入完全相同的所有 key。\n"
    )

    backend = _get_thread_backend(settings)
    had_failures = False
    last_parse_error: str | None = None
    for attempt in range(attempt_limit):
        check_cancelled()
        if not remaining:
            break

        user = json.dumps(remaining, ensure_ascii=False)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        try:
            content = _chat_completion_text(backend, messages)
            obj = _extract_first_json_object(content)
            # Allow wrappers like {"result": {...}} / {"outputs": {...}}
            for wrap_key in ("result", "results", "output", "outputs", "data"):
                if isinstance(obj.get(wrap_key), dict):
                    obj = cast(dict[str, Any], obj[wrap_key])
                    break

            invalid_keys: list[str] = []
            progressed = 0
            for k, src in remaining.items():
                v = obj.get(k, None)
                if v is None:
                    invalid_keys.append(k)
                    continue
                cand = str(v)
                if not _punct_fix_is_valid(src, cand):
                    invalid_keys.append(k)
                    continue
                result[k] = cand
                progressed += 1

            if invalid_keys:
                had_failures = True
                # If no key was accepted in this attempt, warn and stop retrying aggressively.
                # Retrying the exact same payload often wastes minutes and does not improve output.
                if progressed <= 0:
                    logger.warning(
                        f"标点修复校验失败 (尝试={attempt + 1}/{attempt_limit}): "
                        f"invalid_keys={invalid_keys[:5]} (total={len(invalid_keys)})"
                    )
                    break
                remaining = {k: remaining[k] for k in invalid_keys}
                sleep_with_cancel(0.2)
                continue

            remaining = {}
            break
        except (ValueError, JSONDecodeError) as exc:
            last_parse_error = str(exc)
            logger.warning(f"标点修复解析失败 (尝试={attempt + 1}/{attempt_limit}): {exc}")
            sleep_with_cancel(0.3)
        except Exception as exc:
            delay = _handle_sdk_exception(exc, attempt)
            if delay is None:
                had_failures = True
                break
            sleep_with_cancel(delay)

    # Any remaining keys fall back to original text.
    if remaining:
        had_failures = True
        # Avoid log spam; only mention parse errors when we never made progress.
        if last_parse_error and not result:
            logger.warning(f"标点修复失败，将回退到原文（原因: {last_parse_error}）")

    final = dict(payload)
    final.update(result)
    # Ensure strict contract: return must contain all keys.
    if set(final.keys()) != set(payload.keys()):
        # Extremely defensive: if something went wrong, return original.
        return dict(payload), False
    return final, (not had_failures)


def _load_or_create_punctuated_transcript(
    folder: str,
    transcript: list[dict[str, Any]],
    *,
    settings: Settings,
) -> list[dict[str, Any]]:
    """
    Best-effort:
    - If cached punctuated transcript exists and is newer than transcript.json, reuse it.
    - Otherwise, call LLM to fix punctuation (only punctuation allowed to change) and cache it.
    """
    if not transcript:
        return transcript

    # Allow disabling to save cost or for tests.
    if not _read_env_bool("YOUDUB_PUNCTUATION_FIX_BEFORE_TRANSLATE", True):
        return transcript

    src_path = os.path.join(folder, "transcript.json")
    punct_path = os.path.join(folder, _PUNCT_FIX_TRANSCRIPT_FILE)

    if os.path.exists(punct_path) and os.path.exists(src_path):
        try:
            if os.path.getmtime(punct_path) >= os.path.getmtime(src_path):
                with open(punct_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, list) and len(cached) == len(transcript) and all(
                    isinstance(it, dict) and isinstance(it.get("text"), str) for it in cached[:50]
                ):
                    return cast(list[dict[str, Any]], cached)
        except Exception:
            # Ignore and regenerate.
            pass

    try:
        out: list[dict[str, Any]] = [dict(it) for it in transcript]

        # Small batching: one call returns a JSON map of multiple segments.
        chunk_size = max(1, min(_read_int_env("PUNCTUATION_FIX_CHUNK_SIZE", 8), 128))
        attempt_limit = max(1, min(_read_int_env("PUNCTUATION_FIX_ATTEMPT_LIMIT", 2), 10))
        any_changed = False

        for i0 in range(0, len(out), chunk_size):
            check_cancelled()
            idxs = list(range(i0, min(i0 + chunk_size, len(out))))
            payload = {str(i): cast(str, out[i].get("text", "")) for i in idxs}
            fixed, _ok = _punct_fix_chunk(settings, payload, attempt_limit=attempt_limit)
            for i in idxs:
                src = payload[str(i)]
                new = fixed.get(str(i), src)
                if new != src:
                    any_changed = True
                out[i]["text"] = new

        # Cache only if we actually changed something (avoid caching "no-op due to failures").
        if any_changed:
            try:
                with open(punct_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
            except Exception as exc:
                logger.warning(f"写入标点修复缓存失败（忽略）: {exc}")

        return out
    except Exception as exc:  # pylint: disable=broad-except
        # Best-effort: never block translation due to punctuation fix.
        logger.warning(f"翻译前标点修复失败（忽略，继续翻译原始转写）: {exc}")
        return transcript


def _translate_single_text(
    text: str,
    history: list[dict],
    backend: str,
    fixed_message: list[dict],
    attempt_limit: int = 30,
    enable_fallback: bool = True
) -> str:
    """
    翻译单段文本，支持重试和自动拆分降级策略。
    """
    
    # 1. 尝试直接翻译
    for attempt in range(attempt_limit):
        check_cancelled()
        # Keep history short to avoid token limit
        current_history = history[-30:]
        messages = fixed_message + current_history + [
            {'role': 'user', 'content': f'Translate:"{text}"'}
        ]
        
        try:
            translation = _chat_completion_text(backend, messages).replace("\n", "")
            logger.info(f'原文: {text}')
            logger.info(f'译文: {translation}')
            
            is_valid, processed = valid_translation(text, translation)
            if not is_valid:
                raise _TranslationValidationError(processed)
            
            return processed

        except _TranslationValidationError as exc:
            logger.warning(f"翻译校验失败 (尝试={attempt + 1}/{attempt_limit}): {exc}")
            
            # 智能降级策略：如果是因为翻译太短（内容丢失），且允许 fallback，尝试拆分翻译
            if enable_fallback and "too short" in str(exc) and attempt == attempt_limit - 1:
                parts = _pack_source_text_chunks(_split_source_text_into_sentences(text))
                if len(parts) > 1:
                    logger.info(f"检测到内容丢失且重试无效，触发智能分段翻译策略 (分段数: {len(parts)})")
                    full_translation_parts = []
                    try:
                        temp_history = history.copy() # 使用临时 history 以保持分段间的上下文
                        for sub_text in parts:
                            # 递归调用，但禁止再次 fallback 防止死循环
                            sub_translation = _translate_single_text(
                                sub_text, 
                                temp_history, 
                                backend, 
                                fixed_message, 
                                attempt_limit=10, # 分段翻译重试次数可以少一点
                                enable_fallback=False
                            )
                            full_translation_parts.append(sub_translation)
                            # 更新临时上下文，让后续分段知道前面翻译了什么
                            temp_history.append({'role': 'user', 'content': f'Translate:"{sub_text}"'})
                            temp_history.append({'role': 'assistant', 'content': sub_translation})
                        
                        combined_translation = "".join(full_translation_parts)
                        logger.info(f"分段翻译合并成功: {combined_translation}")
                        return combined_translation
                    except Exception as fallback_exc:
                        logger.error(f"智能分段翻译失败: {fallback_exc}，将返回最后一次尝试的不完整结果")
                        # 如果分段翻译也挂了，就抛出最外层的异常或返回最后一次结果
                        pass

            sleep_with_cancel(0.5)
            
        except Exception as exc:
            delay = _handle_sdk_exception(exc, attempt)
            if delay is None:
                raise
            sleep_with_cancel(delay)

    # 如果所有尝试都失败，抛出异常或返回最后一次结果（由调用者决定是否捕获）
    # 这里为了保持行为一致，如果还是失败，我们记录错误并返回最后一次的 translation (如果有) 
    # 但在函数结构中很难获取最后一次 translation，所以重新抛出异常比较合适，
    # 或者我们约定：如果重试耗尽，返回空字符串或抛出异常。
    # 为了兼容旧逻辑（记录错误但继续），我们可以抛出一个特定的 MaxRetryError
    msg = f"达到最大重试次数，翻译失败: {text[:50]}..."
    logger.error(msg)
    # 返回最后一次尝试的翻译结果（即使是不完整的），避免程序崩溃，并在日志中标记
    if 'translation' in locals():
        return translation
    return ""


def split_text_into_sentences(para: str) -> list[str]:
    para = re.sub(r'([。！？\?])([^，。！？\?”’》])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub(r'(\.{6})([^，。！？\?”’》])', r"\1\n\2", para)  # 英文省略号
    para = re.sub(r'(\…{2})([^，。！？\?”’》])', r"\1\n\2", para)  # 中文省略号
    para = re.sub(r'([。！？\?][”’])([^，。！？\?”’》])', r'\1\n\2', para)
    para = para.rstrip()
    return para.split("\n")


def _split_source_text_into_sentences(text: str) -> list[str]:
    s = " ".join((text or "").split()).strip()
    if not s:
        return []
    # Chinese punctuation
    s = re.sub(r"([。！？])([^”’])", r"\1\n\2", s)
    s = re.sub(r"([。！？][”’])([^，。！？])", r"\1\n\2", s)
    # English punctuation: split on .!? followed by whitespace
    s = re.sub(r"([.!?][\"”’']?)\s+(?=\S)", r"\1\n", s)
    out = [x.strip() for x in s.splitlines() if x.strip()]
    return out or ([s] if s else [])


def _pack_source_text_chunks(
    sentences: list[str],
    *,
    max_chars: int = 220,
    min_chars: int = 80,
) -> list[str]:
    """Pack sentence list into moderate-sized chunks for fallback translation."""
    out: list[str] = []
    buf = ""
    for s in [x.strip() for x in sentences if x and x.strip()]:
        if not buf:
            buf = s
            continue
        # Keep chunks reasonably sized: merge short pieces, avoid over-long.
        if len(buf) < min_chars or len(buf) + 1 + len(s) <= max_chars:
            buf = f"{buf} {s}".strip()
        else:
            out.append(buf)
            buf = s
    if buf:
        out.append(buf)
    return out


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _split_source_text_relaxed(text: str) -> list[str]:
    """
    A more aggressive splitter for subtitle alignment.

    Why: ASR sometimes produces very long "sentences" without .!?; in bilingual subtitle mode,
    we still want the source text to be split into smaller readable chunks rather than
    repeating the full paragraph on every translated sentence.
    """
    s = _normalize_ws(text)
    if not s:
        return []

    # English punctuation (comma/semicolon/colon) when followed by whitespace.
    s = re.sub(r"([,;:])\s+(?=\S)", r"\1\n", s)
    # Chinese comma/semicolon/colon (often no whitespace after).
    s = re.sub(r"([，；：])([^，；：])", r"\1\n\2", s)
    # Long dash variants commonly used as clause separator.
    s = re.sub(r"([—–])\s+(?=\S)", r"\1\n", s)

    out = [x.strip() for x in s.splitlines() if x.strip()]
    return out or ([s] if s else [])


def _merge_parts_to_count(parts: list[str], target_count: int) -> list[str]:
    if target_count <= 0:
        return []
    cleaned = [p.strip() for p in parts if p and p.strip()]
    if not cleaned:
        return [""] * target_count
    if len(cleaned) == target_count:
        return cleaned
    if len(cleaned) < target_count:
        return cleaned + [""] * (target_count - len(cleaned))

    n = len(cleaned)
    out: list[str] = []
    for i in range(target_count):
        a = i * n // target_count
        b = (i + 1) * n // target_count
        out.append(" ".join(cleaned[a:b]).strip())
    return out


def _split_words_to_count(text: str, target_count: int) -> list[str]:
    if target_count <= 0:
        return []
    s = _normalize_ws(text)
    if not s:
        return [""] * target_count
    if target_count == 1:
        return [s]

    words = s.split(" ")
    if not words:
        return [""] * target_count

    base = len(words) // target_count
    rem = len(words) % target_count
    out: list[str] = []
    idx = 0
    for i in range(target_count):
        size = base + (1 if i < rem else 0)
        if size <= 0:
            out.append("")
            continue
        out.append(" ".join(words[idx : idx + size]).strip())
        idx += size
    return out


def _map_source_text_to_translation_count(text: str, target_count: int) -> list[str]:
    """
    Best-effort mapping for bilingual subtitles.

    Goal: produce exactly `target_count` source chunks with minimal duplication.
    """
    if target_count <= 0:
        return []

    src_text = _normalize_ws(text)
    if not src_text:
        return [""] * target_count

    strict = [x.strip() for x in _split_source_text_into_sentences(src_text) if x and x.strip()]
    if not strict:
        strict = [src_text]

    if len(strict) >= target_count:
        return _merge_parts_to_count(strict, target_count)

    # Not enough strict sentences: try a relaxed split first.
    relaxed = [x.strip() for x in _split_source_text_relaxed(src_text) if x and x.strip()]
    if relaxed and len(relaxed) >= target_count:
        return _merge_parts_to_count(relaxed, target_count)

    # Still not enough: fallback to word-chunking the full text to avoid repeating paragraphs.
    return _split_words_to_count(src_text, target_count)


def split_sentences(translation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_data = []
    for item in translation:
        start = item['start']
        end = item['end']
        text = item['text']
        speaker = item['speaker']
        translation_text = item.get('translation', '')
        sentences = [s.strip() for s in split_text_into_sentences(str(translation_text or "")) if str(s).strip()]
        if not sentences:
            sentences = [str(translation_text or "").strip()] if str(translation_text or "").strip() else [""]

        mapped_src = _map_source_text_to_translation_count(str(text or ""), len(sentences))
        
        if not translation_text or not sentences:
            # Handle empty translation
            duration_per_char = 0
        else:
            # Use actual split sentence lengths to avoid gaps from rstrip/processing
            total_chars = sum(len(s) for s in sentences)
            if total_chars > 0:
                duration_per_char = (end - start) / total_chars
            else:
                duration_per_char = 0
             
        for i, sentence in enumerate(sentences):
            sentence_len = len(sentence)
            sentence_end = start + duration_per_char * sentence_len
            
            # Ensure the last sentence ends exactly at the original segment end
            is_last = (i == len(sentences) - 1)
            if is_last:
                sentence_end = end

            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": (mapped_src[i] if mapped_src else text),
                "speaker": speaker,
                "translation": sentence
            })

            start = sentence_end
            
    return output_data


def _assign_speakers_by_overlap(
    segments: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    *,
    default_speaker: str = "SPEAKER_00",
) -> bool:
    """Assign `segments[*].speaker` by maximizing time overlap with `turns`.

    This is used to refresh `translation.json` speakers after re-diarization updates
    `transcript.json`, without re-translating text.
    """
    if not segments:
        return False

    if not turns:
        changed = False
        for seg in segments:
            check_cancelled()
            old = str(seg.get("speaker") or "")
            if old != str(default_speaker):
                changed = True
            seg["speaker"] = str(default_speaker)
        return changed

    turns_sorted = sorted(turns, key=lambda x: float(x.get("start", 0.0) or 0.0))
    idx = 0
    changed = False

    for seg in segments:
        check_cancelled()
        try:
            seg_start = float(seg.get("start", 0.0) or 0.0)
            seg_end = float(seg.get("end", seg_start) or seg_start)
        except Exception:
            seg_start, seg_end = 0.0, 0.0

        if seg_end <= seg_start:
            new_speaker = str(default_speaker)
        else:
            while idx < len(turns_sorted) and float(turns_sorted[idx].get("end", 0.0) or 0.0) <= seg_start:
                check_cancelled()
                idx += 1

            best_speaker: str | None = None
            best_overlap = 0.0
            j = idx
            while j < len(turns_sorted) and float(turns_sorted[j].get("start", 0.0) or 0.0) < seg_end:
                check_cancelled()
                t = turns_sorted[j]
                try:
                    t_start = float(t.get("start", 0.0) or 0.0)
                    t_end = float(t.get("end", 0.0) or 0.0)
                except Exception:
                    j += 1
                    continue

                ov = max(0.0, min(seg_end, t_end) - max(seg_start, t_start))
                if ov > best_overlap:
                    best_overlap = ov
                    best_speaker = str(t.get("speaker") or default_speaker)
                j += 1

            new_speaker = best_speaker if best_speaker is not None else str(default_speaker)

        old_speaker = str(seg.get("speaker") or str(default_speaker))
        if old_speaker != str(new_speaker):
            changed = True
        seg["speaker"] = str(new_speaker)

    return changed


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
    s = (raw or "parallel").strip().lower()
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
        "- dont_translate：必须原样保留的 token（如代码/公式/缩写/产品名/专业术语）\n"
        "- notes：额外注意事项\n"
        "硬性要求：\n"
        "- 尽可能保留专业术语原文不翻译（如 API、GPU、CPU、RGB、FFT、MFCC、CNN、RNN、LSTM 等）\n"
        "- 长度控制：目标中文字数 ≈ 英文单词数 × 1.6（为了匹配配音时长，避免过短）\n"
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
        check_cancelled()
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
            logger.warning(f"翻译指南解析失败 (尝试={attempt + 1}/5): {exc}")
            sleep_with_cancel(1)
        except Exception as exc:
            delay = _handle_sdk_exception(exc, attempt)
            if delay is None:
                raise
            sleep_with_cancel(delay)

    logger.warning("翻译指南生成失败，回退到空指南")
    return {"style": [], "glossary": {"agent": "智能体", "Agent": "智能体"}, "dont_translate": ["Q-Learning", "Transformer"], "notes": ""}


def _build_global_glossary(
    summary: dict[str, Any],
    transcript: list[dict[str, Any]],
    target_language: str,
    settings: Settings,
) -> dict[str, Any]:
    """
    通过一次 LLM 调用获得全局术语表（glossary + dont_translate），用于后续所有翻译 prompt。
    返回格式：{"glossary": {term: translation}, "dont_translate": [...], "notes": "..."}。
    """
    max_chars = max(800, _read_int_env("TRANSLATION_GLOSSARY_MAX_CHARS", 3000))
    max_terms = max(10, min(_read_int_env("TRANSLATION_GLOSSARY_MAX_TERMS", 40), 120))

    transcript_text = " ".join(cast(str, line.get("text", "")) for line in transcript).strip()
    transcript_text = ensure_transcript_length(transcript_text, max_length=max_chars)

    title = str(summary.get("title", "")).strip()
    summary_text = str(summary.get("summary", "")).strip()

    system = (
        f"你是专业译者，目标语言：{target_language}。\n"
        "请基于给定的视频信息与字幕样本，抽取一份“全局术语表”，用于后续字幕翻译保持一致。\n"
        "必须只输出 JSON（不要任何额外文本）。\n"
        'JSON 格式：{"glossary": {"term": "translation"}, "dont_translate": ["..."], "notes": "..."}\n'
        f"- glossary：只收录重要专有名词/术语/人名/机构/产品名/库名/函数名等，最多 {max_terms} 条\n"
        "- dont_translate：必须原样保留的 token（如代码/公式/缩写/产品名/专业术语）\n"
        "- notes：额外注意事项（可为空字符串）\n"
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
    try:
        content = _chat_completion_text(backend, [{"role": "system", "content": system}, {"role": "user", "content": user}])
        obj = _extract_first_json_object(content)

        # Allow model to wrap glossary in "terminology"/"translations" etc.
        if isinstance(obj.get("terminology"), dict):
            obj = cast(dict[str, Any], obj["terminology"])

        glossary_raw = obj.get("glossary", obj)
        glossary: dict[str, str] = {}
        if isinstance(glossary_raw, dict):
            for k, v in glossary_raw.items():
                ks = str(k).strip()
                vs = str(v).strip()
                if ks and vs and ks not in {"dont_translate", "notes"}:
                    glossary[ks] = vs

        dont_raw = obj.get("dont_translate", [])
        if isinstance(dont_raw, str):
            dont_translate = [dont_raw.strip()] if dont_raw.strip() else []
        elif isinstance(dont_raw, list):
            dont_translate = [str(x).strip() for x in dont_raw if str(x).strip()]
        else:
            dont_translate = []

        notes = str(obj.get("notes", "")).strip()

        # Always enforce core term preferences even if the glossary omitted them.
        glossary.setdefault("agent", "智能体")
        glossary.setdefault("Agent", "智能体")
        dont_translate = list(dict.fromkeys(dont_translate + ["Q-Learning", "Transformer"]))

        return {"glossary": glossary, "dont_translate": dont_translate, "notes": notes}
    except Exception as exc:
        logger.warning(f"全局术语表生成失败，回退到最小术语表: {exc}")
        return {"glossary": {"agent": "智能体", "Agent": "智能体"}, "dont_translate": ["Q-Learning", "Transformer"], "notes": ""}


def _translate_single_with_guide(
    settings: Settings,
    info: str,
    guide: dict[str, Any],
    text: str,
    target_language: str,
    attempt_limit: int = 30,
    enable_fallback: bool = True,
) -> str:
    src = (text or "").strip()
    if not src:
        return ""

    guide_json = json.dumps(guide, ensure_ascii=False)
    system = (
        f"你是专业字幕翻译，目标语言：{target_language}。\n"
        f"{info}\n"
        f"指南（JSON）：{guide_json}\n"
        "请只输出译文本身：\n"
        "- 尽可能保留专业术语原文不翻译（如 API、GPU、CPU、RGB、FFT、MFCC、CNN、RNN、LSTM 等）\n"
        "- 长度控制：目标中文字数 ≈ 英文单词数 × 1.6（为了匹配配音时长，避免过短）\n"
        "- 标点符号：根据原文语气合理使用标点（疑问用？、强调用！、停顿用……），保持专业，不要过度口语化\n"
        "- 不要包含“翻译”二字\n"
        "- 不要加引号\n"
        "- 不要换行\n"
        "- 不要解释\n"
        "- 必须完整翻译原文所有信息，不要漏译或省略\n"
    )
    user = src

    backend = _get_thread_backend(settings)
    for attempt in range(attempt_limit):
        check_cancelled()
        try:
            cand = _chat_completion_text(backend, [{"role": "system", "content": system}, {"role": "user", "content": user}])
            cand = cand.replace("\n", "").strip()
            ok, processed = valid_translation(src, cand)
            if not ok:
                raise _TranslationValidationError(processed)
            return processed
        except _TranslationValidationError as exc:
            logger.warning(f"翻译校验失败 (尝试={attempt + 1}/{attempt_limit}): {exc}")
            if enable_fallback and "too short" in str(exc) and attempt == attempt_limit - 1:
                parts = _pack_source_text_chunks(_split_source_text_into_sentences(src))
                if len(parts) > 1:
                    logger.info(f"检测到内容丢失且重试无效，触发智能分段翻译策略 (分段数: {len(parts)})")
                    translated_parts: list[str] = []
                    for p in parts:
                        translated_parts.append(
                            _translate_single_with_guide(
                                settings,
                                info,
                                guide,
                                p,
                                target_language,
                                attempt_limit=10,
                                enable_fallback=False,
                            )
                        )
                    combined = translation_postprocess("".join(translated_parts))
                    # combined is already per-part validated; return directly.
                    return combined
            sleep_with_cancel(0.5)
        except Exception as exc:
            delay = _handle_sdk_exception(exc, attempt)
            if delay is None:
                raise
            sleep_with_cancel(delay)

    raise RuntimeError("翻译失败：重试次数已耗尽。")


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
        f"指南（JSON）：{guide_json}\n"
        "你会收到一个 JSON 对象：key 是句子编号（字符串），value 是原文。\n"
        "请只返回 JSON 对象，保持相同 key，value 只包含译文本身：\n"
        "- 尽可能保留专业术语原文不翻译（如 API、GPU、CPU、RGB、FFT、MFCC、CNN、RNN、LSTM 等）\n"
        "- 长度控制：目标中文字数 ≈ 英文单词数 × 1.6（为了匹配配音时长，避免过短）\n"
        "- 标点符号：根据原文语气合理使用标点（疑问用？、强调用！、停顿用……），保持专业，不要过度口语化\n"
        "- 不要包含“翻译”二字\n"
        "- 不要加引号\n"
        "- 不要换行\n"
        "- 不要解释\n"
        "- 必须完整翻译原文所有信息，不要漏译或省略\n"
        "硬性要求：agent -> 智能体；强化学习中写 `Q-Learning`（不要写 Queue Learning）；"
        "变压器指模型时保留 `Transformer`；数学公式用 plain text，不要 LaTeX。\n"
    )
    user = json.dumps(payload, ensure_ascii=False)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    backend = _get_thread_backend(settings)
    for attempt in range(10):
        check_cancelled()
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
            logger.warning(f"块翻译校验失败 (尝试={attempt + 1}/10): {exc}")
            sleep_with_cancel(0.8)
        except (ValueError, JSONDecodeError) as exc:
            logger.warning(f"块翻译解析失败 (尝试={attempt + 1}/10): {exc}")
            sleep_with_cancel(0.8)
        except Exception as exc:
            delay = _handle_sdk_exception(exc, attempt)
            if delay is None:
                raise
            sleep_with_cancel(delay)

    # Fallback: translate each line in this chunk sequentially (still no cross-chunk history).
    logger.warning(f"块翻译失败，回退到逐句翻译: indexes={indexes[:5]}.. (len={len(indexes)})")
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
    # NOTE:
    # - guide_parallel 会用一次额外的 LLM 调用生成全局术语表，给“并发分块翻译”提供一致性约束；
    # - history 串行策略本身依赖上下文累计来保持一致性，这里默认用最小术语表，避免额外一次 LLM 调用
    #   （也方便单元测试只关注“history 是否带上上一轮输出”）。
    if strategy == "guide_parallel":
        terminology = _build_global_glossary(summary, transcript, target_language, settings=cfg)
    else:
        terminology = {"glossary": {"agent": "智能体", "Agent": "智能体"}, "dont_translate": ["Q-Learning", "Transformer"], "notes": ""}
    terminology_json = json.dumps(terminology, ensure_ascii=False)

    if strategy == "guide_parallel":
        max_workers = max(1, min(_read_int_env("TRANSLATION_MAX_CONCURRENCY", 4), 32))
        chunk_size = max(1, min(_read_int_env("TRANSLATION_CHUNK_SIZE", 8), 64))

        guide = _build_translation_guide(summary, transcript, target_language, settings=cfg)
        # Merge global terminology into guide (guide wins on conflicts).
        try:
            g0 = cast(dict[str, str], terminology.get("glossary", {})) if isinstance(terminology.get("glossary"), dict) else {}
            g1 = cast(dict[str, str], guide.get("glossary", {})) if isinstance(guide.get("glossary"), dict) else {}
            merged_glossary = dict(g0)
            merged_glossary.update(g1)
            guide["glossary"] = merged_glossary

            dt0 = terminology.get("dont_translate", [])
            dt1 = guide.get("dont_translate", [])
            dts: list[str] = []
            for arr in (dt0, dt1):
                if isinstance(arr, str) and arr.strip():
                    dts.append(arr.strip())
                elif isinstance(arr, list):
                    dts.extend([str(x).strip() for x in arr if str(x).strip()])
            guide["dont_translate"] = list(dict.fromkeys(dts))
            if not str(guide.get("notes", "")).strip() and str(terminology.get("notes", "")).strip():
                guide["notes"] = str(terminology.get("notes", "")).strip()
        except Exception:
            pass

        results: list[str] = [""] * len(transcript)
        chunks = [list(range(i, min(i + chunk_size, len(transcript)))) for i in range(0, len(transcript), chunk_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_translate_chunk_with_guide, cfg, info, guide, transcript, chunk, target_language): chunk
                for chunk in chunks
            }
            try:
                for future in as_completed(futures):
                    check_cancelled()
                    mapping = future.result()
                    for idx, tr in mapping.items():
                        if 0 <= idx < len(results):
                            results[idx] = tr
                            src = cast(str, transcript[idx].get("text", ""))
                            logger.info(f"原文: {src}")
                            logger.info(f"译文: {tr}")
            except BaseException:
                for f in futures:
                    f.cancel()
                raise

        return results

    backend = _build_chat_backend(cfg)
    full_translation: list[str] = []

    fixed_message = [
        {'role': 'system', 'content': f'You are a expert in the field of this video.\n{info}\nTranslate the sentence into {target_language}.\n全局术语表（JSON）：{terminology_json}\n翻译要求：下面我让你来充当翻译家，你的目标是把任何语言翻译成中文，请翻译时不要带翻译腔，而是要翻译得自然、流畅和地道，使用优美和高雅的表达方式。尽可能保留专业术语原文不翻译（如 API、GPU、CPU、RGB、FFT、MFCC、CNN、RNN、LSTM 等）。长度控制：目标中文字数 ≈ 英文单词数 × 1.6（为了匹配配音时长，避免过短）。标点符号：根据原文语气合理使用标点（疑问用？、强调用！、停顿用……），保持专业，不要过度口语化。请严格参考全局术语表：glossary 中出现的术语优先按对应译法，dont_translate 中的 token 必须原样保留。请将人工智能的“agent”翻译为“智能体”，强化学习中是`Q-Learning`而不是`Queue Learning`。数学公式写成plain text，不要使用latex。确保翻译完整，不要漏译或省略。注意信达雅。只输出译文，不要包含“翻译”二字。'},
        {'role': 'user', 'content': 'Translate:"Knowledge is power."'},
        {'role': 'assistant', 'content': '知识就是力量。'},
        {'role': 'user', 'content': 'Translate:"To be or not to be, that is the question."'},
        {'role': 'assistant', 'content': '生存还是毁灭，这是一个值得考虑的问题。'},
    ]

    history: list[dict[str, str]] = []
    for line in transcript:
        check_cancelled()
        text = cast(str, line.get("text", ""))
        translation = _translate_single_text(
            text=text,
            history=cast(list[dict], history),
            backend=backend,
            fixed_message=cast(list[dict], fixed_message),
            attempt_limit=30,
            enable_fallback=True,
        )
        
        full_translation.append(translation)
        
        history.append({'role': 'user', 'content': f'Translate:"{text}"'})
        history.append({'role': 'assistant', 'content': f'{translation}'})
        # Avoid rate limits
        sleep_with_cancel(0.1)

    return full_translation


def translate_folder(folder: str, target_language: str = '简体中文', settings: Settings | None = None) -> bool:
    check_cancelled()
    translation_path = os.path.join(folder, 'translation.json')
    translation_raw_path = os.path.join(folder, 'translation_raw.json')
    summary_path = os.path.join(folder, 'summary.json')
    transcript_path = os.path.join(folder, 'transcript.json')

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
            logger.warning(f"翻译文件无效，将重新生成: {translation_path} ({exc})")
            try:
                os.remove(translation_path)
            except Exception:
                pass
            translation_ok = False

    def _safe_mtime(path: str) -> float | None:
        try:
            return float(os.path.getmtime(path))
        except Exception:
            return None

    # If both translation + summary exist, we can skip *only if* transcript hasn't changed.
    if translation_ok and os.path.exists(summary_path):
        check_cancelled()
        tr_mtime = _safe_mtime(transcript_path) if os.path.exists(transcript_path) else None
        tl_mtime = _safe_mtime(translation_path)
        transcript_newer = bool(
            (tr_mtime is not None)
            and (tl_mtime is not None)
            and (float(tr_mtime) > (float(tl_mtime) + 1e-6))
        )
        transcript_changed = transcript_newer

        # Defensive: some filesystems have coarse mtime; also catch transcript length mismatch.
        if (not transcript_changed) and os.path.exists(transcript_path) and os.path.exists(translation_raw_path):
            try:
                with open(transcript_path, "r", encoding="utf-8") as f:
                    tr_obj = json.load(f)
                with open(translation_raw_path, "r", encoding="utf-8") as f:
                    raw_obj = json.load(f)
                if isinstance(tr_obj, list) and isinstance(raw_obj, list) and len(tr_obj) != len(raw_obj):
                    transcript_changed = True
            except Exception:
                pass

        if not transcript_changed:
            if not os.path.exists(translation_raw_path):
                logger.warning(
                    f"检测到 {translation_path} 已存在，但缺少对轴前翻译文件: {translation_raw_path}（不会自动重翻译）"
                )
            logger.info(f'翻译已存在于 {folder}')
            return True

        # transcript.json is newer: refresh speakers (and timing) without re-translation when possible.
        if not os.path.exists(transcript_path):
            logger.warning(f"检测到翻译已存在但缺少转录文件，无法刷新 speaker：{transcript_path}")
            return True

        must_retranslate = False
        transcript_obj: Any = None
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_obj = json.load(f)
        except Exception as exc:
            logger.warning(f"读取转录失败，无法刷新 speaker：{transcript_path} ({exc})")
            return True

        if not isinstance(transcript_obj, list):
            logger.warning(f"转录文件结构异常，无法刷新 speaker：{transcript_path}")
            return True

        # If we have translation_raw (1:1 with transcript), we can fully realign without model calls.
        if os.path.exists(translation_raw_path):
            try:
                with open(translation_raw_path, 'r', encoding='utf-8') as f:
                    raw_obj = json.load(f)
            except Exception:
                raw_obj = None

            if isinstance(raw_obj, list) and len(raw_obj) == len(transcript_obj):
                refreshed_raw: list[dict[str, Any]] = []
                for i, tr_line in enumerate(transcript_obj):
                    check_cancelled()
                    tr_d = tr_line if isinstance(tr_line, dict) else {}
                    prev = raw_obj[i] if i < len(raw_obj) else {}
                    prev_d = prev if isinstance(prev, dict) else {}
                    out = dict(prev_d)
                    out["start"] = tr_d.get("start", out.get("start", 0.0))
                    out["end"] = tr_d.get("end", out.get("end", 0.0))
                    out["text"] = tr_d.get("text", out.get("text", ""))
                    out["speaker"] = tr_d.get("speaker", out.get("speaker", "SPEAKER_00")) or "SPEAKER_00"
                    if "translation" not in out:
                        out["translation"] = ""
                    refreshed_raw.append(out)

                # Write refreshed raw + re-split to refresh translation.json speaker labels/timing.
                try:
                    with open(translation_raw_path, 'w', encoding='utf-8') as f:
                        json.dump(refreshed_raw, f, indent=2, ensure_ascii=False)
                except Exception as exc:
                    logger.warning(f"写入 translation_raw.json 失败（忽略）: {exc}")

                refreshed = split_sentences(refreshed_raw)
                try:
                    with open(translation_path, 'w', encoding='utf-8') as f:
                        json.dump(refreshed, f, indent=2, ensure_ascii=False)
                    logger.info(f"检测到转录已更新，已刷新 translation.json speaker/时间轴（无需重翻译）: {folder}")
                except Exception as exc:
                    logger.warning(f"写入刷新后的 translation.json 失败（忽略）: {exc}")
                return True

            # transcript length changed -> must retranslate (keep summary.json).
            logger.info(f"检测到转录长度变化，将重新翻译: {folder}")
            translation_ok = False
            must_retranslate = True

        if not must_retranslate:
            # No translation_raw: best-effort refresh speakers in existing translation.json by overlap.
            try:
                with open(translation_path, 'r', encoding='utf-8') as f:
                    aligned = json.load(f)
            except Exception:
                aligned = None

            if isinstance(aligned, list) and aligned:
                turns = []
                for tr_line in transcript_obj:
                    check_cancelled()
                    if not isinstance(tr_line, dict):
                        continue
                    try:
                        st = float(tr_line.get("start", 0.0) or 0.0)
                        ed = float(tr_line.get("end", 0.0) or 0.0)
                    except Exception:
                        continue
                    if ed <= st:
                        continue
                    turns.append(
                        {
                            "start": st,
                            "end": ed,
                            "speaker": str(tr_line.get("speaker") or "SPEAKER_00"),
                        }
                    )

                changed = _assign_speakers_by_overlap(
                    cast(list[dict[str, Any]], aligned), turns, default_speaker="SPEAKER_00"
                )
                if changed:
                    try:
                        with open(translation_path, 'w', encoding='utf-8') as f:
                            json.dump(aligned, f, indent=2, ensure_ascii=False)
                        logger.info(f"检测到转录已更新，已刷新 translation.json speaker（缺少translation_raw，未重翻译）: {folder}")
                    except Exception as exc:
                        logger.warning(f"写入刷新后的 translation.json 失败（忽略）: {exc}")
                else:
                    logger.info(f"检测到转录已更新，但 translation.json speaker 无变化: {folder}")
                return True

            logger.warning(
                f"检测到转录已更新，但无法加载现有 translation.json 进行 speaker 刷新，将继续执行翻译流程: {folder}"
            )

    info_path = os.path.join(folder, 'download.info.json')
    if not os.path.exists(info_path):
        logger.warning(f"未找到信息文件: {info_path}")
        return False
        
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    info = get_necessary_info(info)
    
    if not os.path.exists(transcript_path):
        logger.warning(f"未找到转录文件: {transcript_path}")
        return False
        
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    
    cfg = settings or _DEFAULT_SETTINGS
    translation_model = cfg.model_name
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            # Backfill translation_model if missing
            if "translation_model" not in summary:
                summary["translation_model"] = translation_model
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception:
            summary = summarize(info, transcript, target_language, settings=settings)
            summary["translation_model"] = translation_model
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
    else:
        summary = summarize(info, transcript, target_language, settings=settings)
        summary["translation_model"] = translation_model
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # If we only needed to backfill summary.json, stop here.
    if translation_ok:
        return True
    
    # Punctuate transcript before translating (strictly punctuation-only changes).
    check_cancelled()
    transcript = _load_or_create_punctuated_transcript(
        str(folder),
        cast(list[dict[str, Any]], transcript),
        settings=cfg,
    )

    # Perform translation
    check_cancelled()
    translations = _translate_content(summary, transcript, target_language, settings=settings)
    
    for i, line in enumerate(transcript):
        line['translation'] = translations[i] if i < len(translations) else ""

    # Save pre-alignment translation (1:1 with transcript.json).
    with open(translation_raw_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
        
    transcript = split_sentences(transcript)
    
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
        
    return True


def translate_all_transcript_under_folder(folder: str, target_language: str, settings: Settings | None = None) -> str:
    count = 0
    skipped = 0
    no_transcript = 0
    for root, dirs, files in os.walk(folder):
        check_cancelled()
        if 'transcript.json' not in files:
            # 只统计有 download.info.json 的文件夹（说明是视频目录）
            if 'download.info.json' in files:
                no_transcript += 1
            continue
        need = ('translation.json' not in files) or ('summary.json' not in files)
        if translate_folder(root, target_language, settings=settings):
            if need:
                count += 1
            else:
                skipped += 1
    parts = []
    if count > 0:
        parts.append(f"翻译 {count} 个")
    if skipped > 0:
        parts.append(f"跳过 {skipped} 个")
    if no_transcript > 0:
        parts.append(f"缺少转录 {no_transcript} 个")
    if parts:
        msg = f"翻译完成: {folder}（{', '.join(parts)}）"
    else:
        msg = f"翻译完成: {folder}（无可处理文件）"
    logger.info(msg)
    return msg
