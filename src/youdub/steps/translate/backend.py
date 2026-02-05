from __future__ import annotations

import json
import os
import threading
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

