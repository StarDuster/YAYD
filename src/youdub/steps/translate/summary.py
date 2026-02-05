from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any, cast

from loguru import logger

from ...config import Settings
from ...interrupts import check_cancelled, sleep_with_cancel
from .backend import _build_chat_backend, _chat_completion_text, _extract_first_json_object, _handle_sdk_exception


_DEFAULT_SETTINGS = Settings()


def get_necessary_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "description": info.get("description"),
        "upload_date": info.get("upload_date"),
        "categories": info.get("categories"),
        "tags": info.get("tags"),
    }


def ensure_transcript_length(transcript: str, max_length: int = 4000) -> str:
    if len(transcript) <= max_length:
        return transcript

    mid = len(transcript) // 2
    length = max_length // 2
    before = transcript[:mid]
    after = transcript[mid:]
    return before[:length] + after[-length:]


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

