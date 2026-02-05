from __future__ import annotations

import json
import os
from json import JSONDecodeError
from typing import Any, cast

from loguru import logger

from ..config import Settings
from ..interrupts import check_cancelled, sleep_with_cancel

_PUNCT_FIX_TRANSCRIPT_FILE = "transcript_punctuated.json"


def _read_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _punct_fix_chunk(
    settings: Settings,
    payload: dict[str, str],
    *,
    attempt_limit: int = 5,
) -> tuple[dict[str, str], bool]:
    """
    Refine transcript chunk via LLM: fix ASR errors and improve punctuation.
    Returns (mapping, ok).
    """
    # NOTE: Keep `translate` helpers as the source of truth so tests can monkeypatch
    # `translate._chat_completion_text` and intercept model calls.
    from . import translate as tr

    # Skip empty-only chunks quickly.
    if not any(str(v or "").strip() for v in payload.values()):
        return dict(payload), True

    # We fill results progressively.
    result: dict[str, str] = {}
    remaining: dict[str, str] = {}
    for k, src in payload.items():
        if not str(src or "").strip():
            # Preserve empties exactly.
            result[k] = str(src or "")
        else:
            remaining[k] = str(src)

    system = (
        "你是专业的 ASR（语音识别）转录文本修复专家。你的任务是修复 Whisper 转录文本中的显著错误并优化标点符号，"
        "使其更符合人类阅读习惯。\n"
        "\n"
        "请遵循以下原则：\n"
        "1. **修正转录错误**：仅在发现明显的拼写错误、同音词混淆（如 \"their\" vs \"there\"）、无意义的重复词（如 \"the the\"）"
        "或上下文完全不通顺的词汇时进行修正。对于专有名词、代码 token、版本号等，请保持谨慎，不要随意修改。\n"
        "2. **优化标点符号**：\n"
        "   - **长句断句**：Whisper 有时会输出非常长的句子（run-on sentences）。请在语法和语义合适的位置插入句号（. 或 。）"
        "或分号（; 或 ；），将长句拆分为更短、更易读的句子。\n"
        "   - **补充语气标点**：根据句子语气补充缺失的问号（?）、感叹号（!）或省略号（...）。\n"
        "   - **保持风格**：尽量沿用原文的标点风格（全角/半角）。\n"
        "3. **保持原意**：不要进行改写、润色或总结，必须保留原文的完整语义。\n"
        "\n"
        "输入是一个 JSON 对象：key 是字符串编号，value 是待处理文本。\n"
        "输出要求：只输出 JSON 对象本身，不要任何解释；必须包含与输入完全相同的所有 key。\n"
    )

    backend = tr._get_thread_backend(settings)  # noqa: SLF001
    had_failures = False
    last_parse_error: str | None = None
    for attempt in range(int(attempt_limit)):
        check_cancelled()
        if not remaining:
            break

        user = json.dumps(remaining, ensure_ascii=False)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        try:
            content = tr._chat_completion_text(backend, messages)  # noqa: SLF001
            obj = tr._extract_first_json_object(content)  # noqa: SLF001
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
                    # Key missing in output
                    invalid_keys.append(k)
                    continue

                # Check for empty output on non-empty input (safety guard)
                cand = str(v)
                if src and not cand.strip():
                    invalid_keys.append(k)
                    continue

                result[k] = cand
                progressed += 1

            if invalid_keys:
                had_failures = True
                # If no key was accepted in this attempt, warn and stop retrying aggressively.
                if progressed <= 0:
                    logger.warning(
                        f"转录修复部分失败 (尝试={attempt + 1}/{attempt_limit}): "
                        f"missing_keys={invalid_keys[:5]} (total={len(invalid_keys)})"
                    )
                    break
                remaining = {k: remaining[k] for k in invalid_keys}
                sleep_with_cancel(0.2)
                continue

            remaining = {}
            break
        except (ValueError, JSONDecodeError) as exc:
            last_parse_error = str(exc)
            logger.warning(f"转录修复解析失败 (尝试={attempt + 1}/{attempt_limit}): {exc}")
            sleep_with_cancel(0.3)
        except Exception as exc:  # pylint: disable=broad-except
            delay = tr._handle_sdk_exception(exc, attempt)  # noqa: SLF001
            if delay is None:
                had_failures = True
                break
            sleep_with_cancel(delay)

    # Any remaining keys fall back to original text.
    if remaining:
        had_failures = True
        # Avoid log spam; only mention parse errors when we never made progress.
        if last_parse_error and not result:
            logger.warning(f"转录修复失败，将回退到原文（原因: {last_parse_error}）")

    final = dict(payload)
    final.update(result)
    # Ensure strict contract: return must contain all keys.
    if set(final.keys()) != set(payload.keys()):
        # Extremely defensive: if something went wrong, return original.
        return dict(payload), False
    return final, (not had_failures)


def load_or_create_punctuated_transcript(
    folder: str,
    transcript: list[dict[str, Any]],
    *,
    settings: Settings,
) -> list[dict[str, Any]]:
    """
    Best-effort:
    - If cached punctuated transcript exists and is newer than transcript.json, reuse it.
    - Otherwise, call LLM to fix punctuation and cache it.
    """
    from . import translate as tr

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
        chunk_size = max(1, min(tr._read_int_env("PUNCTUATION_FIX_CHUNK_SIZE", 8), 128))  # noqa: SLF001
        attempt_limit = max(1, min(tr._read_int_env("PUNCTUATION_FIX_ATTEMPT_LIMIT", 2), 10))  # noqa: SLF001
        any_changed = False
        all_ok = True

        for i0 in range(0, len(out), int(chunk_size)):
            check_cancelled()
            idxs = list(range(i0, min(i0 + int(chunk_size), len(out))))
            payload = {str(i): cast(str, out[i].get("text", "")) for i in idxs}
            fixed, ok = _punct_fix_chunk(settings, payload, attempt_limit=int(attempt_limit))
            all_ok = bool(all_ok and ok)
            for i in idxs:
                src = payload[str(i)]
                new = fixed.get(str(i), src)
                if new != src:
                    any_changed = True
                out[i]["text"] = new

        # Cache when fully successful (even if it's a no-op), so we won't call the model twice
        # in the same pipeline run after extracting this step.
        if all_ok:
            try:
                with open(punct_path, "w", encoding="utf-8") as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(f"写入标点修复缓存失败（忽略）: {exc}")
        else:
            # Keep legacy semantics: don't cache partial/failed fixes.
            if any_changed:
                logger.warning("标点修复存在失败，将不写缓存；后续可重试。")

        return out
    except Exception as exc:  # pylint: disable=broad-except
        # Best-effort: never block translation due to punctuation fix.
        logger.warning(f"翻译前标点修复失败（忽略，继续使用原始转写）: {exc}")
        return transcript


def optimize_transcript_folder(folder: str, settings: Settings | None = None) -> bool:
    """
    Pipeline step: generate/refresh `transcript_punctuated.json` for a single folder.

    This step is best-effort: failures never raise to callers by default.
    """
    check_cancelled()
    transcript_path = os.path.join(folder, "transcript.json")
    if not os.path.exists(transcript_path):
        return False

    cfg = settings or Settings()
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        if not isinstance(transcript, list):
            raise ValueError("transcript.json is not a list")
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(f"转录优化: 读取 transcript.json 失败，跳过: {folder} ({exc})")
        return False

    try:
        _ = load_or_create_punctuated_transcript(folder, cast(list[dict[str, Any]], transcript), settings=cfg)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(f"转录优化失败（忽略）: {folder} ({exc})")
        return False


def optimize_all_transcript_under_folder(folder: str, settings: Settings | None = None) -> str:
    check_cancelled()
    count = 0
    skipped = 0
    for root, _dirs, files in os.walk(folder):
        check_cancelled()
        if "transcript.json" not in files:
            continue
        if optimize_transcript_folder(root, settings=settings):
            count += 1
        else:
            skipped += 1
    parts = []
    if count:
        parts.append(f"优化 {count} 个")
    if skipped:
        parts.append(f"跳过 {skipped} 个")
    msg = f"转录优化完成: {folder}" + (f"（{', '.join(parts)}）" if parts else "")
    logger.info(msg)
    return msg

