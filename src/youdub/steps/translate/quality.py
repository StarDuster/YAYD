from __future__ import annotations

import re


class _TranslationValidationError(ValueError):
    pass


def translation_postprocess(result: str) -> str:
    result = re.sub(r"\（[^）]*\）", "", result)
    result = re.sub(r"(?<=[\u4e00-\u9fff])\s*\([A-Za-z][^)]*\)", "", result)
    result = result.replace("...", "，")
    result = re.sub(r"(?<=\d),(?=\d)", "", result)
    result = result.replace("²", "的平方").replace("————", "：").replace("——", "：").replace("°", "度")
    result = result.replace("AI", "人工智能")
    result = result.replace("变压器", "Transformer")
    return result


def valid_translation(text: str, translation: str) -> tuple[bool, str]:
    if translation.startswith("```") and translation.endswith("```"):
        translation = translation[3:-3]
        return True, translation_postprocess(translation)

    if (translation.startswith("“") and translation.endswith("”")) or (
        translation.startswith('"') and translation.endswith('"')
    ):
        translation = translation[1:-1]
        return True, translation_postprocess(translation)

    # Heuristics to remove prefixes like "翻译：“..."
    if "翻译" in translation and "：“" in translation and "”" in translation:
        translation = translation.split("：“")[-1].split("”")[0]
        return True, translation_postprocess(translation)

    if "翻译" in translation and ':"' in translation and '"' in translation:
        translation = translation.split(':"')[-1].split('"')[0]
        return True, translation_postprocess(translation)

    if len(text) <= 10:
        if len(translation) > 15:
            return False, "Only translate the following sentence and give me the result."
    elif len(translation) > len(text) * 0.75:
        # Keep legacy semantics for compatibility.
        return False, "The translation is too long. Only translate the following sentence and give me the result."

    # Check if translation is too short (content lost during translation)
    # For longer texts (>100 chars), translation should be at least 15% of original length.
    if len(text) > 100 and len(translation) < len(text) * 0.15:
        return False, "The translation is too short, content may be lost. Please translate the complete sentence."

    translation = translation.strip()

    # Newline is always forbidden in translation output
    if "\n" in translation:
        return (
            False,
            "Don't include newlines in the translation. Only translate the following sentence and give me the result.",
        )

    # Check for explanation patterns that indicate LLM is explaining rather than translating.
    explanation_patterns = [
        # "翻译" patterns - reject when used as meta-explanation, allow when part of content
        # e.g. reject "翻译：你好" or "翻译结果是你好", allow "机器翻译技术"
        (r"^翻译[：:]\s*", "翻译："),
        (r"翻译(结果|如下|为|成)[：:]?\s*", "翻译结果/如下/为"),
        (r"(以下|下面)是?.{0,2}翻译", "以下是翻译"),
        # "中文/简体中文" patterns - reject meta-explanation, allow content discussion
        # e.g. reject "中文：你好" or "简体中文翻译：", allow "学习中文" or "中文版本"
        (r"^(简体)?中文[：:]\s*", "中文："),
        (r"(简体)?中文翻译[：:]?\s*", "中文翻译："),
        (r"翻译成?(简体)?中文[：:]?\s*", "翻译成中文："),
        # "这句" patterns - reject explanation, allow normal translation of "this statement"
        (r"这句.{0,3}(的翻译|的意思|翻译成|意思是)", "这句的翻译/意思"),
        # English patterns - in Chinese translation, these usually indicate LLM explaining
        # e.g. "Translation: ..." or "Translate to Chinese: ..."
        (r"[Tt]ranslat(e|ion)[：:]\s*", "Translation:"),
        (r"[Tt]ranslat(e|ion)\s+(to|into)\s+", "Translate to"),
    ]

    for pattern, desc in explanation_patterns:
        if re.search(pattern, translation):
            return (
                False,
                f"Don't include explanation patterns ({desc}) in the translation. Only give the translated result.",
            )

    return True, translation_postprocess(translation)

