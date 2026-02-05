from __future__ import annotations

import re


def normalize_ws(text: str) -> str:
    return " ".join((text or "").split()).strip()


def split_source_text_into_sentences(text: str) -> list[str]:
    """
    Split source (mostly English) text into sentences for bilingual subtitles.

    - Chinese punctuation: 。！？ and their quote variants
    - English punctuation: .!? followed by whitespace
    """
    s = normalize_ws(text)
    if not s:
        return []
    # Chinese punctuation
    s = re.sub(r"([。！？])([^”’])", r"\1\n\2", s)
    s = re.sub(r"([。！？][”’])([^，。！？])", r"\1\n\2", s)
    # English punctuation: split on .!? followed by whitespace
    s = re.sub(r"([.!?][\"”’']?)\s+(?=\S)", r"\1\n", s)
    out = [x.strip() for x in s.splitlines() if x.strip()]
    return out or ([s] if s else [])


def split_source_text_relaxed(text: str) -> list[str]:
    """
    A more aggressive splitter for subtitle alignment (clause-level).

    - English punctuation: ,;: when followed by whitespace
    - Chinese punctuation: ，；： (often without whitespace)
    - Dash variants: —–
    """
    s = normalize_ws(text)
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

