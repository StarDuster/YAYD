from __future__ import annotations

import re


_TTS_PY_DOT_CHAIN_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[A-Za-z_][A-Za-z0-9_]*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
    r"(?![A-Za-z0-9_])"
)
_TTS_PY_UNDERSCORE_IDENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[A-Za-z_][A-Za-z0-9_]*"
    r"_[A-Za-z0-9_]+"
    r"(?![A-Za-z0-9_])"
)


def _tts_prompt_normalize_python_tokens(text: str) -> str:
    """
    Make Python-ish code tokens more TTS friendly.

    Examples:
    - matplotlib.pyplot -> matplotlib dot pyplot
    - os.path.join -> os dot path dot join
    - base_dir -> base dir

    Notes:
    - We only touch identifier-like tokens to avoid breaking decimals like 3.14.
    - This function is used ONLY for TTS prompt text (subtitles are unchanged).
    """
    t = str(text or "")

    def _dot_chain(m: re.Match) -> str:
        s = m.group(0)
        # Normalize snake_case inside each segment too.
        s = s.replace("_", " ")
        s = s.replace(".", " dot ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _underscore_ident(m: re.Match) -> str:
        s = m.group(0)
        s = s.replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # Convert dotted chains first (also handles underscores inside them).
    t = _TTS_PY_DOT_CHAIN_RE.sub(_dot_chain, t)
    # Convert remaining snake_case identifiers.
    t = _TTS_PY_UNDERSCORE_IDENT_RE.sub(_underscore_ident, t)
    return t


def _tts_text_for_attempt(raw_text: str, attempt: int) -> str:
    """
    Text variants for retries:
    - attempt 0: keep original (normalized, but make Python-ish tokens TTS-friendly)
    - attempt 1+: strip markdown-ish code markers and make common code punctuation more TTS-friendly
    - attempt 2+: more aggressive cleanup of uncommon symbols
    """
    t = str(raw_text or "")
    if attempt > 0:
        # Remove fenced blocks and inline backticks.
        t = re.sub(r"```[\s\S]*?```", " ", t)
        t = t.replace("`", "")

    # Always make Python-ish tokens more readable for TTS.
    t = _tts_prompt_normalize_python_tokens(t)

    if attempt >= 2:
        # Keep only: CJK, ASCII alnum, whitespace, and basic punctuation.
        t = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s，。！？,.!?:：；;（）()\-\+*/=]", " ", t)

    t = re.sub(r"\s+", " ", t).strip()
    return t

