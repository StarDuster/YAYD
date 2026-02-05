from __future__ import annotations

import re

# NOTE:
# This module provides lightweight Chinese "NSW" (non-standard words) normalization for TTS.
# It is intentionally much smaller than the historical `cn_tx.py` script:
# - no CLI / corpus processing
# - focus on digits, dates, money, phone numbers, fractions, percentages
#
# The goal is to make TTS read numbers naturally in Chinese.


_DIGITS = "零一二三四五六七八九"
_SMALL_UNITS = ("", "十", "百", "千")
_BIG_UNITS = ("", "万", "亿", "兆")


def _has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", str(s or "")))


def _digit_to_zh(d: int) -> str:
    if 0 <= int(d) <= 9:
        return _DIGITS[int(d)]
    raise ValueError(f"not a digit: {d!r}")


def _digits_to_zh(s: str) -> str:
    out = []
    for ch in str(s or ""):
        if ch.isdigit():
            out.append(_digit_to_zh(int(ch)))
        else:
            out.append(ch)
    return "".join(out)


def _group4_to_zh(group4: str, *, alt_two: bool = True) -> str:
    g = str(group4 or "")
    g = g[-4:].rjust(4, "0")
    out: list[str] = []
    zero_pending = False
    for idx, ch in enumerate(g):
        pos = 3 - idx  # 3:千,2:百,1:十,0:""
        d = int(ch)
        if d == 0:
            zero_pending = True
            continue

        if zero_pending and out:
            out.append("零")
        zero_pending = False

        # "十X" instead of "一十X" for 10-19 inside a 4-digit group.
        if d == 1 and pos == 1 and not out:
            out.append("十")
        else:
            if alt_two and d == 2 and pos >= 2:
                out.append("两")
            else:
                out.append(_digit_to_zh(d))
            out.append(_SMALL_UNITS[pos])
    return "".join(out)


def _int_to_zh_cardinal(int_string: str, *, alt_two: bool = True) -> str:
    s = str(int_string or "").strip()
    if not s:
        return ""
    if not re.fullmatch(r"\d+", s):
        raise ValueError(f"invalid integer string: {int_string!r}")
    s = s.lstrip("0") or "0"
    if s == "0":
        return "零"

    groups: list[str] = []
    while s:
        groups.append(s[-4:])
        s = s[:-4]
    groups = list(reversed(groups))

    parts: list[str] = []
    zero_between = False
    for gi, g in enumerate(groups):
        chunk = _group4_to_zh(g, alt_two=alt_two)
        unit = _BIG_UNITS[len(groups) - 1 - gi] if (len(groups) - 1 - gi) < len(_BIG_UNITS) else ""
        if not chunk:
            zero_between = True
            continue
        if parts:
            # Need a "零" if previous group was 0, or this group has leading zeros (e.g. 10001 -> 一万零一).
            if zero_between or (len(g) == 4 and g.startswith("0")):
                if not parts[-1].endswith("零"):
                    parts.append("零")
        zero_between = False
        parts.append(chunk + unit)
    return "".join(parts).rstrip("零")


def num_to_zh(text_num: str, *, use_units: bool = True, alt_two: bool = True) -> str:
    """
    Convert a numeric string into Chinese readable form.

    - use_units=True: 12 -> 十二, 1234 -> 一千二百三十四
    - use_units=False: 2026 -> 二零二六 (digit-by-digit)
    """
    s = str(text_num or "").strip()
    if not s:
        return ""
    if not re.fullmatch(r"\d+(?:\.\d+)?", s):
        raise ValueError(f"invalid number string: {text_num!r}")

    if "." in s:
        int_part, dec_part = s.split(".", 1)
    else:
        int_part, dec_part = s, ""

    if use_units and len(int_part) > 1:
        head = _int_to_zh_cardinal(int_part, alt_two=alt_two)
    else:
        head = _digits_to_zh(int_part)

    if not dec_part:
        return head
    return head + "点" + _digits_to_zh(dec_part)


def _normalize_date(s: str) -> str:
    raw = str(s or "")
    if not raw:
        return raw

    # Split by 年/月 to keep original suffixes.
    year = ""
    other = raw
    if "年" in raw:
        year, other = raw.split("年", 1)
        year = _digits_to_zh(year) + "年" if year.strip().isdigit() else year + "年"

    month = ""
    day = ""
    if "月" in other:
        m, rest = other.split("月", 1)
        if m.strip().isdigit():
            month = num_to_zh(m, use_units=True) + "月"
        else:
            month = m + "月"
        other = rest

    if other:
        # day ends with 日/号 usually, but keep any trailing char.
        suf = other[-1]
        core = other[:-1]
        if core.strip().isdigit() and suf in ("日", "号"):
            day = num_to_zh(core, use_units=True) + suf
        else:
            day = other

    return (year + month + day) or raw


_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)(%)")
_FRACTION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")

# Currency/quantifier lists are intentionally conservative to avoid over-normalization.
_CURRENCY_UNITS_RE = re.compile(
    r"(人民币|美元|欧元|日元|英镑|港币|澳元|韩元|元|块|¥|￥|\$|USD|EUR|CNY|RMB|JPY|GBP|HKD|AUD|KRW)",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"(\d+(?:\.\d+)?)(?:[多余几]?)(?P<unit>" + _CURRENCY_UNITS_RE.pattern + r")")

_MOBILE_RE = re.compile(r"(?<!\d)(?:\+?86 ?)?1(?:[38]\d|5[0-35-9]|7[678]|9[89])\d{8}(?!\d)")
_FIXED_RE = re.compile(r"(?<!\d)(?:0(?:10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7}(?!\d)")

_QUANTIFIERS = (
    "个只条本次页章集岁年个月月日号天小时分钟秒"
    "米厘米毫米千米公里克千克公斤吨毫升升"
    "片台部张份行字句段"
)
_NUM_WITH_UNIT_RE = re.compile(r"(\d+(?:\.\d+)?)(?:[多余几]?)([" + re.escape(_QUANTIFIERS) + r"]+)")

_DIGIT_SEQ_RE = re.compile(r"(\d{4,32})")
_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
_P2P_RESTORE_RE = re.compile(r"(([A-Za-z]+)二([A-Za-z]+))")


def normalize_zh_nsw(raw_text: str) -> str:
    """
    Normalize non-standard words for Chinese TTS.

    This is best-effort, and only intended for TTS prompt text (NOT subtitles).
    """
    text = str(raw_text or "")
    if not text:
        return text

    # Fast path: no digits-like tokens.
    if not re.search(r"[\d％%/]", text):
        return text

    # Percent first: keep '%' stable.
    text = text.replace("％", "%")

    # Dates: require at least a year or a month marker.
    date_re = re.compile(
        r"(?<!\d)("
        r"(?:(?:[089]\d|(?:19|20)\d{2})年(?:\d{1,2}月(?:\d{1,2}[日号])?)?)"
        r"|(?:\d{1,2}月(?:\d{1,2}[日号])?)"
        r")"
    )
    text = date_re.sub(lambda m: _normalize_date(m.group(1)), text)

    # Money
    def _money(m: re.Match) -> str:
        n = m.group(1)
        unit = m.group("unit")
        try:
            return num_to_zh(n, use_units=True) + str(unit)
        except Exception:
            return m.group(0)

    text = _MONEY_RE.sub(_money, text)

    # Phone numbers
    def _phone(m: re.Match) -> str:
        s = m.group(0)
        s = s.replace("+", "").replace(" ", "").replace("-", "")
        if s.startswith("86"):
            s = s[2:]
        return _digits_to_zh(s)

    text = _MOBILE_RE.sub(_phone, text)
    text = _FIXED_RE.sub(_phone, text)

    # Fractions
    def _frac(m: re.Match) -> str:
        a = m.group(1)
        b = m.group(2)
        try:
            return num_to_zh(b, use_units=True) + "分之" + num_to_zh(a, use_units=True)
        except Exception:
            return m.group(0)

    text = _FRACTION_RE.sub(_frac, text)

    # Percentages
    def _pct(m: re.Match) -> str:
        n = m.group(1)
        try:
            return "百分之" + num_to_zh(n, use_units=True)
        except Exception:
            return m.group(0)

    text = _PERCENT_RE.sub(_pct, text)

    # Number + quantifier: prefer cardinal reading
    def _n_unit(m: re.Match) -> str:
        n = m.group(1)
        unit = m.group(2)
        try:
            return num_to_zh(n, use_units=True) + unit
        except Exception:
            return m.group(0)

    text = _NUM_WITH_UNIT_RE.sub(_n_unit, text)

    # Long digit sequences: ID / years etc -> digit-by-digit
    def _digit_seq(m: re.Match) -> str:
        s = m.group(1)
        return num_to_zh(s, use_units=False, alt_two=False)

    text = _DIGIT_SEQ_RE.sub(_digit_seq, text)

    # Remaining numbers
    def _number(m: re.Match) -> str:
        s = m.group(1)
        try:
            return num_to_zh(s, use_units=True)
        except Exception:
            return s

    text = _NUMBER_RE.sub(_number, text)

    # Restore P2P/O2O/B2C etc.
    text = _P2P_RESTORE_RE.sub(lambda m: f"{m.group(2)}2{m.group(3)}", text)
    return text


def should_normalize_zh_nsw(text: str, *, target_language: str | None = None) -> bool:
    """
    Heuristic: normalize when:
    - there are digits-like tokens
    - and text likely is Chinese (either target_language hints, or contains CJK)
    """
    t = str(text or "")
    if not re.search(r"[\d％%/]", t):
        return False
    lang = str(target_language or "").strip()
    if lang:
        # Settings values are like "简体中文"/"繁体中文"/"English"/...
        if "中文" in lang or lang.lower().startswith("zh"):
            return True
    return _has_cjk(t)

