from __future__ import annotations


def test_normalize_zh_nsw_date():
    from youdub.text_norm_zh import normalize_zh_nsw

    s = "今天是2026年2月5日。"
    out = normalize_zh_nsw(s)
    assert "二零二六年二月五日" in out


def test_normalize_zh_nsw_percent_and_fraction():
    from youdub.text_norm_zh import normalize_zh_nsw

    s = "完成率12.5%，比例3/4。"
    out = normalize_zh_nsw(s)
    assert "百分之十二点五" in out
    assert "四分之三" in out


def test_normalize_zh_nsw_money_and_quantifier():
    from youdub.text_norm_zh import normalize_zh_nsw

    s = "花了100元，买了1234个。"
    out = normalize_zh_nsw(s)
    assert "一百元" in out
    # "两百" is natural for cardinal numbers.
    assert ("一千两百三十四个" in out) or ("一千二百三十四个" in out)


def test_normalize_zh_nsw_phone_and_restore_p2p():
    from youdub.text_norm_zh import normalize_zh_nsw

    s = "P2P模式，号码13800138000。"
    out = normalize_zh_nsw(s)
    assert "P2P" in out
    assert "一三八零零一三八零零零" in out


def test_normalize_zh_nsw_long_digits_as_id():
    from youdub.text_norm_zh import normalize_zh_nsw

    s = "编号2026，ID 12345678。"
    out = normalize_zh_nsw(s)
    assert "二零二六" in out
    assert "一二三四五六七八" in out

