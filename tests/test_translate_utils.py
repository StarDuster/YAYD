import pytest


def test_extract_first_json_object_skips_invalid_and_returns_dict():
    from youdub.core.steps.translate import _extract_first_json_object

    text = 'prefix {bad json} middle {"a": 1, "b": "x"} suffix'
    obj = _extract_first_json_object(text)
    assert obj == {"a": 1, "b": "x"}


def test_extract_first_json_object_raises_when_missing():
    from youdub.core.steps.translate import _extract_first_json_object

    with pytest.raises(ValueError, match="No JSON object found"):
        _extract_first_json_object("no json here")


def test_ensure_transcript_length_truncates_middle_but_preserves_ends():
    from youdub.core.steps.translate import ensure_transcript_length

    s = "A" * 50 + "B" * 50
    out = ensure_transcript_length(s, max_length=20)
    assert len(out) == 20
    assert out.startswith("A" * 10)
    assert out.endswith("B" * 10)


def test_valid_translation_accepts_backticks_and_postprocesses():
    from youdub.core.steps.translate import valid_translation

    ok, processed = valid_translation("Knowledge is power.", "```AI...```")
    assert ok is True
    assert "人工智能" in processed
    assert "，" in processed  # "..." -> "，"


def test_valid_translation_rejects_forbidden_word():
    from youdub.core.steps.translate import valid_translation

    ok, msg = valid_translation("Hello world", "翻译: 你好")
    assert ok is False
    assert "Don't include" in msg


def test_valid_translation_rejects_too_long_for_short_source():
    from youdub.core.steps.translate import valid_translation

    ok, msg = valid_translation("Hi", "This translation is definitely too long.")
    assert ok is False
    assert "Only translate the following sentence" in msg

