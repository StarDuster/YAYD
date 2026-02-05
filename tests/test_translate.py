"""Translate 步骤测试."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #


def test_extract_first_json_object_skips_invalid_and_returns_dict():
    from youdub.steps.translate import _extract_first_json_object

    text = 'prefix {bad json} middle {"a": 1, "b": "x"} suffix'
    obj = _extract_first_json_object(text)
    assert obj == {"a": 1, "b": "x"}


def test_extract_first_json_object_raises_when_missing():
    from youdub.steps.translate import _extract_first_json_object

    with pytest.raises(ValueError, match="No JSON object found"):
        _extract_first_json_object("no json here")


def test_ensure_transcript_length_truncates_middle_but_preserves_ends():
    from youdub.steps.translate import ensure_transcript_length

    s = "A" * 50 + "B" * 50
    out = ensure_transcript_length(s, max_length=20)
    assert len(out) == 20
    assert out.startswith("A" * 10)
    assert out.endswith("B" * 10)


def test_valid_translation_accepts_backticks_and_postprocesses():
    from youdub.steps.translate import valid_translation

    ok, processed = valid_translation("Knowledge is power.", "```AI...```")
    assert ok is True
    assert "人工智能" in processed
    assert "，" in processed  # "..." -> "，"


def test_valid_translation_rejects_explanation_patterns():
    """解释性模式（如 '翻译：...'）应该被拒绝."""
    from youdub.steps.translate import valid_translation

    long_source = "This is a longer sentence for testing purposes."
    
    # "翻译：" 开头
    ok, msg = valid_translation(long_source, "翻译：这是一个用于测试目的的较长句子")
    assert ok is False
    assert "explanation patterns" in msg
    
    # "翻译结果是..."
    ok, msg = valid_translation(long_source, "翻译结果是这是一个用于测试目的的较长句子")
    assert ok is False
    assert "explanation patterns" in msg
    
    # "以下是翻译..."
    ok, msg = valid_translation(long_source, "以下是翻译这是一个用于测试目的的较长句子")
    assert ok is False
    assert "explanation patterns" in msg


def test_valid_translation_rejects_newline():
    """翻译中包含换行符应该被拒绝."""
    from youdub.steps.translate import valid_translation

    ok, msg = valid_translation("Hello world", "你好\n世界")
    assert ok is False
    assert "newline" in msg.lower()


def test_valid_translation_rejects_too_long_for_short_source():
    from youdub.steps.translate import valid_translation

    ok, msg = valid_translation("Hi", "This translation is definitely too long.")
    assert ok is False
    assert "Only translate the following sentence" in msg


def test_valid_translation_accepts_normal_content_with_keywords():
    """正常翻译内容中包含 '翻译'、'中文' 等词应该被接受."""
    from youdub.steps.translate import valid_translation

    # "machine translation" -> "机器翻译" 是合理翻译
    ok, processed = valid_translation(
        "Machine translation technology has improved significantly.",
        "机器翻译技术已经有了显著的进步。"
    )
    assert ok is True
    assert "机器翻译" in processed
    
    # "learn Chinese" -> "学中文" 是合理翻译
    ok, processed = valid_translation(
        "Many people want to learn Chinese these days.",
        "如今很多人都想学中文。"
    )
    assert ok is True
    assert "学中文" in processed
    
    # "Simplified Chinese version" -> "简体中文版本" 是合理翻译
    ok, processed = valid_translation(
        "The Simplified Chinese version is now available.",
        "简体中文版本现已上线。"
    )
    assert ok is True
    assert "简体中文版本" in processed


def test_valid_translation_accepts_zheju_in_normal_translation():
    """'这句' 作为 'This statement' 等的正常翻译应该被接受."""
    from youdub.steps.translate import valid_translation

    # "This statement says..." -> "这句陈述说的是..." 是合理翻译
    ok, processed = valid_translation(
        "This statement says that at any level of your code.",
        "这句陈述说的是在你代码的任何层级。"
    )
    assert ok is True
    assert "这句陈述" in processed


def test_valid_translation_rejects_zheju_in_explanation_patterns():
    """'这句' 出现在解释性模式中（如 '这句的意思是...'）应该被拒绝."""
    from youdub.steps.translate import valid_translation

    long_source = "This is a longer sentence for testing purposes."
    
    # "这句的意思是..." 模式
    ok, msg = valid_translation(long_source, "这句的意思是这是一个用于测试目的的较长句子")
    assert ok is False
    assert "explanation patterns" in msg
    
    # "这句话意思是..." 模式
    ok, msg = valid_translation(long_source, "这句话意思是这是一个用于测试目的的较长句子")
    assert ok is False
    assert "explanation patterns" in msg


def test_valid_translation_rejects_chinese_label_patterns():
    """'中文：' 或 '简体中文翻译：' 等标签模式应该被拒绝."""
    from youdub.steps.translate import valid_translation

    long_source = "This is a longer sentence for testing purposes."
    
    # "中文：" 开头
    ok, msg = valid_translation(long_source, "中文：这是一个用于测试目的的较长句子")
    assert ok is False
    assert "explanation patterns" in msg
    
    # "简体中文翻译：..."
    ok, msg = valid_translation(long_source, "简体中文翻译：这是一个用于测试目的的较长句子")
    assert ok is False
    assert "explanation patterns" in msg


# --------------------------------------------------------------------------- #
# Whisper punctuation fix (before translation)
# --------------------------------------------------------------------------- #


def test_translate_folder_applies_punctuation_fix_before_translate(tmp_path: Path, monkeypatch):
    import youdub.steps.translate as tr

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "download.info.json").write_text(
        json.dumps({"title": "t", "uploader": "u", "upload_date": "20260101"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (folder / "transcript.json").write_text(
        json.dumps(
            [{"start": 0.0, "end": 1.0, "text": "Hello world", "speaker": "SPEAKER_00"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("YOUDUB_PUNCTUATION_FIX_BEFORE_TRANSLATE", "1")

    monkeypatch.setattr(tr, "summarize", lambda *_args, **_kwargs: {"title": "t", "author": "u", "summary": "s", "tags": []})

    def _fake_chat_completion_text(_backend, messages):
        system = str(messages[0].get("content", ""))
        if "转录文本修复专家" not in system:
            raise AssertionError("unexpected model call (not transcription fix)")
        payload = json.loads(str(messages[1].get("content", "")))
        # Modify text (add punctuation AND fix capitalization)
        out = {k: str(v).replace("Hello world", "Hello, World!") for k, v in payload.items()}
        return json.dumps(out, ensure_ascii=False)

    monkeypatch.setattr(tr, "_chat_completion_text", _fake_chat_completion_text)

    captured: dict[str, str] = {}

    def _fake_translate_content(_summary, transcript, *_args, **_kwargs):
        captured["text0"] = str(transcript[0].get("text", ""))
        return ["好"]

    monkeypatch.setattr(tr, "_translate_content", _fake_translate_content)

    ok = tr.translate_folder(str(folder), target_language="简体中文", settings=tr.Settings(openai_api_key="dummy"))
    assert ok is True
    # Verify that the text modification was accepted (even though it changed non-punctuation characters)
    assert captured.get("text0") == "Hello, World!"


def test_punctuated_transcript_cache_is_reused(tmp_path: Path, monkeypatch):
    import youdub.steps.translate as tr

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    transcript = [{"start": 0.0, "end": 1.0, "text": "Hello world", "speaker": "SPEAKER_00"}]
    (folder / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setenv("YOUDUB_PUNCTUATION_FIX_BEFORE_TRANSLATE", "1")

    calls = {"n": 0}

    def _fake_chat_completion_text(_backend, messages):
        calls["n"] += 1
        payload = json.loads(str(messages[1].get("content", "")))
        out = {k: str(v).replace("Hello world", "Hello, World!") for k, v in payload.items()}
        return json.dumps(out, ensure_ascii=False)

    monkeypatch.setattr(tr, "_chat_completion_text", _fake_chat_completion_text)

    settings = tr.Settings(openai_api_key="dummy")
    out1 = tr._load_or_create_punctuated_transcript(str(folder), transcript, settings=settings)  # noqa: SLF001
    assert out1[0]["text"] == "Hello, World!"
    assert calls["n"] == 1
    assert (folder / "transcript_punctuated.json").exists()

    # Second call should use the cached file instead of calling the model again.
    out2 = tr._load_or_create_punctuated_transcript(str(folder), transcript, settings=settings)  # noqa: SLF001
    assert out2[0]["text"] == "Hello, World!"
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# Strategy normalization
# --------------------------------------------------------------------------- #


def test_normalize_translation_strategy_defaults_to_guide_parallel():
    from youdub.steps.translate import _normalize_translation_strategy

    assert _normalize_translation_strategy(None) == "guide_parallel"
    assert _normalize_translation_strategy("") == "guide_parallel"
    assert _normalize_translation_strategy("history") == "history"
    assert _normalize_translation_strategy("serial") == "history"


def test_normalize_translation_strategy_accepts_parallel_aliases():
    from youdub.steps.translate import _normalize_translation_strategy

    assert _normalize_translation_strategy("guide_parallel") == "guide_parallel"
    assert _normalize_translation_strategy("parallel") == "guide_parallel"
    assert _normalize_translation_strategy("2") == "guide_parallel"


# --------------------------------------------------------------------------- #
# Translation content (history strategy)
# --------------------------------------------------------------------------- #


def test_translate_content_default_history_includes_previous_turns(monkeypatch):
    import youdub.steps.translate as tr

    monkeypatch.setattr(tr.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("TRANSLATION_STRATEGY", "history")

    calls = {"n": 0}

    def _fake_chat_completion_text(_backend, messages):
        calls["n"] += 1

        # The first request should not contain a turn with the first translated output yet.
        if calls["n"] == 1:
            assert not any(
                m.get("role") == "assistant" and "你好" in str(m.get("content", ""))
                for m in messages
            )
            return "你好"

        # The second request should include history from the first request.
        assert any(
            m.get("role") == "assistant" and str(m.get("content", "")).strip() == "你好"
            for m in messages
        )
        return "世界"

    monkeypatch.setattr(tr, "_chat_completion_text", _fake_chat_completion_text)

    summary = {"title": "T", "summary": "S"}
    transcript = [{"text": "This is sentence one with enough length."}, {"text": "This is sentence two with enough length."}]

    out = tr._translate_content(summary, transcript, target_language="简体中文", settings=tr.Settings(openai_api_key="dummy"))
    assert out == ["你好", "世界"]


# --------------------------------------------------------------------------- #
# Translation content (guide_parallel strategy)
# --------------------------------------------------------------------------- #


def test_translate_content_guide_parallel_returns_indexed_translations(monkeypatch):
    import youdub.steps.translate as tr

    monkeypatch.setattr(tr.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("TRANSLATION_STRATEGY", "guide_parallel")
    monkeypatch.setenv("TRANSLATION_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("TRANSLATION_CHUNK_SIZE", "2")
    monkeypatch.setenv("TRANSLATION_GUIDE_MAX_CHARS", "2000")

    def _fake_chat_completion_text(_backend, messages):
        system = str(messages[0].get("content", ""))
        user = str(messages[1].get("content", ""))

        # Guide generation step.
        if "翻译指南" in system:
            return json.dumps(
                {
                    "style": ["保持译文简洁自然"],
                    "glossary": {"agent": "智能体"},
                    "dont_translate": ["Q-Learning", "Transformer"],
                    "notes": "",
                },
                ensure_ascii=False,
            )

        # Chunk translation step: user message is a JSON object containing sentences/speakers/contexts.
        obj = json.loads(user)
        payload = obj.get("sentences", obj) if isinstance(obj, dict) else obj
        out = {k: (f"句{k}" if str(v).strip() else "") for k, v in payload.items()}
        return json.dumps(out, ensure_ascii=False)

    monkeypatch.setattr(tr, "_chat_completion_text", _fake_chat_completion_text)

    summary = {"title": "T", "summary": "S"}
    transcript = [
        {"text": "This is sentence number 0 with enough length."},
        {"text": "This is sentence number 1 with enough length."},
        {"text": "This is sentence number 2 with enough length."},
        {"text": "This is sentence number 3 with enough length."},
    ]

    out = tr._translate_content(summary, transcript, target_language="简体中文", settings=tr.Settings(openai_api_key="dummy"))
    assert out == ["句0", "句1", "句2", "句3"]


# --------------------------------------------------------------------------- #
# Subtitle alignment (postprocess)
# --------------------------------------------------------------------------- #


def test_split_sentences_splits_source_text_when_translation_has_more_sentences():
    from youdub.steps.translate import split_sentences

    src = (
        "This is a long run on sentence without punctuation but with enough words "
        "to be split for subtitle alignment instead of being duplicated as a paragraph"
    )
    items = [
        {
            "start": 0.0,
            "end": 9.0,
            "text": src,
            "speaker": "SPEAKER_00",
            "translation": "你好。再见。谢谢。",
        }
    ]

    out = split_sentences(items)
    assert len(out) == 3

    texts = [x["text"] for x in out]
    # Avoid repeating the full paragraph on each translated sentence.
    assert len(set(t for t in texts if t)) > 1
    assert all(len(t) < len(src) for t in texts if t)
    # Preserve word order when concatenated (best-effort chunking).
    assert " ".join(texts).split() == src.split()
    # Timeline boundary must be preserved.
    assert out[0]["start"] == 0.0
    assert out[-1]["end"] == 9.0


# --------------------------------------------------------------------------- #
# Translation schema validation
# --------------------------------------------------------------------------- #


def _assert_translation_items_schema(items: list[dict[str, Any]]) -> None:
    assert isinstance(items, list)
    assert items, "translation.json must be a non-empty list"
    for it in items[:50]:
        assert isinstance(it, dict)
        assert isinstance(it.get("start"), (int, float))
        assert isinstance(it.get("end"), (int, float))
        assert isinstance(it.get("speaker"), str)
        assert isinstance(it.get("text"), str)
        assert isinstance(it.get("translation"), str)


def test_translate_folder_generates_translation_and_summary_schema(tmp_path: Path, monkeypatch):
    import youdub.steps.translate as tr

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "download.info.json").write_text(
        json.dumps({"title": "t", "uploader": "u", "upload_date": "20260101"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (folder / "transcript.json").write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
                {"start": 1.0, "end": 2.0, "text": "world", "speaker": "SPEAKER_00"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(tr, "summarize", lambda *_args, **_kwargs: {"title": "t", "author": "u", "summary": "s", "tags": []})
    # Make the first segment contain two sentences, so postprocess alignment produces more items than transcript.
    monkeypatch.setattr(tr, "_translate_content", lambda *_args, **_kwargs: ["你好。再见。", "世界！"])

    ok = tr.translate_folder(str(folder), target_language="简体中文")
    assert ok is True

    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert isinstance(summary, dict)
    assert isinstance(summary.get("title"), str)
    assert isinstance(summary.get("summary"), str)
    assert isinstance(summary.get("translation_model"), str)

    translation_raw = json.loads((folder / "translation_raw.json").read_text(encoding="utf-8"))
    _assert_translation_items_schema(translation_raw)
    assert len(translation_raw) == 2
    assert translation_raw[0]["translation"] == "你好。再见。"
    assert translation_raw[1]["translation"] == "世界！"

    translation = json.loads((folder / "translation.json").read_text(encoding="utf-8"))
    _assert_translation_items_schema(translation)
    assert len(translation) > len(translation_raw)


def test_translate_folder_skips_when_translation_and_summary_exist(tmp_path: Path, monkeypatch):
    import youdub.steps.translate as tr

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    # Prereqs
    (folder / "download.info.json").write_text(json.dumps({"title": "t", "uploader": "u", "upload_date": "20260101"}), encoding="utf-8")
    (folder / "transcript.json").write_text(json.dumps([{"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_00"}]), encoding="utf-8")

    # Existing translation.json + summary.json (translation_raw.json intentionally missing).
    (folder / "translation.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_00", "translation": "好"}], ensure_ascii=False),
        encoding="utf-8",
    )
    (folder / "summary.json").write_text(
        json.dumps({"title": "t", "author": "u", "summary": "s", "tags": [], "translation_model": "dummy"}, ensure_ascii=False),
        encoding="utf-8",
    )

    def _should_not_call(*_args, **_kwargs):
        raise AssertionError("should not be called when translation+summary already exist")

    monkeypatch.setattr(tr, "summarize", _should_not_call)
    monkeypatch.setattr(tr, "_translate_content", _should_not_call)

    ok = tr.translate_folder(str(folder), target_language="简体中文")
    assert ok is True
    assert not (folder / "translation_raw.json").exists()


def test_translate_folder_refreshes_speaker_when_transcript_updated_and_raw_missing(
    tmp_path: Path, monkeypatch
):
    import os

    import youdub.steps.translate as tr

    folder = tmp_path / "job_refresh_speaker"
    folder.mkdir(parents=True, exist_ok=True)

    # Existing translation + summary (translation_raw.json intentionally missing).
    translation_path = folder / "translation.json"
    translation_path.write_text(
        json.dumps(
            [{"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_00", "translation": "好"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary_path = folder / "summary.json"
    summary_path.write_text(
        json.dumps(
            {"title": "t", "author": "u", "summary": "s", "tags": [], "translation_model": "dummy"},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # transcript.json is updated (speaker label changed) after translation.json.
    transcript_path = folder / "transcript.json"
    transcript_path.write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_01"}], ensure_ascii=False),
        encoding="utf-8",
    )
    # Ensure mtime is newer even on coarse filesystems.
    t_mtime = translation_path.stat().st_mtime
    os.utime(str(transcript_path), (t_mtime + 10.0, t_mtime + 10.0))

    def _should_not_call(*_args, **_kwargs):
        raise AssertionError("should not be called when refreshing speakers only")

    monkeypatch.setattr(tr, "summarize", _should_not_call)
    monkeypatch.setattr(tr, "_translate_content", _should_not_call)

    ok = tr.translate_folder(str(folder), target_language="简体中文")
    assert ok is True

    updated = json.loads(translation_path.read_text(encoding="utf-8"))
    assert isinstance(updated, list)
    assert updated[0]["speaker"] == "SPEAKER_01"


def test_translate_folder_retranslates_when_transcript_length_changes(tmp_path: Path, monkeypatch):
    import youdub.steps.translate as tr

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    # Disable punctuation fix to avoid LLM calls in tests.
    monkeypatch.setenv("YOUDUB_PUNCTUATION_FIX_BEFORE_TRANSLATE", "0")

    (folder / "download.info.json").write_text(
        json.dumps({"title": "t", "uploader": "u", "upload_date": "20260101"}, ensure_ascii=False),
        encoding="utf-8",
    )

    # Existing translation files (from an older transcript segmentation)
    (folder / "translation_raw.json").write_text(
        json.dumps(
            [{"start": 0.0, "end": 1.0, "text": "old", "speaker": "SPEAKER_00", "translation": "旧"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (folder / "translation.json").write_text(
        json.dumps(
            [{"start": 0.0, "end": 1.0, "text": "old", "speaker": "SPEAKER_00", "translation": "旧"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (folder / "summary.json").write_text(
        json.dumps({"title": "t", "author": "u", "summary": "s", "tags": [], "translation_model": "dummy"}, ensure_ascii=False),
        encoding="utf-8",
    )

    # Transcript is newer and has different length -> should trigger re-translation.
    (folder / "transcript.json").write_text(
        json.dumps(
            [
                {"start": 0.0, "end": 1.0, "text": "a", "speaker": "SPEAKER_00"},
                {"start": 1.0, "end": 2.0, "text": "b", "speaker": "SPEAKER_00"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calls = {"n": 0}

    def _fake_translate_content(_summary, transcript, *_args, **_kwargs):
        calls["n"] += 1
        return ["好"] * len(transcript)

    def _should_not_call(*_args, **_kwargs):
        raise AssertionError("summarize should not be called when summary.json already exists")

    monkeypatch.setattr(tr, "summarize", _should_not_call)
    monkeypatch.setattr(tr, "_translate_content", _fake_translate_content)

    ok = tr.translate_folder(str(folder), target_language="简体中文", settings=tr.Settings(openai_api_key="dummy"))
    assert ok is True
    assert calls["n"] == 1

    raw = json.loads((folder / "translation_raw.json").read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    assert len(raw) == 2


# --------------------------------------------------------------------------- #
# Backfill summary when translation exists
# --------------------------------------------------------------------------- #


def test_translate_folder_backfills_summary_when_translation_exists(tmp_path: Path, monkeypatch):
    import youdub.steps.translate as tr

    folder = tmp_path / "job"
    folder.mkdir(parents=True, exist_ok=True)

    # Prereqs
    (folder / "download.info.json").write_text(json.dumps({"title": "t", "uploader": "u", "upload_date": "20260101"}), encoding="utf-8")
    (folder / "transcript.json").write_text(json.dumps([{"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_00"}]), encoding="utf-8")

    # Existing translation.json but missing summary.json
    (folder / "translation.json").write_text(
        json.dumps([{"start": 0.0, "end": 1.0, "text": "x", "speaker": "SPEAKER_00", "translation": "好"}], ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(tr, "summarize", lambda *_args, **_kwargs: {"title": "t", "author": "u", "summary": "s", "tags": []})

    ok = tr.translate_folder(str(folder), target_language="简体中文")
    assert ok is True
    assert (folder / "summary.json").exists()
