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


def test_valid_translation_rejects_forbidden_word():
    from youdub.steps.translate import valid_translation

    ok, msg = valid_translation("Hello world", "翻译: 你好")
    assert ok is False
    assert "Don't include" in msg


def test_valid_translation_rejects_too_long_for_short_source():
    from youdub.steps.translate import valid_translation

    ok, msg = valid_translation("Hi", "This translation is definitely too long.")
    assert ok is False
    assert "Only translate the following sentence" in msg


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

        # Chunk translation step: user message is a JSON map {idx: text}.
        payload = json.loads(user)
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
    monkeypatch.setattr(tr, "_translate_content", lambda *_args, **_kwargs: ["你好", "世界"])

    ok = tr.translate_folder(str(folder), target_language="简体中文")
    assert ok is True

    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert isinstance(summary, dict)
    assert isinstance(summary.get("title"), str)
    assert isinstance(summary.get("summary"), str)
    assert isinstance(summary.get("translation_model"), str)

    translation = json.loads((folder / "translation.json").read_text(encoding="utf-8"))
    _assert_translation_items_schema(translation)


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
