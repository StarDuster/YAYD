import json


def test_normalize_translation_strategy_defaults_to_guide_parallel():
    from youdub.core.steps.translate import _normalize_translation_strategy

    assert _normalize_translation_strategy(None) == "guide_parallel"
    assert _normalize_translation_strategy("") == "guide_parallel"
    assert _normalize_translation_strategy("history") == "history"
    assert _normalize_translation_strategy("serial") == "history"


def test_normalize_translation_strategy_accepts_parallel_aliases():
    from youdub.core.steps.translate import _normalize_translation_strategy

    assert _normalize_translation_strategy("guide_parallel") == "guide_parallel"
    assert _normalize_translation_strategy("parallel") == "guide_parallel"
    assert _normalize_translation_strategy("2") == "guide_parallel"


def test_translate_content_default_history_includes_previous_turns(monkeypatch):
    import youdub.core.steps.translate as tr

    monkeypatch.setattr(tr.time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("TRANSLATION_STRATEGY", "history")

    calls = {"n": 0}

    def _fake_chat_completion_text(_backend, messages):
        calls["n"] += 1

        # The first request should not contain a turn with the first translated output yet.
        if calls["n"] == 1:
            assert not any(
                m.get("role") == "assistant" and "翻译：“你好”" in str(m.get("content", ""))
                for m in messages
            )
            return "你好"

        # The second request should include history from the first request.
        assert any(
            m.get("role") == "assistant" and "翻译：“你好”" in str(m.get("content", ""))
            for m in messages
        )
        return "世界"

    monkeypatch.setattr(tr, "_chat_completion_text", _fake_chat_completion_text)

    summary = {"title": "T", "summary": "S"}
    transcript = [{"text": "This is sentence one with enough length."}, {"text": "This is sentence two with enough length."}]

    out = tr._translate_content(summary, transcript, target_language="简体中文", settings=tr.Settings(openai_api_key="dummy"))
    assert out == ["你好", "世界"]


def test_translate_content_guide_parallel_returns_indexed_translations(monkeypatch):
    import youdub.core.steps.translate as tr

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

