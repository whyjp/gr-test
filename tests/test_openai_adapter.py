"""Unit tests for OpenAIAdapter that never make network calls."""

from __future__ import annotations

import pytest

from htb.llm import LLMAdapter, OpenAIAdapter


def test_adapter_satisfies_protocol():
    adapter = OpenAIAdapter()
    assert isinstance(adapter, LLMAdapter)


def test_adapter_defers_client_construction():
    """Creating the adapter must NOT require OPENAI_API_KEY; only complete()
    does. This keeps import-time safe on hosts without the key."""
    adapter = OpenAIAdapter(api_key_env="NONEXISTENT_KEY_FOR_TEST")
    # No exception on construction.
    assert adapter._client is None


def test_complete_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = OpenAIAdapter()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        adapter.complete("hello")


def test_complete_uses_model_override(monkeypatch):
    """When caller passes model=, it should override the default."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMessage(content)

    class FakeResponse:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def __init__(self, adapter):
            self.adapter = adapter

        def create(self, **kwargs):
            self.adapter._last_kwargs = kwargs
            return FakeResponse("hello from " + kwargs["model"])

    class FakeChat:
        def __init__(self, adapter):
            self.completions = FakeCompletions(adapter)

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = FakeChat(self)

    import htb.llm.openai_adapter as mod

    monkeypatch.setattr(mod, "OpenAI", FakeClient, raising=False)

    adapter = OpenAIAdapter(default_model="gpt-4o-mini")
    # force client construction via module-patched FakeClient
    adapter._client = FakeClient()
    result = adapter.complete("hi", model="gpt-5-test")
    assert result == "hello from gpt-5-test"


def test_factory_functions_respect_env_model(monkeypatch):
    monkeypatch.setenv("EXTRACT_MODEL", "gpt-4o-mini-custom")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4o-custom")
    from htb.llm import make_default_extract_adapter, make_default_judge_adapter

    assert make_default_extract_adapter().default_model == "gpt-4o-mini-custom"
    assert make_default_judge_adapter().default_model == "gpt-4o-custom"


def test_factories_default_to_recommended_models(monkeypatch):
    monkeypatch.delenv("EXTRACT_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    from htb.llm import make_default_extract_adapter, make_default_judge_adapter

    assert make_default_extract_adapter().default_model == "gpt-4o-mini"
    assert make_default_judge_adapter().default_model == "gpt-4o"
