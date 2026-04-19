"""Unit tests for baseline Pipeline adapter stubs — offline only.

Verifies every adapter:
- Imports without side effects (zero network, zero upstream dep at import time)
- Satisfies the Pipeline protocol
- Raises a clear PipelineNotReadyError with a descriptive readiness hint
  when ingest/retrieve/answer is called before wire-up
- Exposes a status() introspection dict
"""

from __future__ import annotations

import pytest

from htb.data.locomo import Conversation, QAPair, Session, Turn
from htb.eval.interfaces import Pipeline, RetrievalResult
from systems.baselines import (
    BaselineAdapter,
    EverMemOSAdapter,
    GaamaAdapter,
    HippoRAGAdapter,
    HyperGraphRAGAdapter,
    HyperMemAdapter,
    PipelineNotReadyError,
)

ALL_ADAPTERS = [
    GaamaAdapter,
    HippoRAGAdapter,
    HyperGraphRAGAdapter,
    HyperMemAdapter,
    EverMemOSAdapter,
]


def _tiny_conv() -> Conversation:
    t = Turn(speaker="A", dia_id="D1:1", text="hello", session_index=1)
    s = Session(index=1, date_time="t", turns=(t,))
    return Conversation(
        sample_id="c1",
        speaker_a="A",
        speaker_b="B",
        sessions=(s,),
        qa=(QAPair(question="q", answer="a", category=1),),
    )


@pytest.mark.parametrize("cls", ALL_ADAPTERS)
def test_adapter_satisfies_pipeline_protocol(cls):
    adapter = cls()
    assert isinstance(adapter, Pipeline)
    assert isinstance(adapter, BaselineAdapter)


@pytest.mark.parametrize("cls", ALL_ADAPTERS)
def test_adapter_reset_does_not_raise(cls):
    adapter = cls()
    adapter.reset()
    # reset should clear the last-ingested-conv id even on a fresh adapter
    assert adapter.status()["last_ingested_conv_id"] is None


@pytest.mark.parametrize("cls", ALL_ADAPTERS)
def test_ingest_raises_pipeline_not_ready(cls):
    adapter = cls()
    with pytest.raises(PipelineNotReadyError) as excinfo:
        adapter.ingest(_tiny_conv())
    # The message must carry the adapter name AND a non-trivial hint
    msg = str(excinfo.value)
    assert adapter.name in msg
    assert "not ready" in msg.lower()
    assert len(msg) > 100  # non-trivial hint


@pytest.mark.parametrize("cls", ALL_ADAPTERS)
def test_retrieve_raises_pipeline_not_ready(cls):
    adapter = cls()
    with pytest.raises(PipelineNotReadyError):
        adapter.retrieve("anything")


@pytest.mark.parametrize("cls", ALL_ADAPTERS)
def test_answer_raises_pipeline_not_ready(cls):
    adapter = cls()
    empty_ctx = RetrievalResult(context="", word_count=0)
    with pytest.raises(PipelineNotReadyError):
        adapter.answer("q", empty_ctx)


@pytest.mark.parametrize("cls", ALL_ADAPTERS)
def test_external_path_points_to_real_clone_or_submodule(cls):
    adapter = cls()
    path = adapter.external_path
    # Path must point under external/, exist, and have a README.md / any child
    assert "external" in str(path).replace("\\", "/")


@pytest.mark.parametrize("cls", ALL_ADAPTERS)
def test_status_dict_shape(cls):
    adapter = cls()
    status = adapter.status()
    assert set(status.keys()) >= {"name", "ready", "external_path"}
    assert status["ready"] is False  # stubs are never ready


def test_names_are_distinct():
    """MultiSystemRunner keys off Pipeline.name — must be unique."""
    names = {cls().name for cls in ALL_ADAPTERS}
    assert len(names) == len(ALL_ADAPTERS)


def test_evermemos_honours_env_url(monkeypatch):
    monkeypatch.setenv("EVERMEMOS_API_URL", "http://remote.internal:8080")
    adapter = EverMemOSAdapter()
    assert adapter.api_base_url == "http://remote.internal:8080"


def test_evermemos_default_url_when_env_unset(monkeypatch):
    monkeypatch.delenv("EVERMEMOS_API_URL", raising=False)
    adapter = EverMemOSAdapter()
    assert adapter.api_base_url == "http://localhost:1995"


def test_each_adapter_readiness_hint_mentions_path_or_api():
    """Hint text must give the operator enough to act on."""
    for cls in ALL_ADAPTERS:
        adapter = cls()
        hint = adapter._readiness_hint()
        # Must describe either the upstream path, an API key, or docker-compose
        indicators = ("external/", "API_KEY", "docker", "sys.path", "vLLM")
        assert any(
            ind in hint for ind in indicators
        ), f"{adapter.name} hint lacks actionable indicator: {hint[:200]}"
