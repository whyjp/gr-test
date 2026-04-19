"""Tests for MockLLMAdapter — verifies GAAMA-compatible behaviour without any API."""

from __future__ import annotations

import json

import pytest

from htb.llm import (
    LLMAdapter,
    MockLLMAdapter,
    canned_fact_generation_response,
    canned_node_set_generation_response,
    canned_reflection_generation_response,
    make_gaama_mock,
    make_hyper_triplet_mock,
)


def test_protocol_satisfied():
    mock = MockLLMAdapter()
    assert isinstance(mock, LLMAdapter)


def test_raises_on_unmatched_prompt_when_no_default():
    mock = MockLLMAdapter()
    with pytest.raises(LookupError):
        mock.complete("anything")


def test_default_used_when_no_rules_match():
    mock = MockLLMAdapter(default="fallback")
    assert mock.complete("anything") == "fallback"


def test_substring_matcher_first_wins():
    mock = MockLLMAdapter(
        rules=[("alpha", "A"), ("beta", "B"), ("alpha beta", "AB")],
    )
    assert mock.complete("only alpha here") == "A"
    assert mock.complete("only beta here") == "B"
    # "alpha beta" contains both — first rule wins because 'alpha' matches first
    assert mock.complete("alpha beta combined") == "A"


def test_callable_matcher_and_responder_receive_kwargs():
    seen: dict = {}

    def matcher(p: str) -> bool:
        return "use-callable" in p

    def responder(prompt: str, **kwargs) -> str:
        seen.update(kwargs)
        return f"echoed-{len(prompt)}"

    mock = MockLLMAdapter(rules=[(matcher, responder)])
    out = mock.complete(
        "use-callable here",
        system="SYS",
        max_tokens=100,
        model="m",
        temperature=0.2,
    )
    assert out == f"echoed-{len('use-callable here')}"
    assert seen == {
        "system": "SYS",
        "max_tokens": 100,
        "model": "m",
        "temperature": 0.2,
    }


def test_call_log_populated():
    mock = MockLLMAdapter(rules=[("x", "y")])
    mock.complete("x1", max_tokens=50)
    mock.complete("x2", max_tokens=60)
    assert len(mock.calls) == 2
    assert mock.calls[0].prompt == "x1"
    assert mock.calls[0].max_tokens == 50
    assert mock.calls[0].response == "y"
    assert mock.calls[0].matched_rule_index == 0


def test_canned_fact_generation_parses_as_expected_schema():
    raw = canned_fact_generation_response()
    data = json.loads(raw)
    assert set(data.keys()) == {"facts", "concepts"}
    assert data["facts"] and data["concepts"]
    fact = data["facts"][0]
    assert {"fact_text", "belief", "source_episode_ids", "concepts"}.issubset(fact.keys())
    concept = data["concepts"][0]
    assert {"concept_label", "episode_ids"}.issubset(concept.keys())


def test_canned_reflection_generation_parses_as_expected_schema():
    raw = canned_reflection_generation_response()
    data = json.loads(raw)
    assert list(data.keys()) == ["reflections"]
    refl = data["reflections"][0]
    assert {"reflection_text", "belief", "source_fact_ids"}.issubset(refl.keys())


def test_canned_node_set_generation_parses_as_expected_schema():
    raw = canned_node_set_generation_response()
    data = json.loads(raw)
    assert list(data.keys()) == ["node_sets"]
    ns = data["node_sets"][0]
    assert "fact" in ns and "qualifiers" in ns
    fact = ns["fact"]
    assert {"subject", "predicate", "object"} == set(fact.keys())
    qualifiers = ns["qualifiers"]
    assert {"location", "participants", "activity_type", "time_reference", "mood", "topic"}.issubset(
        qualifiers.keys()
    )


def test_gaama_mock_routes_fact_and_reflection_prompts():
    mock = make_gaama_mock()

    fact_prompt = (
        "# Extract facts and concepts from conversation episodes\n\nYou are ..."
    )
    refl_prompt = (
        "# Generate new reflections from facts\n\nYou are an insight generation system..."
    )

    fact_raw = mock.complete(fact_prompt)
    refl_raw = mock.complete(refl_prompt)

    fact_data = json.loads(fact_raw)
    refl_data = json.loads(refl_raw)

    assert "facts" in fact_data and "concepts" in fact_data
    assert "reflections" in refl_data
    assert [c.matched_rule_index for c in mock.calls] == [0, 1]


def test_gaama_mock_rejects_node_set_prompt():
    mock = make_gaama_mock()
    with pytest.raises(LookupError):
        mock.complete("Please produce a node_set")


def test_hyper_triplet_mock_routes_node_set_and_reflection():
    mock = make_hyper_triplet_mock()

    ns_prompt = "Generate a node_set for each fact. qualifiers: ..."
    refl_prompt = "You are an insight generation system ..."

    ns_raw = mock.complete(ns_prompt)
    refl_raw = mock.complete(refl_prompt)
    assert json.loads(ns_raw)["node_sets"]
    assert json.loads(refl_raw)["reflections"]


def test_custom_canned_payload_passthrough():
    custom_facts = [
        {
            "fact_text": "Alice speaks Spanish",
            "belief": 1.0,
            "source_episode_ids": ["ep-xyz"],
            "concepts": ["language_skill"],
        }
    ]
    mock = make_gaama_mock(facts=custom_facts)
    raw = mock.complete("Extract facts and concepts from conversation episodes ...")
    data = json.loads(raw)
    assert data["facts"] == custom_facts


def test_complete_with_full_signature_matches_gaama_protocol():
    """GAAMA's LLMAdapter.complete uses keyword-only kwargs — ensure we accept them."""
    mock = MockLLMAdapter(rules=[("x", "y")])
    out = mock.complete(
        "x test",
        system="you are a tester",
        max_tokens=100,
        model="gpt-test",
        temperature=0.0,
    )
    assert out == "y"
    rec = mock.calls[0]
    assert rec.system == "you are a tester"
    assert rec.model == "gpt-test"
    assert rec.temperature == 0.0
