"""Offline smoke test for LLMNodeSetExtractor via MockLLMAdapter."""

from __future__ import annotations

import json

import pytest

from htb.llm import (
    MockLLMAdapter,
    canned_node_set_generation_response,
    make_hyper_triplet_mock,
)
from systems.hyper_triplet.extractors import (
    EpisodeRef,
    FactRef,
    LLMNodeSetExtractor,
    _format_episodes,
    _format_facts,
    _format_qualifiers_by_type,
    render_prompt,
)
from systems.hyper_triplet.types import NodeSet, Qualifiers, merge_key


def test_render_prompt_substitutes_all_placeholders():
    template = "Hello {{name}}, you are {{role}}."
    out = render_prompt(template, {"name": "Alice", "role": "engineer"})
    assert out == "Hello Alice, you are engineer."


def test_format_episodes_empty_and_with_date():
    assert _format_episodes([]) == "(none)"
    eps = [
        EpisodeRef(id="ep-1", text="hello", session_date="2023-05-08"),
        EpisodeRef(id="ep-2", text="world"),
    ]
    out = _format_episodes(eps)
    assert "[ep-1] [2023-05-08] hello" in out
    assert "[ep-2] world" in out


def test_format_facts_empty_and_populated():
    assert _format_facts([]) == "(none)"
    facts = [FactRef(id="f-1", text="A is B")]
    assert _format_facts(facts) == "[f-1] A is B"


def test_format_qualifiers_by_type_groups_properly():
    qmap = {
        "location": ["cabin by the lake", "downtown Seattle"],
        "participant": ["Melanie", "Caroline"],
        "mood": [],  # skipped
    }
    out = _format_qualifiers_by_type(qmap)
    assert "location:" in out
    assert "  - cabin by the lake" in out
    assert "participant:" in out
    assert "mood:" not in out  # empty list skipped


def test_format_qualifiers_by_type_all_empty_returns_none():
    assert _format_qualifiers_by_type({}) == "(none)"
    assert _format_qualifiers_by_type({"location": [], "mood": []}) == "(none)"


def test_node_set_validation_from_canned_response():
    raw = canned_node_set_generation_response()
    data = json.loads(raw)
    ns = NodeSet.model_validate(data["node_sets"][0])
    assert ns.fact.subject == "Melanie"
    assert ns.qualifiers.location == "cabin by the lake"
    assert ns.qualifiers.participants == ("Melanie",)
    assert ns.belief == 0.9


def test_qualifiers_iter_typed_values():
    q = Qualifiers(
        location="Paris",
        participants=("Alice", "Bob"),
        mood="happy",
        topic="travel",
    )
    pairs = list(q.iter_typed_values())
    assert ("location", "Paris") in pairs
    assert ("participant", "Alice") in pairs
    assert ("participant", "Bob") in pairs
    assert ("mood", "happy") in pairs
    assert ("topic", "travel") in pairs
    assert len(pairs) == 5
    # No activity_type or time_reference because they weren't set
    types = [t for t, _ in pairs]
    assert "activity_type" not in types


def test_qualifiers_strips_whitespace_and_skips_empties():
    q = Qualifiers(location="  ", participants=("", "Alice", "  "), mood="  happy  ")
    pairs = list(q.iter_typed_values())
    assert pairs == [("participant", "Alice"), ("mood", "happy")]


def test_merge_key_normalises_case_and_whitespace():
    assert merge_key("location", "Cabin By The Lake") == ("location", "cabin by the lake")
    assert merge_key("topic", "  Painting_Hobby  ") == ("topic", "painting_hobby")


def test_belief_clamped_to_unit_interval():
    ns = NodeSet.model_validate(
        {"fact": {"subject": "a", "predicate": "b", "object": "c"}, "belief": 1.5}
    )
    assert ns.belief == 1.0
    ns2 = NodeSet.model_validate(
        {"fact": {"subject": "a", "predicate": "b", "object": "c"}, "belief": -0.2}
    )
    assert ns2.belief == 0.0
    ns3 = NodeSet.model_validate(
        {"fact": {"subject": "a", "predicate": "b", "object": "c"}, "belief": "not a number"}
    )
    assert ns3.belief == 1.0


def test_end_to_end_extract_with_mock_llm():
    mock = make_hyper_triplet_mock()
    extractor = LLMNodeSetExtractor(llm=mock)

    results = extractor.extract_node_sets(
        new_episodes=[EpisodeRef(id="ep-mock-1", text="Melanie painted a lake sunrise.")],
    )

    assert len(results) == 1
    ns = results[0]
    assert ns.fact.subject == "Melanie"
    assert ns.qualifiers.location == "cabin by the lake"
    assert ns.qualifiers.topic == "artistic_creation"
    # MockLLM recorded the call with the rendered prompt
    assert len(mock.calls) == 1
    assert "node_set" in mock.calls[0].prompt
    assert "ep-mock-1" in mock.calls[0].prompt


def test_end_to_end_extract_substitutes_context_into_prompt():
    mock = make_hyper_triplet_mock()
    extractor = LLMNodeSetExtractor(llm=mock)

    extractor.extract_node_sets(
        new_episodes=[EpisodeRef(id="ep-new", text="Alice moved to Tokyo.")],
        related_episodes=[EpisodeRef(id="ep-old", text="Alice studied Japanese.")],
        existing_facts=[FactRef(id="f-1", text="Alice is a software engineer")],
        existing_qualifiers_by_type={"location": ["Seoul", "Tokyo"]},
    )

    prompt = mock.calls[0].prompt
    assert "[ep-new] Alice moved to Tokyo." in prompt
    assert "[ep-old] Alice studied Japanese." in prompt
    assert "[f-1] Alice is a software engineer" in prompt
    assert "location:" in prompt
    assert "  - Seoul" in prompt
    assert "  - Tokyo" in prompt


def test_empty_node_sets_array_is_valid():
    mock = MockLLMAdapter(rules=[("node_set", '{"node_sets": []}')])
    extractor = LLMNodeSetExtractor(llm=mock)
    assert extractor.extract_node_sets([EpisodeRef(id="e", text="t")]) == []


def test_malformed_item_skipped_not_crashed():
    # Two items: first is malformed (no fact), second is valid
    canned = json.dumps(
        {
            "node_sets": [
                {"belief": 0.5},  # missing fact — ValidationError
                {
                    "fact": {"subject": "X", "predicate": "is", "object": "Y"},
                    "source_episode_ids": ["e1"],
                    "belief": 0.9,
                },
            ]
        }
    )
    mock = MockLLMAdapter(rules=[("node_set", canned)])
    extractor = LLMNodeSetExtractor(llm=mock)
    results = extractor.extract_node_sets([EpisodeRef(id="e1", text="t")])
    assert len(results) == 1
    assert results[0].fact.subject == "X"


def test_json_fenced_response_is_unwrapped():
    fenced = "```json\n" + canned_node_set_generation_response() + "\n```"
    mock = MockLLMAdapter(rules=[("node_set", fenced)])
    extractor = LLMNodeSetExtractor(llm=mock)
    results = extractor.extract_node_sets([EpisodeRef(id="e1", text="t")])
    assert len(results) == 1


def test_non_dict_response_returns_empty():
    mock = MockLLMAdapter(rules=[("node_set", "this is not JSON")])
    extractor = LLMNodeSetExtractor(llm=mock, max_tokens=100)
    # Retry path kicks in but mock always returns the same non-JSON; should
    # exhaust retries and return empty.
    results = extractor.extract_node_sets([EpisodeRef(id="e1", text="t")])
    assert results == []
    # Original + 2 retries = 3 calls
    assert len(mock.calls) == 3


def test_top_level_list_response_accepted():
    # Some LLMs skip the wrapper and return just the array.
    list_json = json.dumps(
        [
            {
                "fact": {"subject": "A", "predicate": "B", "object": "C"},
                "source_episode_ids": ["e1"],
                "belief": 1.0,
            }
        ]
    )
    mock = MockLLMAdapter(rules=[("node_set", list_json)])
    extractor = LLMNodeSetExtractor(llm=mock)
    results = extractor.extract_node_sets([EpisodeRef(id="e1", text="t")])
    assert len(results) == 1
    assert results[0].fact.subject == "A"


def test_prompt_template_override():
    """Passing a custom template bypasses the markdown file."""
    custom = "TEST TEMPLATE node_set episodes={{new_episodes}}"
    mock = MockLLMAdapter(rules=[("node_set", '{"node_sets": []}')])
    extractor = LLMNodeSetExtractor(llm=mock, prompt_template=custom)
    extractor.extract_node_sets([EpisodeRef(id="e", text="t")])
    prompt = mock.calls[0].prompt
    assert prompt.startswith("TEST TEMPLATE node_set episodes=")
    assert "[e] t" in prompt


@pytest.mark.parametrize(
    "custom_ns",
    [
        [
            {
                "fact": {"subject": "S1", "predicate": "P1", "object": "O1"},
                "source_episode_ids": ["e1"],
                "belief": 0.8,
                "qualifiers": {"location": "home"},
            },
            {
                "fact": {"subject": "S2", "predicate": "P2", "object": "O2"},
                "source_episode_ids": ["e1", "e2"],
                "belief": 1.0,
                "qualifiers": {"participants": ["Alice", "Bob"], "topic": "t"},
            },
        ],
    ],
)
def test_extract_preserves_order_and_fields(custom_ns):
    raw = canned_node_set_generation_response(node_sets=custom_ns)
    mock = MockLLMAdapter(rules=[("node_set", raw)])
    extractor = LLMNodeSetExtractor(llm=mock)
    results = extractor.extract_node_sets([EpisodeRef(id="e1", text="t")])
    assert len(results) == 2
    assert results[0].fact.subject == "S1"
    assert results[1].qualifiers.participants == ("Alice", "Bob")
    assert results[1].qualifiers.topic == "t"
