"""Regression tests anchoring the Hyper Triplet pipeline to hand-crafted
extractions from real LoCoMo-10 conv-26 session_1 turns.

Purpose:
  1. Lock the expected shape of node_set extraction (fact triple + qualifier
     keys) before any real LLM run so future prompt edits are reviewable.
  2. Drive the full LTMCreator pipeline with realistic input/output shapes,
     not just trivial synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from htb.llm import build_replay_mock, load_fixture
from systems.hyper_triplet.extractors import EpisodeRef, LLMNodeSetExtractor
from systems.hyper_triplet.ltm_creator import HyperTripletLTMCreator
from systems.hyper_triplet.types import NodeSet

FIXTURE = Path(__file__).parent / "fixtures" / "locomo_conv26_session1_gold.json"


@pytest.fixture(scope="module")
def gold() -> dict:
    return load_fixture(FIXTURE)


def test_fixture_shape(gold):
    assert gold["source"].startswith("conv-26 session_1")
    assert gold["session_date"] == "8 May 2023"
    assert len(gold["chunks"]) == 3
    for chunk in gold["chunks"]:
        assert chunk["chunk_id"]
        assert chunk["marker"].startswith("D1:")
        assert chunk["episodes"]
        assert chunk["gold_node_sets"]


def test_gold_node_sets_parse_to_valid_models(gold):
    """Every hand-crafted node_set must pass the production pydantic validator."""
    for chunk in gold["chunks"]:
        for raw in chunk["gold_node_sets"]:
            NodeSet.model_validate(raw)


def test_gold_covers_expected_qa_evidence(gold):
    """The three QA pairs in conv-26 that cite D1:3/5/9/11/14 should have
    matching gold node_sets covering their source episode ids."""
    ids_covered = set()
    for chunk in gold["chunks"]:
        for ns in chunk["gold_node_sets"]:
            for sid in ns["source_episode_ids"]:
                ids_covered.add(sid)
    # From the QA inspection earlier: D1:3 (LGBTQ date), D1:5 (identity),
    # D1:9 / D1:11 (career fields), D1:14 (painting year)
    assert {"D1:3", "D1:5", "D1:9", "D1:11", "D1:14"}.issubset(ids_covered)


def test_replay_mock_dispatches_by_chunk_marker(gold):
    mock = build_replay_mock(gold)
    # Each chunk's marker dia_id must be present in its prompt.
    prompt_c1 = "You are an extractor. [D1:3] Caroline: went to support group..."
    prompt_c3 = "You are an extractor. [D1:13] Caroline: thanks for the painting..."
    import json as _json

    r1 = _json.loads(mock.complete(prompt_c1))
    r3 = _json.loads(mock.complete(prompt_c3))

    assert r1["node_sets"][0]["fact"]["subject"] == "Caroline"
    assert r3["node_sets"][0]["fact"]["subject"] == "Melanie"


def test_replay_mock_default_empty_for_unmatched():
    fixture = {"chunks": []}
    mock = build_replay_mock(fixture)
    import json as _json
    assert _json.loads(mock.complete("unrelated prompt"))["node_sets"] == []


def test_end_to_end_pipeline_builds_expected_graph(gold):
    """Drive each chunk through the real LTMCreator using fixture-replay LLM.
    Assert: total fact nodes, cross-chunk qualifier MERGE, edge counts.
    """
    mock = build_replay_mock(gold)
    extractor = LLMNodeSetExtractor(llm=mock)
    creator = HyperTripletLTMCreator(extractor=extractor)

    for chunk in gold["chunks"]:
        episodes = [
            EpisodeRef(id=e["id"], text=e["text"], session_date=gold["session_date"])
            for e in chunk["episodes"]
        ]
        creator.create_from_episodes(new_episodes=episodes)

    stats = creator.graph.stats()

    # Episodes: 8 + 4 + 4 = 16 total (all unique dia_ids)
    assert stats["nodes.episode"] == 16

    # Facts: 3 + 2 + 2 = 7 distinct
    assert stats["nodes.fact"] == 7

    # DERIVED_FROM: one per (fact, source_episode) pair.
    # All 7 facts reference exactly one source episode each.
    assert stats["edges.DERIVED_FROM"] == 7

    # NEXT edges within each chunk only (chunks are ingested separately):
    # chunk1: 7 NEXT, chunk2: 3 NEXT, chunk3: 3 NEXT = 13
    assert stats["edges.NEXT"] == 13

    # Topic MERGE check: "lgbtq_support" appears twice in chunk1 (NS1 and NS3).
    # Must collapse to a single topic node, with 2 ABOUT_TOPIC edges to it.
    topic_nodes = creator.graph.qualifier_nodes("topic")
    topic_values = sorted(n.content for n in topic_nodes)
    # Distinct topics across all chunks:
    #   lgbtq_support, lgbtq_identity, education_plan,
    #   career_plan, artistic_creation
    assert topic_values == [
        "artistic_creation",
        "career_plan",
        "education_plan",
        "lgbtq_identity",
        "lgbtq_support",
    ]

    # lgbtq_support has 2 incoming ABOUT_TOPIC edges (from fact 1 and fact 3)
    from systems.hyper_triplet.graph import qualifier_node_id

    lgbtq_support_id = qualifier_node_id("topic", "lgbtq_support")
    about_topic_edges_to_lgbtq = [
        e for e in creator.graph.edges_of_type("ABOUT_TOPIC") if e.target_id == lgbtq_support_id
    ]
    assert len(about_topic_edges_to_lgbtq) == 2

    # artistic_creation has 2 incoming ABOUT_TOPIC edges (from chunk3 NS1 and NS2)
    art_id = qualifier_node_id("topic", "artistic_creation")
    about_topic_art = [
        e for e in creator.graph.edges_of_type("ABOUT_TOPIC") if e.target_id == art_id
    ]
    assert len(about_topic_art) == 2

    # activity_type "painting" MERGE across chunk3 NS1 and NS2
    painting_id = qualifier_node_id("activity_type", "painting")
    painting_edges = [
        e for e in creator.graph.edges_of_type("ACTIVITY_TYPE") if e.target_id == painting_id
    ]
    assert len(painting_edges) == 2


def test_temporal_qualifier_preserved(gold):
    """The painting year (QA cat 2 gold answer: 2022) must surface as a
    time_reference qualifier on the painting fact."""
    mock = build_replay_mock(gold)
    extractor = LLMNodeSetExtractor(llm=mock)
    # Feed only chunk 3
    chunk3 = next(c for c in gold["chunks"] if c["chunk_id"].endswith("c3"))
    episodes = [EpisodeRef(id=e["id"], text=e["text"]) for e in chunk3["episodes"]]
    results = extractor.extract_node_sets(new_episodes=episodes)

    painting_fact = next(
        ns for ns in results if "painted" in ns.fact.predicate and "lake sunrise" in ns.fact.object
    )
    assert painting_fact.qualifiers.time_reference == "2022"


def test_transgender_identity_inferred_with_lower_belief(gold):
    """Cat 1 QA gold: 'Transgender woman' at evidence D1:5. The hand-crafted
    extraction marks this as inferred (belief < explicit-statement floor)."""
    chunk1 = next(c for c in gold["chunks"] if c["chunk_id"].endswith("c1"))
    ns_identity = next(
        ns for ns in chunk1["gold_node_sets"] if ns["fact"]["object"] == "a transgender woman"
    )
    # Inferred facts should have belief < 0.9 (we chose 0.75)
    assert ns_identity["belief"] < 0.9
