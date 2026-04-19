"""Offline tests for HyperTripletLTMCreator and HyperTripletGraph.

Exercises MERGE semantics, typed-edge shape, and cross-chunk qualifier reuse
without touching GAAMA, SQLite, or any LLM API.
"""

from __future__ import annotations

import json

from htb.llm import MockLLMAdapter, canned_node_set_generation_response
from systems.hyper_triplet.extractors import EpisodeRef, LLMNodeSetExtractor
from systems.hyper_triplet.graph import (
    EDGE_FACT_TO_EPISODE,
    EDGE_NEXT_EPISODE,
    EDGE_TYPE_BY_QUALIFIER,
    HyperTripletGraph,
    episode_node_id,
    fact_node_id,
    qualifier_node_id,
)
from systems.hyper_triplet.ltm_creator import HyperTripletLTMCreator
from systems.hyper_triplet.types import Fact, NodeSet, Qualifiers


def _make_creator_with_nodesets(node_sets: list[dict]) -> HyperTripletLTMCreator:
    raw = canned_node_set_generation_response(node_sets=node_sets)
    mock = MockLLMAdapter(rules=[("node_set", raw)])
    return HyperTripletLTMCreator(extractor=LLMNodeSetExtractor(llm=mock))


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------


def test_empty_graph_stats():
    g = HyperTripletGraph()
    s = g.stats()
    assert s["n_nodes"] == 0
    assert s["n_edges"] == 0


def test_merge_qualifier_normalises_case_and_whitespace():
    g = HyperTripletGraph()
    id1 = g.merge_qualifier("location", "Seattle")
    id2 = g.merge_qualifier("location", "  SEATTLE  ")
    id3 = g.merge_qualifier("location", "seattle")
    assert id1 == id2 == id3
    assert len(g.qualifier_nodes("location")) == 1


def test_merge_qualifier_different_types_dont_collide():
    g = HyperTripletGraph()
    loc = g.merge_qualifier("location", "Alice")
    participant = g.merge_qualifier("participant", "Alice")
    assert loc != participant
    assert len(g.qualifier_nodes()) == 2


def test_merge_qualifier_empty_value_returns_empty_string():
    g = HyperTripletGraph()
    assert g.merge_qualifier("location", "") == ""
    assert g.merge_qualifier("location", "   ") == ""
    assert g.nodes == {}


def test_fact_node_id_deterministic():
    id1 = fact_node_id("Alice enjoys painting")
    id2 = fact_node_id("Alice enjoys painting")
    id3 = fact_node_id("alice enjoys painting")
    assert id1 == id2
    assert id1 == id3  # case-folded


def test_qualifier_node_id_deterministic():
    id1 = qualifier_node_id("location", "paris")
    id2 = qualifier_node_id("location", "paris")
    assert id1 == id2


# ---------------------------------------------------------------------------
# ingest_episodes
# ---------------------------------------------------------------------------


def test_ingest_episodes_creates_nodes_and_next_chain():
    creator = _make_creator_with_nodesets([])
    eps = [
        EpisodeRef(id="ep-1", text="turn 1"),
        EpisodeRef(id="ep-2", text="turn 2"),
        EpisodeRef(id="ep-3", text="turn 3"),
    ]
    ids = creator.ingest_episodes(eps)
    assert ids == [episode_node_id("ep-1"), episode_node_id("ep-2"), episode_node_id("ep-3")]

    g = creator.graph
    assert len(g.nodes_by_kind("episode")) == 3
    next_edges = g.edges_of_type(EDGE_NEXT_EPISODE)
    assert len(next_edges) == 2
    assert (next_edges[0].source_id, next_edges[0].target_id) == (ids[0], ids[1])
    assert (next_edges[1].source_id, next_edges[1].target_id) == (ids[1], ids[2])


def test_ingest_single_episode_no_next_edge():
    creator = _make_creator_with_nodesets([])
    creator.ingest_episodes([EpisodeRef(id="ep-only", text="solo")])
    assert creator.graph.edges_of_type(EDGE_NEXT_EPISODE) == []


# ---------------------------------------------------------------------------
# materialise_node_sets — single node_set
# ---------------------------------------------------------------------------


def test_materialise_single_nodeset_creates_fact_qualifiers_and_edges():
    creator = _make_creator_with_nodesets([])
    creator.ingest_episodes([EpisodeRef(id="ep-1", text="Alice went to Paris with Bob.")])

    ns = NodeSet(
        fact=Fact(subject="Alice", predicate="visited", object="Paris"),
        source_episode_ids=("ep-1",),
        belief=0.9,
        qualifiers=Qualifiers(
            location="Paris",
            participants=("Bob",),
            activity_type="trip",
            mood="excited",
            topic="travel",
        ),
    )
    fact_ids = creator.materialise_node_sets([ns])
    assert len(fact_ids) == 1

    g = creator.graph
    assert len(g.nodes_by_kind("fact")) == 1
    # 5 qualifier nodes: location, participant, activity_type, mood, topic (time_reference not set)
    assert len(g.qualifier_nodes()) == 5

    # Edges: 1 DERIVED_FROM + 5 typed qualifier edges
    assert len(g.edges_of_type(EDGE_FACT_TO_EPISODE)) == 1
    for qtype, edge_type in EDGE_TYPE_BY_QUALIFIER.items():
        if qtype == "time_reference":
            continue
        assert len(g.edges_of_type(edge_type)) == 1, f"missing edge for {qtype}"


def test_materialise_skips_edges_to_unknown_episodes():
    """If source_episode_id references a non-existent episode, no edge is made.
    Fact node is still created."""
    creator = _make_creator_with_nodesets([])
    ns = NodeSet(
        fact=Fact(subject="A", predicate="B", object="C"),
        source_episode_ids=("ep-ghost",),
    )
    creator.materialise_node_sets([ns])
    assert len(creator.graph.nodes_by_kind("fact")) == 1
    assert creator.graph.edges_of_type(EDGE_FACT_TO_EPISODE) == []


# ---------------------------------------------------------------------------
# Cross-chunk MERGE behaviour
# ---------------------------------------------------------------------------


def test_two_chunks_same_location_merge_to_one_node():
    creator = _make_creator_with_nodesets([])
    creator.ingest_episodes(
        [
            EpisodeRef(id="ep-1", text="Alice visited Paris."),
            EpisodeRef(id="ep-2", text="Bob also went to paris."),
        ]
    )
    ns_a = NodeSet(
        fact=Fact(subject="Alice", predicate="visited", object="Paris"),
        source_episode_ids=("ep-1",),
        qualifiers=Qualifiers(location="Paris"),
    )
    ns_b = NodeSet(
        fact=Fact(subject="Bob", predicate="went to", object="Paris"),
        source_episode_ids=("ep-2",),
        qualifiers=Qualifiers(location="paris"),  # different case
    )
    creator.materialise_node_sets([ns_a, ns_b])

    g = creator.graph
    assert len(g.nodes_by_kind("fact")) == 2
    assert len(g.qualifier_nodes("location")) == 1, "location merged to one node"
    at_loc = g.edges_of_type("AT_LOCATION")
    assert len(at_loc) == 2, "both facts edge to the shared location"
    # Both AT_LOCATION edges point to the same target
    assert at_loc[0].target_id == at_loc[1].target_id


def test_two_chunks_same_fact_do_not_duplicate():
    creator = _make_creator_with_nodesets([])
    ns = NodeSet(fact=Fact(subject="A", predicate="B", object="C"))
    # Materialise same node_set twice in different calls
    creator.materialise_node_sets([ns])
    creator.materialise_node_sets([ns])
    assert len(creator.graph.nodes_by_kind("fact")) == 1


def test_materialise_returns_fact_ids_dedup_within_batch():
    """If a node_set batch contains duplicate facts, only one id returned."""
    creator = _make_creator_with_nodesets([])
    ns = NodeSet(fact=Fact(subject="A", predicate="B", object="C"))
    ids = creator.materialise_node_sets([ns, ns, ns])
    assert len(ids) == 1


# ---------------------------------------------------------------------------
# Participants fan-out
# ---------------------------------------------------------------------------


def test_participants_fan_out_to_separate_nodes():
    creator = _make_creator_with_nodesets([])
    ns = NodeSet(
        fact=Fact(subject="Group", predicate="gathered", object="at park"),
        qualifiers=Qualifiers(participants=("Alice", "Bob", "Carol")),
    )
    creator.materialise_node_sets([ns])

    g = creator.graph
    assert len(g.qualifier_nodes("participant")) == 3
    with_p = g.edges_of_type("WITH_PARTICIPANT")
    assert len(with_p) == 3
    # All point to distinct targets
    targets = {e.target_id for e in with_p}
    assert len(targets) == 3


def test_participant_across_facts_shares_one_node():
    creator = _make_creator_with_nodesets([])
    ns_a = NodeSet(
        fact=Fact(subject="Alice", predicate="met", object="Bob"),
        qualifiers=Qualifiers(participants=("Bob",)),
    )
    ns_b = NodeSet(
        fact=Fact(subject="Carol", predicate="called", object="Bob"),
        qualifiers=Qualifiers(participants=("Bob",)),
    )
    creator.materialise_node_sets([ns_a, ns_b])
    assert len(creator.graph.qualifier_nodes("participant")) == 1


# ---------------------------------------------------------------------------
# End-to-end through extractor
# ---------------------------------------------------------------------------


def test_end_to_end_create_from_episodes_uses_mock_llm():
    custom = [
        {
            "fact": {"subject": "Melanie", "predicate": "painted", "object": "lake"},
            "source_episode_ids": ["ep-1"],
            "belief": 0.9,
            "qualifiers": {
                "location": "cabin by the lake",
                "participants": ["Melanie"],
                "topic": "art",
            },
        }
    ]
    raw = canned_node_set_generation_response(node_sets=custom)
    mock = MockLLMAdapter(rules=[("node_set", raw)])
    creator = HyperTripletLTMCreator(extractor=LLMNodeSetExtractor(llm=mock))

    fact_ids = creator.create_from_episodes(
        new_episodes=[EpisodeRef(id="ep-1", text="Melanie painted a lake sunrise.")],
    )
    assert len(fact_ids) == 1

    stats = creator.graph.stats()
    assert stats["nodes.episode"] == 1
    assert stats["nodes.fact"] == 1
    assert stats["nodes.qualifier"] == 3  # location, participant, topic
    assert stats["edges.DERIVED_FROM"] == 1
    assert stats["edges.AT_LOCATION"] == 1
    assert stats["edges.WITH_PARTICIPANT"] == 1
    assert stats["edges.ABOUT_TOPIC"] == 1


def test_end_to_end_no_nodesets_yields_only_episodes():
    mock = MockLLMAdapter(rules=[("node_set", json.dumps({"node_sets": []}))])
    creator = HyperTripletLTMCreator(extractor=LLMNodeSetExtractor(llm=mock))
    creator.create_from_episodes(
        new_episodes=[EpisodeRef(id="ep-1", text="uninteresting chatter")],
    )
    stats = creator.graph.stats()
    assert stats["nodes.episode"] == 1
    assert stats.get("nodes.fact", 0) == 0
    assert stats.get("nodes.qualifier", 0) == 0
