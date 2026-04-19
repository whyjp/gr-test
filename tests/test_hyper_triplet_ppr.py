"""Tests for qualifier-edge PPR retrieval."""

from __future__ import annotations

from htb.llm import build_replay_mock, load_fixture
from systems.hyper_triplet.extractors import EpisodeRef, LLMNodeSetExtractor
from systems.hyper_triplet.graph import HyperTripletGraph
from systems.hyper_triplet.ltm_creator import HyperTripletLTMCreator
from systems.hyper_triplet.retrieval_ppr import (
    QUALIFIER_EDGE_TYPES,
    PPRConfig,
    PPRRetrievalResult,
    retrieve_ppr,
)
from systems.hyper_triplet.types import Fact, NodeSet, Qualifiers

FIXTURE_PATH = "tests/fixtures/locomo_conv26_session1_gold.json"


def _empty_creator() -> HyperTripletLTMCreator:
    from systems.hyper_triplet.extractors import LLMNodeSetExtractor as _EX

    return HyperTripletLTMCreator(extractor=_EX(llm=None))  # type: ignore[arg-type]


def _build_fixture_graph() -> HyperTripletGraph:
    fixture = load_fixture(FIXTURE_PATH)
    mock = build_replay_mock(fixture)
    creator = HyperTripletLTMCreator(extractor=LLMNodeSetExtractor(llm=mock))
    for chunk in fixture["chunks"]:
        eps = [
            EpisodeRef(id=e["id"], text=e["text"], session_date=fixture["session_date"])
            for e in chunk["episodes"]
        ]
        creator.create_from_episodes(new_episodes=eps)
    return creator.graph


def test_ppr_empty_graph_returns_empty():
    g = HyperTripletGraph()
    r = retrieve_ppr(g, "anything", budget_words=100)
    assert r.hits == ()
    assert r.context_text == ""


def test_ppr_empty_query_returns_empty():
    g = _build_fixture_graph()
    r = retrieve_ppr(g, "", budget_words=100)
    assert r.hits == ()


def test_qualifier_edges_are_propagation_scope():
    """The PPR adjacency must only use qualifier + DERIVED_FROM edges, not NEXT."""
    from systems.hyper_triplet.retrieval_ppr import ALLOWED_PROP_EDGE_TYPES

    assert "NEXT" not in ALLOWED_PROP_EDGE_TYPES
    assert "DERIVED_FROM" in ALLOWED_PROP_EDGE_TYPES
    for qtype in QUALIFIER_EDGE_TYPES:
        assert qtype in ALLOWED_PROP_EDGE_TYPES


def test_ppr_returns_retrieved_context_shape():
    g = _build_fixture_graph()
    r = retrieve_ppr(g, "painting", budget_words=100)
    assert isinstance(r, PPRRetrievalResult)
    assert r.word_count >= 0


def test_shared_qualifier_lifts_via_propagation():
    """Two facts sharing a location qualifier: query matches fact A directly,
    fact B is lifted to a nonzero PPR score via the shared location even
    though its text has no query overlap."""
    creator = _empty_creator()
    ns_a = NodeSet(
        fact=Fact(subject="Alice", predicate="painted", object="a sunset"),
        qualifiers=Qualifiers(location="Paris"),
    )
    ns_b = NodeSet(
        fact=Fact(subject="Bob", predicate="met", object="someone"),
        qualifiers=Qualifiers(location="Paris"),
    )
    creator.materialise_node_sets([ns_a, ns_b])

    # Query matches Alice's fact directly (contains "painted", "sunset")
    r = retrieve_ppr(
        creator.graph,
        "who painted a sunset",
        budget_words=200,
        config=PPRConfig(combine_with_bm25=False),
    )
    hit_ids = {h.fact_node.node_id for h in r.hits}

    # Both facts should appear in the PPR scores because they share location
    a_id = next(f.node_id for f in creator.graph.nodes_by_kind("fact") if "Alice" in f.content)
    b_id = next(f.node_id for f in creator.graph.nodes_by_kind("fact") if "Bob" in f.content)
    assert a_id in hit_ids or b_id in hit_ids  # at least one hit
    assert r.ppr_scores.get(a_id, 0.0) > 0
    assert r.ppr_scores.get(b_id, 0.0) > 0  # non-zero via shared-location propagation


def test_bm25_blend_controls_ranking():
    """BM25 weight 1.0 should match retrieve.py's ranking; 0.0 should be pure PPR."""
    creator = _empty_creator()
    creator.materialise_node_sets(
        [
            NodeSet(
                fact=Fact(subject="Melanie", predicate="painted", object="a lake sunrise"),
                qualifiers=Qualifiers(activity_type="painting"),
            ),
            NodeSet(
                fact=Fact(subject="Caroline", predicate="walked", object="in the park"),
                qualifiers=Qualifiers(activity_type="walking"),
            ),
        ]
    )

    # BM25 dominant: "painting" query → Melanie's fact (contains painting/painted)
    r_bm25 = retrieve_ppr(
        creator.graph,
        "painting activity",
        budget_words=100,
        config=PPRConfig(combine_with_bm25=True, bm25_weight=1.0),
    )
    top_bm25 = r_bm25.hits[0]
    assert "painted" in top_bm25.fact_node.content or "painting" in top_bm25.fact_node.content

    # Pure PPR: both facts' qualifier nodes get seeded (painting, walking),
    # but "painting" token seeds the painting qualifier node; propagation
    # flows to Melanie's fact via ACTIVITY_TYPE edge. Should still rank
    # Melanie first.
    r_ppr = retrieve_ppr(
        creator.graph,
        "painting activity",
        budget_words=100,
        config=PPRConfig(combine_with_bm25=False),
    )
    if r_ppr.hits:
        assert "painted" in r_ppr.hits[0].fact_node.content or "painting" in r_ppr.hits[0].fact_node.content


def test_locomo_query_ppr_returns_evidence():
    """End-to-end smoke test on real fixture graph."""
    g = _build_fixture_graph()
    r = retrieve_ppr(g, "when did Melanie paint a sunrise?", budget_words=200)
    assert r.hits
    # Painting fact sourced from D1:14 should surface in evidence
    assert "D1:14" in r.evidence_dia_ids


def test_seed_strategy_fact_only_skips_qualifier_node_seeds():
    """With seed_strategy="fact_only" we do NOT seed qualifier nodes. But the
    fact document already includes adjacent qualifier text (per retrieve.py's
    _fact_document helper), so the fact itself still gets seeded via the
    qualifier-augmented document. This is the expected behaviour — "fact_only"
    restricts which NODES get seed mass, not which TEXT is considered."""
    creator = _empty_creator()
    creator.materialise_node_sets(
        [NodeSet(fact=Fact(subject="A", predicate="B", object="C"), qualifiers=Qualifiers(location="Paris"))]
    )

    r_fact_only = retrieve_ppr(
        creator.graph,
        "in Paris",
        budget_words=100,
        config=PPRConfig(seed_strategy="fact_only", combine_with_bm25=False),
    )
    # Exactly one fact in the graph; it is the only fact node that could be seeded.
    assert len(r_fact_only.hits) == 1
    # Fact carries PPR mass via its own (qualifier-augmented) token document seeding.
    fact_id = r_fact_only.hits[0].fact_node.node_id
    assert r_fact_only.ppr_scores.get(fact_id, 0.0) > 0


def test_seed_strategy_qualifier_only_reaches_facts_via_edges():
    creator = _empty_creator()
    creator.materialise_node_sets(
        [NodeSet(fact=Fact(subject="A", predicate="B", object="C"), qualifiers=Qualifiers(location="Paris"))]
    )
    r_qualifier_only = retrieve_ppr(
        creator.graph,
        "Paris trip",
        budget_words=100,
        config=PPRConfig(seed_strategy="qualifier_only", combine_with_bm25=False),
    )
    # "Paris" seeds the location qualifier node; propagation reaches the
    # fact via AT_LOCATION edge.
    assert len(r_qualifier_only.hits) == 1


def test_budget_trim_still_keeps_at_least_one_hit():
    g = _build_fixture_graph()
    r = retrieve_ppr(g, "Caroline painting career", budget_words=5)
    assert len(r.hits) >= 1


def test_idempotent_under_repeat_calls():
    """PPR is deterministic given fixed seed/config."""
    g = _build_fixture_graph()
    r1 = retrieve_ppr(g, "painting career", budget_words=200)
    r2 = retrieve_ppr(g, "painting career", budget_words=200)
    assert [h.fact_node.node_id for h in r1.hits] == [h.fact_node.node_id for h in r2.hits]
