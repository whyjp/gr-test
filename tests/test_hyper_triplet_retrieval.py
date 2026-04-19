"""Tests for graph retrieval — offline, no LLM."""

from __future__ import annotations

from htb.llm import build_replay_mock, load_fixture
from systems.hyper_triplet.extractors import EpisodeRef, LLMNodeSetExtractor
from systems.hyper_triplet.graph import HyperTripletGraph
from systems.hyper_triplet.ltm_creator import HyperTripletLTMCreator
from systems.hyper_triplet.retrieval import (
    RetrievedContext,
    _token_set,
    format_memory_pack,
    retrieve,
    retrieve_facts,
)
from systems.hyper_triplet.types import Fact, NodeSet, Qualifiers

FIXTURE_PATH = "tests/fixtures/locomo_conv26_session1_gold.json"


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


def test_token_set_filters_stopwords():
    assert _token_set("The quick brown fox") == {"quick", "brown", "fox"}
    assert _token_set("") == set()


def test_retrieve_empty_graph_returns_empty():
    g = HyperTripletGraph()
    ctx = retrieve(g, "anything", budget_words=100)
    assert ctx.hits == ()
    assert ctx.context_text == ""


def test_retrieve_empty_query_returns_empty():
    g = _build_fixture_graph()
    ctx = retrieve(g, "", budget_words=100)
    assert ctx.hits == ()


def _empty_creator():
    from systems.hyper_triplet.extractors import LLMNodeSetExtractor as _EX
    from systems.hyper_triplet.ltm_creator import HyperTripletLTMCreator as _LC
    return _LC(extractor=_EX(llm=None))  # type: ignore[arg-type]


def test_retrieve_finds_exact_match():
    ns = NodeSet(
        fact=Fact(subject="Alice", predicate="painted", object="a sunset"),
        source_episode_ids=("ep-1",),
        qualifiers=Qualifiers(location="Paris", topic="artistic_creation"),
    )
    creator = _empty_creator()
    creator.materialise_node_sets([ns])

    ctx = retrieve(creator.graph, "Who painted a sunset?", budget_words=100)
    assert len(ctx.hits) == 1
    assert "painted" in ctx.hits[0].fact_node.content
    assert "painted" in ctx.hits[0].matched_query_tokens


def test_qualifier_text_boosts_retrieval():
    """Query using a qualifier-only term (no overlap with fact text) should
    still retrieve because qualifier content is merged into the searchable doc.
    """
    ns = NodeSet(
        fact=Fact(subject="Alice", predicate="met", object="Bob"),
        qualifiers=Qualifiers(location="Tokyo"),
    )
    creator = _empty_creator()
    creator.materialise_node_sets([ns])

    ctx = retrieve(creator.graph, "who was in tokyo?", budget_words=100)
    # The fact text has no "tokyo" but the location qualifier does.
    assert len(ctx.hits) == 1
    assert "tokyo" in ctx.hits[0].matched_query_tokens


def test_locomo_query_retrieves_painting_fact():
    """Real-data smoke: asking about Melanie's painting surfaces the painting
    fact from the hand-crafted fixture."""
    g = _build_fixture_graph()
    ctx = retrieve(g, "When did Melanie paint a sunrise?", budget_words=200)
    assert ctx.hits, "expected at least one hit"
    top = ctx.hits[0]
    assert "painted" in top.fact_node.content or "painting" in top.fact_node.content
    # Evidence should surface a D1:1x id from session 1
    assert any(eid.startswith("D1:") for eid in ctx.evidence_dia_ids)


def test_locomo_query_retrieves_career_fact():
    g = _build_fixture_graph()
    ctx = retrieve(g, "What fields is Caroline interested in for her career?", budget_words=200)
    assert ctx.hits
    # Evidence should reference D1:9 or D1:11 (the career-related turns)
    assert any(eid in {"D1:9", "D1:11"} for eid in ctx.evidence_dia_ids)


def test_locomo_query_retrieves_lgbtq_fact():
    g = _build_fixture_graph()
    ctx = retrieve(g, "When did Caroline attend a support group?", budget_words=200)
    assert ctx.hits
    # Evidence should reference D1:3 (LGBTQ group turn)
    assert "D1:3" in ctx.evidence_dia_ids


def test_retrieve_ranks_belief_higher_when_scores_close():
    """With identical keyword overlap, the higher-belief fact should rank first."""
    high = NodeSet(
        fact=Fact(subject="Alice", predicate="likes", object="coffee"),
        belief=1.0,
    )
    low = NodeSet(
        fact=Fact(subject="Bob", predicate="likes", object="coffee"),
        belief=0.3,
    )
    creator = _empty_creator()
    creator.materialise_node_sets([high, low])

    ctx = retrieve(creator.graph, "who likes coffee", budget_words=100)
    assert len(ctx.hits) == 2
    assert ctx.hits[0].fact_node.belief >= ctx.hits[1].fact_node.belief


def test_format_memory_pack_includes_qualifiers():
    g = _build_fixture_graph()
    hits = retrieve_facts(g, "painting lake sunrise", top_k=5)
    pack = format_memory_pack(g, hits, include_qualifiers=True)
    assert "## Facts" in pack
    assert "painted" in pack.lower() or "painting" in pack.lower()
    # Should surface at least one qualifier line starting with "    - "
    assert any(line.strip().startswith("- ") and ":" in line for line in pack.split("\n"))


def test_budget_trim_respects_word_limit():
    g = _build_fixture_graph()
    ctx_small = retrieve(g, "Caroline Melanie painting career", budget_words=8)
    ctx_full = retrieve(g, "Caroline Melanie painting career", budget_words=1000)
    assert ctx_small.word_count <= ctx_full.word_count
    # At least one hit always kept so a single fact can exceed budget
    assert len(ctx_small.hits) >= 1


def test_retrieve_returns_retrieved_context_type():
    g = _build_fixture_graph()
    ctx = retrieve(g, "painting", budget_words=100)
    assert isinstance(ctx, RetrievedContext)
    assert hasattr(ctx, "context_text")
    assert hasattr(ctx, "evidence_dia_ids")
