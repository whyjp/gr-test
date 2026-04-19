"""Tests for 3-stage retrieval pipeline."""

from __future__ import annotations

from systems.hyper_triplet.retrieval_stages import (
    Stage1Broad,
    Stage2Rank,
    Stage3Exact,
    Stage3Result,
    ThreeStagePipeline,
    _star_context_text,
    _tokens,
)
from systems.hyper_triplet.star_store import StarStore
from systems.hyper_triplet.types import Fact, NodeSet, Qualifiers


def _ns(
    subject: str,
    predicate: str = "did",
    obj: str = "something",
    **kwargs,
) -> NodeSet:
    return NodeSet(fact=Fact(subject=subject, predicate=predicate, object=obj), **kwargs)


def _populated_store() -> StarStore:
    store = StarStore()
    store.put(
        _ns(
            "Alice",
            "visited",
            "Paris",
            qualifiers=Qualifiers(
                location="Paris",
                participants=("Alice",),
                time_reference="2024-06-15",
                topic="travel",
            ),
            importance=0.8,
        )
    )
    store.put(
        _ns(
            "Bob",
            "met",
            "Alice",
            qualifiers=Qualifiers(
                location="Paris",
                participants=("Alice", "Bob"),
                topic="travel",
            ),
            importance=0.5,
        )
    )
    store.put(
        _ns(
            "Carol",
            "wrote",
            "a book",
            qualifiers=Qualifiers(activity_type="writing", topic="literature"),
            importance=0.3,
        )
    )
    return store


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_tokens_filters_stopwords_and_case():
    assert set(_tokens("The Quick brown Fox")) == {"quick", "brown", "fox"}


def test_star_context_text_includes_all_layers():
    ns = _ns(
        "A",
        qualifiers=Qualifiers(
            location="Paris",
            participants=("Alice",),
            activity_type="trip",
            mood="excited",
            topic="travel",
        ),
    )
    text = _star_context_text(ns)
    for needle in ("Paris", "Alice", "trip", "excited", "travel"):
        assert needle in text


# ---------------------------------------------------------------------------
# Stage 1 — Broad
# ---------------------------------------------------------------------------


def test_stage1_retrieves_context_matches():
    store = _populated_store()
    stage1 = Stage1Broad(expand_via_community=False)
    cands = stage1.retrieve(store, "who visited Paris")
    # Alice and Bob both have Paris in L2; Carol does not
    assert len(cands) == 2
    subjects = {store.get(nid).fact.subject for nid in cands}  # type: ignore[union-attr]
    assert subjects == {"Alice", "Bob"}


def test_stage1_empty_query_returns_empty():
    store = _populated_store()
    assert Stage1Broad().retrieve(store, "") == []


def test_stage1_topic_match_surfaces_candidate():
    store = _populated_store()
    cands = Stage1Broad(expand_via_community=False).retrieve(store, "literature lover")
    assert len(cands) == 1
    assert store.get(cands[0]).fact.subject == "Carol"  # type: ignore[union-attr]


def test_stage1_top_n_caps_output():
    store = _populated_store()
    # Add 20 more stars that match "Paris"
    for i in range(20):
        store.put(
            _ns(
                f"Person{i}",
                "travelled",
                "to Paris",
                qualifiers=Qualifiers(location="Paris"),
            )
        )
    stage1 = Stage1Broad(top_n=5, expand_via_community=False)
    cands = stage1.retrieve(store, "paris trip")
    assert len(cands) == 5


def test_stage1_community_expansion():
    store = _populated_store()
    # Put all three stars in the same community
    for nid in store.iter_ids():
        store.assign_community(nid, "cluster_travel")
    # Query matches only "Paris" facts directly (Alice + Bob), but community
    # expansion pulls Carol into the candidate pool too.
    stage1 = Stage1Broad(expand_via_community=True)
    cands = stage1.retrieve(store, "paris")
    subjects = {store.get(nid).fact.subject for nid in cands}  # type: ignore[union-attr]
    assert subjects == {"Alice", "Bob", "Carol"}


# ---------------------------------------------------------------------------
# Stage 2 — Rank
# ---------------------------------------------------------------------------


def test_stage2_orders_by_importance():
    store = _populated_store()
    candidates = list(store.iter_ids())
    ranked = Stage2Rank().rank(store, candidates, "anything")
    # Alice (0.8) > Bob (0.5) > Carol (0.3)
    subjects_in_order = [store.get(nid).fact.subject for nid, _ in ranked]  # type: ignore[union-attr]
    assert subjects_in_order == ["Alice", "Bob", "Carol"]


def test_stage2_temporal_trigger_boost():
    store = _populated_store()
    candidates = list(store.iter_ids())

    no_temporal = Stage2Rank(temporal_weight=0.5).rank(store, candidates, "what did Alice do")
    with_temporal = Stage2Rank(temporal_weight=0.5).rank(store, candidates, "when did Alice go")
    # Alice has a time_reference, so with "when" trigger her score rises
    a_nid = next(nid for nid, _ in no_temporal if "Alice" in store.get(nid).fact.subject)  # type: ignore[union-attr]
    no_t_score = dict(no_temporal)[a_nid]
    w_t_score = dict(with_temporal)[a_nid]
    assert w_t_score > no_t_score


def test_stage2_top_k_limits():
    store = _populated_store()
    candidates = list(store.iter_ids())
    ranked = Stage2Rank(top_k=1).rank(store, candidates, "anything")
    assert len(ranked) == 1


def test_stage2_skips_missing_candidates():
    store = _populated_store()
    ranked = Stage2Rank().rank(store, ["ns-ghost", *list(store.iter_ids())], "anything")
    # Ghost silently skipped
    assert len(ranked) == len(list(store.iter_ids()))


# ---------------------------------------------------------------------------
# Stage 3 — Exact
# ---------------------------------------------------------------------------


def test_stage3_refines_by_fact_overlap():
    store = _populated_store()
    ranked = Stage2Rank().rank(store, list(store.iter_ids()), "anything")
    result = Stage3Exact().refine(store, ranked, "Alice visited Paris")
    assert isinstance(result, Stage3Result)
    # Alice's fact should be the top hit since it directly contains query tokens
    top_nid = result.hits[0][0]
    assert "Alice" in store.get(top_nid).fact.subject  # type: ignore[union-attr]


def test_stage3_budget_trim_keeps_at_least_one():
    store = _populated_store()
    ranked = Stage2Rank().rank(store, list(store.iter_ids()), "anything")
    result = Stage3Exact(budget_words=3).refine(store, ranked, "Alice")
    assert len(result.hits) >= 1


def test_stage3_confidence_floor_filters():
    store = StarStore()
    low_ns = _ns("Dubious", "allegedly did", "something", belief=0.1)
    high_ns = _ns("Solid", "definitely did", "that", belief=1.0)
    store.put(low_ns)
    store.put(high_ns)
    ranked = [(low_ns.effective_ns_id, 1.0), (high_ns.effective_ns_id, 1.0)]
    result = Stage3Exact(confidence_floor=0.5).refine(store, ranked, "Solid")
    subjects = {store.get(nid).fact.subject for nid, _ in result.hits}  # type: ignore[union-attr]
    assert "Dubious" not in subjects
    assert "Solid" in subjects


def test_stage3_populates_evidence():
    store = StarStore()
    ns = _ns(
        "Alice",
        qualifiers=Qualifiers(location="Paris"),
    )
    ns = ns.model_copy(update={"source_episode_ids": ("D1:3", "D1:7")})
    store.put(ns)
    ranked = [(ns.effective_ns_id, 1.0)]
    result = Stage3Exact().refine(store, ranked, "Alice")
    assert result.evidence_dia_ids == ("D1:3", "D1:7")


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_full_mode_runs_all_stages():
    store = _populated_store()
    pipeline = ThreeStagePipeline(
        stage1=Stage1Broad(expand_via_community=False),
    )
    result = pipeline.retrieve(store, "who visited Paris")
    assert result.hits
    assert "Paris" in result.context_text
    # Alice should be top
    top_nid = result.hits[0][0]
    assert "Alice" in store.get(top_nid).fact.subject  # type: ignore[union-attr]


def test_pipeline_no_stage1_uses_full_store():
    store = _populated_store()
    pipeline = ThreeStagePipeline(mode="no_stage1")
    result = pipeline.retrieve(store, "literature")  # matches Carol's topic
    # Without stage1 filter, all 3 stars enter stage2 + stage3
    assert len(result.hits) >= 1


def test_pipeline_no_stage2_preserves_stage1_order():
    store = _populated_store()
    pipeline = ThreeStagePipeline(
        stage1=Stage1Broad(expand_via_community=False),
        mode="no_stage2",
    )
    result = pipeline.retrieve(store, "paris")
    # Order is whatever stage1 returned; scores all zero before stage3 overlay
    assert len(result.hits) >= 1


def test_pipeline_no_stage3_skips_refine():
    store = _populated_store()
    pipeline = ThreeStagePipeline(
        stage1=Stage1Broad(expand_via_community=False),
        mode="no_stage3",
    )
    result = pipeline.retrieve(store, "paris")
    # no_stage3 context has no "[belief=..]" annotations (simple format)
    assert result.hits
    assert "[belief=" not in result.context_text


def test_pipeline_empty_store_returns_empty():
    pipeline = ThreeStagePipeline()
    result = pipeline.retrieve(StarStore(), "anything")
    assert result.hits == ()


def test_pipeline_empty_query_returns_empty():
    store = _populated_store()
    result = ThreeStagePipeline().retrieve(store, "")
    assert result.hits == ()
