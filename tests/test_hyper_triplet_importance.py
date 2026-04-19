"""Tests for ImportanceScorer."""

from __future__ import annotations

import math

from systems.hyper_triplet.importance_scorer import (
    AccessEvent,
    ImportanceConfig,
    ImportanceScorer,
    build_access_events_from_retrieval,
    score_importance,
)
from systems.hyper_triplet.star_store import StarStore
from systems.hyper_triplet.types import Fact, NodeSet


def _mk_ns(subject: str, belief: float = 1.0) -> NodeSet:
    return NodeSet(
        fact=Fact(subject=subject, predicate="did", object="something"),
        belief=belief,
    )


def test_score_with_no_events_uses_neutral_recency():
    ns = _mk_ns("A")
    s = score_importance(ns, [], current_time=0.0)
    # No frequency => log1p(0) = 0 + recency 1.0 => base = recency_weight, times belief
    assert math.isclose(s, 1.0)


def test_score_increases_with_frequency():
    ns = _mk_ns("A")
    events = [AccessEvent(ns_id=ns.effective_ns_id, at_time=1.0) for _ in range(5)]
    s = score_importance(ns, events, current_time=1.0)
    assert s > 1.0


def test_score_decays_with_time_since_last_access():
    ns = _mk_ns("A")
    events = [AccessEvent(ns_id=ns.effective_ns_id, at_time=0.0)]
    fresh = score_importance(ns, events, current_time=0.0)
    stale = score_importance(ns, events, current_time=100.0)
    assert stale < fresh


def test_score_scaled_by_belief():
    ns_high = _mk_ns("H", belief=1.0)
    ns_low = _mk_ns("L", belief=0.3)
    s_high = score_importance(ns_high, [], current_time=0.0)
    s_low = score_importance(ns_low, [], current_time=0.0)
    assert s_low < s_high
    assert math.isclose(s_low, s_high * 0.3)


def test_score_ignores_events_for_other_ns():
    ns_a = _mk_ns("A")
    ns_b = _mk_ns("B")
    # All events target B; A should see empty history
    events = [AccessEvent(ns_id=ns_b.effective_ns_id, at_time=1.0) for _ in range(10)]
    s_a = score_importance(ns_a, events, current_time=1.0)
    assert math.isclose(s_a, 1.0)  # neutral


def test_scorer_writes_back_to_store():
    store = StarStore()
    ns_a = _mk_ns("A")
    ns_b = _mk_ns("B")
    store.put(ns_a)
    store.put(ns_b)
    events = [AccessEvent(ns_id=ns_a.effective_ns_id, at_time=0.0) for _ in range(3)]
    scorer = ImportanceScorer()
    scores = scorer.score_all(store, events, current_time=0.0)
    assert scores[ns_a.effective_ns_id] > scores[ns_b.effective_ns_id]
    # Store now reflects the update
    assert store.get(ns_a.effective_ns_id).importance == scores[ns_a.effective_ns_id]  # type: ignore[union-attr]


def test_rank_sorts_descending():
    store = StarStore()
    ns_list = [_mk_ns(s) for s in ("A", "B", "C")]
    for ns in ns_list:
        store.put(ns)
    events = (
        [AccessEvent(ns_id=ns_list[0].effective_ns_id, at_time=0.0) for _ in range(10)]
        + [AccessEvent(ns_id=ns_list[1].effective_ns_id, at_time=0.0) for _ in range(3)]
    )
    scorer = ImportanceScorer()
    scorer.score_all(store, events, current_time=0.0)
    ranked = scorer.rank(store)
    assert ranked[0][0] == ns_list[0].effective_ns_id
    assert ranked[1][0] == ns_list[1].effective_ns_id
    assert ranked[2][0] == ns_list[2].effective_ns_id


def test_rank_top_k_limits_output():
    store = StarStore()
    for s in "ABCDE":
        store.put(_mk_ns(s))
    scorer = ImportanceScorer()
    ranked = scorer.rank(store, top_k=2)
    assert len(ranked) == 2


def test_build_access_events_from_retrieval():
    events = build_access_events_from_retrieval({"ns-a": 0.9, "ns-b": 0.7}, at_time=5.0)
    assert len(events) == 2
    assert all(e.at_time == 5.0 for e in events)
    assert {e.ns_id for e in events} == {"ns-a", "ns-b"}


def test_custom_config_changes_balance():
    """With decay_rate=0 recency doesn't decay; frequency fully dominates."""
    ns = _mk_ns("A")
    events = [AccessEvent(ns_id=ns.effective_ns_id, at_time=0.0) for _ in range(10)]
    cfg = ImportanceConfig(decay_rate=0.0)
    fresh = score_importance(ns, events, current_time=0.0, config=cfg)
    stale = score_importance(ns, events, current_time=1000.0, config=cfg)
    assert math.isclose(fresh, stale)


def test_score_importance_clamped_to_nonneg():
    """ImportanceScorer never produces a negative score even with extreme inputs."""
    ns = _mk_ns("A", belief=0.0)
    s = score_importance(ns, [], current_time=0.0)
    assert s == 0.0  # belief=0 zeros the score
