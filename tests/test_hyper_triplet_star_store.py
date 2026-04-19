"""Phase A1 — StarStore abstraction tests."""

from __future__ import annotations

from systems.hyper_triplet.star_store import StarStore
from systems.hyper_triplet.types import Fact, NodeSet, Qualifiers


def _make_ns(subject: str, obj: str = "sunset", **kwargs) -> NodeSet:
    return NodeSet(
        fact=Fact(subject=subject, predicate="painted", object=obj),
        **kwargs,
    )


def test_put_and_get():
    store = StarStore()
    ns = _make_ns("Melanie")
    ns_id = store.put(ns)
    assert ns_id == ns.effective_ns_id
    assert store.get(ns_id) is ns
    assert ns_id in store
    assert len(store) == 1


def test_put_is_idempotent_for_same_fact():
    store = StarStore()
    ns1 = _make_ns("Melanie")
    ns2 = _make_ns("melanie")  # case-folded collapses to same ns_id
    store.put(ns1)
    store.put(ns2)
    assert len(store) == 1


def test_iter_stars():
    store = StarStore()
    store.put_many([_make_ns("A"), _make_ns("B"), _make_ns("C")])
    subjects = sorted(ns.fact.subject for ns in store.iter_stars())
    assert subjects == ["A", "B", "C"]


def test_qualifier_index_finds_shared_qualifier():
    store = StarStore()
    ns_a = _make_ns(
        "Alice", qualifiers=Qualifiers(location="Paris", participants=("Alice",))
    )
    ns_b = _make_ns(
        "Bob", qualifiers=Qualifiers(location="paris", participants=("Bob",))
    )
    ns_c = _make_ns("Carol", qualifiers=Qualifiers(location="Tokyo"))
    store.put_many([ns_a, ns_b, ns_c])

    shared_paris = store.stars_with_qualifier("location", "Paris")
    assert len(shared_paris) == 2  # Alice + Bob via case-folded match
    assert {ns_a.effective_ns_id, ns_b.effective_ns_id} == shared_paris


def test_qualifier_index_cleaned_on_delete():
    store = StarStore()
    ns_a = _make_ns("A", qualifiers=Qualifiers(location="Paris"))
    ns_b = _make_ns("B", qualifiers=Qualifiers(location="Paris"))
    store.put_many([ns_a, ns_b])
    assert len(store.stars_with_qualifier("location", "Paris")) == 2
    assert store.delete(ns_a.effective_ns_id) is True
    assert len(store.stars_with_qualifier("location", "Paris")) == 1
    assert ns_b.effective_ns_id in store.stars_with_qualifier("location", "Paris")


def test_qualifier_index_cleaned_on_reput():
    """Re-putting the same ns_id with different qualifiers should update
    the qualifier index correctly."""
    store = StarStore()
    ns_v1 = NodeSet(
        fact=Fact(subject="Subject", predicate="lives", object="in city"),
        qualifiers=Qualifiers(location="Paris"),
    )
    store.put(ns_v1)
    assert len(store.stars_with_qualifier("location", "Paris")) == 1

    ns_v2 = NodeSet(
        fact=Fact(subject="Subject", predicate="lives", object="in city"),
        qualifiers=Qualifiers(location="Tokyo"),
    )
    store.put(ns_v2)
    assert len(store.stars_with_qualifier("location", "Paris")) == 0
    assert len(store.stars_with_qualifier("location", "Tokyo")) == 1


def test_episode_index_tracks_source_ids():
    store = StarStore()
    store.put(
        NodeSet(
            fact=Fact(subject="A", predicate="B", object="C"),
            source_episode_ids=("D1:3", "D1:7"),
        )
    )
    store.put(
        NodeSet(
            fact=Fact(subject="X", predicate="Y", object="Z"),
            source_episode_ids=("D1:3",),
        )
    )
    ids_for_D13 = store.stars_with_episode("D1:3")
    assert len(ids_for_D13) == 2
    ids_for_D17 = store.stars_with_episode("D1:7")
    assert len(ids_for_D17) == 1


def test_assign_community_indexes_correctly():
    store = StarStore()
    ns_a = _make_ns("A")
    ns_b = _make_ns("B")
    store.put_many([ns_a, ns_b])

    assert store.assign_community(ns_a.effective_ns_id, "cluster_001") is True
    assert store.assign_community(ns_b.effective_ns_id, "cluster_001") is True
    assert store.stars_in_community("cluster_001") == {
        ns_a.effective_ns_id,
        ns_b.effective_ns_id,
    }


def test_assign_community_moves_on_reassign():
    store = StarStore()
    ns = _make_ns("A")
    store.put(ns)
    store.assign_community(ns.effective_ns_id, "c1")
    assert store.stars_in_community("c1") == {ns.effective_ns_id}
    store.assign_community(ns.effective_ns_id, "c2")
    assert store.stars_in_community("c1") == set()
    assert store.stars_in_community("c2") == {ns.effective_ns_id}
    assert store.community_of(ns.effective_ns_id) == "c2"


def test_assign_community_none_clears():
    store = StarStore()
    ns = _make_ns("A")
    store.put(ns)
    store.assign_community(ns.effective_ns_id, "c1")
    store.assign_community(ns.effective_ns_id, None)
    assert store.community_of(ns.effective_ns_id) is None
    assert store.stars_in_community("c1") == set()


def test_assign_community_missing_star_returns_false():
    store = StarStore()
    assert store.assign_community("ns-ghost", "c1") is False


def test_update_importance_rewrites_star():
    store = StarStore()
    ns = _make_ns("A")
    store.put(ns)
    updated = store.update_importance(ns.effective_ns_id, 0.87)
    assert updated is not None
    assert updated.importance == 0.87
    # Identity check: new object but same ns_id + same fact
    retrieved = store.get(ns.effective_ns_id)
    assert retrieved is not None
    assert retrieved.importance == 0.87
    assert retrieved.fact == ns.fact


def test_update_importance_missing_star_returns_none():
    store = StarStore()
    assert store.update_importance("ns-ghost", 1.0) is None


def test_delete_nonexistent_returns_false():
    store = StarStore()
    assert store.delete("ns-ghost") is False


def test_stats_reflects_index_state():
    store = StarStore()
    store.put_many(
        [
            _make_ns("A", qualifiers=Qualifiers(location="Paris")),
            _make_ns("B", qualifiers=Qualifiers(location="Paris", participants=("Alice",))),
            _make_ns("C"),
        ]
    )
    store.assign_community("fake-id", "c1")  # no-op
    s = store.stats()
    assert s["n_stars"] == 3
    # Qualifier index keys: (location, paris) + (participant, alice) = 2
    assert s["n_qualifier_index_entries"] == 2


def test_iter_ids_matches_ids():
    store = StarStore()
    a = _make_ns("A")
    b = _make_ns("B")
    store.put_many([a, b])
    assert set(store.iter_ids()) == {a.effective_ns_id, b.effective_ns_id}
