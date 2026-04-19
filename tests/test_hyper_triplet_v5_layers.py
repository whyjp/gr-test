"""Phase A0 — v5 layer views + ns_id + importance tests.

Verifies that the added L0/L1/L2/L3 projections correctly split the
existing NodeSet's flat Qualifiers into the four functional layers,
while legacy construction (no ns_id / no importance) keeps working.
"""

from __future__ import annotations

from systems.hyper_triplet.types import (
    Fact,
    L0Fact,
    L1TemporalImportance,
    L2Context,
    L3Auxiliary,
    NodeSet,
    Qualifiers,
    _compute_ns_id,
)


def _base_nodeset() -> NodeSet:
    return NodeSet(
        fact=Fact(subject="Alice", predicate="visited", object="Paris"),
        source_episode_ids=("ep-1",),
        belief=0.9,
        qualifiers=Qualifiers(
            location="Paris",
            participants=("Alice", "Bob"),
            activity_type="trip",
            time_reference="June 2024",
            mood="excited",
            topic="travel",
        ),
    )


def test_l0_view_is_the_triple():
    ns = _base_nodeset()
    l0 = ns.l0
    assert isinstance(l0, L0Fact)
    assert l0.subject == "Alice"
    assert l0.predicate == "visited"
    assert l0.object == "Paris"
    assert l0.edge_qualifiers == {}


def test_l1_view_carries_temporal_and_importance():
    ns = _base_nodeset()
    l1 = ns.l1
    assert isinstance(l1, L1TemporalImportance)
    assert l1.time_reference == "June 2024"
    assert l1.belief == 0.9
    assert l1.importance == 0.0  # default


def test_l2_view_carries_context_only():
    ns = _base_nodeset()
    l2 = ns.l2
    assert isinstance(l2, L2Context)
    assert l2.location == "Paris"
    assert l2.participants == ("Alice", "Bob")
    assert l2.activity_type == "trip"
    assert l2.mood == "excited"
    # Topic does NOT appear in L2 — it belongs to L3
    assert not hasattr(l2, "topic")


def test_l3_view_carries_topic():
    ns = _base_nodeset()
    l3 = ns.l3
    assert isinstance(l3, L3Auxiliary)
    assert l3.topic == "travel"
    assert l3.community_id is None  # uninitialised placeholder


def test_layers_are_disjoint_views():
    """No single Qualifiers field should end up in two layers."""
    ns = _base_nodeset()
    l1_attrs = set(ns.l1.model_fields_set)
    l2_attrs = set(ns.l2.model_fields_set)
    l3_attrs = set(ns.l3.model_fields_set)
    # time_reference only in L1; location/participants/activity/mood only in L2; topic only in L3
    assert "time_reference" in l1_attrs or ns.l1.time_reference is not None
    assert "location" in l2_attrs or ns.l2.location is not None
    assert "topic" in l3_attrs or ns.l3.topic is not None
    # And conversely: no cross-pollination
    assert ns.l2.location is not None and "location" not in {
        f for f in l1_attrs | l3_attrs
    }


def test_importance_field_defaults_to_zero():
    ns = NodeSet(fact=Fact(subject="A", predicate="B", object="C"))
    assert ns.importance == 0.0
    assert ns.l1.importance == 0.0


def test_importance_clamped_and_coerced():
    ns1 = NodeSet(fact=Fact(subject="A", predicate="B", object="C"), importance=-5)
    assert ns1.importance == 0.0
    ns2 = NodeSet(fact=Fact(subject="A", predicate="B", object="C"), importance="not a number")
    assert ns2.importance == 0.0
    ns3 = NodeSet(fact=Fact(subject="A", predicate="B", object="C"), importance=0.42)
    assert ns3.importance == 0.42
    assert ns3.l1.importance == 0.42


def test_effective_ns_id_deterministic():
    ns_a = NodeSet(fact=Fact(subject="A", predicate="B", object="C"))
    ns_b = NodeSet(fact=Fact(subject="A", predicate="B", object="C"))
    ns_c = NodeSet(fact=Fact(subject="a", predicate="b", object="c"))  # lowercase
    assert ns_a.effective_ns_id == ns_b.effective_ns_id
    assert ns_a.effective_ns_id == ns_c.effective_ns_id  # case-folded
    # Different triple → different id
    ns_d = NodeSet(fact=Fact(subject="A", predicate="B", object="D"))
    assert ns_a.effective_ns_id != ns_d.effective_ns_id


def test_explicit_ns_id_overrides_default():
    ns = NodeSet(fact=Fact(subject="A", predicate="B", object="C"), ns_id="ns-custom")
    assert ns.effective_ns_id == "ns-custom"


def test_compute_ns_id_helper_matches_property():
    fact = Fact(subject="A", predicate="B", object="C")
    ns = NodeSet(fact=fact)
    assert ns.effective_ns_id == _compute_ns_id(fact)


def test_legacy_construction_still_works():
    """Critical backwards-compat: v5 adds only optional fields and computed
    properties. Existing test-style construction must keep passing."""
    ns = NodeSet(
        fact=Fact(subject="Melanie", predicate="painted", object="a lake sunrise"),
        source_episode_ids=("D1:14",),
        belief=0.95,
        qualifiers=Qualifiers(
            time_reference="2022",
            activity_type="painting",
            topic="artistic_creation",
            mood="fond",
        ),
    )
    # legacy access
    assert ns.fact.subject == "Melanie"
    assert ns.qualifiers.time_reference == "2022"
    # v5 layer views
    assert ns.l1.time_reference == "2022"
    assert ns.l2.activity_type == "painting"
    assert ns.l3.topic == "artistic_creation"
    assert ns.l2.mood == "fond"
    assert ns.l1.belief == 0.95


def test_layer_views_are_frozen():
    """L0-L3 should be immutable so retrieval code can't accidentally mutate
    the canonical NodeSet state."""
    ns = _base_nodeset()
    import pydantic

    try:
        ns.l0.subject = "Evil"  # type: ignore[misc]
    except (AttributeError, ValueError, pydantic.ValidationError):
        pass
    else:
        raise AssertionError("L0 should be frozen (immutable)")
