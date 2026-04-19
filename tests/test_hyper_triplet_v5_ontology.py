"""Phase A2 — L3Auxiliary ontology axis tests.

Per docs/grouping-node-principle.md, L3 gains a 3rd grouping axis beyond
temporal (L1) and context (L2): ontology_type + ontology_properties
(Palantir-style Object/Property/Link schema).
"""

from __future__ import annotations

from systems.hyper_triplet.types import L3Auxiliary, NodeSet


def test_l3_aux_default_ontology_fields_empty():
    l3 = L3Auxiliary()
    assert l3.ontology_type is None
    assert l3.ontology_properties == ()


def test_l3_aux_ontology_round_trip():
    l3 = L3Auxiliary(
        ontology_type="Player",
        ontology_properties=("level:42", "class:wizard"),
    )
    assert l3.ontology_type == "Player"
    assert l3.ontology_properties == ("level:42", "class:wizard")


def test_l3_aux_immutable():
    import pydantic

    l3 = L3Auxiliary(ontology_type="Item")
    try:
        l3.ontology_type = "Player"  # type: ignore[misc]
    except (AttributeError, ValueError, pydantic.ValidationError):
        pass
    else:
        raise AssertionError("L3Auxiliary should be frozen")


def test_nodeset_l3_view_defaults_to_empty_ontology():
    """Legacy NodeSet construction never touches ontology fields; L3 view
    returns the defaults (None / empty tuple) without error."""
    from systems.hyper_triplet.types import Fact, Qualifiers

    ns = NodeSet(
        fact=Fact(subject="A", predicate="B", object="C"),
        qualifiers=Qualifiers(topic="travel"),
    )
    assert ns.l3.topic == "travel"
    assert ns.l3.ontology_type is None
    assert ns.l3.ontology_properties == ()


def test_ontology_properties_accepts_sequence_coercion():
    """tuple coercion should accept list input (pydantic default behaviour)."""
    l3 = L3Auxiliary(ontology_properties=["a", "b", "c"])
    assert l3.ontology_properties == ("a", "b", "c")
