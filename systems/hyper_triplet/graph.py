"""In-memory hyper-relational graph for Hyper Triplet.

This is a standalone module — no GAAMA dependency — so the data structure and
MERGE semantics can be unit-tested offline. Phase 3 production will swap the
in-memory graph for GAAMA's SqliteMemoryStore (same primitives: node upsert,
edge upsert, MERGE key lookup).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal

from systems.hyper_triplet.types import QualifierType

NodeKind = Literal["episode", "fact", "qualifier"]

EDGE_TYPE_BY_QUALIFIER: dict[QualifierType, str] = {
    "location": "AT_LOCATION",
    "participant": "WITH_PARTICIPANT",
    "activity_type": "ACTIVITY_TYPE",
    "time_reference": "AT_TIME",
    "mood": "IN_MOOD",
    "topic": "ABOUT_TOPIC",
}

EDGE_FACT_TO_EPISODE: str = "DERIVED_FROM"
EDGE_NEXT_EPISODE: str = "NEXT"


def _short_hash(value: str, length: int = 16) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()[:length]


def episode_node_id(episode_id: str) -> str:
    return f"ep-{episode_id}" if not episode_id.startswith("ep-") else episode_id


def fact_node_id(fact_text: str) -> str:
    """Deterministic id for a fact — same text produces same id (natural MERGE).

    Production GAAMA uses `canonical_id_entity`; we use md5 for determinism in
    the offline skeleton.
    """
    return f"fact-{_short_hash(fact_text.strip().lower())}"


def qualifier_node_id(qualifier_type: QualifierType, normalized_value: str) -> str:
    return f"qual-{qualifier_type}-{_short_hash(normalized_value)}"


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    kind: NodeKind
    content: str
    qualifier_type: QualifierType | None = None
    belief: float = 1.0
    source_episode_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: str


@dataclass
class HyperTripletGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: list[GraphEdge] = field(default_factory=list)
    # (qualifier_type, normalized_value) -> qualifier_node_id for MERGE lookup
    _qualifier_index: dict[tuple[str, str], str] = field(default_factory=dict)

    def upsert_node(self, node: GraphNode) -> str:
        """Insert node if absent; if present with same id keep earliest.
        Returns the node_id."""
        if node.node_id not in self.nodes:
            self.nodes[node.node_id] = node
        return node.node_id

    def add_edge(self, source_id: str, target_id: str, edge_type: str) -> GraphEdge:
        edge = GraphEdge(source_id=source_id, target_id=target_id, edge_type=edge_type)
        self.edges.append(edge)
        return edge

    def merge_qualifier(
        self,
        qualifier_type: QualifierType,
        value: str,
        *,
        source_episode_ids: Iterable[str] = (),
    ) -> str:
        """Return the node_id for this qualifier, creating it if absent.

        Normalisation: `value.strip().lower()` — so "Seattle" and "seattle"
        collapse to the same node. Empty / whitespace-only values return "".
        """
        normalized = value.strip().lower()
        if not normalized:
            return ""
        key = (qualifier_type, normalized)
        if key in self._qualifier_index:
            return self._qualifier_index[key]
        nid = qualifier_node_id(qualifier_type, normalized)
        node = GraphNode(
            node_id=nid,
            kind="qualifier",
            content=value.strip(),
            qualifier_type=qualifier_type,
            source_episode_ids=tuple(source_episode_ids),
        )
        self.upsert_node(node)
        self._qualifier_index[key] = nid
        return nid

    # --- query helpers ---

    def nodes_by_kind(self, kind: NodeKind) -> list[GraphNode]:
        return [n for n in self.nodes.values() if n.kind == kind]

    def qualifier_nodes(
        self, qualifier_type: QualifierType | None = None
    ) -> list[GraphNode]:
        out: list[GraphNode] = []
        for n in self.nodes.values():
            if n.kind != "qualifier":
                continue
            if qualifier_type is not None and n.qualifier_type != qualifier_type:
                continue
            out.append(n)
        return out

    def edges_from(self, source_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.source_id == source_id]

    def edges_to(self, target_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.target_id == target_id]

    def edges_of_type(self, edge_type: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.edge_type == edge_type]

    def iter_edges(self) -> Iterator[GraphEdge]:
        return iter(self.edges)

    def stats(self) -> dict[str, int]:
        by_kind: dict[str, int] = {}
        for n in self.nodes.values():
            by_kind[n.kind] = by_kind.get(n.kind, 0) + 1
        by_edge_type: dict[str, int] = {}
        for e in self.edges:
            by_edge_type[e.edge_type] = by_edge_type.get(e.edge_type, 0) + 1
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            **{f"nodes.{k}": v for k, v in by_kind.items()},
            **{f"edges.{k}": v for k, v in by_edge_type.items()},
        }
