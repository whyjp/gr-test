"""HyperTripletLTMCreator — drives extractor -> graph materialisation.

Mirrors GAAMA's `LTMCreator.create_from_events()` layout (episodes first, then
fact + qualifier node_sets). Phase 3 production wires this to a GAAMA-backed
SDK; this skeleton uses the in-memory `HyperTripletGraph` so behaviour is
unit-testable without LLM or SQLite.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from systems.hyper_triplet.extractors import EpisodeRef, FactRef, LLMNodeSetExtractor
from systems.hyper_triplet.graph import (
    EDGE_FACT_TO_EPISODE,
    EDGE_NEXT_EPISODE,
    EDGE_TYPE_BY_QUALIFIER,
    GraphNode,
    HyperTripletGraph,
    episode_node_id,
    fact_node_id,
)
from systems.hyper_triplet.types import NodeSet


@dataclass
class HyperTripletLTMCreator:
    extractor: LLMNodeSetExtractor
    graph: HyperTripletGraph = field(default_factory=HyperTripletGraph)

    def ingest_episodes(self, episodes: Sequence[EpisodeRef]) -> list[str]:
        """Add episode nodes + NEXT chain. Returns the episode node_ids in order."""
        ids: list[str] = []
        prev_id: str | None = None
        for ep in episodes:
            nid = episode_node_id(ep.id)
            self.graph.upsert_node(
                GraphNode(node_id=nid, kind="episode", content=ep.text)
            )
            if prev_id is not None and prev_id != nid:
                self.graph.add_edge(prev_id, nid, EDGE_NEXT_EPISODE)
            ids.append(nid)
            prev_id = nid
        return ids

    def materialise_node_sets(self, node_sets: Sequence[NodeSet]) -> list[str]:
        """Insert fact + qualifier nodes + typed edges for each node_set.

        Returns list of newly created fact node ids (duplicates dedup'd).
        Qualifier nodes are deduplicated by (type, normalized-value) — same
        location across chunks becomes the same node.
        """
        created_fact_ids: list[str] = []
        seen_fact_ids: set[str] = set()
        for ns in node_sets:
            fid = fact_node_id(ns.fact.to_text())
            if fid not in self.graph.nodes:
                self.graph.upsert_node(
                    GraphNode(
                        node_id=fid,
                        kind="fact",
                        content=ns.fact.to_text(),
                        belief=ns.belief,
                        source_episode_ids=ns.source_episode_ids,
                    )
                )
            if fid not in seen_fact_ids:
                created_fact_ids.append(fid)
                seen_fact_ids.add(fid)

            for src_ep_id in ns.source_episode_ids:
                ep_nid = episode_node_id(src_ep_id)
                if ep_nid in self.graph.nodes:
                    self.graph.add_edge(fid, ep_nid, EDGE_FACT_TO_EPISODE)

            for qtype, value in ns.qualifiers.iter_typed_values():
                qid = self.graph.merge_qualifier(
                    qtype, value, source_episode_ids=ns.source_episode_ids
                )
                if qid:
                    edge_type = EDGE_TYPE_BY_QUALIFIER[qtype]
                    self.graph.add_edge(fid, qid, edge_type)

        return created_fact_ids

    def create_from_episodes(
        self,
        new_episodes: Sequence[EpisodeRef],
        related_episodes: Sequence[EpisodeRef] = (),
        existing_facts: Sequence[FactRef] = (),
        existing_qualifiers_by_type: dict[str, Sequence[str]] | None = None,
    ) -> list[str]:
        """High-level: ingest episodes, extract, materialise. Returns fact ids."""
        self.ingest_episodes(new_episodes)
        node_sets = self.extractor.extract_node_sets(
            new_episodes=new_episodes,
            related_episodes=related_episodes,
            existing_facts=existing_facts,
            existing_qualifiers_by_type=existing_qualifiers_by_type,
        )
        return self.materialise_node_sets(node_sets)
