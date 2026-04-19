"""CommunityDetector — Louvain over the shared-qualifier graph.

Per plan v5 Phase D and `grouping-node-principle.md` Invariant #8
(LLM-as-classifier-only), community detection is a classifier-style
operation: it assigns membership labels to existing facts via a
deterministic graph algorithm, never generating new summarised content.

Graph construction:
- Nodes = node_sets (one per ns_id)
- Edges = two ns_ids share at least one qualifier value (any type)
- Edge weight = count of shared qualifier values

Algorithm: networkx's built-in Louvain community detection
(`louvain_communities`) with a fixed seed for determinism. Falls back to
connected-components when graph is empty or networkx is unavailable.

The detector writes community_ids back to the StarStore via
``StarStore.assign_community()`` and returns the assignment mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from systems.hyper_triplet.star_store import StarStore


@dataclass(slots=True, frozen=True)
class CommunityConfig:
    seed: int = 42
    resolution: float = 1.0
    community_prefix: str = "c"
    # If a community has fewer than min_community_size members, it is absorbed
    # into a shared "c-singletons" label (helps keep the ablation clean).
    min_community_size: int = 2


@dataclass(slots=True)
class CommunityDetector:
    config: CommunityConfig = CommunityConfig()

    def build_graph(self, store: StarStore) -> nx.Graph:
        """Undirected graph: ns_id -> ns_id edge when they share any
        qualifier value. Weight = number of shared qualifier values."""
        g: nx.Graph = nx.Graph()
        for ns_id in store.iter_ids():
            g.add_node(ns_id)

        # For each qualifier bucket, pairwise-link every ns_id inside it.
        for bucket in store._qualifier_index.values():  # type: ignore[attr-defined]
            members = list(bucket)
            n = len(members)
            if n < 2:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = members[i], members[j]
                    if g.has_edge(a, b):
                        g[a][b]["weight"] += 1
                    else:
                        g.add_edge(a, b, weight=1)
        return g

    def detect(self, store: StarStore) -> dict[str, str]:
        """Run Louvain; assign community_ids back to the store; return
        mapping ns_id -> community_id (or "c-singletons" when the node's
        community is below min_community_size)."""
        graph = self.build_graph(store)
        if graph.number_of_nodes() == 0:
            return {}

        try:
            communities = nx.community.louvain_communities(
                graph,
                weight="weight",
                resolution=self.config.resolution,
                seed=self.config.seed,
            )
        except Exception:
            # Fallback: every connected component becomes a community
            communities = list(nx.connected_components(graph))

        # Sort communities by first ns_id for deterministic labelling across runs
        sorted_communities = sorted(communities, key=lambda c: min(c) if c else "")
        assignment: dict[str, str] = {}
        singleton_label = f"{self.config.community_prefix}-singletons"
        for idx, members in enumerate(sorted_communities):
            if len(members) < self.config.min_community_size:
                label = singleton_label
            else:
                label = f"{self.config.community_prefix}-{idx:04d}"
            for ns_id in members:
                assignment[ns_id] = label

        # Write back to store
        for ns_id, label in assignment.items():
            store.assign_community(ns_id, label)

        return assignment
