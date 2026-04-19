"""StarStore — one NodeSet = one star subgraph, O(1) KV-friendly retrieval.

Per plan v5 §3.4 (star-native storage): each node_set is a star whose centre
is an L0 fact and leaves are L1/L2/L3 values. The StarStore gives direct
dict-based access to a whole star by ns_id (no graph walk needed) plus
inverse indices for qualifier / community / episode lookups so inter-star
joins stay cheap.

The in-memory backend is a plain dict; a production deployment would swap
this for Redis Hash (HGETALL → full star) / FalkorDB / Memgraph.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from systems.hyper_triplet.types import NodeSet


def _qualifier_index_key(qualifier_type: str, value: str) -> tuple[str, str]:
    """Same MERGE semantics as graph.merge_qualifier: case-folded stripped value."""
    return qualifier_type, value.strip().lower()


@dataclass
class StarStore:
    _stars: dict[str, NodeSet] = field(default_factory=dict)
    _qualifier_index: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    _community_index: dict[str, set[str]] = field(default_factory=dict)
    _episode_index: dict[str, set[str]] = field(default_factory=dict)
    _community_overrides: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def put(self, node_set: NodeSet) -> str:
        ns_id = node_set.effective_ns_id
        if ns_id in self._stars:
            self._remove_from_indices(self._stars[ns_id])
        self._stars[ns_id] = node_set
        self._add_to_indices(node_set)
        return ns_id

    def put_many(self, node_sets: Iterable[NodeSet]) -> list[str]:
        return [self.put(ns) for ns in node_sets]

    def _add_to_indices(self, ns: NodeSet) -> None:
        ns_id = ns.effective_ns_id
        for qtype, value in ns.qualifiers.iter_typed_values():
            key = _qualifier_index_key(qtype, value)
            self._qualifier_index.setdefault(key, set()).add(ns_id)
        cid = ns.l3.community_id
        if cid:
            self._community_index.setdefault(cid, set()).add(ns_id)
        for ep_id in ns.source_episode_ids:
            self._episode_index.setdefault(ep_id, set()).add(ns_id)

    def _remove_from_indices(self, ns: NodeSet) -> None:
        ns_id = ns.effective_ns_id
        for qtype, value in ns.qualifiers.iter_typed_values():
            key = _qualifier_index_key(qtype, value)
            bucket = self._qualifier_index.get(key)
            if bucket is not None:
                bucket.discard(ns_id)
                if not bucket:
                    del self._qualifier_index[key]
        cid = ns.l3.community_id
        if cid:
            bucket = self._community_index.get(cid)
            if bucket is not None:
                bucket.discard(ns_id)
                if not bucket:
                    del self._community_index[cid]
        for ep_id in ns.source_episode_ids:
            bucket = self._episode_index.get(ep_id)
            if bucket is not None:
                bucket.discard(ns_id)
                if not bucket:
                    del self._episode_index[ep_id]

    def delete(self, ns_id: str) -> bool:
        ns = self._stars.pop(ns_id, None)
        if ns is None:
            return False
        self._remove_from_indices(ns)
        self._community_overrides.pop(ns_id, None)
        return True

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get(self, ns_id: str) -> NodeSet | None:
        return self._stars.get(ns_id)

    def __contains__(self, ns_id: str) -> bool:
        return ns_id in self._stars

    def __len__(self) -> int:
        return len(self._stars)

    def iter_stars(self) -> Iterator[NodeSet]:
        return iter(self._stars.values())

    def iter_ids(self) -> Iterator[str]:
        return iter(self._stars.keys())

    def stars_with_qualifier(self, qualifier_type: str, value: str) -> set[str]:
        key = _qualifier_index_key(qualifier_type, value)
        return set(self._qualifier_index.get(key, ()))

    def stars_in_community(self, community_id: str) -> set[str]:
        return set(self._community_index.get(community_id, ()))

    def stars_with_episode(self, episode_dia_id: str) -> set[str]:
        return set(self._episode_index.get(episode_dia_id, ()))

    # ------------------------------------------------------------------
    # Background-workflow updates (community + importance)
    # ------------------------------------------------------------------

    def assign_community(self, ns_id: str, community_id: str | None) -> bool:
        """Attach or clear a community_id for an existing star. Updates the
        community index and records in the override dict so `community_of()`
        reflects it even though NodeSet.l3 doesn't carry the field yet.
        (Phase D will add it to the NodeSet itself.)"""
        if ns_id not in self._stars:
            return False
        # Drop from previous community
        prev = self._community_overrides.get(ns_id)
        if prev is not None:
            bucket = self._community_index.get(prev)
            if bucket is not None:
                bucket.discard(ns_id)
                if not bucket:
                    del self._community_index[prev]
        if community_id:
            self._community_overrides[ns_id] = community_id
            self._community_index.setdefault(community_id, set()).add(ns_id)
        else:
            self._community_overrides.pop(ns_id, None)
        return True

    def community_of(self, ns_id: str) -> str | None:
        return self._community_overrides.get(ns_id)

    def update_importance(self, ns_id: str, importance: float) -> NodeSet | None:
        existing = self.get(ns_id)
        if existing is None:
            return None
        updated = existing.model_copy(update={"importance": float(importance)})
        self._stars[ns_id] = updated
        return updated

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "n_stars": len(self._stars),
            "n_qualifier_index_entries": len(self._qualifier_index),
            "n_communities": len(self._community_index),
            "n_episodes_indexed": len(self._episode_index),
            "n_community_overrides": len(self._community_overrides),
        }
