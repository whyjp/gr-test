"""ImportanceScorer — ACT-R inspired activation for NodeSet importance.

Per plan v5 Phase B, each NodeSet carries an ``importance`` scalar on its L1
layer. The score combines three classifier-style signals:

1. **Base activation** — ``log(frequency + 1)`` where frequency is the count
   of times the fact was accessed (retrieved, or referenced by a new
   extraction) since ingestion.
2. **Recency decay** — ``exp(-decay_rate * time_since_last_access)`` where
   time is measured in session-count units for LoCoMo.
3. **Belief multiplier** — the LLM-reported confidence ``belief`` scales
   the whole score (a low-belief fact should not dominate even when
   frequently accessed).

ACT-R activation rule (simplified):
    A_i = log(sum_k t_k^-d) + epsilon

We approximate with ``log(frequency + 1) * exp(-d * dt)`` which keeps the
monotonic behaviour without the summation overhead. Parameters match the
HyperMem / EverMemOS observation that exponential decay with a single
global rate is sufficient for dialogue-scale memory.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from systems.hyper_triplet.star_store import StarStore
from systems.hyper_triplet.types import NodeSet


@dataclass(slots=True, frozen=True)
class ImportanceConfig:
    decay_rate: float = 0.1
    belief_weight: float = 1.0
    frequency_weight: float = 1.0
    recency_weight: float = 1.0


_DEFAULT_CONFIG: ImportanceConfig = None  # type: ignore[assignment]  # set below


def _default_config() -> ImportanceConfig:
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = ImportanceConfig()
    return _DEFAULT_CONFIG


@dataclass(slots=True)
class AccessEvent:
    ns_id: str
    at_time: float  # unit-agnostic; caller defines (session count, real seconds, etc.)


def score_importance(
    node_set: NodeSet,
    access_events: Iterable[AccessEvent],
    *,
    current_time: float,
    config: ImportanceConfig | None = None,
) -> float:
    """Compute a single NodeSet's importance from its access history."""
    if config is None:
        config = _default_config()
    events = [e for e in access_events if e.ns_id == node_set.effective_ns_id]
    frequency = len(events)
    if frequency == 0:
        recency_component = 1.0  # neutral when no history
    else:
        last_access_time = max(e.at_time for e in events)
        dt = max(0.0, current_time - last_access_time)
        recency_component = math.exp(-config.decay_rate * dt)
    freq_component = math.log1p(frequency)
    base = config.frequency_weight * freq_component + config.recency_weight * recency_component
    return base * (config.belief_weight * node_set.belief)


@dataclass
class ImportanceScorer:
    """Applies importance scores to every star in a StarStore."""

    config: ImportanceConfig = field(default_factory=ImportanceConfig)

    def score_all(
        self,
        store: StarStore,
        access_events: Iterable[AccessEvent],
        *,
        current_time: float,
    ) -> dict[str, float]:
        """Returns mapping ns_id -> new importance. Also writes back via
        StarStore.update_importance so StarStore.get() returns the scored
        NodeSet."""
        events_list = list(access_events)
        events_by_ns: dict[str, list[AccessEvent]] = {}
        for e in events_list:
            events_by_ns.setdefault(e.ns_id, []).append(e)

        scores: dict[str, float] = {}
        for ns_id in store.iter_ids():
            ns = store.get(ns_id)
            if ns is None:
                continue
            ns_events = events_by_ns.get(ns_id, [])
            new_score = score_importance(
                ns,
                ns_events,
                current_time=current_time,
                config=self.config,
            )
            scores[ns_id] = new_score
            store.update_importance(ns_id, new_score)
        return scores

    def rank(
        self,
        store: StarStore,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Return (ns_id, importance) sorted descending. Uses whatever
        importance is currently stored on each NodeSet (must be called
        after score_all for meaningful results)."""
        pairs = [(ns_id, store.get(ns_id).importance) for ns_id in store.iter_ids() if store.get(ns_id) is not None]  # type: ignore[union-attr]
        pairs.sort(key=lambda p: -p[1])
        if top_k is not None:
            pairs = pairs[:top_k]
        return pairs


def build_access_events_from_retrieval(
    retrieval_hits: Mapping[str, float],
    at_time: float,
) -> list[AccessEvent]:
    """Convenience: every retrieval hit (ns_id -> score) produces one access
    event at ``at_time``. Useful when you want retrieval rounds to feed
    importance back into the store."""
    return [AccessEvent(ns_id=nid, at_time=at_time) for nid in retrieval_hits]
