"""Three-stage retrieval pipeline over StarStore.

Per plan v5 §6 and `grouping-principle-integration.md`:

1. **Stage 1 (Broad)** — query → L2 context lexical match + L3 community
   expansion → candidate ns_id pool (default 500).
2. **Stage 2 (Rank)** — L1 importance + query-temporal alignment → top-K
   (default 30).
3. **Stage 3 (Exact)** — L0 fact overlap + edge-qualifier feature re-rank →
   final context under a word budget.

Each stage is a standalone class with explicit hyperparameters so Phase E's
ablation harness can toggle them individually (no_stage1, no_hybrid_index,
etc.).

Operates on ``StarStore`` directly — works independently from
``HyperTripletGraph``. Import-free from existing ``retrieval.py`` to keep
the ablation surface clean; small helpers are duplicated intentionally.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from systems.hyper_triplet.star_store import StarStore
from systems.hyper_triplet.types import NodeSet

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
        "does", "for", "from", "has", "have", "how", "i", "in", "is",
        "it", "its", "of", "on", "or", "so", "that", "the", "this", "to",
        "was", "were", "what", "when", "where", "who", "why", "will",
        "with", "you", "your",
    }
)

_WHEN_TRIGGERS: frozenset[str] = frozenset({"when", "since", "until", "during"})


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS]


def _token_set(text: str) -> frozenset[str]:
    return frozenset(_tokens(text))


def _raw_tokens(text: str) -> frozenset[str]:
    """Lowercase tokenisation WITHOUT stopword filtering. Used for trigger
    detection ("when", "since", etc.) since those ARE stopwords in the
    search-relevance sense but carry semantic intent for query routing."""
    return frozenset(_WORD_RE.findall(text.lower()))


def _star_context_text(ns: NodeSet) -> str:
    """Concatenate L2 context + L3 topic/ontology for stage-1 lexical match."""
    parts: list[str] = []
    if ns.l2.location:
        parts.append(ns.l2.location)
    parts.extend(ns.l2.participants)
    if ns.l2.activity_type:
        parts.append(ns.l2.activity_type)
    if ns.l2.mood:
        parts.append(ns.l2.mood)
    if ns.l3.topic:
        parts.append(ns.l3.topic)
    if ns.l3.ontology_type:
        parts.append(ns.l3.ontology_type)
    parts.extend(ns.l3.ontology_properties)
    return " ".join(parts)


def _star_fact_text(ns: NodeSet) -> str:
    return ns.fact.to_text()


# ---------------------------------------------------------------------------
# Stage 1 — Broad
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Stage1Broad:
    top_n: int = 500
    include_fact_text: bool = True
    expand_via_community: bool = True
    min_overlap_tokens: int = 1

    def retrieve(self, store: StarStore, query: str) -> list[str]:
        """Return candidate ns_ids. Score = count of query tokens that appear
        in the star's L2/L3 context text (+ optional L0 fact text). Breaks
        ties by ns_id for determinism."""
        query_tokens = _token_set(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, str]] = []
        for ns in store.iter_stars():
            ctx_tokens = _token_set(_star_context_text(ns))
            fact_tokens = _token_set(_star_fact_text(ns)) if self.include_fact_text else frozenset()
            overlap = (ctx_tokens | fact_tokens) & query_tokens
            if len(overlap) < self.min_overlap_tokens:
                continue
            scored.append((float(len(overlap)), ns.effective_ns_id))

        if self.expand_via_community and scored:
            # Walk community index: every ns_id already in `scored` expands to
            # its community siblings with a small bonus
            seed_ids = {nid for _, nid in scored}
            bonus = 0.5
            for nid in list(seed_ids):
                cid = store.community_of(nid)
                if cid is None:
                    continue
                for sibling in store.stars_in_community(cid):
                    if sibling in seed_ids:
                        continue
                    scored.append((bonus, sibling))
                    seed_ids.add(sibling)

        scored.sort(key=lambda p: (-p[0], p[1]))
        return [nid for _, nid in scored[: self.top_n]]


# ---------------------------------------------------------------------------
# Stage 2 — Rank
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Stage2Rank:
    top_k: int = 30
    importance_weight: float = 1.0
    temporal_weight: float = 0.5

    def rank(
        self,
        store: StarStore,
        candidates: Iterable[str],
        query: str,
    ) -> list[tuple[str, float]]:
        """Return (ns_id, score) sorted descending. Score combines
        L1.importance with an optional temporal-alignment boost when the
        query contains a `when`-style trigger."""
        is_temporal_query = bool(_WHEN_TRIGGERS & _raw_tokens(query))

        results: list[tuple[str, float]] = []
        for nid in candidates:
            ns = store.get(nid)
            if ns is None:
                continue
            score = self.importance_weight * ns.importance
            if is_temporal_query and ns.l1.time_reference:
                # Any non-null time_reference in a time-query gets a boost;
                # a stricter implementation would parse the reference and
                # compare to a query-inferred window.
                score += self.temporal_weight
            results.append((nid, score))

        results.sort(key=lambda p: (-p[1], p[0]))
        return results[: self.top_k]


# ---------------------------------------------------------------------------
# Stage 3 — Exact
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Stage3Result:
    hits: tuple[tuple[str, float], ...]  # (ns_id, final_score)
    context_text: str
    word_count: int
    evidence_dia_ids: tuple[str, ...]


@dataclass(slots=True)
class Stage3Exact:
    budget_words: int = 1000
    fact_match_weight: float = 1.0
    confidence_floor: float = 0.3
    include_qualifiers_in_context: bool = True

    def refine(
        self,
        store: StarStore,
        ranked: Iterable[tuple[str, float]],
        query: str,
    ) -> Stage3Result:
        query_tokens = _token_set(query)

        hits: list[tuple[str, float]] = []
        for nid, stage2_score in ranked:
            ns = store.get(nid)
            if ns is None:
                continue
            if ns.belief < self.confidence_floor:
                continue
            fact_tokens = _token_set(_star_fact_text(ns))
            fact_overlap = len(fact_tokens & query_tokens)
            final = stage2_score + self.fact_match_weight * fact_overlap
            hits.append((nid, final))

        hits.sort(key=lambda p: (-p[1], p[0]))

        lines: list[str] = []
        if hits:
            lines.append("## Facts")
        kept: list[tuple[str, float]] = []
        used_words = 0
        evidence: list[str] = []
        seen_evidence: set[str] = set()

        for nid, score in hits:
            ns = store.get(nid)
            if ns is None:
                continue
            preview: list[str] = []
            belief_tag = f" [belief={ns.belief:.2f}]" if ns.belief < 1.0 else ""
            preview.append(f"- {_star_fact_text(ns)}{belief_tag}")
            if self.include_qualifiers_in_context:
                for qtype, value in ns.qualifiers.iter_typed_values():
                    preview.append(f"    - {qtype}: {value}")
            preview_words = sum(len(line.split()) for line in preview)
            if used_words + preview_words > self.budget_words and kept:
                break
            kept.append((nid, score))
            lines.extend(preview)
            used_words += preview_words
            for ep_id in ns.source_episode_ids:
                if ep_id not in seen_evidence:
                    seen_evidence.add(ep_id)
                    evidence.append(ep_id)

        return Stage3Result(
            hits=tuple(kept),
            context_text="\n".join(lines),
            word_count=used_words,
            evidence_dia_ids=tuple(evidence),
        )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


PipelineMode = Literal["full", "no_stage1", "no_stage2", "no_stage3"]


@dataclass(slots=True)
class ThreeStagePipeline:
    stage1: Stage1Broad = field(default_factory=Stage1Broad)
    stage2: Stage2Rank = field(default_factory=Stage2Rank)
    stage3: Stage3Exact = field(default_factory=Stage3Exact)
    mode: PipelineMode = "full"

    def retrieve(self, store: StarStore, query: str) -> Stage3Result:
        if self.mode == "no_stage1":
            candidates = list(store.iter_ids())
        else:
            candidates = self.stage1.retrieve(store, query)

        if not candidates:
            return Stage3Result(hits=(), context_text="", word_count=0, evidence_dia_ids=())

        if self.mode == "no_stage2":
            # Pass-through scores (zero) preserving candidate order
            ranked: list[tuple[str, float]] = [(nid, 0.0) for nid in candidates]
        else:
            ranked = self.stage2.rank(store, candidates, query)

        if not ranked:
            return Stage3Result(hits=(), context_text="", word_count=0, evidence_dia_ids=())

        if self.mode == "no_stage3":
            # Build minimal context from stage-2 top results without re-ranking
            kept: list[tuple[str, float]] = []
            lines: list[str] = []
            used_words = 0
            evidence: list[str] = []
            seen_evidence: set[str] = set()
            if ranked:
                lines.append("## Facts")
            for nid, score in ranked:
                ns = store.get(nid)
                if ns is None:
                    continue
                preview = [f"- {_star_fact_text(ns)}"]
                preview_words = sum(len(line.split()) for line in preview)
                if used_words + preview_words > self.stage3.budget_words and kept:
                    break
                kept.append((nid, score))
                lines.extend(preview)
                used_words += preview_words
                for ep_id in ns.source_episode_ids:
                    if ep_id not in seen_evidence:
                        seen_evidence.add(ep_id)
                        evidence.append(ep_id)
            return Stage3Result(
                hits=tuple(kept),
                context_text="\n".join(lines),
                word_count=used_words,
                evidence_dia_ids=tuple(evidence),
            )

        return self.stage3.refine(store, ranked, query)
