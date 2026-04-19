"""Qualifier-edge graph propagation retrieval.

Closes HINGE invariant #7 for Hyper Triplet — retrieval must traverse
qualifier edges, not just BM25 over flattened text.

Approach: Personalized PageRank-style propagation over the
HyperTripletGraph, seeded by fact nodes that directly match the query
plus qualifier nodes whose value matches the query. Propagation flows
along typed qualifier edges (AT_LOCATION / WITH_PARTICIPANT / ...) so
facts sharing a qualifier with a matched fact bubble up even when the
fact text has no lexical overlap with the query.

This is a lightweight PPR (no neural embeddings) so it works offline
and fits the HINGE-minimal probe role — the research question becomes
"does typed qualifier edge traversal add signal over token-bag
retrieval?" measurable as an A/B in Phase 5 ablation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from systems.hyper_triplet.graph import (
    EDGE_FACT_TO_EPISODE,
    EDGE_TYPE_BY_QUALIFIER,
    HyperTripletGraph,
)
from systems.hyper_triplet.retrieval import (
    RetrievalHit,
    _build_idf,
    _evidence_ids,
    _fact_document,
    _token_set,
    format_memory_pack,
)

# Qualifier edges are bidirectional for propagation: a fact → its qualifiers
# and qualifiers → facts that share them.
QUALIFIER_EDGE_TYPES: frozenset[str] = frozenset(EDGE_TYPE_BY_QUALIFIER.values())
ALLOWED_PROP_EDGE_TYPES: frozenset[str] = frozenset(
    [*QUALIFIER_EDGE_TYPES, EDGE_FACT_TO_EPISODE]
)

SeedStrategy = Literal["fact_and_qualifier", "fact_only", "qualifier_only"]


@dataclass(slots=True, frozen=True)
class PPRConfig:
    """PPR hyperparameters kept explicit for ablation sweeps."""
    alpha: float = 0.15           # teleport probability
    iterations: int = 20          # power-iteration depth
    seed_strategy: SeedStrategy = "fact_and_qualifier"
    qualifier_match_token_threshold: int = 1
    combine_with_bm25: bool = True
    bm25_weight: float = 0.5      # weight on IDF score vs PPR score
    belief_weight: float = 0.5


_WORD_SPLIT = re.compile(r"\W+")


def _build_adjacency(
    graph: HyperTripletGraph,
) -> dict[str, list[tuple[str, float]]]:
    """Bidirectional adjacency over qualifier edges + fact→episode edges.

    Weights are uniform 1.0 — no learned importance. Self-loops omitted.
    Edge types outside ALLOWED_PROP_EDGE_TYPES (e.g. NEXT) are excluded so
    episode NEXT chains don't dominate the walk.
    """
    adj: dict[str, list[tuple[str, float]]] = {nid: [] for nid in graph.nodes}
    for e in graph.edges:
        if e.edge_type not in ALLOWED_PROP_EDGE_TYPES:
            continue
        if e.source_id == e.target_id:
            continue
        adj.setdefault(e.source_id, []).append((e.target_id, 1.0))
        adj.setdefault(e.target_id, []).append((e.source_id, 1.0))
    return adj


def _seed_distribution(
    graph: HyperTripletGraph,
    query_tokens: frozenset[str],
    config: PPRConfig,
) -> dict[str, float]:
    """Seed mass on fact nodes + qualifier nodes matching query tokens."""
    seeds: dict[str, float] = {}

    # Qualifier seeds: qualifier node whose content has >=N tokens in common with query
    if config.seed_strategy in ("fact_and_qualifier", "qualifier_only"):
        for q in graph.qualifier_nodes():
            qtokens = _token_set(q.content)
            overlap = qtokens & query_tokens
            if len(overlap) >= config.qualifier_match_token_threshold:
                seeds[q.node_id] = seeds.get(q.node_id, 0.0) + float(len(overlap))

    # Fact seeds: facts whose fact_document tokens overlap query
    if config.seed_strategy in ("fact_and_qualifier", "fact_only"):
        docs: dict[str, frozenset[str]] = {
            f.node_id: _token_set(_fact_document(graph, f)) for f in graph.nodes_by_kind("fact")
        }
        idf = _build_idf(graph.nodes_by_kind("fact"), docs)
        for f in graph.nodes_by_kind("fact"):
            overlap = docs[f.node_id] & query_tokens
            if not overlap:
                continue
            seeds[f.node_id] = seeds.get(f.node_id, 0.0) + sum(
                idf.get(t, 1.0) for t in overlap
            ) * (1.0 + config.belief_weight * f.belief)

    # Normalise to a distribution summing to 1. If empty, return empty.
    total = sum(seeds.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in seeds.items()}


def _ppr(
    adjacency: dict[str, list[tuple[str, float]]],
    seeds: dict[str, float],
    *,
    alpha: float,
    iterations: int,
) -> dict[str, float]:
    """Power-iteration PPR.

    r_{t+1} = alpha * seeds + (1-alpha) * P^T r_t
    where P is the row-normalised adjacency transition matrix.
    """
    if not seeds:
        return {}
    # Initialise scores uniformly over seeds.
    r = dict(seeds)
    # Pre-compute row sums for normalisation
    out_sum: dict[str, float] = {nid: sum(w for _, w in nbrs) for nid, nbrs in adjacency.items()}

    for _ in range(iterations):
        new_r: dict[str, float] = {nid: alpha * seeds.get(nid, 0.0) for nid in adjacency}
        for nid, score in r.items():
            nbrs = adjacency.get(nid)
            if not nbrs:
                continue
            denom = out_sum.get(nid, 0.0)
            if denom <= 0:
                continue
            spread = (1.0 - alpha) * score
            for tgt, w in nbrs:
                new_r[tgt] = new_r.get(tgt, 0.0) + spread * (w / denom)
        r = new_r

    return r


@dataclass(slots=True)
class PPRRetrievalResult:
    hits: tuple[RetrievalHit, ...] = ()
    context_text: str = ""
    word_count: int = 0
    evidence_dia_ids: tuple[str, ...] = ()
    ppr_scores: dict[str, float] = field(default_factory=dict)


def retrieve_ppr(
    graph: HyperTripletGraph,
    query: str,
    *,
    budget_words: int = 1000,
    top_k: int = 20,
    config: PPRConfig | None = None,
) -> PPRRetrievalResult:
    """PPR-based retrieval. Returns the top-k facts by final score, with
    optional BM25 blend (per config.combine_with_bm25)."""
    cfg = config or PPRConfig()
    query_tokens = frozenset(_token_set(query))
    if not query_tokens:
        return PPRRetrievalResult()

    facts = graph.nodes_by_kind("fact")
    if not facts:
        return PPRRetrievalResult()

    adjacency = _build_adjacency(graph)
    seeds = _seed_distribution(graph, query_tokens, cfg)
    if not seeds:
        return PPRRetrievalResult()

    ppr = _ppr(adjacency, seeds, alpha=cfg.alpha, iterations=cfg.iterations)

    # BM25-ish fallback score (reused from sum-of-IDF in retrieve.py)
    docs: dict[str, frozenset[str]] = {
        f.node_id: _token_set(_fact_document(graph, f)) for f in facts
    }
    idf = _build_idf(facts, docs)

    hits: list[RetrievalHit] = []
    for f in facts:
        ppr_score = ppr.get(f.node_id, 0.0)
        bm25_score = 0.0
        matched = docs[f.node_id] & query_tokens
        if matched:
            bm25_score = sum(idf.get(t, 1.0) for t in matched) * (
                1.0 + cfg.belief_weight * f.belief
            )
        if cfg.combine_with_bm25:
            # Normalise both to roughly comparable scale — PPR sums to ~1
            # across all nodes, bm25 can be O(10). Scale bm25 by 1/n_facts.
            final_score = (1.0 - cfg.bm25_weight) * ppr_score * len(facts) + (
                cfg.bm25_weight * bm25_score
            )
        else:
            final_score = ppr_score
        if final_score <= 0:
            continue
        hits.append(
            RetrievalHit(
                fact_node=f,
                score=final_score,
                matched_query_tokens=tuple(sorted(matched)),
                evidence_episode_ids=_evidence_ids(graph, f.node_id),
            )
        )

    hits.sort(key=lambda h: (-h.score, h.fact_node.node_id))
    hits = hits[:top_k]

    # Word-budget trim via successive preview (reuse retrieve.py's helper logic
    # inline to avoid pulling in the whole function)
    kept: list[RetrievalHit] = []
    used_words = 0
    for h in hits:
        preview = format_memory_pack(graph, [*kept, h], include_qualifiers=True)
        if len(preview.split()) > budget_words and kept:
            break
        kept.append(h)
        used_words = len(preview.split())

    context_text = format_memory_pack(graph, kept, include_qualifiers=True)

    seen: set[str] = set()
    ev: list[str] = []
    for h in kept:
        for eid in h.evidence_episode_ids:
            if eid not in seen:
                seen.add(eid)
                ev.append(eid)

    return PPRRetrievalResult(
        hits=tuple(kept),
        context_text=context_text,
        word_count=used_words,
        evidence_dia_ids=tuple(ev),
        ppr_scores=ppr,
    )
