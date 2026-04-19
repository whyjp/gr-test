"""Keyword-scored retrieval over HyperTripletGraph.

Lightweight standalone — no LLM, no external vector store. Given a query,
ranks fact nodes by TF-IDF-like scoring over fact text + adjacent qualifier
content, then trims to a word budget.

Intent: good enough to smoke-test the full Pipeline end-to-end before real
embedding-based retrieval is wired in Phase 3.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from systems.hyper_triplet.graph import (
    EDGE_FACT_TO_EPISODE,
    EDGE_TYPE_BY_QUALIFIER,
    GraphNode,
    HyperTripletGraph,
)

_WORD = re.compile(r"[a-z0-9]+")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "by",
        "did", "do", "does", "for", "from", "has", "have", "in",
        "is", "it", "its", "of", "on", "or", "that", "the", "to",
        "was", "were", "will", "with", "this", "these", "those",
        "what", "when", "where", "who", "why", "how", "which",
    }
)


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS]


def _token_set(text: str) -> frozenset[str]:
    return frozenset(_tokens(text))


@dataclass(slots=True, frozen=True)
class RetrievalHit:
    fact_node: GraphNode
    score: float
    matched_query_tokens: tuple[str, ...]
    evidence_episode_ids: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class RetrievedContext:
    hits: tuple[RetrievalHit, ...]
    context_text: str
    word_count: int
    evidence_dia_ids: tuple[str, ...]


def _adjacent_qualifier_text(graph: HyperTripletGraph, fact_id: str) -> str:
    """Concatenate content of qualifier nodes connected to this fact via any
    typed qualifier edge. Used to extend the fact's searchable text."""
    typed_edges = set(EDGE_TYPE_BY_QUALIFIER.values())
    parts: list[str] = []
    for e in graph.edges_from(fact_id):
        if e.edge_type in typed_edges:
            q = graph.nodes.get(e.target_id)
            if q is not None and q.content:
                parts.append(q.content)
    return " ".join(parts)


def _evidence_ids(graph: HyperTripletGraph, fact_id: str) -> tuple[str, ...]:
    """Dia_ids of episodes this fact derives from (DERIVED_FROM)."""
    out: list[str] = []
    for e in graph.edges_from(fact_id):
        if e.edge_type == EDGE_FACT_TO_EPISODE:
            # episode node id shape: "ep-D1:3" -> extract "D1:3"
            tid = e.target_id
            out.append(tid[3:] if tid.startswith("ep-") else tid)
    return tuple(out)


def _fact_document(graph: HyperTripletGraph, fact: GraphNode) -> str:
    """Searchable text for a fact = fact content + adjacent qualifier values."""
    return f"{fact.content} {_adjacent_qualifier_text(graph, fact.node_id)}".strip()


def _build_idf(facts: Iterable[GraphNode], docs: dict[str, frozenset[str]]) -> dict[str, float]:
    n_docs = sum(1 for _ in facts)
    df: dict[str, int] = {}
    for tokens in docs.values():
        for t in tokens:
            df[t] = df.get(t, 0) + 1
    # Standard add-1 smoothed IDF
    return {t: math.log((n_docs + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()}


def retrieve_facts(
    graph: HyperTripletGraph,
    query: str,
    *,
    top_k: int = 20,
    belief_weight: float = 0.5,
) -> list[RetrievalHit]:
    """Return top-k fact hits ranked by IDF-weighted query-term overlap.

    Score = sum(idf[t] for t in matched) * (1 + belief_weight * fact.belief).
    Qualifier content adjacent to the fact is included in the searchable text
    so a query about "painting" surfaces facts whose activity_type is painting.
    """
    query_tokens = _token_set(query)
    if not query_tokens:
        return []

    facts = graph.nodes_by_kind("fact")
    if not facts:
        return []

    doc_tokens: dict[str, frozenset[str]] = {
        f.node_id: _token_set(_fact_document(graph, f)) for f in facts
    }
    idf = _build_idf(facts, doc_tokens)

    hits: list[RetrievalHit] = []
    for f in facts:
        matched = doc_tokens[f.node_id] & query_tokens
        if not matched:
            continue
        base = sum(idf.get(t, 1.0) for t in matched)
        score = base * (1.0 + belief_weight * f.belief)
        hits.append(
            RetrievalHit(
                fact_node=f,
                score=score,
                matched_query_tokens=tuple(sorted(matched)),
                evidence_episode_ids=_evidence_ids(graph, f.node_id),
            )
        )

    hits.sort(key=lambda h: (-h.score, h.fact_node.node_id))
    return hits[:top_k]


def format_memory_pack(
    graph: HyperTripletGraph,
    hits: Iterable[RetrievalHit],
    *,
    include_qualifiers: bool = True,
) -> str:
    """Render hits into a readable context string with fact + its qualifiers."""
    lines: list[str] = []
    if not hits:
        return ""
    lines.append("## Facts")
    for h in hits:
        belief_tag = f" [belief={h.fact_node.belief:.2f}]" if h.fact_node.belief < 1.0 else ""
        lines.append(f"- {h.fact_node.content}{belief_tag}")
        if include_qualifiers:
            typed_edges = set(EDGE_TYPE_BY_QUALIFIER.values())
            for e in graph.edges_from(h.fact_node.node_id):
                if e.edge_type in typed_edges:
                    q = graph.nodes.get(e.target_id)
                    if q is not None:
                        lines.append(f"    - {q.qualifier_type}: {q.content}")
    return "\n".join(lines)


def retrieve(
    graph: HyperTripletGraph,
    query: str,
    *,
    budget_words: int = 1000,
    top_k: int = 20,
    include_qualifiers: bool = True,
) -> RetrievedContext:
    """End-to-end: rank facts, format, trim to word budget, dedup evidence."""
    hits = retrieve_facts(graph, query, top_k=top_k)

    kept: list[RetrievalHit] = []
    used_words = 0
    for h in hits:
        pack_preview = format_memory_pack(graph, [*kept, h], include_qualifiers=include_qualifiers)
        if len(pack_preview.split()) > budget_words and kept:
            break
        kept.append(h)
        used_words = len(pack_preview.split())

    context_text = format_memory_pack(graph, kept, include_qualifiers=include_qualifiers)

    # Dedup evidence dia_ids preserving order
    seen: set[str] = set()
    ev: list[str] = []
    for h in kept:
        for eid in h.evidence_episode_ids:
            if eid not in seen:
                seen.add(eid)
                ev.append(eid)

    return RetrievedContext(
        hits=tuple(kept),
        context_text=context_text,
        word_count=used_words,
        evidence_dia_ids=tuple(ev),
    )
