"""HyperTripletPipeline — adapter that makes HyperTriplet implement the
benchmark `Pipeline` protocol (reset / ingest / retrieve / answer).

Production wiring (Phase 3) will swap the in-memory HyperTripletGraph for
GAAMA's SqliteMemoryStore and replace the smoke answerer with an LLM
answerer using GAAMA's answer_from_memory prompt. The Pipeline surface is
unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from htb.data import Conversation
from htb.eval.interfaces import AnswerResult, RetrievalResult
from systems.hyper_triplet.extractors import EpisodeRef, LLMNodeSetExtractor
from systems.hyper_triplet.graph import HyperTripletGraph
from systems.hyper_triplet.ltm_creator import HyperTripletLTMCreator
from systems.hyper_triplet.retrieval import retrieve

Answerer = Callable[[str, str], str]
"""Callable signature (query, context) -> answer."""


def context_passthrough_answerer(query: str, context: str) -> str:
    """Trivial answerer returning the retrieved context verbatim. Judges that
    check token overlap (e.g. KeywordMockJudge) give credit when the right
    fact was retrieved. Intended for smoke tests."""
    return context or "(no context)"


_TIME_REF_RE = re.compile(r"time_reference:\s*([^\n]+)", re.IGNORECASE)
_FACT_LINE_RE = re.compile(r"^- (?P<fact>.+?)(?:\s*\[belief=[\d.]+\])?$", re.MULTILINE)


def template_answerer(query: str, context: str) -> str:
    """Light extract-from-context answerer for smoke tests.

    - 'when' questions → first `time_reference: ...` qualifier.
    - other questions → first fact bullet line.
    - fallback → first 200 chars of context.
    """
    q = query.lower().strip()
    if q.startswith("when"):
        m = _TIME_REF_RE.search(context)
        if m:
            return m.group(1).strip()
    m = _FACT_LINE_RE.search(context)
    if m:
        return m.group("fact").strip()
    return context[:200] if context else "(no context)"


def _session_to_chunks(
    session_turns: Sequence,
    session_date: str,
    turns_per_chunk: int,
) -> list[list[EpisodeRef]]:
    chunks: list[list[EpisodeRef]] = []
    for i in range(0, len(session_turns), turns_per_chunk):
        chunks.append(
            [
                EpisodeRef(
                    id=t.dia_id,
                    text=f"{t.speaker}: {t.text}",
                    session_date=session_date,
                )
                for t in session_turns[i : i + turns_per_chunk]
            ]
        )
    return chunks


@dataclass
class HyperTripletPipeline:
    extractor: LLMNodeSetExtractor
    answerer: Answerer = context_passthrough_answerer
    name: str = "hyper-triplet"
    budget_words: int = 1000
    turns_per_chunk: int = 8

    _creator: HyperTripletLTMCreator | None = field(default=None, init=False)

    @property
    def graph(self) -> HyperTripletGraph:
        if self._creator is None:
            raise RuntimeError("Pipeline.reset() must be called before graph access.")
        return self._creator.graph

    def reset(self) -> None:
        self._creator = HyperTripletLTMCreator(extractor=self.extractor)

    def ingest(self, conversation: Conversation) -> None:
        if self._creator is None:
            self.reset()
        assert self._creator is not None
        for session in conversation.sessions:
            for chunk in _session_to_chunks(
                session.turns, session.date_time, self.turns_per_chunk
            ):
                if chunk:
                    self._creator.create_from_episodes(new_episodes=chunk)

    def ingest_chunks(self, chunks: Sequence[Sequence[EpisodeRef]]) -> None:
        """Ingest pre-segmented chunks (used in tests that align chunking with
        a fixture's extraction boundaries)."""
        if self._creator is None:
            self.reset()
        assert self._creator is not None
        for chunk in chunks:
            if chunk:
                self._creator.create_from_episodes(new_episodes=list(chunk))

    def retrieve(self, query: str, budget_words: int | None = None) -> RetrievalResult:
        if self._creator is None:
            return RetrievalResult(context="", word_count=0)
        budget = budget_words if budget_words is not None else self.budget_words
        ctx = retrieve(self._creator.graph, query, budget_words=budget)
        return RetrievalResult(
            context=ctx.context_text,
            word_count=ctx.word_count,
            evidence_dia_ids=ctx.evidence_dia_ids,
        )

    def answer(self, query: str, retrieval: RetrievalResult) -> AnswerResult:
        return AnswerResult(text=self.answerer(query, retrieval.context))
