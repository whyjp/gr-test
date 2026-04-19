"""HyperTripletPipelineV5 — Pipeline protocol adapter using all v5 components.

Wires together the new v5 stack:
- BoundaryDetector (chunk splitting)
- LLMNodeSetExtractor (extraction)
- StarStore (storage)
- CommunityDetector (L3 grouping)
- ImportanceScorer (L1 activation)
- ThreeStagePipeline (retrieval)

All driven by a single HyperTripletConfig so ablation presets flip
any component on/off without code changes.

Co-exists with the legacy `HyperTripletPipeline` in pipeline.py; tests
that predate v5 continue to use the legacy wire-up.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from htb.data import Conversation
from htb.eval.interfaces import AnswerResult, RetrievalResult
from systems.hyper_triplet.boundary_detector import BoundaryDetector
from systems.hyper_triplet.community_detector import CommunityDetector
from systems.hyper_triplet.config import DEFAULT_CONFIG, HyperTripletConfig
from systems.hyper_triplet.extractors import EpisodeRef, LLMNodeSetExtractor
from systems.hyper_triplet.importance_scorer import (
    ImportanceScorer,
    build_access_events_from_retrieval,
)
from systems.hyper_triplet.pipeline import context_passthrough_answerer
from systems.hyper_triplet.retrieval_stages import (
    ThreeStagePipeline,
)
from systems.hyper_triplet.star_store import StarStore

Answerer = Callable[[str, str], str]


@dataclass
class HyperTripletPipelineV5:
    extractor: LLMNodeSetExtractor
    config: HyperTripletConfig = DEFAULT_CONFIG
    answerer: Answerer = context_passthrough_answerer
    name: str = "hyper-triplet-v5"

    _store: StarStore | None = field(default=None, init=False)
    _boundary: BoundaryDetector | None = field(default=None, init=False)
    _three_stage: ThreeStagePipeline | None = field(default=None, init=False)
    _community: CommunityDetector | None = field(default=None, init=False)
    _importance: ImportanceScorer | None = field(default=None, init=False)
    _ingest_time_counter: float = field(default=0.0, init=False)

    # --------------------------------------------------------------
    # Pipeline protocol
    # --------------------------------------------------------------

    def reset(self) -> None:
        self._store = StarStore()
        self._ingest_time_counter = 0.0
        self._boundary = BoundaryDetector(self.config.boundary)
        self._community = CommunityDetector(self.config.community)
        self._importance = ImportanceScorer(self.config.importance)
        self._three_stage = ThreeStagePipeline(
            stage1=self.config.stage1,
            stage2=self.config.stage2,
            stage3=self.config.stage3,
            mode=self.config.retrieval_pipeline_mode,  # type: ignore[arg-type]
        )

    def ingest(self, conversation: Conversation) -> None:
        if self._store is None:
            self.reset()
        assert self._store is not None
        assert self._boundary is not None

        for session in conversation.sessions:
            turns = list(session.turns)
            if self.config.use_boundary_detector:
                chunks_of_turns = self._boundary.segment(turns)
            else:
                chunks_of_turns = [
                    turns[i : i + self.config.boundary.max_turns_per_chunk]
                    for i in range(0, len(turns), self.config.boundary.max_turns_per_chunk)
                ]

            for chunk in chunks_of_turns:
                if not chunk:
                    continue
                eps = [
                    EpisodeRef(
                        id=t.dia_id,
                        text=f"{t.speaker}: {t.text}",
                        session_date=session.date_time,
                    )
                    for t in chunk
                ]
                node_sets = self.extractor.extract_node_sets(new_episodes=eps)
                self._store.put_many(node_sets)
                self._ingest_time_counter += 1.0

        # Post-ingest background pass: community + importance
        if self.config.use_community and self._community is not None:
            self._community.detect(self._store)
        if self.config.use_importance and self._importance is not None:
            # No retrieval history yet; score with zero access events so that
            # ns.importance at least reflects the (neutral recency) baseline
            self._importance.score_all(
                self._store,
                access_events=[],
                current_time=self._ingest_time_counter,
            )

    def retrieve(self, query: str, budget_words: int | None = None) -> RetrievalResult:
        if self._store is None or self._three_stage is None:
            return RetrievalResult(context="", word_count=0)
        if budget_words is not None:
            # Temporarily override stage3 budget for this call
            original_budget = self._three_stage.stage3.budget_words
            self._three_stage.stage3.budget_words = budget_words
        try:
            result = self._three_stage.retrieve(self._store, query)
        finally:
            if budget_words is not None:
                self._three_stage.stage3.budget_words = original_budget

        # Feed retrieval hits back into importance for next round
        if self.config.use_importance and self._importance is not None and result.hits:
            scores_by_nid = {nid: score for nid, score in result.hits}
            events = build_access_events_from_retrieval(
                scores_by_nid, at_time=self._ingest_time_counter
            )
            # Aggregate historical events (none persisted yet) with this round's
            self._importance.score_all(
                self._store,
                access_events=events,
                current_time=self._ingest_time_counter,
            )

        return RetrievalResult(
            context=result.context_text,
            word_count=result.word_count,
            evidence_dia_ids=result.evidence_dia_ids,
        )

    def answer(self, query: str, retrieval: RetrievalResult) -> AnswerResult:
        return AnswerResult(text=self.answerer(query, retrieval.context))

    # --------------------------------------------------------------
    # Convenience accessors (read-only, for tests/debugging)
    # --------------------------------------------------------------

    @property
    def store(self) -> StarStore:
        if self._store is None:
            raise RuntimeError("Pipeline not reset yet; call reset() first.")
        return self._store
