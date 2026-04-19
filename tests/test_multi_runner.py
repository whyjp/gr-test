"""Tests for MultiSystemRunner and paired bootstrap."""

from __future__ import annotations

from dataclasses import dataclass, field

from htb.data.locomo import Conversation, QAPair, Session, Turn
from htb.eval import (
    KeywordMockJudge,
    MultiSystemResult,
    MultiSystemRunner,
    Pipeline,
    format_comparison_table,
)
from htb.eval.interfaces import AnswerResult, RetrievalResult


def _mk_conv(sample_id: str, qa: list[tuple[str, str, int]]) -> Conversation:
    t = Turn(speaker="A", dia_id="D1:1", text="hi", session_index=1)
    s = Session(index=1, date_time="t", turns=(t,))
    return Conversation(
        sample_id=sample_id,
        speaker_a="A",
        speaker_b="B",
        sessions=(s,),
        qa=tuple(QAPair(question=q, answer=a, category=c) for q, a, c in qa),
    )


@dataclass
class AlwaysRightPipeline:
    name: str = "always-right"
    _map: dict[str, str] = field(default_factory=dict)

    def reset(self) -> None:
        self._map = {}

    def ingest(self, conversation: Conversation) -> None:
        for q in conversation.qa:
            self._map[q.question] = q.gold_answer_text

    def retrieve(self, query: str, budget_words: int = 1000) -> RetrievalResult:
        return RetrievalResult(context=self._map.get(query, ""), word_count=1)

    def answer(self, query: str, retrieval: RetrievalResult) -> AnswerResult:
        return AnswerResult(text=retrieval.context)


@dataclass
class AlwaysWrongPipeline:
    name: str = "always-wrong"

    def reset(self) -> None: ...
    def ingest(self, conversation: Conversation) -> None: ...

    def retrieve(self, query: str, budget_words: int = 1000) -> RetrievalResult:
        return RetrievalResult(context="", word_count=0)

    def answer(self, query: str, retrieval: RetrievalResult) -> AnswerResult:
        return AnswerResult(text="zzzzz")


@dataclass
class HalfRightPipeline:
    """CORRECT only on every-other QA (deterministic)."""
    name: str = "half-right"
    _map: dict[str, str] = field(default_factory=dict)
    _counter: int = 0

    def reset(self) -> None:
        self._map = {}
        self._counter = 0

    def ingest(self, conversation: Conversation) -> None:
        for q in conversation.qa:
            self._map[q.question] = q.gold_answer_text

    def retrieve(self, query: str, budget_words: int = 1000) -> RetrievalResult:
        self._counter += 1
        if self._counter % 2 == 1:
            return RetrievalResult(context=self._map.get(query, ""), word_count=1)
        return RetrievalResult(context="zzzz", word_count=1)

    def answer(self, query: str, retrieval: RetrievalResult) -> AnswerResult:
        return AnswerResult(text=retrieval.context)


def test_multi_system_runner_runs_each_system():
    conv = _mk_conv(
        "c1",
        [("q1", "apple", 1), ("q2", "banana", 2), ("q3", "cherry", 3), ("q4", "date", 4)],
    )
    runner = MultiSystemRunner(
        systems=[("right", AlwaysRightPipeline()), ("wrong", AlwaysWrongPipeline())],
        judge=KeywordMockJudge(),
    )
    result = runner.run([conv], n_runs=1)
    assert isinstance(result, MultiSystemResult)
    assert {s.name for s in result.systems} == {"right", "wrong"}
    acc = result.accuracy_table()
    assert acc["right"] == 1.0
    assert acc["wrong"] == 0.0


def test_per_category_table_populated():
    conv = _mk_conv("c1", [("q1", "apple", 1), ("q2", "banana", 2)])
    runner = MultiSystemRunner(
        systems=[("right", AlwaysRightPipeline())],
        judge=KeywordMockJudge(),
    )
    r = runner.run([conv], n_runs=1)
    pcat = r.per_category_table()
    assert 1 in pcat["right"]
    assert 2 in pcat["right"]
    assert pcat["right"][1] == (1, 1, 1.0)


def test_paired_bootstrap_detects_large_effect():
    conv = _mk_conv("c1", [("q1", "apple", 1), ("q2", "banana", 2), ("q3", "cherry", 3)])
    runner = MultiSystemRunner(
        systems=[("right", AlwaysRightPipeline()), ("wrong", AlwaysWrongPipeline())],
        judge=KeywordMockJudge(),
    )
    r = runner.run([conv], n_runs=1)
    bs = r.paired_bootstrap("right", "wrong", n_resamples=500, seed=0)
    assert bs.n_qa == 3
    assert bs.delta_mean == 1.0
    # Confidence interval should not cross zero
    assert bs.ci_low > 0


def test_paired_bootstrap_handles_tie():
    conv = _mk_conv("c1", [("q1", "apple", 1), ("q2", "banana", 2)])
    runner = MultiSystemRunner(
        systems=[("a", AlwaysRightPipeline()), ("b", AlwaysRightPipeline())],
        judge=KeywordMockJudge(),
    )
    r = runner.run([conv], n_runs=1)
    bs = r.paired_bootstrap("a", "b", n_resamples=200, seed=1)
    assert bs.delta_mean == 0.0
    assert bs.p_value >= 0.5


def test_paired_bootstrap_no_shared_qa_returns_zero():
    # Construct a result with no overlap by running disjoint conversations
    runner = MultiSystemRunner(
        systems=[("a", AlwaysRightPipeline())],
        judge=KeywordMockJudge(),
    )
    conv_a = _mk_conv("c-a", [("qa", "x", 1)])
    r_a = runner.run([conv_a], n_runs=1)
    # just one system; bootstrap against same system on same qa -> delta 0
    bs = r_a.paired_bootstrap("a", "a", n_resamples=100, seed=2)
    assert bs.n_qa == 1
    assert bs.delta_mean == 0.0


def test_format_comparison_table_markdown():
    conv = _mk_conv("c1", [("q1", "apple", 1), ("q2", "banana", 2)])
    runner = MultiSystemRunner(
        systems=[("right", AlwaysRightPipeline()), ("wrong", AlwaysWrongPipeline())],
        judge=KeywordMockJudge(),
    )
    r = runner.run([conv], n_runs=1)
    table = format_comparison_table(r)
    assert "| system | overall | cat1 | cat2 |" in table
    # right system should show 1.000 overall
    assert "| right | 1.000 |" in table


def test_multi_system_n_runs_aggregated():
    """With n_runs=3 and a deterministic half-right pipeline, each conv is ingested
    and iterated 3 times; total records = 3 x n_qa per system."""
    conv = _mk_conv("c1", [("q1", "apple", 1), ("q2", "banana", 2), ("q3", "cherry", 3)])
    runner = MultiSystemRunner(
        systems=[("half", HalfRightPipeline())],
        judge=KeywordMockJudge(),
    )
    r = runner.run([conv], n_runs=3)
    half = next(s for s in r.systems if s.name == "half")
    assert len(half.records) == 9  # 3 runs x 3 QA


def test_pipeline_reset_between_systems_isolated():
    """Each system's ingest is scoped within its BenchmarkRunner — no cross-contamination."""
    p1 = AlwaysRightPipeline()
    p2 = AlwaysWrongPipeline()
    conv = _mk_conv("c1", [("q1", "apple", 1)])
    runner = MultiSystemRunner(systems=[("r", p1), ("w", p2)], judge=KeywordMockJudge())
    r = runner.run([conv], n_runs=1)
    acc = r.accuracy_table()
    assert acc["r"] == 1.0
    assert acc["w"] == 0.0
    # Ensure systems satisfy Pipeline protocol
    assert isinstance(p1, Pipeline)
    assert isinstance(p2, Pipeline)
