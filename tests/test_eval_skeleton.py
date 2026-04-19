"""Offline end-to-end test of the eval skeleton.

Uses a trivial Pipeline (remembers the gold answer) + KeywordMockJudge to verify:
  - Pipeline/Judge protocols are satisfiable
  - Runner iterates category 1-4 QA (excludes cat 5)
  - Metrics aggregate correctly across runs

No network, no data file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from htb.data.locomo import Conversation, QAPair, Session, Turn
from htb.eval.interfaces import AnswerResult, Pipeline, RetrievalResult
from htb.eval.judge import KeywordMockJudge
from htb.eval.runner import BenchmarkRunner


def _make_conv(sample_id: str, qa_specs: list[tuple[str, str, int]]) -> Conversation:
    """Build a tiny Conversation fixture with 1 session, 1 turn, and given QA."""
    turn = Turn(speaker="A", dia_id="D1:1", text="hello world", session_index=1)
    session = Session(index=1, date_time="t=0", turns=(turn,))
    qa = tuple(
        QAPair(question=q, answer=a, evidence=("D1:1",), category=cat)
        for q, a, cat in qa_specs
    )
    return Conversation(
        sample_id=sample_id,
        speaker_a="A",
        speaker_b="B",
        sessions=(session,),
        qa=qa,
    )


@dataclass
class OracleEchoPipeline:
    """Pipeline that memorises the last ingested conv's QA and echoes the
    gold answer for every query. Always CORRECT under KeywordMockJudge.
    """

    name: str = "oracle-echo"
    _answers: dict[str, str] = field(default_factory=dict)

    def reset(self) -> None:
        self._answers = {}

    def ingest(self, conversation: Conversation) -> None:
        for q in conversation.qa:
            self._answers[q.question] = q.gold_answer_text

    def retrieve(self, query: str, budget_words: int = 1000) -> RetrievalResult:
        ctx = self._answers.get(query, "")
        return RetrievalResult(context=ctx, word_count=len(ctx.split()))

    def answer(self, query: str, retrieval: RetrievalResult) -> AnswerResult:
        return AnswerResult(text=retrieval.context)


@dataclass
class EmptyPipeline:
    """Returns empty answers — never CORRECT under any non-trivial gold."""

    name: str = "empty"

    def reset(self) -> None: ...
    def ingest(self, conversation: Conversation) -> None: ...

    def retrieve(self, query: str, budget_words: int = 1000) -> RetrievalResult:
        return RetrievalResult(context="", word_count=0)

    def answer(self, query: str, retrieval: RetrievalResult) -> AnswerResult:
        return AnswerResult(text="")


def test_pipeline_and_judge_satisfy_protocols():
    assert isinstance(OracleEchoPipeline(), Pipeline)
    assert isinstance(EmptyPipeline(), Pipeline)
    judge = KeywordMockJudge()
    # Judge doesn't need isinstance check (Protocol with methods)
    assert judge.judge("q", "apple", "I ate an apple") == "CORRECT"
    assert judge.judge("q", "apple", "I ate a banana") == "WRONG"


def test_oracle_pipeline_perfect_score():
    conv = _make_conv(
        "conv-test-1",
        [
            ("What fruit?", "apple pie", 1),
            ("What city?", "tokyo", 2),
            ("When?", "march 2023", 3),
            ("Why?", "because rain", 4),
            ("Adversarial?", "nonsense", 5),  # should be excluded
        ],
    )
    runner = BenchmarkRunner(pipeline=OracleEchoPipeline(), judge=KeywordMockJudge())
    results = runner.run([conv], n_runs=2)

    assert results.system_name == "oracle-echo"
    assert results.judge_name == "keyword-mock"
    # 4 categories kept per run x 2 runs = 8 records
    assert len(results.per_run) == 2
    assert all(len(run) == 4 for run in results.per_run)
    assert all(r.correct for run in results.per_run for r in run)

    agg = results.aggregate_runs()
    assert agg.mean == 1.0
    assert agg.stddev == 0.0
    assert agg.n_runs == 2


def test_empty_pipeline_zero_score():
    conv = _make_conv("conv-test-2", [("q", "apple", 1), ("q2", "banana", 2)])
    runner = BenchmarkRunner(pipeline=EmptyPipeline(), judge=KeywordMockJudge())
    results = runner.run([conv], n_runs=1)
    agg = results.aggregate_runs()
    assert agg.mean == 0.0


def test_per_category_breakdown():
    conv = _make_conv(
        "conv-test-3",
        [
            ("q1", "gold1", 1),
            ("q2", "gold2", 1),
            ("q3", "gold3", 2),
            ("q4", "gold4", 4),
        ],
    )
    runner = BenchmarkRunner(pipeline=OracleEchoPipeline(), judge=KeywordMockJudge())
    results = runner.run([conv], n_runs=1)
    merged = results.aggregate_merged()
    assert merged.n == 4
    assert merged.n_correct == 4
    # categories 1, 2, 4 present; 3 absent
    assert set(merged.per_category.keys()) == {1, 2, 4}
    assert merged.per_category[1] == (2, 2, 1.0)
    assert merged.per_category[2] == (1, 1, 1.0)
    assert merged.per_category[4] == (1, 1, 1.0)


def test_cat5_always_excluded():
    """Adversarial category must never appear even if pipeline would answer."""
    conv = _make_conv("conv-test-4", [("q_adv", "gold", 5)])
    runner = BenchmarkRunner(pipeline=OracleEchoPipeline(), judge=KeywordMockJudge())
    results = runner.run([conv], n_runs=3)
    for run in results.per_run:
        assert run == []


def test_keyword_mock_judge_stopword_handling():
    j = KeywordMockJudge()
    # gold is all stopwords -> content_tokens empty -> CORRECT only if gen is empty too
    assert j.judge("q", "the of and", "") == "CORRECT"
    assert j.judge("q", "the of and", "apple") == "WRONG"
    # case insensitive
    assert j.judge("q", "Apple", "apple pie") == "CORRECT"
