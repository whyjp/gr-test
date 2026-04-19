"""Offline tests for AblationRunner using MockLLM."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from htb.data.locomo import Conversation, QAPair, Session, Turn
from htb.eval import (
    AblationRunner,
    AblationSweepResult,
    KeywordMockJudge,
    Pipeline,
    format_ablation_report,
)
from htb.eval.interfaces import AnswerResult, RetrievalResult
from systems.hyper_triplet.ablation import ABLATION_NAMES
from systems.hyper_triplet.config import HyperTripletConfig


def _mk_conv(sample_id: str) -> Conversation:
    t = Turn(speaker="A", dia_id="D1:1", text="hi", session_index=1)
    s = Session(index=1, date_time="t", turns=(t,))
    return Conversation(
        sample_id=sample_id,
        speaker_a="A",
        speaker_b="B",
        sessions=(s,),
        qa=tuple(
            QAPair(question=q, answer=a, category=cat)
            for q, a, cat in [
                ("q1", "apple", 1),
                ("q2", "banana", 2),
                ("q3", "cherry", 3),
                ("q4", "date", 4),
            ]
        ),
    )


@dataclass
class OracleEchoPipeline:
    """Trivial Pipeline that echoes gold answers — gives 100% accuracy."""

    name: str = "oracle"
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
class CorrectnessByConfigPipeline:
    """Pipeline whose correctness depends on use_community flag — simulates
    an ablation that actually changes behaviour."""

    name: str = "cfg-dependent"
    config: HyperTripletConfig = field(default_factory=HyperTripletConfig)
    _map: dict[str, str] = field(default_factory=dict)

    def reset(self) -> None:
        self._map = {}

    def ingest(self, conversation: Conversation) -> None:
        for q in conversation.qa:
            self._map[q.question] = q.gold_answer_text

    def retrieve(self, query: str, budget_words: int = 1000) -> RetrievalResult:
        # When community is on, retrieval returns the right answer; when off,
        # return something wrong.
        if self.config.use_community:
            return RetrievalResult(context=self._map.get(query, ""), word_count=1)
        return RetrievalResult(context="zzz", word_count=1)

    def answer(self, query: str, retrieval: RetrievalResult) -> AnswerResult:
        return AnswerResult(text=retrieval.context)


def test_ablation_runner_runs_every_preset():
    conv = _mk_conv("c1")

    def factory(cfg: HyperTripletConfig) -> Pipeline:
        return OracleEchoPipeline()

    runner = AblationRunner(pipeline_factory=factory, judge=KeywordMockJudge())
    sweep = runner.run([conv], n_runs=1)
    assert isinstance(sweep, AblationSweepResult)
    assert {r.name for r in sweep.runs} == set(ABLATION_NAMES)


def test_ablation_runner_accuracy_table_shape():
    conv = _mk_conv("c1")
    runner = AblationRunner(
        pipeline_factory=lambda cfg: OracleEchoPipeline(),
        judge=KeywordMockJudge(),
    )
    sweep = runner.run([conv], n_runs=1)
    accuracy = sweep.accuracy_table()
    assert "baseline" in accuracy
    assert accuracy["baseline"] == 1.0  # oracle gets them all right


def test_ablation_runner_delta_shows_behaviour_difference():
    conv = _mk_conv("c1")
    runner = AblationRunner(
        pipeline_factory=lambda cfg: CorrectnessByConfigPipeline(config=cfg),
        judge=KeywordMockJudge(),
    )
    sweep = runner.run([conv], n_runs=1)
    accuracy = sweep.accuracy_table()
    assert accuracy["baseline"] == 1.0
    # no_community disables use_community -> pipeline returns wrong answer -> 0 accuracy
    assert accuracy["no_community"] == 0.0
    deltas = sweep.delta_vs_baseline()
    assert deltas["no_community"] == pytest.approx(-1.0)


def test_ablation_runner_restricts_presets_by_name():
    conv = _mk_conv("c1")
    runner = AblationRunner(
        pipeline_factory=lambda cfg: OracleEchoPipeline(),
        judge=KeywordMockJudge(),
        preset_names=("baseline", "no_community"),
    )
    sweep = runner.run([conv], n_runs=1)
    assert {r.name for r in sweep.runs} == {"baseline", "no_community"}


def test_format_ablation_report_is_markdown():
    conv = _mk_conv("c1")
    runner = AblationRunner(
        pipeline_factory=lambda cfg: CorrectnessByConfigPipeline(config=cfg),
        judge=KeywordMockJudge(),
        preset_names=("baseline", "no_community", "no_importance"),
    )
    sweep = runner.run([conv], n_runs=1)
    report = format_ablation_report(sweep)
    assert "## Ablation accuracy table" in report
    assert "## Delta vs baseline" in report
    assert "no_community" in report
    assert "HINGE invariant violated" in report


def test_sweep_baseline_is_found():
    conv = _mk_conv("c1")
    runner = AblationRunner(
        pipeline_factory=lambda cfg: OracleEchoPipeline(),
        judge=KeywordMockJudge(),
    )
    sweep = runner.run([conv], n_runs=1)
    baseline = sweep.baseline
    assert baseline is not None
    assert baseline.name == "baseline"


def test_sweep_as_multi_system_result_enables_bootstrap():
    conv = _mk_conv("c1")
    runner = AblationRunner(
        pipeline_factory=lambda cfg: CorrectnessByConfigPipeline(config=cfg),
        judge=KeywordMockJudge(),
        preset_names=("baseline", "no_community"),
    )
    sweep = runner.run([conv], n_runs=1)
    bs = sweep.paired_bootstrap_against_baseline("no_community", n_resamples=100, seed=0)
    assert bs is not None
    # no_community performs worse than baseline -> delta_mean negative
    assert bs.delta_mean < 0


def test_runner_handles_empty_preset_names():
    conv = _mk_conv("c1")
    runner = AblationRunner(
        pipeline_factory=lambda cfg: OracleEchoPipeline(),
        judge=KeywordMockJudge(),
        preset_names=(),
    )
    sweep = runner.run([conv], n_runs=1)
    assert sweep.runs == []


def test_unknown_preset_names_are_skipped():
    conv = _mk_conv("c1")
    runner = AblationRunner(
        pipeline_factory=lambda cfg: OracleEchoPipeline(),
        judge=KeywordMockJudge(),
        preset_names=("baseline", "no_such_thing"),
    )
    sweep = runner.run([conv], n_runs=1)
    assert {r.name for r in sweep.runs} == {"baseline"}
