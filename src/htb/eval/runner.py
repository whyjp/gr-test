"""BenchmarkRunner — drives a Pipeline + Judge across LoCoMo-10 for N runs.

Pure-Python, no network. Deterministic given a deterministic Pipeline+Judge.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from htb.data import QA_CATEGORIES_BENCHMARK, Conversation
from htb.data.locomo import QAPair
from htb.eval.interfaces import Judge, Pipeline
from htb.eval.metrics import (
    AggregateScore,
    RunAggregate,
    ScoreRecord,
    aggregate_run,
    aggregate_runs,
)


@dataclass(slots=True)
class RunResults:
    system_name: str
    judge_name: str
    per_run: list[list[ScoreRecord]] = field(default_factory=list)

    def accuracies(self) -> list[float]:
        return [aggregate_run(run).accuracy for run in self.per_run]

    def aggregate_runs(self) -> RunAggregate:
        return aggregate_runs(self.accuracies())

    def merged_records(self) -> list[ScoreRecord]:
        out: list[ScoreRecord] = []
        for run in self.per_run:
            out.extend(run)
        return out

    def aggregate_merged(self) -> AggregateScore:
        return aggregate_run(self.merged_records())


@dataclass(slots=True)
class BenchmarkRunner:
    pipeline: Pipeline
    judge: Judge
    budget_words: int = 1000
    categories: frozenset[int] = QA_CATEGORIES_BENCHMARK

    def _score_qa(
        self,
        run_id: int,
        conv_id: str,
        qa: QAPair,
    ) -> ScoreRecord:
        t0 = time.perf_counter()
        retrieval = self.pipeline.retrieve(qa.question, budget_words=self.budget_words)
        t1 = time.perf_counter()
        answer = self.pipeline.answer(qa.question, retrieval)
        t2 = time.perf_counter()
        gold = qa.gold_answer_text
        judgment = self.judge.judge(qa.question, gold, answer.text)
        return ScoreRecord(
            run_id=run_id,
            conv_id=conv_id,
            question=qa.question,
            gold_answer=gold,
            generated_answer=answer.text,
            category=qa.category,
            judgment=judgment,
            retrieval_ms=(t1 - t0) * 1000.0,
            answer_ms=(t2 - t1) * 1000.0,
        )

    def run(
        self,
        conversations: Iterable[Conversation],
        n_runs: int = 1,
    ) -> RunResults:
        convs = list(conversations)
        results = RunResults(
            system_name=getattr(self.pipeline, "name", type(self.pipeline).__name__),
            judge_name=getattr(self.judge, "name", type(self.judge).__name__),
        )
        for run_id in range(n_runs):
            run_records: list[ScoreRecord] = []
            for conv in convs:
                self.pipeline.reset()
                self.pipeline.ingest(conv)
                for qa in conv.qa:
                    if qa.category not in self.categories:
                        continue
                    run_records.append(self._score_qa(run_id, conv.sample_id, qa))
            results.per_run.append(run_records)
        return results
