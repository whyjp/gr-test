"""Scoring records and aggregation across QA / categories / runs."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from htb.eval.interfaces import Judgment


@dataclass(slots=True, frozen=True)
class ScoreRecord:
    run_id: int
    conv_id: str
    question: str
    gold_answer: str
    generated_answer: str
    category: int
    judgment: Judgment
    retrieval_ms: float = 0.0
    answer_ms: float = 0.0

    @property
    def correct(self) -> bool:
        return self.judgment == "CORRECT"


@dataclass(slots=True, frozen=True)
class AggregateScore:
    n: int
    n_correct: int
    accuracy: float
    per_category: dict[int, tuple[int, int, float]] = field(default_factory=dict)


def _accuracy(records: Iterable[ScoreRecord]) -> tuple[int, int, float]:
    n = 0
    correct = 0
    for r in records:
        n += 1
        if r.correct:
            correct += 1
    acc = correct / n if n > 0 else 0.0
    return n, correct, acc


def per_category_accuracy(records: Iterable[ScoreRecord]) -> dict[int, tuple[int, int, float]]:
    by_cat: dict[int, list[ScoreRecord]] = {}
    for r in records:
        by_cat.setdefault(r.category, []).append(r)
    return {cat: _accuracy(rs) for cat, rs in sorted(by_cat.items())}


def aggregate_run(records: Iterable[ScoreRecord]) -> AggregateScore:
    rs = list(records)
    n, correct, acc = _accuracy(rs)
    return AggregateScore(
        n=n, n_correct=correct, accuracy=acc, per_category=per_category_accuracy(rs)
    )


@dataclass(slots=True, frozen=True)
class RunAggregate:
    mean: float
    stddev: float
    n_runs: int
    per_run: tuple[float, ...]


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var)


def aggregate_runs(per_run_accuracies: list[float]) -> RunAggregate:
    mean, std = _mean_std(per_run_accuracies)
    return RunAggregate(
        mean=mean, stddev=std, n_runs=len(per_run_accuracies), per_run=tuple(per_run_accuracies)
    )
