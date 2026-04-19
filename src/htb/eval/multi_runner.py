"""MultiSystemRunner — drives N Pipelines through identical evaluation for side-by-side comparison.

Reuses `BenchmarkRunner` per system, then aggregates per-system / per-category
scores and computes paired-bootstrap confidence intervals so v4's decomposition
ablation can be expressed with statistical rigour rather than single-point
accuracy numbers.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from htb.data import QA_CATEGORIES_BENCHMARK, Conversation
from htb.eval.interfaces import Judge, Pipeline
from htb.eval.metrics import ScoreRecord, per_category_accuracy
from htb.eval.runner import BenchmarkRunner, RunResults


@dataclass(slots=True, frozen=True)
class SystemResult:
    name: str
    run_results: RunResults

    @property
    def records(self) -> list[ScoreRecord]:
        return self.run_results.merged_records()

    def per_qa_correct(self) -> dict[tuple[str, str], bool]:
        """Map (conv_id, question) -> correct-across-all-runs (majority)."""
        by_qa: dict[tuple[str, str], list[bool]] = {}
        for r in self.records:
            by_qa.setdefault((r.conv_id, r.question), []).append(r.correct)
        return {k: sum(v) > len(v) / 2 for k, v in by_qa.items()}


@dataclass(slots=True, frozen=True)
class PairedBootstrapResult:
    system_a: str
    system_b: str
    n_qa: int
    delta_mean: float
    ci_low: float
    ci_high: float
    p_value: float


@dataclass
class MultiSystemResult:
    systems: list[SystemResult] = field(default_factory=list)

    def accuracy_table(self) -> dict[str, float]:
        """System name → overall accuracy across all runs."""
        out: dict[str, float] = {}
        for s in self.systems:
            recs = s.records
            out[s.name] = sum(1 for r in recs if r.correct) / len(recs) if recs else 0.0
        return out

    def per_category_table(self) -> dict[str, dict[int, tuple[int, int, float]]]:
        """System name → {category: (n, correct, accuracy)}."""
        return {s.name: per_category_accuracy(s.records) for s in self.systems}

    def paired_bootstrap(
        self,
        name_a: str,
        name_b: str,
        *,
        n_resamples: int = 1000,
        ci_alpha: float = 0.05,
        seed: int | None = 42,
    ) -> PairedBootstrapResult:
        """QA-paired bootstrap of (system_a_correct - system_b_correct).

        For each QA shared between the two systems, produce an outcome pair
        (a_correct, b_correct). Resample with replacement; each resample's
        delta_mean is one bootstrap sample. 95% CI + two-sided p-value.
        """
        a = next(s for s in self.systems if s.name == name_a)
        b = next(s for s in self.systems if s.name == name_b)
        a_map = a.per_qa_correct()
        b_map = b.per_qa_correct()
        shared_keys = sorted(set(a_map.keys()) & set(b_map.keys()))
        pairs = [(1 if a_map[k] else 0, 1 if b_map[k] else 0) for k in shared_keys]
        if not pairs:
            return PairedBootstrapResult(
                system_a=name_a, system_b=name_b, n_qa=0,
                delta_mean=0.0, ci_low=0.0, ci_high=0.0, p_value=1.0,
            )

        observed_delta = sum(ai - bi for ai, bi in pairs) / len(pairs)

        rng = random.Random(seed)
        n = len(pairs)
        deltas: list[float] = []
        more_extreme = 0
        for _ in range(n_resamples):
            total = 0
            for _j in range(n):
                ai, bi = pairs[rng.randrange(n)]
                total += ai - bi
            delta = total / n
            deltas.append(delta)
            # two-sided: |delta - observed| >= |observed|
            if abs(delta) >= abs(observed_delta):
                more_extreme += 1

        deltas.sort()
        low_idx = int((ci_alpha / 2) * n_resamples)
        high_idx = int((1 - ci_alpha / 2) * n_resamples) - 1
        low_idx = max(0, min(n_resamples - 1, low_idx))
        high_idx = max(0, min(n_resamples - 1, high_idx))

        return PairedBootstrapResult(
            system_a=name_a,
            system_b=name_b,
            n_qa=n,
            delta_mean=observed_delta,
            ci_low=deltas[low_idx],
            ci_high=deltas[high_idx],
            p_value=(more_extreme + 1) / (n_resamples + 1),
        )


@dataclass
class MultiSystemRunner:
    systems: list[tuple[str, Pipeline]]
    judge: Judge
    budget_words: int = 1000
    categories: frozenset[int] = QA_CATEGORIES_BENCHMARK

    def run(
        self,
        conversations: Iterable[Conversation],
        n_runs: int = 1,
    ) -> MultiSystemResult:
        convs = list(conversations)
        results: list[SystemResult] = []
        for name, pipeline in self.systems:
            runner = BenchmarkRunner(
                pipeline=pipeline,
                judge=self.judge,
                budget_words=self.budget_words,
                categories=self.categories,
            )
            rr = runner.run(convs, n_runs=n_runs)
            results.append(SystemResult(name=name, run_results=rr))
        return MultiSystemResult(systems=results)


def format_comparison_table(result: MultiSystemResult) -> str:
    """Markdown-friendly comparison table of accuracy + per-category."""
    lines: list[str] = []
    cats = sorted({cat for per_cat in result.per_category_table().values() for cat in per_cat})
    header = ["system", "overall"] + [f"cat{cat}" for cat in cats]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    overall = result.accuracy_table()
    per_cat = result.per_category_table()
    for s in result.systems:
        row = [s.name, f"{overall[s.name]:.3f}"]
        for cat in cats:
            stats = per_cat[s.name].get(cat)
            row.append(f"{stats[2]:.3f}" if stats else "-")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
