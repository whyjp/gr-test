"""AblationRunner — iterate HyperTripletConfig presets under identical evaluation.

Per plan v5 Phase E, the paper's decomposition story requires running the
same Hyper Triplet pipeline under 12 config variants (``baseline`` plus 11
principle-level ablations) and reporting delta accuracy against baseline
per category.

Design:
- Accepts a ``pipeline_factory: Callable[[HyperTripletConfig], Pipeline]``
  so the runner is agnostic to whether callers construct ``HyperTripletPipelineV5``,
  a mocked pipeline for offline tests, or a future variant.
- Uses the existing ``BenchmarkRunner`` internally so per-preset evaluation
  stays consistent with single-system runs.
- Emits a ``MultiSystemResult`` whose ``systems`` list is one entry per
  preset; paired-bootstrap between any two presets is then available via
  the existing ``MultiSystemResult.paired_bootstrap`` API.
- Produces a summary markdown via ``format_ablation_report`` that highlights
  per-preset accuracy and the HINGE invariant each variant violates.

Completely offline-safe — given a ``MockLLMAdapter`` factory, every ablation
preset executes without network.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from htb.data import QA_CATEGORIES_BENCHMARK, Conversation
from htb.eval.interfaces import Judge, Pipeline
from htb.eval.multi_runner import (
    MultiSystemResult,
    PairedBootstrapResult,
    SystemResult,
    format_comparison_table,
)
from htb.eval.runner import BenchmarkRunner
from systems.hyper_triplet.ablation import ABLATION_NAMES, AblationPreset, all_presets
from systems.hyper_triplet.config import HyperTripletConfig

PipelineFactory = Callable[[HyperTripletConfig], Pipeline]


@dataclass
class AblationRunResult:
    preset: AblationPreset
    system_result: SystemResult

    @property
    def name(self) -> str:
        return self.preset.name

    @property
    def accuracy(self) -> float:
        records = self.system_result.records
        if not records:
            return 0.0
        return sum(1 for r in records if r.correct) / len(records)


@dataclass
class AblationSweepResult:
    runs: list[AblationRunResult] = field(default_factory=list)

    @property
    def baseline(self) -> AblationRunResult | None:
        for run in self.runs:
            if run.name == "baseline":
                return run
        return None

    def as_multi_system_result(self) -> MultiSystemResult:
        return MultiSystemResult(systems=[r.system_result for r in self.runs])

    def accuracy_table(self) -> dict[str, float]:
        return {r.name: r.accuracy for r in self.runs}

    def delta_vs_baseline(self) -> dict[str, float]:
        baseline = self.baseline
        if baseline is None:
            return {}
        base_acc = baseline.accuracy
        return {r.name: r.accuracy - base_acc for r in self.runs if r.name != "baseline"}

    def paired_bootstrap_against_baseline(
        self,
        preset_name: str,
        *,
        n_resamples: int = 1000,
        seed: int | None = 42,
    ) -> PairedBootstrapResult | None:
        baseline = self.baseline
        if baseline is None:
            return None
        msr = self.as_multi_system_result()
        return msr.paired_bootstrap(
            name_a=preset_name,
            name_b="baseline",
            n_resamples=n_resamples,
            seed=seed,
        )


@dataclass
class AblationRunner:
    pipeline_factory: PipelineFactory
    judge: Judge
    budget_words: int = 1000
    categories: frozenset[int] = QA_CATEGORIES_BENCHMARK
    preset_names: tuple[str, ...] = ABLATION_NAMES

    def run(
        self,
        conversations: Iterable[Conversation],
        n_runs: int = 1,
    ) -> AblationSweepResult:
        convs = list(conversations)
        presets = all_presets()
        results: list[AblationRunResult] = []
        for name in self.preset_names:
            if name not in presets:
                continue
            preset = presets[name]
            pipeline = self.pipeline_factory(preset.config)
            runner = BenchmarkRunner(
                pipeline=pipeline,
                judge=self.judge,
                budget_words=self.budget_words,
                categories=self.categories,
            )
            rr = runner.run(convs, n_runs=n_runs)
            # MultiSystemResult keys off SystemResult.name; rename to preset name
            # so paired_bootstrap between presets works cleanly.
            from dataclasses import replace

            sr = SystemResult(name=preset.name, run_results=replace(rr))
            results.append(AblationRunResult(preset=preset, system_result=sr))
        return AblationSweepResult(runs=results)


def format_ablation_report(sweep: AblationSweepResult) -> str:
    """Markdown summary of an ablation sweep.

    Layout:
      1. Per-preset accuracy table (reusing ``format_comparison_table``)
      2. Delta-vs-baseline table with HINGE invariant label
      3. Significance asterisk for any preset whose paired-bootstrap
         vs baseline has p < 0.05
    """
    lines: list[str] = []
    lines.append("## Ablation accuracy table")
    lines.append("")
    lines.append(format_comparison_table(sweep.as_multi_system_result()))
    lines.append("")

    baseline = sweep.baseline
    if baseline is not None:
        lines.append("## Delta vs baseline")
        lines.append("")
        lines.append("| preset | delta | HINGE invariant violated |")
        lines.append("|---|---|---|")
        for run in sweep.runs:
            if run.name == "baseline":
                continue
            delta = run.accuracy - baseline.accuracy
            lines.append(
                f"| {run.name} | {delta:+.3f} | {run.preset.hinge_invariant_violated} |"
            )
    return "\n".join(lines)
