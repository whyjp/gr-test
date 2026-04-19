"""Result serialisation matching plan v5 §10.3 layout.

- Per-system results go to ``{dir}/{dataset}_{system}_{seed}.json``.
- A roll-up ``{dir}/summary.md`` is generated from every JSON in the dir.

The schema stays minimal so it survives Phase F changes — we capture what
any paper reader needs: per-QA records + accuracy + per-category + metadata.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from htb.eval.ablation_runner import AblationSweepResult
from htb.eval.metrics import ScoreRecord, per_category_accuracy
from htb.eval.multi_runner import SystemResult


@dataclass(slots=True)
class RunMetadata:
    dataset: str
    system: str
    seed: int
    n_runs: int = 1
    extract_model: str | None = None
    judge_model: str | None = None
    notes: str = ""


def _record_to_dict(r: ScoreRecord) -> dict[str, Any]:
    return {
        "run_id": r.run_id,
        "conv_id": r.conv_id,
        "question": r.question,
        "gold_answer": r.gold_answer,
        "generated_answer": r.generated_answer,
        "category": r.category,
        "judgment": r.judgment,
        "retrieval_ms": round(r.retrieval_ms, 3),
        "answer_ms": round(r.answer_ms, 3),
    }


def _dict_to_record(d: dict[str, Any]) -> ScoreRecord:
    return ScoreRecord(
        run_id=int(d["run_id"]),
        conv_id=str(d["conv_id"]),
        question=str(d["question"]),
        gold_answer=str(d["gold_answer"]),
        generated_answer=str(d["generated_answer"]),
        category=int(d["category"]),
        judgment=str(d["judgment"]),  # type: ignore[arg-type]
        retrieval_ms=float(d.get("retrieval_ms", 0.0)),
        answer_ms=float(d.get("answer_ms", 0.0)),
    )


def serialize_system_result(
    system_result: SystemResult,
    metadata: RunMetadata,
) -> dict[str, Any]:
    records = system_result.records
    n = len(records)
    n_correct = sum(1 for r in records if r.correct)
    accuracy = n_correct / n if n else 0.0
    per_cat = {
        str(cat): {"n": stats[0], "n_correct": stats[1], "accuracy": stats[2]}
        for cat, stats in per_category_accuracy(records).items()
    }
    return {
        "metadata": asdict(metadata),
        "accuracy": accuracy,
        "n": n,
        "n_correct": n_correct,
        "per_category": per_cat,
        "records": [_record_to_dict(r) for r in records],
    }


def save_system_result(
    system_result: SystemResult,
    metadata: RunMetadata,
    results_dir: Path,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{metadata.dataset}_{metadata.system}_{metadata.seed}.json"
    path = results_dir / filename
    payload = serialize_system_result(system_result, metadata)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_ablation_sweep(
    sweep: AblationSweepResult,
    *,
    dataset: str,
    seed: int,
    results_dir: Path,
    extract_model: str | None = None,
    judge_model: str | None = None,
) -> list[Path]:
    """Save each preset's result under ``{dataset}_{preset}_{seed}.json``."""
    out: list[Path] = []
    for run in sweep.runs:
        metadata = RunMetadata(
            dataset=dataset,
            system=run.name,
            seed=seed,
            extract_model=extract_model,
            judge_model=judge_model,
            notes=run.preset.description,
        )
        out.append(save_system_result(run.system_result, metadata, results_dir))
    return out


@dataclass(slots=True, frozen=True)
class LoadedResult:
    metadata: RunMetadata
    accuracy: float
    n: int
    n_correct: int
    per_category: dict[int, tuple[int, int, float]]
    records: tuple[ScoreRecord, ...] = field(default_factory=tuple)


def load_system_result(path: Path) -> LoadedResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    md = payload["metadata"]
    metadata = RunMetadata(
        dataset=md["dataset"],
        system=md["system"],
        seed=int(md["seed"]),
        n_runs=int(md.get("n_runs", 1)),
        extract_model=md.get("extract_model"),
        judge_model=md.get("judge_model"),
        notes=md.get("notes", ""),
    )
    per_cat_raw = payload.get("per_category", {}) or {}
    per_cat: dict[int, tuple[int, int, float]] = {
        int(cat): (int(v["n"]), int(v["n_correct"]), float(v["accuracy"]))
        for cat, v in per_cat_raw.items()
    }
    return LoadedResult(
        metadata=metadata,
        accuracy=float(payload["accuracy"]),
        n=int(payload["n"]),
        n_correct=int(payload["n_correct"]),
        per_category=per_cat,
        records=tuple(_dict_to_record(r) for r in payload.get("records", [])),
    )


def load_all_in_dir(results_dir: Path) -> list[LoadedResult]:
    out: list[LoadedResult] = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "summary.md":
            continue
        try:
            out.append(load_system_result(p))
        except (KeyError, json.JSONDecodeError, ValueError):
            continue
    return out


def format_summary_markdown(loaded: Iterable[LoadedResult]) -> str:
    """Markdown summary aggregating every system/seed result in a dir.

    Layout:
    1. Overall accuracy table (system x seed -> overall) mean ± std across seeds
    2. Per-category best-seed numbers per system
    """
    items = list(loaded)
    by_system: dict[str, list[LoadedResult]] = {}
    for it in items:
        by_system.setdefault(it.metadata.system, []).append(it)

    lines: list[str] = []
    lines.append("# LoCoMo results — auto-generated summary")
    lines.append("")
    lines.append("## Overall accuracy by system (mean across seeds)")
    lines.append("")
    lines.append("| system | n_seeds | mean | min | max |")
    lines.append("|---|---:|---:|---:|---:|")
    for system in sorted(by_system.keys()):
        seeds = by_system[system]
        accs = [r.accuracy for r in seeds]
        if not accs:
            continue
        mean = sum(accs) / len(accs)
        lines.append(
            f"| {system} | {len(accs)} | {mean:.3f} | {min(accs):.3f} | {max(accs):.3f} |"
        )

    lines.append("")
    lines.append("## Per-category accuracy (best seed per system)")
    lines.append("")
    # Union of categories across all loaded
    cats: set[int] = set()
    for it in items:
        cats.update(it.per_category.keys())
    cat_list = sorted(cats)
    header = "| system |" + "".join(f" cat{c} |" for c in cat_list) + " overall |"
    divider = "|---|" + "".join("---:|" for _ in cat_list) + "---:|"
    lines.append(header)
    lines.append(divider)
    for system in sorted(by_system.keys()):
        best = max(by_system[system], key=lambda r: r.accuracy)
        cells = [system]
        for c in cat_list:
            stats = best.per_category.get(c)
            cells.append(f"{stats[2]:.3f}" if stats else "-")
        cells.append(f"{best.accuracy:.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def write_summary(results_dir: Path) -> Path:
    loaded = load_all_in_dir(results_dir)
    summary = format_summary_markdown(loaded)
    out_path = results_dir / "summary.md"
    out_path.write_text(summary, encoding="utf-8")
    return out_path


def make_run_metadata_from_env(
    *,
    dataset: str,
    system: str,
    seed: int,
    notes: str = "",
) -> RunMetadata:
    """Convenience: populate extract_model / judge_model from env vars so the
    JSON payload carries provenance automatically."""
    return RunMetadata(
        dataset=dataset,
        system=system,
        seed=seed,
        extract_model=os.environ.get("EXTRACT_MODEL"),
        judge_model=os.environ.get("JUDGE_MODEL"),
        notes=notes,
    )
