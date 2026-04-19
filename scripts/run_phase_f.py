"""Phase F orchestrator — 12-preset ablation sweep over LoCoMo.

Runs the full v5 decomposition ablation per plan v5 §6 + 7.

Modes:
    --dry-run       use MockLLM + KeywordMockJudge (no network, fast sanity
                    check; expected accuracy is not meaningful but every code
                    path executes)
    real            (default) uses OpenAIAdapter for extract + OpenAIJudge
                    for scoring; requires OPENAI_API_KEY in .env

Typical invocations:
    uv run python scripts/run_phase_f.py --dry-run
    uv run --extra llm python scripts/run_phase_f.py \
        --sample-ids conv-26 --presets baseline no_community --seeds 42

Artefacts: ``results/phase_f/{dataset}_{preset}_{seed}.json`` + summary.md
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from htb.data import load_locomo10  # noqa: E402
from htb.eval import (  # noqa: E402
    AblationRunner,
    KeywordMockJudge,
    OpenAIJudge,
    save_ablation_sweep,
    write_summary,
)
from htb.llm import (  # noqa: E402
    MockLLMAdapter,
    OpenAIAdapter,
    canned_node_set_generation_response,
)
from systems.hyper_triplet.ablation import ABLATION_NAMES  # noqa: E402
from systems.hyper_triplet.config import HyperTripletConfig  # noqa: E402
from systems.hyper_triplet.extractors import LLMNodeSetExtractor  # noqa: E402
from systems.hyper_triplet.pipeline import template_answerer  # noqa: E402
from systems.hyper_triplet.pipeline_v5 import HyperTripletPipelineV5  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use MockLLM / KeywordMockJudge — no network, no cost.",
    )
    parser.add_argument(
        "--dataset",
        default="locomo10",
        help="Dataset tag embedded in result filenames.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="LoCoMo sample_ids to include (default: all 10).",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=list(ABLATION_NAMES),
        help="Ablation preset names to run (default: all 12).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 1337, 2024],
        help="RNG seeds; script runs one sweep per seed.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1,
        help="BenchmarkRunner n_runs per seed (default 1).",
    )
    parser.add_argument(
        "--budget-words",
        type=int,
        default=1000,
        help="Retrieval budget in words.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "phase_f"),
        help="Output directory for per-preset JSON + summary.md.",
    )
    return parser.parse_args()


def _build_llm_components(dry_run: bool):
    if dry_run:
        mock = MockLLMAdapter(
            rules=[("node_set", canned_node_set_generation_response())],
            default=canned_node_set_generation_response(),
        )
        judge_mock = MockLLMAdapter(default="CORRECT")
        return mock, OpenAIJudge(llm=judge_mock), "mock-node-set", "mock-correct"
    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY missing. Populate .env or rerun with --dry-run."
        )
    extract_model = os.environ.get("EXTRACT_MODEL", "gpt-4o-mini")
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-4o")
    extract_llm = OpenAIAdapter(default_model=extract_model)
    judge_llm = OpenAIAdapter(default_model=judge_model)
    return extract_llm, OpenAIJudge(llm=judge_llm, model=judge_model), extract_model, judge_model


def main() -> int:
    args = _parse_args()
    data_path = REPO_ROOT / "data" / "locomo10.json"
    if not data_path.exists():
        print(f"[error] dataset missing at {data_path}", file=sys.stderr)
        return 1

    convs = load_locomo10(data_path)
    if args.sample_ids:
        want = set(args.sample_ids)
        convs = [c for c in convs if c.sample_id in want]
        if not convs:
            print(f"[error] no convs matched {args.sample_ids}", file=sys.stderr)
            return 1

    extract_llm, judge, extract_model, judge_model = _build_llm_components(args.dry_run)
    extractor = LLMNodeSetExtractor(llm=extract_llm)

    def pipeline_factory(cfg: HyperTripletConfig):
        return HyperTripletPipelineV5(
            extractor=extractor,
            config=cfg,
            answerer=template_answerer,
        )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # KeywordMockJudge on dry_run is cheaper and deterministic;
    # OpenAIJudge even via MockLLM works but KeywordMockJudge is more typical
    # for smoke testing the wiring.
    actual_judge = KeywordMockJudge() if args.dry_run else judge

    print(
        f"[phase_f] mode={'dry-run' if args.dry_run else 'live'} "
        f"extract={extract_model} judge={judge_model} "
        f"convs={len(convs)} presets={len(args.presets)} seeds={args.seeds}"
    )

    overall_t0 = time.time()
    for seed in args.seeds:
        print(f"\n[phase_f] seed={seed}")
        runner = AblationRunner(
            pipeline_factory=pipeline_factory,
            judge=actual_judge,
            budget_words=args.budget_words,
            preset_names=tuple(args.presets),
        )
        t0 = time.time()
        sweep = runner.run(convs, n_runs=args.n_runs)
        elapsed = time.time() - t0
        print(f"[phase_f] seed={seed} elapsed={elapsed:.1f}s")

        seed_dir = results_dir / str(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        paths = save_ablation_sweep(
            sweep,
            dataset=args.dataset,
            seed=seed,
            results_dir=seed_dir,
            extract_model=extract_model,
            judge_model=judge_model,
        )
        print(f"[phase_f] seed={seed} wrote {len(paths)} files to {seed_dir}")

    # Roll-up: aggregate all seeds into a top-level summary
    all_dir = results_dir
    # Gather files from every per-seed subdir into a flat view for the summary
    flat_dir = results_dir / "_flat"
    flat_dir.mkdir(exist_ok=True)
    for seed_dir in results_dir.iterdir():
        if not seed_dir.is_dir() or seed_dir.name in ("_flat",):
            continue
        for p in seed_dir.glob("*.json"):
            target = flat_dir / p.name
            target.write_bytes(p.read_bytes())
    summary_path = write_summary(flat_dir)
    print(f"\n[phase_f] summary written to {summary_path}")
    print(f"[phase_f] total_elapsed={time.time() - overall_t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
