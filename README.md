# hyper-triplet-bench

[![CI](https://github.com/whyjp/gr-test/actions/workflows/ci.yml/badge.svg)](https://github.com/whyjp/gr-test/actions/workflows/ci.yml)

HippoRAG2 vs GAAMA vs **Hyper Triplet** on LoCoMo-10.

**Project north star:** [`docs/hinge-north-star.md`](docs/hinge-north-star.md) — HINGE (WWW 2020, *Beyond Triplets*) defines the hyper-relational invariant this project preserves at every layer. Every design decision is judged against it, not against any current SOTA system.

See [`docs/hyper-triplet-implementation-plan-v4.md`](docs/hyper-triplet-implementation-plan-v4.md) for the live plan.
(v1/v2/v3 preserved as history — v4 supersedes after the EverMemOS finding.)

Supporting design notes:
- [`docs/hypergraph-memory-lineage.md`](docs/hypergraph-memory-lineage.md) — two-lineage map (pre-EverMemOS analysis)
- [`docs/evermemos-reference-notes.md`](docs/evermemos-reference-notes.md) — **EverMemOS = true LoCoMo SOTA 93.05%**, repo map, storage stack, Phase 2B plan
- [`docs/gaama-reference-notes.md`](docs/gaama-reference-notes.md) — GAAMA upstream code map
- [`docs/gaama-fork-points.md`](docs/gaama-fork-points.md) — GAAMA replace/extend/reuse map
- [`docs/hypergraph-rag-reference-notes.md`](docs/hypergraph-rag-reference-notes.md) — HyperGraphRAG (NeurIPS 2025) code map
- [`docs/hipporag-reference-notes.md`](docs/hipporag-reference-notes.md) — HippoRAG 2 code map + caveat about LoCoMo score
- [`docs/hypermem-reference-notes.md`](docs/hypermem-reference-notes.md) — HyperMem paper, now repositioned as EverMemOS's conversation-memory sub-module

## Layout

```
data/                   LoCoMo-10 dataset (gitignored; re-download via scripts/)
systems/
  hipporag2/            Phase 1 baseline
  gaama/                Phase 2 baseline (forks/reuses external/gaama)
  hyper_triplet/        Phase 3 proposal
eval/
  judge.py              LLM-as-judge
  metrics.py            Accuracy, per-category, mean+std
  runner.py             3-system evaluation driver
results/                Per-system scores (gitignored)
ablation/               A1..A5 variants (gitignored)
scripts/                Dataset + DB + bench utilities
external/               3rd-party clones, gitignored (e.g. external/gaama)
src/htb/                Shared library code (data loaders, types, utils)
tests/                  pytest — offline, no API required
docs/                   Plans + design notes
```

## Quickstart

```bash
# 1. Install Python 3.11 and create venv (uv manages everything)
uv python install 3.11
uv venv --python 3.11
source .venv/Scripts/activate   # bash on Windows

# 2. Sync dependencies
uv sync                          # core only
uv sync --extra llm              # + OpenAI (needs OPENAI_API_KEY)
uv sync --extra llm --extra embed

# 3. Dataset
bash scripts/fetch-locomo10.sh

# 4. Offline tests
uv run pytest
```

## Status

Offline foundation (API not required):

- [x] Repo scaffold
- [x] Python 3.11 venv (uv)
- [x] LoCoMo-10 dataset fetch script + loader with schema validation
- [x] GAAMA reference clone + code map + fork points
- [x] Offline eval skeleton (Pipeline/Judge protocols, MockJudge, metrics, runner)
- [x] MockLLMAdapter for offline smoke tests
- [x] Plan v2 reframed around hyper-relational KG (fact vs memory motivation)
- [x] Hyper Triplet node_set prompt + LLMNodeSetExtractor + typed models
- [x] HyperTripletLTMCreator skeleton with in-memory graph + MERGE semantics
- [x] Hand-crafted LoCoMo conv-26 gold fixture + fixture-replay MockLLM
- [x] In-memory keyword+IDF retrieval over the hyper-relational graph
- [x] HyperTripletPipeline adapter — full ingest/retrieve/answer loop runs end-to-end through BenchmarkRunner
- [x] Baseline references cloned: HyperGraphRAG, HippoRAG 2 (+ HyperMem paper notes)

API-required phases:

- [ ] Phase 1 HippoRAG2 baseline
- [ ] Phase 2 GAAMA reproduction (target 78.9% ± 2%p)
- [ ] Phase 3 Hyper Triplet — wire LTMCreator skeleton to GAAMA's SqliteMemoryStore
- [ ] Phase 4 full 3-system × N-run sweep
- [ ] Phase 5 ablation A0–A5
- [ ] Phase 6 efficiency data collection

See the [plan v2](docs/hyper-triplet-implementation-plan-v2.md) for details.

## Running tests

```bash
uv run pytest -v                      # 98 tests, all offline, <1s
uv run ruff check src tests systems
```
