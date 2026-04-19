# hyper-triplet-bench

[![CI](https://github.com/whyjp/gr-test/actions/workflows/ci.yml/badge.svg)](https://github.com/whyjp/gr-test/actions/workflows/ci.yml)

Decompose the LoCoMo-10 accuracy gap between triplet-based memory systems
(HippoRAG, GAAMA ≤ 79%) and hypergraph systems (HyperMem, EverMemOS ≥ 93%)
into per-invariant contributions via a **12-preset ablation**. Anchored on
HINGE's hyper-relational principle (Rosso et al., WWW 2020) plus a
classifier-only grouping rule (Invariant #8, derived on 2026-04-19).

> **Project north star:** [`docs/hinge-north-star.md`](docs/hinge-north-star.md) — 8 invariants judged against every design decision.
> **Current entry point:** [`docs/work-log-status.md`](docs/work-log-status.md) — what's built, what's blocked, what's next.
> **Live plan:** [`docs/hyper-triplet-implementation-plan-v5.md`](docs/hyper-triplet-implementation-plan-v5.md)
> **Phase F runbook:** [`docs/phase-f-runbook.md`](docs/phase-f-runbook.md)

---

## Status (2026-04-20)

- **34 commits**, **312 pytest cases** (all offline, <1.5 s), ruff clean
- Plan v5 Phases A0-E complete + F-pre (pipeline wiring, adapters, orchestrator) complete
- Phase F (real LoCoMo sweep) **blocked on OpenAI billing** — code path verified up to first API call
- 6 external systems cloned: GAAMA, HippoRAG 2, HyperGraphRAG, HyperMem, EverMemOS, HINGE/sHINGE

### What's done (offline)

| Category | Artefact |
|---|---|
| Data | LoCoMo-10 loader with evidence normalisation; hand-crafted conv-26 regression fixture |
| Hyper Triplet core | L0/L1/L2/L3 layered NodeSet, StarStore, BoundaryDetector, ImportanceScorer, CommunityDetector, 3-stage retrieval |
| Pipeline | `HyperTripletPipelineV5` composing all v5 components via `HyperTripletConfig` |
| LLM glue | MockLLMAdapter + fixture-replay + OpenAIAdapter + OpenAIJudge |
| Benchmark | BenchmarkRunner, MultiSystemRunner, AblationRunner, paired-bootstrap CI, result I/O + summary auto-gen |
| Ablation presets | 12 named `AblationPreset` (baseline + 11 variants, each labelled against a HINGE invariant) |
| Baselines | 5 Pipeline-protocol adapter stubs with readiness hints |
| Orchestration | `scripts/run_phase_f.py` (dry-run verified end-to-end) |
| Docs | 25 markdown files: plan v1-v5 history, 5 external system reference notes, compliance audit, paper outline, runbook |

### What's blocked (API-gated)

| Step | Blocker | Unblock path |
|---|---|---|
| Smoke test | OpenAI 429 `insufficient_quota` | Top up `platform.openai.com/billing` ($5 min) OR swap `.env` to OpenRouter |
| Phase F full sweep | Same | ~$30–50 budget for 3 seeds × 12 presets × 10 convs |
| Baseline adapters | Each needs upstream wire-up (readiness hints cite exact glue) | Incremental; simplest first: GAAMA → HyperGraphRAG → HippoRAG → HyperMem → EverMemOS |

---

## Quickstart

```bash
# 1. Install Python 3.11 + sync deps via uv
uv python install 3.11
uv venv --python 3.11
uv sync                          # core (offline only)
uv sync --extra llm              # + OpenAI SDK for real-API runs

# 2. Download dataset (2.8 MB)
bash scripts/fetch-locomo10.sh

# 3. Offline tests (should print 312 passed)
uv run pytest
uv run ruff check src tests systems

# 4. Offline dry-run of Phase F orchestrator
uv run python scripts/run_phase_f.py --dry-run \
  --sample-ids conv-26 --presets baseline no_community --seeds 42 \
  --results-dir /tmp/phase_f_dry
```

For live Phase F (requires `OPENAI_API_KEY` in `.env`):

```bash
uv run --extra llm python scripts/smoke_test_openai.py
uv run --extra llm python scripts/run_phase_f.py \
  --sample-ids conv-26 --presets baseline no_community --seeds 42
```

Detailed steps in [`docs/phase-f-runbook.md`](docs/phase-f-runbook.md).

---

## Layout

```
data/                       LoCoMo-10 (gitignored; fetch via scripts/)
external/                   upstream clones (gitignored): gaama, hipporag,
                            hypergraph_rag, everos, hinge, shinge
scripts/                    fetch + docker + smoke + Phase F orchestrator
src/htb/
  data/                     LoCoMo loader + typed models
  llm/                      LLMAdapter + Mock + OpenAI + fixture replay
  eval/                     Pipeline/Judge protocols, runners, judge, ablation,
                            result I/O, summary generator
systems/
  hyper_triplet/            v5 implementation (types, store, retrieval, pipeline,
                            config, ablation, extractors)
  baselines/                5 Pipeline-protocol adapter stubs
results/                    per-run JSON + summary.md (gitignored)
docs/                       plans, reference notes, audits, runbook
tests/                      pytest (312 cases, offline)
.github/workflows/ci.yml    ubuntu + windows matrix on push/PR
```

---

## The 8 invariants (design rules)

Every system is judged against these (see [`docs/hinge-north-star.md`](docs/hinge-north-star.md)):

1. **Atomic extraction** — fact + qualifiers in ONE LLM response
2. **No flat-concept reduction** — typed qualifier schema
3. **Qualifier typing is load-bearing** — location ≠ participant ≠ time
4. **MERGE on value identity** — same value → same node
5. **Evaluation exposes correlation** — per-category primary
6. **Storage keeps qualifiers first-class** — typed edges, not JSON payload
7. **Retrieval traverses qualifier edges** — not just BM25 over flat text
8. **LLM-as-classifier only for grouping** — no encoder-style summarisation

Per-system compliance verdicts: [`docs/hinge-compliance-audit.md`](docs/hinge-compliance-audit.md).

---

## Documentation index

**Live**
- [`work-log-status.md`](docs/work-log-status.md) — session snapshot, next steps
- [`hyper-triplet-implementation-plan-v5.md`](docs/hyper-triplet-implementation-plan-v5.md) — current plan
- [`hinge-north-star.md`](docs/hinge-north-star.md) — 8 invariants
- [`phase-f-runbook.md`](docs/phase-f-runbook.md) — execution guide
- [`paper-outline-draft.md`](docs/paper-outline-draft.md) — paper structure

**Audits & evidence**
- [`hinge-compliance-audit.md`](docs/hinge-compliance-audit.md) — 6-system × 8-invariant matrix
- [`hinge-technical-notes.md`](docs/hinge-technical-notes.md) — HINGE + sHINGE math
- [`evermemos-architecture-deep-dive.md`](docs/evermemos-architecture-deep-dive.md) — SOTA internal wiring
- [`baseline-numbers.md`](docs/baseline-numbers.md) — prior reported LoCoMo scores

**External system reference**
- [`gaama-reference-notes.md`](docs/gaama-reference-notes.md) + [`gaama-fork-points.md`](docs/gaama-fork-points.md)
- [`hypergraph-rag-reference-notes.md`](docs/hypergraph-rag-reference-notes.md)
- [`hipporag-reference-notes.md`](docs/hipporag-reference-notes.md)
- [`hypermem-reference-notes.md`](docs/hypermem-reference-notes.md)
- [`evermemos-reference-notes.md`](docs/evermemos-reference-notes.md) + [`evermemos-setup-checklist.md`](docs/evermemos-setup-checklist.md)

**User-supplied design docs (reconciled)**
- [`grouping-node-principle.md`](docs/grouping-node-principle.md) + [`grouping-principle-integration.md`](docs/grouping-principle-integration.md)
- [`hypergraph-memory-lineage.md`](docs/hypergraph-memory-lineage.md)
- [`my-own-test-design-spec.md`](docs/my-own-test-design-spec.md) + [`my-own-spec-vs-current-direction.md`](docs/my-own-spec-vs-current-direction.md)

**Plan history (superseded)**
- v1 [`hyper-triplet-implementation-plan.md`](docs/hyper-triplet-implementation-plan.md)
- v2 [`hyper-triplet-implementation-plan-v2.md`](docs/hyper-triplet-implementation-plan-v2.md)
- v3 [`hyper-triplet-implementation-plan-v3.md`](docs/hyper-triplet-implementation-plan-v3.md)
- v4 [`hyper-triplet-implementation-plan-v4.md`](docs/hyper-triplet-implementation-plan-v4.md)
