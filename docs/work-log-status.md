# Work Log & Status — 2026-04-20

Snapshot of where the project is, what's blocked, and the exact next steps.
This doc is the **starting point** for any future session picking up the work.

Read this first, then see:
- [`hyper-triplet-implementation-plan-v5.md`](./hyper-triplet-implementation-plan-v5.md) for the live plan
- [`phase-f-runbook.md`](./phase-f-runbook.md) for the executable Phase F workflow
- [`paper-outline-draft.md`](./paper-outline-draft.md) for the paper structure

---

## 1. Project metrics

| Metric | Value |
|---|---|
| Git commits | 34 |
| Pytest cases | **312**, all offline, <1.5 s |
| Lint | ruff clean |
| Docs in `docs/` | 24 markdown files |
| External clones in `external/` | 6 (gaama, hypergraph_rag, hipporag, everos, hinge, shinge) |
| LLM calls made so far | 0 (offline only; real API blocked on billing) |

## 2. Principles confirmed

The project has crystallised around **8 invariants** derived from HINGE + the
grouping-node principle, documented in [`hinge-north-star.md`](./hinge-north-star.md):

1. **Atomic extraction** — fact + qualifiers in one LLM response
2. **No flat-concept reduction** — typed qualifier schema
3. **Qualifier typing is load-bearing** — location ≠ participant ≠ time
4. **MERGE on value identity** — same value → same node
5. **Evaluation exposes correlation** — per-category breakdown primary
6. **Storage keeps qualifiers first-class** — typed edges, not JSON metadata
7. **Retrieval traverses qualifier edges** — not just BM25 over flat text
8. **LLM-as-classifier only for grouping** — no encoder-style summarisation
   (from [`grouping-node-principle.md`](./grouping-node-principle.md))

Plus a **4-layer functional separation** inside every NodeSet: L0 fact /
L1 temporal + importance / L2 context / L3 auxiliary (topic, community,
ontology).

The **research contribution** is no longer "we invented X". It is "we
decompose the GAAMA (78.9%) → EverMemOS (93.05%) LoCoMo gap into per-
invariant contributions via a 12-preset ablation, and argue the correct
axis is correlation preservation, not hypergraph-vs-triple".

## 3. What's built (offline, all green)

### 3.1 Core v5 stack

```
systems/hyper_triplet/
├── types.py                   L0Fact / L1TemporalImportance / L2Context
│                              / L3Auxiliary + NodeSet (w/ ns_id hash)
├── graph.py                   legacy HyperTripletGraph (kept for v4 tests)
├── star_store.py              O(1) KV store + qualifier/community/episode indices
├── boundary_detector.py       classifier-style chunk segmentation
├── importance_scorer.py       ACT-R activation (log(freq) * exp(-d*dt) * belief)
├── community_detector.py      Louvain over shared-qualifier graph
├── retrieval.py               legacy keyword retrieval (token-IDF)
├── retrieval_ppr.py           PPR over qualifier edges
├── retrieval_stages.py        3-stage pipeline (broad / rank / exact)
├── extractors.py              LLMNodeSetExtractor
├── ltm_creator.py             HyperTripletLTMCreator (legacy graph)
├── pipeline.py                legacy HyperTripletPipeline (v0)
├── pipeline_v5.py             v5 pipeline wiring ALL components via
│                              HyperTripletConfig
├── config.py                  HyperTripletConfig (composes all sub-configs)
├── ablation.py                12 named AblationPreset instances
└── prompts/
    └── node_set_generation.md HINGE-annotated extraction prompt
```

### 3.2 Benchmark infrastructure

```
src/htb/
├── data/locomo.py             LoCoMo-10 loader + evidence normalisation
├── llm/
│   ├── interfaces.py          LLMAdapter protocol
│   ├── mock.py                MockLLMAdapter + canned responses
│   ├── fixture_replay.py      fixture-driven mock for end-to-end smokes
│   └── openai_adapter.py      OpenAIAdapter (gpt-4o-mini/gpt-4o)
└── eval/
    ├── interfaces.py          Pipeline / Judge protocols
    ├── judge.py               KeywordMockJudge (offline fallback)
    ├── llm_judge.py           OpenAIJudge (gpt-4o, paper-standard prompt)
    ├── metrics.py             ScoreRecord + per-category aggregation
    ├── runner.py              BenchmarkRunner (single-system)
    ├── multi_runner.py        MultiSystemRunner + paired bootstrap
    ├── ablation_runner.py     AblationRunner over 12 presets
    └── result_io.py           JSON schema + summary.md generator
```

### 3.3 Baseline adapters (Phase F wire-up targets)

```
systems/baselines/
├── base.py                    BaselineAdapter + PipelineNotReadyError
├── gaama_adapter.py           stub + readiness hint
├── hypergraph_rag_adapter.py  stub + readiness hint
├── hipporag_adapter.py        stub + readiness hint + LoCoMo caveat
├── hypermem_adapter.py        stub + readiness hint + 3080 Ti note
└── evermemos_adapter.py       stub + REST client + EVERMEMOS_API_URL env
```

Each stub satisfies `Pipeline` protocol at import time and raises
`PipelineNotReadyError` with a descriptive hint until wire-up.

### 3.4 Scripts

```
scripts/
├── fetch-locomo10.sh          LoCoMo-10 dataset download
├── evermemos-{up,down,status,logs}.sh  WSL docker lifecycle
├── smoke_test_openai.py       1-conv end-to-end (blocked on API credit)
└── run_phase_f.py             12-preset × N-seed orchestrator, --dry-run works
```

### 3.5 Regression fixture

`tests/fixtures/locomo_conv26_session1_gold.json` — hand-crafted extraction
for 16 real LoCoMo turns across 3 chunks with 7 node_sets. Used for offline
pipeline regression via `fixture_replay.build_replay_mock()`.

### 3.6 CI

`.github/workflows/ci.yml` runs `uv sync + ruff + pytest` on ubuntu +
windows, Python 3.11.

## 4. What's blocked

### 4.1 Phase F execution — OpenAI billing

- Status: `.env` has `OPENAI_API_KEY=sk-...` but account shows 429
  `insufficient_quota` on first call.
- Unblock: top up credits at `platform.openai.com/billing` (minimum $5), OR
  swap to OpenRouter by adding `OPENAI_BASE_URL=https://openrouter.ai/api/v1`
  to `.env` with an OpenRouter key.
- Once unblocked: `scripts/smoke_test_openai.py` should complete in ~1–3 min
  and cost ~$0.10; full 3-seed × 12-preset × 10-conv sweep estimated $30–50.

### 4.2 Baseline adapters — upstream wire-up

5 stubs in `systems/baselines/` with detailed readiness hints but no actual
ingest/retrieve implementations. Each hint cites the specific module /
method / config the wire-up needs. Order of difficulty (easiest first):

1. **GaamaAdapter** — we already understand the SDK. Python-only, SQLite
   storage, uses `gaama.api.AgenticMemorySDK`.
2. **HyperGraphRAGAdapter** — Python-only, `hypergraphrag.HyperGraphRAG`.
3. **HippoRAGAdapter** — Python-only, but LoCoMo-score caveat (HippoRAG 2
   doesn't report LoCoMo natively; choose between GAAMA's inline port or
   running HippoRAG 2 as a new datapoint).
4. **HyperMemAdapter** — Python + local vLLM for Qwen3 embedders (3080 Ti
   may not fit both at FP16 — prefer DeepInfra fallback per readiness hint).
5. **EverMemOSAdapter** — requires full docker-compose stack (MongoDB + ES
   + Milvus + Redis). `scripts/evermemos-up.sh` already exists; just needs
   API key in `methods/evermemos/.env` and `make run`.

## 5. Exact next steps (ordered)

### Immediate (API-blocked)

1. **Unblock OpenAI or OpenRouter** credits (see §4.1).
2. Run `uv run --extra llm python scripts/smoke_test_openai.py` — verifies
   code path end-to-end; if anything breaks, fix before Phase F.

### Phase F proper (API-gated)

3. Run `scripts/run_phase_f.py --sample-ids conv-26 --presets baseline no_community --seeds 42`
   to validate Phase F orchestrator against real OpenAI. ~$1-2 cost.
4. Decide: implement which baseline adapters before the full sweep, or run
   Hyper Triplet solo first and add baselines incrementally.
5. Full sweep: `scripts/run_phase_f.py` (3 seeds, 12 presets, 10 convs) —
   expect 1-2 hours wall time, $30–50.
6. Populate `docs/paper-outline-draft.md` §7.1 and §7.2 with real numbers.

### Offline work still available

Each of these can be done without API credits:

- **Per-category paired bootstrap report format** — extend
  `format_ablation_report` with cat1/cat2/cat3/cat4 deltas + CIs.
- **Cost tracker** — record OpenAI token usage per call; accumulate into
  `results/phase_f/cost.log`.
- **GAAMA adapter real wire-up** — `_readiness_hint` contents already
  describe the exact glue; would require adding `external/gaama` to
  sys.path and invoking the SDK. Unit-test with MockLLM.
- **HyperGraphRAG adapter real wire-up** — similar shape, lighter deps.
- **Judge output cache** — memoize `(question, gold, generated)` → verdict
  so retries don't re-pay judging costs.

## 6. Known gotchas

1. **`systems/hyper_triplet/pipeline_v5.py` does NOT compose the graph** —
   v5 uses StarStore, not HyperTripletGraph. Legacy pipelines still use the
   graph. Don't mix.
2. **BoundaryDetector chunks may differ from the gold fixture's 3 chunks** —
   fixture tests that ingest conv-26 via full pipeline may not match every
   gold node_set; write assertions against observed behaviour, not fixture
   shape.
3. **AblationPreset.config is FROZEN** — `replace()` / `with_overrides()`
   return a new instance.
4. **StarStore.assign_community** keeps an override dict; NodeSet.l3 does
   NOT reflect community_id until Phase D persists it on the NodeSet (see
   comment in star_store.py). Callers should use `store.community_of(ns_id)`
   not `ns.l3.community_id`.
5. **`systems/hyper_triplet/` is the module name** — user-provided spec uses
   `my_own` but that was reconciled to `hyper_triplet` per user decision
   2026-04-19 (see `my-own-test-design-spec.md` header banner).
6. **HINGE citation** — Rosso et al., WWW 2020. The `arxiv 2208.14322` cited
   in the new design spec is HoLmES, not HINGE. Use
   [`hinge-north-star.md`](./hinge-north-star.md) citation, not the spec's.
7. **EverMemOS is the true LoCoMo SOTA** (93.05%), not HyperMem (92.73% —
   a sub-module). See
   [`evermemos-reference-notes.md`](./evermemos-reference-notes.md).
8. **OpenAI account has 0 credits** as of 2026-04-20 PM. First API call
   returns HTTP 429 `insufficient_quota`. Smoke test script otherwise
   verified correct up to that call.

## 7. Document index (pointer list)

**Live plan & north-star**
- [`hyper-triplet-implementation-plan-v5.md`](./hyper-triplet-implementation-plan-v5.md)
- [`hinge-north-star.md`](./hinge-north-star.md)
- [`grouping-principle-integration.md`](./grouping-principle-integration.md)
- [`paper-outline-draft.md`](./paper-outline-draft.md)

**Audits & evidence**
- [`hinge-compliance-audit.md`](./hinge-compliance-audit.md)
- [`hinge-technical-notes.md`](./hinge-technical-notes.md)
- [`evermemos-architecture-deep-dive.md`](./evermemos-architecture-deep-dive.md)
- [`baseline-numbers.md`](./baseline-numbers.md)

**External system reference notes**
- [`gaama-reference-notes.md`](./gaama-reference-notes.md) + [`gaama-fork-points.md`](./gaama-fork-points.md)
- [`hypergraph-rag-reference-notes.md`](./hypergraph-rag-reference-notes.md)
- [`hipporag-reference-notes.md`](./hipporag-reference-notes.md)
- [`hypermem-reference-notes.md`](./hypermem-reference-notes.md)
- [`evermemos-reference-notes.md`](./evermemos-reference-notes.md)
- [`evermemos-setup-checklist.md`](./evermemos-setup-checklist.md)

**User-supplied design docs (reconciled)**
- [`grouping-node-principle.md`](./grouping-node-principle.md)
- [`hypergraph-memory-lineage.md`](./hypergraph-memory-lineage.md)
- [`my-own-test-design-spec.md`](./my-own-test-design-spec.md) (+ [`my-own-spec-vs-current-direction.md`](./my-own-spec-vs-current-direction.md))

**Plan history (superseded)**
- v1 [`hyper-triplet-implementation-plan.md`](./hyper-triplet-implementation-plan.md)
- v2 [`hyper-triplet-implementation-plan-v2.md`](./hyper-triplet-implementation-plan-v2.md)
- v3 [`hyper-triplet-implementation-plan-v3.md`](./hyper-triplet-implementation-plan-v3.md)
- v4 [`hyper-triplet-implementation-plan-v4.md`](./hyper-triplet-implementation-plan-v4.md)

**This session summary** (you are here): `work-log-status.md`.
