# Hyper Triplet: Implementation & Benchmark Plan — v5

> **North star**: [`hinge-north-star.md`](./hinge-north-star.md) — HINGE (Rosso et al., WWW 2020) hyper-relational invariant.
>
> Replaces [v4](./hyper-triplet-implementation-plan-v4.md) (2026-04-19 PM).
> v5 integrates the user-provided [`my-own-test-design-spec.md`](./my-own-test-design-spec.md) and the EverMemOS SOTA finding. Module name stays `hyper_triplet` per user decision 2026-04-19.

---

## What v5 consolidates

From v4:
- 6-system benchmark (HippoRAG / GAAMA / HyperGraphRAG / Hyper Triplet / HyperMem / EverMemOS)
- Decomposition framing — measure each HINGE invariant's contribution
- HINGE north-star invariants as design rules

From the new `my-own-test-design-spec.md`:
- **4-layer functional separation** (L0 fact / L1 temporal-importance / L2 context / L3 auxiliary) as explicit type structure inside `NodeSet`
- **Star-native storage** — one node_set = one star subgraph, O(1) KV-friendly lookup
- **Hyper-relational edges** with per-edge qualifier scoring (not just filters)
- **3-stage retrieval**: broad (L2/L3) → rank (L1) → exact (L0 + edge qualifier)
- **Boundary detector** replacing fixed-turn chunking
- **Importance scorer** (ACT-R activation) on L1
- **Community detector** (Leiden) on L3
- **7 principle-level ablations** in addition to v4's decomposition ablations
- **3-seed evaluation** `[42, 1337, 2024]` with strict "beat-on-all-3" rule

Decisions baked in (user, 2026-04-19):
- Module path: `systems/hyper_triplet/` (no rename to `my_own/`)
- Baselines: include EverMemOS (true LoCoMo SOTA 93.05%)
- HINGE citation: Rosso 2020 primary
- LLM provider: OpenAI
- LongMemEval / MuSiQue / HotpotQA: Phase 6 (LoCoMo first)

---

## Architecture — updated (v5)

### Data model (extends v4's NodeSet)

Full structural separation keeping backward-compatible construction:

```
L0Fact:
  subject, predicate, object               # the triple
  edge_qualifiers: dict[str, Any]          # relation-level qualifiers
                                            #   (confidence, valid_from on the edge itself)

L1TemporalImportance:
  timestamp: str | None                    # observation time (session date)
  time_reference: str | None               # narrative time from dialogue
  valid_from, valid_until: str | None      # Foresight-style validity interval
  duration_days: int | None
  importance: float                        # ACT-R activation + recency + freq
  belief: float                            # LLM-reported fact confidence

L2Context:
  location: str | None
  participants: tuple[str, ...]
  activity_type: str | None
  mood: str | None

L3Auxiliary:
  topic: str | None                        # semantic cluster label
  community_id: str | None                 # Leiden/LPA detected community
  embedding_ref: str | None                # pointer if embeddings stored elsewhere
  source_ref: str | None                   # raw signal/doc id

NodeSet:
  ns_id: str                               # deterministic hash of L0Fact.to_text()
  l0: L0Fact
  l1: L1TemporalImportance
  l2: L2Context
  l3: L3Auxiliary
  source_episode_ids: tuple[str, ...]
```

Backwards compat: keep `Fact` / `Qualifiers` as aliases + conversion helpers so existing 117 tests pass during the refactor.

### Star store

```
StarStore:
  put(ns: NodeSet) -> str                     # upsert by ns_id
  get(ns_id: str) -> NodeSet | None            # O(1)
  iter_stars() -> Iterator[NodeSet]
  qualifier_index[type, value] -> set[ns_id]   # inter-star joins
  community_index[community_id] -> set[ns_id]
```

Implementation: in-memory dict for the bench; production would swap for Memgraph / FalkorDB but that's out of scope for gr-test.

### Retrieval — 3-stage

```
stage1_broad(query, top_n=500) -> candidate_ns_ids:
  1. extract query entities + keywords
  2. lexical (BM25) match against L2 context text
  3. semantic (embedding) match against fact verbalisation
  4. expand via L3 community_id

stage2_rank(candidate_ns_ids, top_k=30) -> ranked:
  1. L1 importance score
  2. query-temporal alignment (if query contains "when"/"since"/etc.,
     favour facts inside the query's time window)
  3. combined score

stage3_exact(ranked, budget_words) -> final_context:
  1. L0 fact text vs query
  2. edge_qualifier feature re-rank (confidence, source_type, temporal_validity)
  3. word-budget trim
  4. assemble final context for answer LLM
```

Each stage is a standalone module and toggleable for ablation.

---

## Phase roster (v5)

### Phase A0 — Layer refactor (offline, immediate)
- Introduce `L0Fact / L1TemporalImportance / L2Context / L3Auxiliary` types
- `NodeSet` gains `ns_id` (deterministic hash) + `importance: float` field
- Keep `Fact` / `Qualifiers` as compatibility aliases; provide migration accessors (`ns.l0`, `ns.l1`, ...)
- All 117 existing tests continue to pass
- **One commit**

### Phase A1 — StarStore abstraction (offline)
- `systems/hyper_triplet/star_store.py` with in-memory backend
- Qualifier / community indices for inter-star joins
- Replaces direct `HyperTripletGraph.nodes` access in ltm_creator + pipeline
- **One commit**

### Phase B — Extraction upgrades (offline + mock LLM)
- `boundary_detector.py` — temporal coherence + entity overlap detector
- `importance_scorer.py` — ACT-R base activation (`log(freq) + recency_decay`)
- Extend `LLMNodeSetExtractor` + prompt to emit L1/L2/L3 fields separately
- Update gold fixture to v2 with the expanded shape
- **One commit**

### Phase C — 3-stage retrieval (offline)
- `retrieval/stage1_broad.py`, `retrieval/stage2_rank.py`, `retrieval/stage3_exact.py`, `retrieval/pipeline.py`
- Existing `retrieve()` becomes stage-3 fallback; `retrieve_ppr()` becomes stage-1 option
- Wire `HyperTripletPipeline` to use the 3-stage pipeline
- **One commit**

### Phase D — Background workers (offline)
- `background/community_detector.py` — Leiden / Louvain via networkx
- `background/importance_scorer.py` — periodic re-score
- Integration tests with fixture graph
- **One commit**

### Phase E — Config + ablations (offline)
- `systems/hyper_triplet/config.py` — all hyperparameters centralised
- `ablation_runner.py` wiring 7 principle-level toggles (no_node_set / no_layer_separation / no_hyper_edge / no_star_storage / no_stage1 / no_hybrid_index / no_community)
- Extend `MultiSystemRunner` to run ablations across shared infrastructure
- **One commit**

### Phase F — Benchmark execution (API-gated)
- OpenAI key + embedding provider configured
- 3-seed × 6-system × LoCoMo-10 full sweep
- `eval/results/{dataset}_{system}_{seed}.json` layout
- `eval/results/summary.md` auto-generator
- Paired-bootstrap confidence intervals via `MultiSystemRunner`

### Phase 6 — Extra benchmarks (deferred)
- LongMemEval adapter
- MuSiQue / HotpotQA adapters
- Cross-benchmark consistency analysis

---

## Ablation roster (v5)

From `my-own-test-design-spec.md` §7 plus v4 decomposition points:

| Ablation | Disabled component | HINGE invariant tested |
|---|---|---|
| `no_node_set` | atomic ingestion (emit independent triples) | #1 atomicity |
| `no_layer_separation` | L0/L1/L2/L3 types (flat) | spec's layer principle |
| `no_hyper_edge` | edge qualifiers (plain triple edges) | #3 qualifier typing |
| `no_star_storage` | star index (flat graph) | #6 first-class qualifiers |
| `no_stage1` | stage-1 broad | spec's pipeline principle |
| `no_hybrid_index` | lexical OR semantic only | retrieval signal combination |
| `no_community` | L3 community_id | high-level structure |
| `no_importance` | L1 importance scoring | temporal-importance layer |
| `no_boundary_detector` | fixed chunking | boundary-aware ingest |

Plus v4 decomposition against EverMemOS:
| Ablation | vs EverMemOS | |
|---|---|---|
| `D-mc` | MemCell-level only | isolates atomic bundle |
| `D-mc+scene` | + MemScene | +theme |
| `D-mc+foresight` | + validity interval | +temporal discipline |
| `D-recollection` | + agentic retrieval | +coarse-to-fine |
| `D-full` | EverMemOS native | reference |

---

## Evaluation protocol (v5)

- **Seeds**: `[42, 1337, 2024]` (3 seeds, full eval each)
- **Aggregation**: mean + std across seeds; `my_own` must beat each baseline on **every** seed
- **Metric**: LLM-as-judge accuracy (primary), F1 + EM + Recall@K + latency + ingest throughput (secondary)
- **Report layout**: `eval/results/{dataset}_{system}_{seed}.json` + `eval/results/summary.md`
- **Baselines per run**: HippoRAG 2, GAAMA, HyperGraphRAG, HyperMem (standalone), EverMemOS, **Hyper Triplet (ours)** + ablations
- **Statistical significance**: paired bootstrap via `MultiSystemRunner.paired_bootstrap()` — 95% CI on per-QA delta

---

## What stays unchanged from v4

- HINGE as north star (Rosso 2020)
- `external/` clones read-only
- Offline tests run under 1 second
- Commit-push at each meaningful milestone, README + docs updated in the same commit
- Pydantic models, `uv` Python 3.11 toolchain

---

## Open risks

1. **Gold fixture churn**: expanding NodeSet to L0/L1/L2/L3 requires re-authoring `tests/fixtures/locomo_conv26_session1_gold.json`. Keep the flat `Qualifiers` shape as a convenience for test authors; auto-split into layers.
2. **3-stage retrieval regressions**: current `retrieve_ppr` tests pass because PPR is single-pass. Moving to 3-stage may change retrieval ranks in edge cases.
3. **EverMemOS execution**: docker-compose infrastructure (4-5 GB RAM) plus LLM + embedding + rerank API usage. Phase F will need a cost budget estimate before running.
4. **Importance scoring reference**: ACT-R parameters from cognitive science aren't tuned for LoCoMo's 10-session scale. Pick simple defaults (recency decay τ=session count, log-frequency) and sweep in Phase E.
