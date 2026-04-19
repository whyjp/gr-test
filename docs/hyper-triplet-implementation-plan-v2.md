# Hyper Triplet: Implementation & Benchmark Plan — v2

> **⚠️ SUPERSEDED BY [v3](./hyper-triplet-implementation-plan-v3.md) — 2026-04-19**
>
> v2's "hyper-relational KG" novelty framing is invalidated by
> `docs/hypergraph-memory-lineage.md`: HINGE/StarE/HyperGraphRAG/HyperMem
> already established this structure, and HyperMem is LoCoMo SOTA at 92.73%.
>
> v3 reframes Hyper Triplet as an ablation probe within Lineage A and the
> benchmark as a decomposition of the +14pp gap between GAAMA and HyperMem.

---

> Replaces [v1](./hyper-triplet-implementation-plan.md) (2026-04-19).
> v2 reflects findings from actual LoCoMo-10 data inspection, GAAMA code deep-read,
> and user-confirmed framing around hyper-relational KGs.

---

## What changed vs v1

| # | v1 claim | v2 correction | Source |
|---|----------|---------------|--------|
| 1 | LoCoMo JSON is a dict keyed by `conv-XX` | **List of 10 samples**; conversation nested under `sample.conversation` | [data schema memo](../../../.claude/projects/D--github-gr-test/memory/project_locomo10_schema.md) / `src/htb/data/locomo.py` |
| 2 | GAAMA makes **3 LLM calls** per chunk (fact / concept / reflection) | GAAMA makes **2 LLM calls** per chunk; fact+concept share one prompt `prompts/fact_generation.md` | [fork points doc](./gaama-fork-points.md) + `external/gaama/services/llm_extractors.py:160` |
| 3 | Contribution framed as "~33% fewer LLM calls" | Contribution reframed around **hyper-relational KG structure** (Beyond Triplets lineage); call count is irrelevant | User directive + hyper-relational KG literature |
| 4 | Session count "~32 per conv" | **19–32 range**, mean ~27 | Inspection of 10 convs |
| 5 | Evidence strings assumed canonical `D<N>:<M>` | **6 non-canonical entries** + 2 genuine errata (ref to non-existent turns) | [evidence errata memo](../../../.claude/projects/D--github-gr-test/memory/project_locomo_evidence_errata.md) |
| 6 | Phase 3 creates 6 new typed node kinds | **Use a single `qualifier` kind + `tags["qualifier_type"]` discriminator** to minimise upstream churn; SQLite schema already supports this with zero migration | `adapters/sqlite_memory.py` — payload is JSON, node_class is plain TEXT |
| 7 | Ablation A5: "LLM call count control" | **Dropped** — GAAMA and Hyper Triplet already have equal call counts | Finding #2 |
| 8 | Phase 6 contribution: cost efficiency | **Dropped as primary contribution**; keep cost data as transparency |  |

---

## Goal (unchanged)

Compare three systems on LoCoMo-10 under identical retrieval / answer / evaluation
protocol; vary only the **IE output unit**:

```
HippoRAG2    flat (s,p,o) triple KG                    → 69.9% (reported)
GAAMA        (s,p,o) + flat concept labels             → 78.9% (reported)
Hyper Triplet (s,p,o) + typed qualifier pairs (NEW)    → ???%
```

**Prove**: replacing flat concept labels with typed qualifier pairs at extraction
time yields richer graph topology and better retrieval-driven QA accuracy.

---

## Motivation — fact vs memory

Typical RAG pipelines accumulate knowledge as a list of bare facts:

- `a relates-to b`
- `b interacts-with d`

Real sentences, however, carry **process** (how / why it happened) and
**environment** (where / when / with whom / in what mood). Classical
`(subject, predicate, object)` triples preserve only the atomic causality and
drop this surrounding context at extraction time. The result is a quality
gap between a **fact** and a **memory**:

- **Fact-sufficient domains** (definition lookup, factoid QA) → triples suffice.
- **Context-requiring domains** (conversational memory, episodic recall) → the
  gap becomes visible in retrieval quality.

LoCoMo-10's QA categories are a natural instrument to measure exactly this:

| Category | Nature | Predicted Hyper Triplet lift vs GAAMA |
|---|---|---|
| 1 single-hop | fact-sufficient | minimal / noise-level |
| 2 multi-hop | needs connecting context | moderate |
| 3 temporal | needs time qualifiers | **large** |
| 4 open-domain | needs environmental/topical context | **large** |

If the per-category pattern matches this prediction, it is stronger evidence
than any single headline accuracy number — it validates the fact vs memory
distinction mechanistically.

### Research framing — hyper-relational KG

Classical triple KG: facts are `(subject, predicate, object)`.

**Hyper-relational KG** (cf. Beyond Triplets: Hyper-Relational KG Embedding for
Link Prediction): each triple carries a set of **qualifier key–value pairs**:

```
(subject, predicate, object, {(qkey_1, qval_1), (qkey_2, qval_2), ...})
```

Mapping:

| Qualifier key | Value type | Source in episode |
|---|---|---|
| `location` | place name | spatial reference in turn |
| `participants` | person names | speakers / mentioned entities |
| `activity_type` | verb phrase | implied action |
| `time_reference` | date / relative | temporal phrase |
| `mood` | emotion label | sentiment tone |
| `topic` | snake_case label | thematic grouping (= GAAMA's concept) |

GAAMA's single `concept` label is the degenerate case: one qualifier type only.
Hyper Triplet generalises it to a typed qualifier schema and binds all
qualifiers **atomically** at extraction time.

---

## Phase 0 — Environment & data (DONE)

- ✅ Repo scaffold — `hyper-triplet-bench/` layout at repo root
- ✅ Python 3.11 venv via uv; `htb` package installs
- ✅ LoCoMo-10 downloaded + loader with normalisation (`src/htb/data/locomo.py`)
- ✅ Offline eval skeleton (`src/htb/eval/{interfaces,judge,metrics,runner}.py`)
- ✅ GAAMA clone @ `d9987ea` at `external/gaama/`
- ✅ Fork points documented: [`docs/gaama-fork-points.md`](./gaama-fork-points.md)

### Controls (updated)

| Variable | Value | Rationale |
|---|---|---|
| Extract / answer LLM | `gpt-4o-mini` | per plan v1 + verified in GAAMA `config.py` |
| Judge LLM | `gpt-4o` | fair upper-tier judge |
| Embedding | `text-embedding-3-small` | GAAMA default |
| Retrieval budget | 1,000 words | GAAMA default |
| Runs | 10 × 3 systems | mean ± stddev |
| Temperature | 0.0 extract, 0.1 judge | reproducibility |
| Categories | {1, 2, 3, 4}; exclude 5 (adversarial) | all comparison papers |
| **Ingestion caching** | adapter-level cache by (conv_id, chunk_hash) | avoid re-running LLM across the 10 runs |
| **Seeding** | seed OpenAI request IDs where possible; record per-run LLM call hashes | reproducibility audit |

**Cost pre-check**: run 1 conversation end-to-end before full sweep.
If per-conv cost × 10 > budget target, cap `n_runs` or use EasyLocomo-style
subset for A/B tuning.

---

## Phase 1 — HippoRAG2 baseline

Unchanged from v1 except:

- Target 69.9% ± 3%p (per GAAMA paper's HippoRAG row)
- If reproduction fails, Phase 1 **is not blocking** — the primary comparison is
  GAAMA ↔ Hyper Triplet. HippoRAG becomes a lower-bound anchor.
- Consider a thin "classical-triple" control that reuses GAAMA's code with the
  concept prompt disabled; treats it like an in-repo HippoRAG analogue, avoiding
  full HippoRAG reimplementation.

---

## Phase 2 — GAAMA reproduction

Unchanged execution (`external/gaama/evals/locomo/run_*.py`), but:

- **Ship-or-stop gate**: if reproduction falls > 2%p below 78.9%, stop and audit
  before proceeding to Phase 3. Without a trustworthy GAAMA baseline, no claim
  about Hyper Triplet is defensible.
- Record per-chunk LLM token counts and latency for later comparison.

### Phase 2 reference (from deep read)

| Call site | File:Line | LLM calls |
|---|---|---|
| Step 1: Episode ingest | `services/ltm_creator.py:92-152` | 0 |
| Step 2: Fact+Concept | `services/ltm_creator.py:209-409` | **1** (not 2!) |
| Step 3: Reflection | `services/ltm_creator.py:411-518` | 1 |
| Retrieval budget derivation | `services/orchestrator.py:246-312` | 1 per query (optional) |
| Answer generation | `services/answer_from_memory.py` | 1 per query |

---

## Phase 3 — Hyper Triplet (reframed)

### 3.1 Strategy

Fork GAAMA. Replace only **Step 2 extraction + prompt**; reuse everything else.
Total new LOC: ~400 (see fork-points doc).

### 3.2 New prompt — `prompts/node_set_generation.md`

```
For each distinct fact in the new episodes, output:
{
  "fact": {"subject": "...", "predicate": "...", "object": "..."},
  "source_episode_ids": [...],
  "belief": 0..1,
  "qualifiers": {
    "location":        "...",
    "participants":    [...],
    "activity_type":   "...",
    "time_reference":  "...",
    "mood":            "...",
    "topic":           "..."
  }
}
```

Rules mirror GAAMA's fact_generation.md:
- Do not duplicate existing facts.
- Resolve relative dates to absolute using episode timestamp.
- Omit qualifier keys when not identifiable (no hallucination).
- Reuse existing qualifier values when applicable (MERGE semantics).

### 3.3 Storage (zero schema change)

Qualifiers materialise as `MemoryNode(kind="qualifier", concept_label=value,
tags={"qualifier_type": key})`. Edges use new types:

| Edge type | From | To |
|---|---|---|
| `AT_LOCATION` | fact | qualifier(location) |
| `WITH_PARTICIPANT` | fact | qualifier(participant) |
| `ACTIVITY_TYPE` | fact | qualifier(activity_type) |
| `AT_TIME` | fact | qualifier(time_ref) |
| `IN_MOOD` | fact | qualifier(mood) |
| `ABOUT_TOPIC` | fact | qualifier(topic) |

MERGE key: `(qualifier_type, normalised_value)` where `normalised_value =
value.strip().lower()`.

### 3.4 Extractor + Creator

| New symbol | Location |
|---|---|
| `LLMNodeSetExtractor` | append to `systems/hyper_triplet/llm_extractors.py` |
| `HyperTripletLTMCreator` | `systems/hyper_triplet/ltm_creator.py` — subclasses `LTMCreator`, overrides only `_step2_generate_*` |
| SDK wiring | `systems/hyper_triplet/sdk.py` — factory returning a GAAMA `AgenticMemorySDK` with the replacement creator |

### 3.5 Variables held equal with GAAMA

- Retrieval: same node-KNN + PPR + memory-pack trim
- Answer generation: same `answer_from_memory.py`
- Evaluation: identical judge + categories
- LLM call count: 2 per chunk (same as GAAMA)
- Token budget per call: match GAAMA's `MAX_TOKENS_PER_CHUNK` setting

---

## Phase 4 — Evaluation (unchanged protocol)

Identical for all three systems:

```
For run_id in 0..N-1:
  For conv in conversations:
    system.reset()
    system.ingest(conv)                 # cached across runs when possible
    For qa in conv.qa where qa.cat in {1,2,3,4}:
      ctx = system.retrieve(qa.q, budget=1000 words)
      ans = system.answer(qa.q, ctx)
      judgment = llm_judge(qa.q, qa.gold, ans)
      record(conv, qa, judgment, latencies, tokens)
```

Statistical testing: paired bootstrap over QA pairs (not per-run t-test) —
correctly controls for within-QA variance.

---

## Phase 5 — Ablation (redesigned)

### Isolated effects

| Name | Description | Isolates |
|---|---|---|
| **A0: GAAMA baseline** | unchanged | reference |
| **A1: Concept-only Hyper** | Hyper Triplet with only the `topic` qualifier (= GAAMA's concept under a new prompt) | **effect of prompt wording alone** |
| **A2: Typed without atomicity** | extract qualifiers in a second pass after facts | **effect of atomic binding** |
| **A3: Drop environment** | keep only context qualifiers (location/participants/activity/time), no mood/topic | **marginal value of environmental qualifiers** |
| **A4: Drop typing** | collapse all qualifier types into one `context` label set | **effect of typed edge structure** |
| **A5: MERGE off** | disable cross-chunk qualifier node MERGE | **effect of entity reuse in graph** |

**Dropped from v1:** LLM-call-count-control (redundant now).

### Expected pattern (if hypothesis holds)

```
GAAMA ≤ A1 (prompt only)  <  A4 (untyped qualifiers)  <  A2 (typed, non-atomic)
                                                       <  A5 (typed, atomic, no MERGE)
                                                       <  Hyper Triplet (full)
```

If A1 alone already matches or beats GAAMA, the prompt structure is doing the
work and typed edges are decorative — a negative result worth publishing.

---

## Phase 6 — Efficiency analysis (downgraded)

No longer a contribution claim. Collect for transparency:

- Per-chunk LLM tokens (in/out)
- Per-chunk latency
- Nodes / edges created (by kind / edge type)
- Retrieval latency
- Answer generation latency

Expect Hyper Triplet to have slightly higher extraction tokens (richer JSON
output) but same call count.

---

## Phase 7 — Paper connection

Primary claim (v2):

> Classical triples preserve facts but discard the process and environment
> surrounding them. A hyper-relational memory graph that binds typed qualifier
> pairs to facts at extraction time closes the fact→memory gap and
> disproportionately improves retrieval on context-requiring questions.

Lineage: Beyond Triplets (hyper-relational KG embedding) × GAAMA-style agentic
memory.

**Report per-category accuracy FIRST, not overall.** The prediction is a shape
(large lift on cat 3/4, small lift on cat 1), not a single number.

Backup stories if overall delta is marginal (< 2%p) but category pattern holds:

1. "The improvement concentrates exactly where context matters (temporal,
   open-domain) — overall averages understate the effect because ~40% of QA is
   fact-sufficient anyway."
2. Qualitative case studies: show one qualifier-enriched conv graph and walk
   through how temporal / participant qualifiers unlocked specific correct
   retrievals.

If the category pattern is NOT met (lift flat across categories or
concentrated in cat 1), that is a meaningful negative result: the structure is
not doing the work we hypothesised, and the paper becomes a study of what
does work.

---

## Checklist delta from v1

Completed (offline, API-free):

- [x] Phase 0 scaffold + data
- [x] Loader + tests
- [x] Offline eval skeleton
- [x] GAAMA clone + fork-points analysis

Next (still API-free):

- [ ] MockLLM adapter for smoke tests
- [ ] Hyper Triplet prompt draft (offline, hand-review)
- [ ] Hyper Triplet extractor stub (can be unit-tested with MockLLM)
- [ ] `HyperTripletLTMCreator` shell (import-only; no LLM call path exercised)

Blocked on API access:

- [ ] Phase 2 GAAMA reproduction run
- [ ] Phase 1 HippoRAG2 run
- [ ] Full 3-system × 10-run sweep
- [ ] Phase 5 ablations A0–A5
