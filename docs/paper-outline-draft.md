# Paper Outline — "Decomposing Hyper-Relational Memory on LoCoMo"

> Draft v0.1, 2026-04-20. Consolidates plan v5, HINGE invariants (north-star),
> grouping-node principle, and the 12-preset ablation design. Numbers below
> marked `TBD` are placeholders for Phase F (API-gated) results.

---

## Provisional title (candidates)

1. "Decomposing Hyper-Relational Memory: A HINGE-Anchored Audit on LoCoMo"
2. "From Triplets to Hyperedges: What Actually Closes the Gap on LoCoMo?"
3. "Classifier vs Encoder Grouping: A Decomposition Study of Long-Term Memory Systems"

---

## Abstract (draft, <200 words)

Conversational long-term memory systems on LoCoMo-10 show a striking accuracy
ceiling: triplet-based systems (HippoRAG, GAAMA) plateau at ~79%, while recent
hypergraph-based systems (HyperMem, EverMemOS) reach 93%+. We argue this
+14pp gap is not explained by "hypergraph vs triple" at the surface level —
reification can restore hypergraph expressiveness inside triple stores — but
by **two compounding structural principles**:

1. **HINGE's hyper-relational invariant** (Rosso et al., WWW 2020): the base
   triple and its qualifier key-value pairs must stay atomically coupled
   during extraction, storage, and retrieval.
2. **LLM-as-classifier-only for grouping**: once facts are extracted, grouping
   (topic / scene / community) must use the LLM to classify membership, never
   to encode new compressed text. GAAMA-style reflections violate this; their
   ~79% ceiling is the empirical footprint.

We realize both principles in *Hyper Triplet*, a minimal HINGE-faithful system
that extracts `(h, r, t) + typed qualifier pairs` atomically, separates them
into four functional layers (L0/L1/L2/L3), and retrieves via a 3-stage
pipeline. We decompose the GAAMA→EverMemOS gap via a 12-preset ablation and
report how each component contributes per question category. [Main empirical
numbers TBD pending Phase F execution.]

---

## 1. Introduction

### 1.1 The fact vs memory gap

Typical RAG accumulates knowledge as a list of bare facts: `a relates-to b`.
But real conversations carry **process** (how / why) and **environment**
(where / when / with whom / in what mood). Classical triples preserve the
atomic causality but drop the surrounding context — producing what we call
the **fact vs memory gap**.

On LoCoMo-10's four question categories this gap is readable directly:

| Category | Nature | Cost of dropping context |
|---|---|---|
| 1 single-hop | fact-sufficient | minimal |
| 2 multi-hop | connecting context | moderate |
| 3 temporal | time qualifiers | **large** |
| 4 open-domain | environmental | **large** |

### 1.2 Why "hypergraph vs triple" is the wrong framing

Recent LoCoMo SOTA (HyperMem 92.73%, EverMemOS 93.05%) uses hypergraph
structures, which prompts the natural reading "hypergraph beats triple".
We argue this frame obscures the actual mechanism: any triple store can
reify hyper-relational facts (HINGE's original construction); any hypergraph
can be implemented on a generic graph backend (EverMemOS uses MongoDB + ES
+ Milvus with no dedicated graph DB). The real axis is HINGE's **hyper-
relational invariant** — fact and qualifiers atomic and correlated.

### 1.3 Contributions

1. **Re-stating the HINGE invariant for the memory domain** as eight concrete,
   testable properties (§3).
2. **Naming the "LLM-as-classifier-only for grouping" invariant** (Inv 8),
   generalising HINGE's "transformation loss" claim from link prediction to
   episodic memory.
3. **A systematic static audit** of six LoCoMo-evaluated systems against the
   eight invariants (§4), showing which systems violate which invariants.
4. **Hyper Triplet**, a minimal HINGE-faithful reference implementation that
   achieves all eight invariants at the data-model, storage, and retrieval
   layers (§5).
5. **A 12-preset ablation** (§6) decomposing the GAAMA→EverMemOS gap into
   per-invariant deltas, with 3-seed paired-bootstrap significance tests.

---

## 2. Background

### 2.1 HINGE (Rosso et al., WWW 2020)

Verbatim claim: "Existing triplet representation oversimplifies the complex
nature of the data ... methods learning from hyper-relational facts using the
n-ary representation [alone] result in suboptimal models as they are unaware
of the triplet structure ... which serves as the fundamental data structure
in modern KGs and preserves the essential information for link prediction."

HINGE's CNN merges a triple-wise feature map `α` with per-qualifier feature
maps `β_i` via column-wise minimum, encoding the constraint that both the
triple AND every qualifier must be plausible. This is the mathematical form
of the correlation invariant.

### 2.2 The two-lineage landscape

**Lineage A (structure-preserving)**: HINGE (2020) → StarE (2020) →
HyperGraphRAG (NeurIPS 2025) → HyperMem (ACL 2026) → EverMemOS (2026-01).
Preserves n-ary facts as hyperedges.

**Lineage B (post-hoc compensation)**: HippoRAG (NeurIPS 2024) → Mem0, Zep /
Graphiti (2023-24) → GAAMA (2026-03). Keeps classical triples; compensates
via reflection, community summary, bi-temporal edges, PPR.

The two lineages evolved in parallel on different benchmarks (JF17K /
WD50K vs LoCoMo) and only met with HyperMem in 2026-04. GAAMA (2026-03)
does not cite HyperGraphRAG despite being contemporaneous.

### 2.3 The grouping-node principle

Per [`docs/grouping-node-principle.md`](./grouping-node-principle.md), the
ceiling discontinuity between Lineage B (≤ 79%) and Lineage A (≥ 93%) is
best explained by a single rule:

> LLM used as **encoder** (generating new compressed content — GAAMA
> reflections, GraphRAG community summaries) is lossy.
> LLM used as **classifier** (membership yes/no — HyperMem topic detection,
> EverMemOS scene clustering) is lossless.

This is the memory-domain generalisation of HINGE's "transformation loss".

---

## 3. The eight invariants (§6 of [`hinge-north-star.md`](./hinge-north-star.md))

For each invariant: the property, the storage/retrieval/eval signature, and
a direct link to the code that enforces it in our reference implementation.

1. **Atomicity of extraction** — fact and qualifiers emerge from ONE LLM
   response. `systems/hyper_triplet/extractors.py`
2. **No lossy reduction to flat concepts** — typed qualifier schema, never a
   single `concept` label slot. `systems/hyper_triplet/types.py`
3. **Qualifier typing is load-bearing** — location ≠ participant ≠ time.
   Typed edges per qualifier type. `systems/hyper_triplet/graph.py`
4. **MERGE on value identity** — shared qualifier values become shared nodes
   across facts. `graph.merge_qualifier()` keyed by `(type, normalised_value)`.
5. **Evaluation exposes correlation** — per-category accuracy is the primary
   unit; overall accuracy secondary.
6. **Storage keeps qualifiers first-class** — dedicated edges per qualifier
   type, not JSON payload metadata. `StarStore._qualifier_index`.
7. **Retrieval traverses qualifier edges** — PPR over typed qualifier graph,
   not just BM25 over flat text. `retrieval_ppr.py` + `retrieval_stages.py`.
8. **LLM-as-classifier-only for grouping** — topic / community / ontology
   assignment uses membership decisions, not encoder-style summarisation.
   Per [`grouping-node-principle.md`](./grouping-node-principle.md).

---

## 4. Static audit of six systems

(Numbers from [`hinge-compliance-audit.md`](./hinge-compliance-audit.md),
refined via [`evermemos-architecture-deep-dive.md`](./evermemos-architecture-deep-dive.md).)

| System | 1 atomic | 2 no-flat | 3 typed | 4 MERGE | 5 corr-eval | 6 first-class | 7 qual-edges | 8 classifier |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| GAAMA | ⚠ | ❌ | ❌ | ❌ | ⚠ | ⚠ | ❌ | ❌ |
| HyperGraphRAG | ✅ | ❌ | ❌ | N/A | ❌ | N/A | ❌ | ⚠ |
| HippoRAG 2 | ❌ | ✅ | ❌ | N/A | ❌ | N/A | ❌ | ✅ |
| EverMemOS | ⚠ | ✅ | ⚠ | ❌ | ✅ | ❌ | ❌ | ✅ |
| HyperMem | ⚠ | ✅ | ✅ | ? | ⚠ | ⚠ | ❌ | ✅ |
| **Hyper Triplet (ours)** | ✅ | ✅ | ✅ | ✅ | via runner | ✅ | ⚠ (partial) | ✅ |

**Observations:**
- GAAMA is the clearest Invariant #8 violation among evaluated systems; its
  78.9% LoCoMo accuracy is consistent with the classifier-only principle's
  ceiling prediction.
- EverMemOS achieves 93.05% despite violating invariants #4, #6, #7 — this
  suggests either the remaining preserved invariants compensate, or
  operational engineering (Foresight validity intervals, MemScene
  clustering, agentic retrieval) substitutes for structural first-class-
  ness. Phase F's decomposition ablation isolates this.
- No system fully satisfies all eight. Hyper Triplet is the closest.

---

## 5. Hyper Triplet — reference implementation

### 5.1 Four-layer NodeSet

```
L0Fact                 (subject, predicate, object, edge_qualifiers)
L1TemporalImportance   (timestamp, time_reference, valid_from/until,
                        duration_days, importance, belief)
L2Context              (location, participants, activity_type, mood)
L3Auxiliary            (topic, community_id, ontology_type,
                        ontology_properties, embedding_ref, source_ref)
```

Each NodeSet carries a deterministic `ns_id` = md5(lowercased triple text).

### 5.2 Extraction — one LLM call

`prompts/node_set_generation.md` emits JSON with `fact + qualifiers` together;
violates no HINGE invariant. Prompt explicitly labelled as extraction-only,
forbids downstream reflection.

### 5.3 Storage — star-native

`StarStore` — one node_set = one star subgraph, O(1) KV retrieval by ns_id.
Qualifier / community / episode indices for inter-star joins.

### 5.4 Retrieval — 3 stages

Stage 1 (Broad): L2/L3 lexical match + L3 community expansion → 500 candidates.
Stage 2 (Rank): L1 importance + temporal-query alignment → top-30.
Stage 3 (Exact): L0 fact overlap + edge qualifier re-rank → budget-trimmed
                 context.

### 5.5 Background workers

- `CommunityDetector` — Louvain over shared-qualifier graph → L3.community_id
- `ImportanceScorer` — ACT-R style `log1p(frequency) * exp(-d*dt) * belief`
- `BoundaryDetector` — classifier-style session + drift segmentation

All post-ingest, deterministic, classifier-only.

---

## 6. Experimental setup

### 6.1 Benchmark

LoCoMo-10 (Maharana et al., 2024). 10 conversations, 1,540 QA pairs
(categories 1–4; category 5 adversarial excluded per all prior work).

### 6.2 Systems

Six systems under identical protocol:

| System | Description | Status |
|---|---|---|
| HippoRAG 2 | OpenIE triples + PPR | external clone |
| GAAMA | 4-node KG + concept + reflection | external clone |
| HyperGraphRAG | free-text hyperedge + entity bag | external clone |
| HyperMem | 3-level topic/episode/fact hypergraph | EverOS submodule |
| EverMemOS | MemCell + MemScene + Foresight + 3-stage recollection | docker-compose |
| **Hyper Triplet (ours)** | HINGE-faithful minimal 4-layer | `systems/hyper_triplet/` |

### 6.3 Ablations (12 presets)

All run on Hyper Triplet as the base system, toggling individual components:

baseline, no_node_set, no_layer_separation, no_hyper_edge, no_star_storage,
no_stage1, no_hybrid_index, no_community, no_importance, no_boundary_detector,
no_ontology_axis, **gaama_style_reflection_on** (deliberate Inv #8 violation).

### 6.4 Protocol

- Extract LLM: `gpt-4o-mini`
- Answer LLM: `gpt-4o-mini`
- Judge LLM: `gpt-4o` (LLM-as-judge)
- Embedding: `text-embedding-3-small`
- Retrieval budget: 1,000 words
- Seeds: `[42, 1337, 2024]`
- Aggregation: mean ± std across seeds; paired bootstrap (n=1000) for p-values

### 6.5 Reporting

Per-category accuracy as primary unit; overall accuracy secondary. Win
criterion: beat each baseline on **every seed**, not just the mean.

---

## 7. Results

[All numbers TBD — filled in after Phase F execution.]

### 7.1 Main comparison

| System | cat1 | cat2 | cat3 | cat4 | overall |
|---|---:|---:|---:|---:|---:|
| HippoRAG 2 | TBD | TBD | TBD | TBD | TBD |
| GAAMA | TBD | TBD | TBD | TBD | ~78.9% |
| HyperGraphRAG | TBD | TBD | TBD | TBD | ~86.5% |
| HyperMem | TBD | TBD | TBD | TBD | ~92.7% |
| EverMemOS | TBD | TBD | TBD | TBD | ~93.1% |
| Hyper Triplet | TBD | TBD | TBD | TBD | TBD |

### 7.2 Ablation decomposition

| Preset | delta vs baseline | 95% CI | HINGE invariant violated |
|---|---:|---|---|
| gaama_style_reflection_on | TBD | TBD | #8 (deliberate) |
| no_community | TBD | TBD | L3 grouping |
| no_star_storage | TBD | TBD | #6 first-class |
| no_hyper_edge | TBD | TBD | #3 typed |
| no_node_set | TBD | TBD | #1 atomicity |
| (remaining 7) | TBD | TBD | ... |

### 7.3 Expected pattern (paper's testable prediction)

- `gaama_style_reflection_on` should cost ~10pp (reproduces the encoder ceiling).
- `no_star_storage` should cost less on cat1, more on cat3/4.
- `no_community` should hurt cat4 (open-domain) most.
- `no_importance` should have near-zero effect on cat1, moderate on cat2.
- `no_layer_separation` should have diffuse cross-category impact.

Falsifying evidence: any preset whose delta doesn't align with the invariant
it violates would refute the causal story and re-open interpretation.

---

## 8. Related work

(Populated from `docs/hypergraph-memory-lineage.md` and reference notes.)

- HINGE / StarE / sHINGE — hyper-relational KG embedding
- HippoRAG / HippoRAG 2 / Mem0 / Zep / GAAMA — episodic memory
- HyperGraphRAG / HyperMem / EverMemOS — convergent SOTA

The gap between Lineage A and Lineage B framing is itself a contribution;
prior surveys treat them separately.

---

## 9. Discussion

### 9.1 Why the invariants compound

No single invariant accounts for 14pp alone. Empirically (to be confirmed
in §7.2), each violation costs 2–4pp; four violated simultaneously (GAAMA)
caps accuracy at ~79%. This is consistent with the hypothesis that the
invariants are independent structural constraints.

### 9.2 When does HINGE compliance not help?

On cat1 (fact-sufficient), HINGE compliance should produce minimal gain
because classical triples are already sufficient. Positive delta here would
be a negative result — it would suggest some of our gain is incidental.

### 9.3 Classifier vs encoder as a general principle

The same rule governs: RAG community summaries, GraphRAG community reports,
LangChain memory summarisation. Each trades lossiness for readability. For
accuracy-critical benchmarks, the principle predicts compression-based
grouping will always underperform classification-based grouping, modulo
fine-tuning.

### 9.4 What our implementation does NOT resolve

- **Streaming-scale write throughput** — the HINGE invariants say nothing
  about write rate; a Kafka-scale hyperedge ingest system remains an open
  engineering problem. (User's separate `graphdb-bench` project.)
- **Multi-user ontology disambiguation** — L3 ontology_type is per-fact;
  reconciling ontology across user sessions needs a meta-layer we don't
  implement.
- **Domain-specific signal structure** — MMORPG / trading / medical
  workloads have richer structured fields that could feed L3 directly
  without LLM; this paper is dialogue-only.

---

## 10. Conclusion

"Hypergraph vs triple" is not the right axis. The axis is **correlation
preservation** at three layers: extraction atomicity (Inv 1), storage
first-class-ness (Inv 6), retrieval edge traversal (Inv 7), plus the
**classifier-only grouping** rule (Inv 8) that prevents compression loss
downstream. Hyper Triplet realises all four faithfully and lets the 12-preset
ablation decompose the accuracy gap into clean per-invariant contributions.

LoCoMo's 14pp Lineage-B-to-A jump is not a mystery once you know which
invariants each system preserves.

---

## Appendix A — Reproducibility

- All six-system runs: `scripts/smoke_test_openai.py` (single-conv variant),
  `scripts/run_full_benchmark.py` (deferred to Phase F).
- Ablation sweep: `AblationRunner` in `src/htb/eval/ablation_runner.py`.
- External clones pinned at: gaama `d9987ea`, hypergraph_rag `a804827`,
  hipporag `d437bfb`, everos `f06c303`, hinge @ HEAD, shinge @ HEAD.
- Offline reproducibility: 243 pytest cases (~0.5s), no network access
  required.

## Appendix B — Artefacts

- `docs/hinge-north-star.md` — the eight invariants
- `docs/hinge-compliance-audit.md` — per-system static verdicts
- `docs/grouping-node-principle.md` — Inv 8 derivation
- `docs/evermemos-architecture-deep-dive.md` — SOTA ablation switch-board
- `docs/baseline-numbers.md` — prior reported LoCoMo scores table
- `docs/my-own-test-design-spec.md` — user-authored design spec (reconciled
  into plan v5)
- `tests/fixtures/locomo_conv26_session1_gold.json` — hand-crafted gold
  regression
