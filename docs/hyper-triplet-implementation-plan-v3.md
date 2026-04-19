# Hyper Triplet: Implementation & Benchmark Plan — v3

> Replaces [v2](./hyper-triplet-implementation-plan-v2.md) (2026-04-19).
> v3 integrates `docs/hypergraph-memory-lineage.md` findings: HyperMem (2026-04) is LoCoMo SOTA at **92.73%**, and the hyper-relational structure underlying v2's "novel contribution" is an established lineage (HINGE 2020 → HyperGraphRAG 2025 → HyperMem 2026-04). v3 reframes the project as a **decomposition benchmark** across two research lineages.

---

## What changed vs v2

| # | v2 claim | v3 correction |
|---|---|---|
| 1 | Hyper Triplet (typed qualifiers) is a novel structural contribution | **Rescinded.** Typed qualifier pairs = HINGE/StarE/HyperGraphRAG hyperedge construction. Already established since WWW 2020. |
| 2 | The hypothesis "triple vs hyper-relational" is to be empirically tested for the first time | **Already tested.** HyperMem demonstrates +14pp over GAAMA on LoCoMo via 3-level hypergraph. Result: hypergraph structure clearly improves. |
| 3 | Phase 3 is where the original research happens | **Phase 3 becomes an ablation probe** within Lineage A. Full research question shifts to "what decomposes the +14pp gap?" |
| 4 | Comparison is HippoRAG / GAAMA / Hyper Triplet (3 systems) | **5 systems, 2 lineages**: HippoRAG, GAAMA (Lineage B) + HyperGraphRAG, HyperMem, Hyper Triplet (Lineage A variants) |
| 5 | Per-category pattern (cat 3/4 lift) is the primary validation axis | **Retained** but now as ablation within Lineage A rather than "our novelty". The pattern still reveals where hypergraph structure matters most. |

---

## The two-lineage picture

Hyper-relational knowledge has two disjoint research communities that only crossed in 2026-04 via HyperMem:

### Lineage A — Structure-preserving
Preserve n-ary facts via direct hyperedge representation. Qualifier pairs attached to base triple. Metrics: MRR / Hits@10 on JF17K / WD50K.

- **HINGE** (WWW 2020) — Rosso et al. — n-ary fact = base triple + k-v qualifiers
- **StarE** (EMNLP 2020) — message-passing GNN, qualifier count unbounded
- **Hy-Transformer / QUAD / NeuInfer** (2021–2023) — representation learning refinements
- **sHINGE** (TKDE 2024) — schema-aware, reconfirms +29.3% over transformation-based methods
- **HyperGraphRAG** (NeurIPS 2025, github.com/LHRLAB/HyperGraphRAG) — full RAG pipeline, medical/legal/agriculture domains
- **HyperMem** (arxiv 2604.08256, 2026-04) — episodic 3-level hypergraph, **LoCoMo SOTA 92.73%**

### Lineage B — Post-hoc compensation
Keep classical triples; compensate via summarisation / reflection / community detection / PPR. Metrics: F1 / LLM-judge on LoCoMo / LongMemEval / MSC.

- **HippoRAG** (NeurIPS 2024) — OpenIE triple + synonym edge + PPR
- **HippoRAG 2** (ICML 2025) — add passage node + query-to-triple matching
- **Mem0**, **Zep / Graphiti** (2023–2024) — fact extraction + temporal KG
- **GAAMA** (2026-03, github.com/swarna-kpaul/gaama) — 4-node KG + PPR

---

## LoCoMo-10 standings (as of 2026-04)

| System | Lineage | LoCoMo | Notes |
|---|---|---|---|
| **HyperMem** | A+B bridge | **92.73%** | SOTA; 3-level hypergraph |
| MemMachine | B | 84.87% | commercial product |
| **GAAMA** | B | 78.9% | concept + reflection |
| Tuned RAG | baseline | 75.0% | — |
| **HippoRAG** | B | 69.9% | OpenIE triple + PPR |
| Nemori | B | 52.1% | narrative memory |
| A-Mem | B | 47.2% | agentic memory |
| **HyperGraphRAG** | A | not reported on LoCoMo | medical/legal benchmarks |

**Observation**: within Lineage B the ceiling looks like ~85% (MemMachine). HyperMem's +14pp jump over GAAMA attributes the headroom to **structure preservation** rather than further post-hoc compensation.

---

## New research question

Given HyperMem = 92.73%, the benchmark's purpose is no longer "prove typed qualifiers help". It is:

> **Decompose the +14pp gap between GAAMA (78.9%) and HyperMem (92.73%).**
>
> How much comes from (a) typed qualifier pairs, (b) 3-level hierarchy, (c) hypergraph-embedding retrieval propagation, (d) multi-session aggregation?

Answering this requires implementing the intermediate design points cleanly so the contribution of each axis can be read off a comparison table.

---

## System roster (v3)

| Label | Role | Primary effect isolated |
|---|---|---|
| **HippoRAG** | Lineage B baseline | Flat triple floor |
| **GAAMA** | Lineage B top | Post-hoc concept + reflection |
| **HyperGraphRAG** | Lineage A baseline | Hyperedge without episodic memory |
| **Hyper Triplet (ours)** | Lineage A ablation | Typed qualifier hyperedges + episodic memory, NO 3-level hierarchy |
| **HyperMem** | Lineage A SOTA target | Full 3-level hypergraph |

"Ours" now means "HyperGraphRAG-style hyperedges applied to episodic memory graph without HyperMem's 3-level hierarchy" — this is the natural intermediate design point between the two systems and isolates the hierarchy's contribution.

---

## Phase roster (v3)

### Phase 0 (DONE)
- Repo scaffold, uv Python 3.11, LoCoMo loader, offline eval skeleton, MockLLM, fixture-replay adapter, gold regression fixture, GAAMA clone, Hyper Triplet extractor + LTMCreator skeleton.
- Plan v2 and fork-points docs retained; v3 supersedes v2 framing.

### Phase 1 — Lineage B reproduction
- **1A** HippoRAG (target 69.9%) — optional; GAAMA's paper already reports this row under identical protocol.
- **1B** GAAMA (target 78.9% ± 2%p) — mandatory. Our Lineage B anchor.

### Phase 2 — Lineage A baseline
- **2A** HyperGraphRAG — clone `github.com/LHRLAB/HyperGraphRAG`, adapt from medical/legal to LoCoMo input format. Primary deliverable: hyperedge extraction over LoCoMo sessions. **LoCoMo score currently unreported in the upstream paper — our number here is a new measurement.**
- **2B** HyperMem reproduction — check if code is available. If paper-only, implement the 3-level hierarchy from the paper description. Target 92.73% ± 2%p.

### Phase 3 — Hyper Triplet as ablation probe
- Our existing code (`systems/hyper_triplet/`) now lives as a Lineage A variant **without** HyperMem's 3-level hierarchy. Wire it to GAAMA's SqliteMemoryStore + PPR.
- Interpretation: "HyperGraphRAG-style hyperedges + GAAMA-style retrieval, no topic layer".

### Phase 4 — 5-system evaluation
Identical LoCoMo protocol across all systems. Per-category breakdown for every run. Paired bootstrap for significance.

### Phase 5 — Ablation (redesigned for decomposition)
Isolate each axis contributing to the 92.73%:

| Name | Description | Isolates |
|---|---|---|
| **A-struct1** | GAAMA baseline (concept labels, post-hoc) | Lineage B ceiling |
| **A-struct2** | Hyper Triplet (typed qualifiers, no hierarchy) | Qualifier pairs alone |
| **A-struct3** | Hyper Triplet + topic hierarchy | Add 1 hierarchy level |
| **A-struct4** | Hyper Triplet + topic + episode hierarchy = HyperMem | Full 3-level |
| **A-retrieval1** | PPR retrieval | Graph propagation |
| **A-retrieval2** | Hypergraph embedding propagation | Hyperedge-specific retrieval |
| **A-cat** | Per-category report for all 5 systems | Fact vs memory hypothesis |

---

## Per-category hypothesis (retained from v2)

Per `fact_vs_memory_motivation` memory: the hypothesis predicts the structural gap concentrates in context-requiring categories. With five systems the prediction is sharper:

| Cat | Nature | Expected GAAMA→HyperMem shape |
|---|---|---|
| 1 single-hop | fact-sufficient | minimal lift at each step |
| 2 multi-hop | connecting context | moderate lift per structure axis |
| 3 temporal | time qualifiers | **large** lift for typed qualifiers, modest for hierarchy |
| 4 open-domain | environmental / topical | **large** lift for hierarchy, moderate for qualifiers |

If this pattern holds, we decompose the +14pp gap mechanistically. If it doesn't, the gap is explained by some other factor (training prompts, ingestion chunking, retrieval hyperparameters) and the result is still publishable as a negative decomposition.

---

## What to clone / integrate (offline work still possible)

| Item | Source | Action | API needed? |
|---|---|---|---|
| HyperGraphRAG | `github.com/LHRLAB/HyperGraphRAG` | clone to `external/hypergraph_rag/` | no (clone) |
| HyperMem | arxiv 2604.08256; check OpenReview / GitHub | search for code | no |
| HippoRAG 2 | `github.com/OSU-NLP-Group/HippoRAG` | clone to `external/hipporag/` | no |
| HINGE / StarE / sHINGE references | link list in lineage doc | document only | no |

New code location convention:
- `systems/gaama_reproduction/` — thin wrapper around `external/gaama/`
- `systems/hyper_triplet/` — our ablation probe (existing code)
- `systems/hypergraph_rag_reproduction/` — adapter for `external/hypergraph_rag/`
- `systems/hypermem_reproduction/` — new implementation if no code

---

## Paper positioning (v3)

Primary claim candidates:

1. **"Decomposition of hypergraph episodic memory"** — present the 4–5 design points between GAAMA and HyperMem. Report per-axis contribution. Publishable even if numbers are modest.
2. **"First fair cross-lineage comparison"** — lineage doc §3 notes GAAMA didn't cite HyperGraphRAG. Side-by-side evaluation under identical LoCoMo protocol is itself a contribution.
3. **"Community boundary" framing** — paper introduction explains why triple-based (Lineage B) and hyperedge-based (Lineage A) communities didn't overlap until HyperMem.

No "our structure is novel" claim. The novelty is the systematic decomposition / fair comparison.

---

## Open questions before Phase 2 starts

1. Is HyperMem code publicly available? Authors / GitHub link in arxiv 2604.08256 abstract should be verified.
2. HyperGraphRAG's input format vs LoCoMo dialogue turns — will need format adapter. Upstream uses narrative documents.
3. Should we also clone HippoRAG 2 for a stronger Lineage B anchor? Plan v2 treated HippoRAG(1) as optional; v3 may want HippoRAG 2 explicitly.
4. If HyperMem's 3 levels are critical, and HyperGraphRAG is flat, which level ordering does our "Hyper Triplet + partial hierarchy" ablation use?

These are answered in Phase 2A/B execution; v3 documents them as deferred.
