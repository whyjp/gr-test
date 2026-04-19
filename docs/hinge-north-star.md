# HINGE as North Star

> **Project-wide guiding principle (2026-04-19, user directive):**
> HINGE (Rosso, Yang, Cudré-Mauroux — *Beyond Triplets: Hyper-Relational Knowledge Graph Embedding for Link Prediction*, WWW 2020) is not merely a cited ancestor — it is the **ideal data model this project aspires to realize** at every layer. Every architectural decision, extraction prompt, storage schema, retrieval mechanism, and evaluation protocol must be judged against HINGE's hyper-relational invariant.

## 1. The HINGE paper — verbatim abstract

> Knowledge Graph (KG) embeddings are a powerful tool for predicting missing links in KGs. Existing techniques typically represent a KG as a set of triplets, where each triplet (h, r, t) links two entities h and t through a relation r, and learn entity/relation embeddings from such triplets while preserving such a structure. However, this triplet representation **oversimplifies the complex nature of the data stored in the KG**, in particular for hyper-relational facts, where each fact contains not only a base triplet (h, r, t), but also the associated key-value pairs (k, v). Even though a few recent techniques tried to learn from such data by transforming a hyper-relational fact into an n-ary representation (i.e., a set of key-value pairs only without triplets), **they result in suboptimal models as they are unaware of the triplet structure, which serves as the fundamental data structure in modern KGs and preserves the essential information for link prediction**. To address this issue, we propose **HINGE**, a hyper-relational KG embedding model, which directly learns from hyper-relational facts in a KG. HINGE captures not only the primary structural information of the KG encoded in the triplets, **but also the correlation between each triplet and its associated key-value pairs**. Our extensive evaluation shows the superiority of HINGE on various link prediction tasks over KGs. In particular, HINGE consistently outperforms not only the KG embedding methods learning from triplets only (by 0.81–41.45%), but also the methods learning from hyper-relational facts using the n-ary representation (by 13.2–84.1%).

Source: [ACM WWW 2020 DOI](https://dl.acm.org/doi/10.1145/3366423.3380257); verbatim abstract retrieved via Semantic Scholar API 2026-04-19.

## 2. The HINGE ideal in three sentences

1. **A fact is NOT a triplet.** It is a base triplet `(h, r, t)` **plus** a set of qualifier key–value pairs `{(k_i, v_i)}` that contextualize the triplet.
2. **Neither part may be dropped.** Triplet-only representations oversimplify; n-ary / k-v-only representations discard the fundamental structure. Both are suboptimal.
3. **The correlation between the triplet and its qualifiers is the learning signal.** Encoding each part independently is insufficient — the joint structure is what distinguishes hyper-relational KG learning.

## 3. The canonical HINGE data model

```
hyper-relational fact
    = (h, r, t)                          ← base triplet (fundamental)
    + { (k_1, v_1), (k_2, v_2), …, (k_n, v_n) }   ← qualifier k-v pairs
                                         ← correlation(triplet, qualifiers) preserved
```

This is the target shape for **every** unit of memory this project emits, stores, retrieves, and evaluates.

## 4. Why HINGE is the right north star (and not EverMemOS / HyperMem / GAAMA)

| Aspect | HINGE (north star) | EverMemOS / HyperMem (current SOTA engineering) | GAAMA / HippoRAG (Lineage B) |
|---|---|---|---|
| Unit of memory | hyper-relational fact (triplet + qualifiers) | MemCell (atomic fact + Foresight + episodic trace) | (h, r, t) triple + optional concept label |
| Qualifier treatment | **first-class correlation** | bundled + typed, but structure largely implicit | flat concept labels, post-hoc |
| Why this | **data-model principle** — should be invariant across systems | engineered system — implements the principle with operational choices | older compromise — needs reframing |

HINGE is a **principle**; EverMemOS is an **instantiation** of the principle at SOTA scale; GAAMA is a **degenerate** instantiation (qualifiers collapsed to single concept labels).

The project is not trying to reinvent HINGE — it is trying to **preserve HINGE's invariant** while benchmarking how different systems trade it off.

## 5. Alignment audit — how the current codebase already matches HINGE

| HINGE primitive | Current code | File |
|---|---|---|
| base triplet `(h, r, t)` | `Fact(subject, predicate, object)` | `systems/hyper_triplet/types.py:37-47` |
| qualifier k-v pair | `Qualifiers` with typed keys (location / participants / activity_type / time_reference / mood / topic) | `systems/hyper_triplet/types.py:50-88` |
| hyper-relational fact | `NodeSet(fact, source_episode_ids, belief, qualifiers)` | `systems/hyper_triplet/types.py:91-108` |
| "triplet + qualifiers in ONE extraction call" (atomic correlation) | `LLMNodeSetExtractor.extract_node_sets()` — single prompt, single JSON response, never emit qualifiers without fact | `systems/hyper_triplet/extractors.py:82-110` |
| qualifier MERGE (preserve correlation across chunks) | `HyperTripletGraph.merge_qualifier()` keyed on `(qualifier_type, normalized_value)` | `systems/hyper_triplet/graph.py:90-117` |
| typed edge per qualifier (correlation exposed to retrieval) | `EDGE_TYPE_BY_QUALIFIER` map, materialised in `HyperTripletLTMCreator._materialise_node_sets` | `systems/hyper_triplet/graph.py:22-29` + `ltm_creator.py:46-81` |

**Verdict**: the codebase is HINGE-aligned by construction. This is not accidental — it is the outcome of treating `(fact, qualifiers)` as the atomic unit from the extraction prompt down.

## 6. Design principles derived from HINGE (project invariants)

These are **hard rules** for any future design decision. Violations should trigger a review.

1. **Atomicity of extraction.** The LLM must emit a fact and its qualifiers **in the same response**. Never two calls. Never qualifiers inferred after the fact from a second pass. Our `node_set_generation.md` prompt preserves this; do not split it.
2. **No lossy reduction to flat concepts.** A qualifier schema with a single `concept` label type (GAAMA-style) violates HINGE's correlation — it collapses N qualifier types into one. Any ablation that reduces qualifier richness must be labelled as a degenerate HINGE variant, not a simplification.
3. **Qualifier typing is load-bearing.** `location` ≠ `participant` ≠ `time_reference`. Collapsing to an untyped set of strings loses the correlation structure HINGE argues for. Typed edges in the graph exist to carry this.
4. **MERGE on value identity preserves the correlation graph.** The same location across facts MUST become the same qualifier node — otherwise facts are not linked through their shared context, and the graph degenerates to independent hyper-relational facts with no topology.
5. **Evaluation must expose the correlation story.** Per-category accuracy decomposition is the unit of paper-level findings, not a single overall number. Categories that need context (temporal / multi-hop / open-domain) are where HINGE's advantage over flat triples should appear.
6. **Storage is not allowed to drop qualifiers for efficiency.** If the storage layer stores only `(h, r, t)` with qualifiers as document metadata, retrieval over qualifiers silently degrades. Storage schemas must have first-class edges / links for each qualifier type. Our in-memory graph does this; GAAMA's SQLite + concept edges is a borderline case we must audit when wiring Phase 3.
7. **Retrieval must traverse qualifier edges, not just fact BM25.** A query "who was in Paris" should surface the participant qualifier network, not just lexical overlap on fact text. Our `retrieval.py` already merges adjacent qualifier content into the searchable document — this is the minimum; graph propagation (PPR-over-qualifier-edges) is the ideal.
8. **LLM-as-classifier only for grouping.** Derived from [`grouping-node-principle.md`](./grouping-node-principle.md) (2026-04-19). Extraction is allowed to use the LLM as an encoder (one-shot, atomic, per-fact); **grouping** (topic / scope / community / ontology assignment) MUST use the LLM as a classifier — membership/yes-no decisions — never as an encoder producing new compressed text (reflection, summary, community description). Encoder-style grouping accumulates compression loss and caps accuracy; classifier-style grouping is lossless because original facts stay verbatim. This is the generalisation of HINGE's "transformation loss" claim to the memory domain. **Violation signature**: an LLM call whose output is stored as authoritative memory AND whose input was a set of already-extracted facts.

## 7. What this means for each Phase in plan v4

### Phase 1 (GAAMA reproduction)
GAAMA operates below HINGE's ideal — single concept-label qualifier, post-hoc binding. That's the point of Phase 1: establish the **non-HINGE** baseline. Report what accuracy costs the HINGE violation.

### Phase 2A (HyperGraphRAG reproduction)
HyperGraphRAG hyperedges are free-text knowledge segments + entity bag — they have the **multi-qualifier spirit** but dropped the typed correlation. Measure whether typed qualifiers (our Hyper Triplet) beat HyperGraphRAG on the categories where correlation matters (multi-hop, temporal).

### Phase 2B (EverMemOS)
EverMemOS is closer to HINGE at the MemCell layer (atomic fact + Foresight + episodic trace = base triplet + typed qualifiers). Phase 2B confirms that the SOTA system IS implicitly HINGE-aligned at the data-model level. Our role: isolate **which correlation structure** contributes how much.

### Phase 3 (Hyper Triplet probe)
Our Hyper Triplet is **deliberately HINGE-minimal**: typed qualifiers, atomic extraction, MERGE, typed edges, and nothing else. No topic hierarchy, no reflections, no Foresight beyond time_reference. Phase 3 measures: how much of EverMemOS's 93% comes purely from HINGE-aligned extraction + retrieval?

### Phase 5 (Ablation)
All ablations are labeled against the HINGE invariant:
- **D-gaama** (no typing) = deliberately violate invariant #3
- **D-mc-noscene** (our Hyper Triplet, no topic grouping) = pure HINGE
- **D-mc+scene** = HINGE + one hierarchy level
- **D-full** (EverMemOS) = HINGE + full EverMemOS engineering
Reading the ablation table tells us where HINGE alone suffices vs where engineering matters.

## 8. Writing protocol — how to cite HINGE in the paper

- Introduction: HINGE is the **motivation** for rejecting flat triples. Cite it first in the related work.
- Method: our `NodeSet` is HINGE's hyper-relational fact. Say so explicitly; do not invent parallel terminology.
- Ablation: label each row against the HINGE invariants it preserves or violates.
- Discussion: HINGE set the data-model principle; this paper measures how far we get on LoCoMo with the principle alone, before operational engineering layers are added.

## 9. What we do NOT claim

- Not "we invented hyper-relational memory" — HINGE did, in 2020.
- Not "typed qualifiers are novel" — StarE (EMNLP 2020) already generalized qualifier counts; HINGE's schema-aware successor sHINGE (TKDE 2024) further formalized it.
- Not "our system beats SOTA" — EverMemOS is SOTA.
- We claim: systematic **measurement** of how faithfully different systems realize the HINGE ideal, and per-category decomposition of where that faithfulness pays off.

## 10. References

- **HINGE** (original): Rosso, Yang, Cudré-Mauroux. *Beyond Triplets: Hyper-Relational Knowledge Graph Embedding for Link Prediction*. WWW 2020. [ACM](https://dl.acm.org/doi/10.1145/3366423.3380257)
- **StarE**: Galkin et al. EMNLP 2020. [arxiv 2009.10847](https://arxiv.org/abs/2009.10847)
- **sHINGE** (schema-aware successor): Rosso, Yang, Cudré-Mauroux. TKDE 2024. [exascale.info](https://exascale.info/assets/pdf/KnowledgeGraphEmbeddings_TKDE2024.pdf)
- **HyperGraphRAG**: Luo et al. NeurIPS 2025. [arxiv 2503.21322](https://arxiv.org/abs/2503.21322)
- **HyperMem**: Yue et al. ACL 2026 (submitted 2026-04). [arxiv 2604.08256](https://arxiv.org/abs/2604.08256)
- **EverMemOS**: 2026-01. [arxiv 2601.02163](https://arxiv.org/abs/2601.02163)
