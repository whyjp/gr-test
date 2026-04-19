# HINGE Compliance Audit — 5 Systems

Static analysis of 5 cloned memory/KG systems against the 7 HINGE invariants
from [`hinge-north-star.md`](./hinge-north-star.md) §6.

Method: read-only inspection of extraction code, prompt templates, storage
schema, and retrieval logic. Scored by a subagent on 2026-04-19.

## Consolidated compliance matrix

| System | Inv 1: Atomic extraction | Inv 2: No flat-concept reduction | Inv 3: Typed qualifiers | Inv 4: MERGE on value | Inv 5: Correlation-exposing eval | Inv 6: Qualifiers first-class in storage | Inv 7: Retrieval traverses qualifier edges | Verdict |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **GAAMA** | ⚠ | ❌ | ❌ | ❌ | ⚠ | ⚠ | ❌ | Degenerate HINGE |
| **HyperGraphRAG** | ✅ | ❌ | ❌ | N/A | ❌ | N/A | ❌ | No hyper-relational qualifier structure |
| **HippoRAG 2** | ❌ | ✅ | ❌ | N/A | ❌ | N/A | ❌ | Pure triplets |
| **EverMemOS** | ⚠ | ✅ | ✅ | ? | ⚠ | ❌ | ❌ | Typed qualifiers, but embedded as document metadata |
| **HyperMem** | ⚠ | ✅ | ✅ | ? | ⚠ | ⚠ | ❌ | SOTA-adjacent, hierarchical but qualifiers not first-class edges |
| **Hyper Triplet (ours)** | ✅ | ✅ | ✅ | ✅ | via runner | ✅ | ⚠ | Minimal but HINGE-faithful at data-model level |

Legend: ✅ preserved · ⚠ partial · ❌ violated · N/A no qualifiers in system · ? code path not fully inspected

## Per-system findings (condensed)

### GAAMA — `external/gaama/`
- Fact = single `fact_text` string on a `MemoryNode(kind="fact")` (`core/types.py:107`). Not a triplet.
- Qualifiers = flat `concept_label` nodes with generic `HAS_CONCEPT` / `ABOUT_CONCEPT` edges (`core/types.py:22-25, 122-123`).
- One prompt yields facts + concepts in one JSON (`fact_generation.md`, `services/llm_extractors.py:160-220`), but concepts are logically independent from facts — not a joint encoding.
- No qualifier-value MERGE. SQLite stores nodes by id but no qualifier dedup.
- Retrieval does not walk concept edges with typed semantics.
- **Verdict**: deliberately non-HINGE. Serves as the "degenerate baseline" that v4 uses to measure what HINGE-typing buys.

### HyperGraphRAG — `external/hypergraph_rag/`
- Fact = free-text `knowledge_segment` + completeness score + entity bag (`prompt.py:13-45, operate.py:115-131`). No `(h, r, t)`.
- Entities have `entity_type` (person / object / location / …) but these are **entity types, not qualifier types binding entities to hyperedges**.
- Atomic extraction ✅ — hyperedges + entities in one LLM call.
- No MERGE on qualifier value (entity MERGE by name is unrelated to HINGE qualifiers).
- Storage = JSON KV + NanoVectorDB; no typed edge per qualifier.
- **Verdict**: multi-qualifier spirit without typed correlation. Violates invariants 2, 3, 5, 7.

### HippoRAG 2 — `external/hipporag/`
- Fact = pure `(h, r, t)` triplet (`information_extraction/openie_openai.py:81-100`, `utils/misc_utils.py:22-28`).
- Qualifiers = none. Named entities extracted separately but not bound to triplets as qualifiers.
- **Atomicity violated**: NER and triple extraction are two separate LLM calls.
- Graph stores entities + triples only; no qualifier schema.
- **Verdict**: the "triplet-only representation" HINGE explicitly rejects. Useful as Lineage B floor.

### EverMemOS — `external/everos/methods/evermemos/`
- Fact = `AtomicFact(time, atomic_fact, ...)` (`api_specs/memory_types.py:289-310`). Free-text claim with time qualifier, embedded.
- Qualifiers = typed fields on `MemCell` / `Foresight` / `AtomicFact` (participants, sender_ids, start_time, end_time, duration_days, evidence). Typed ✅ but stored as **document fields**, not graph nodes or edges.
- `AtomicFactExtractor` and `ForesightExtractor` are separate passes — atomicity violated.
- MERGE evidence not visible in extractor code; may live in `memory_manager.py`.
- Retrieval via vector + BM25 hybrid on document text; no typed-edge traversal.
- **Verdict**: close to HINGE at the data-model level (typed qualifiers preserved) but falls off at storage and retrieval — qualifiers are metadata, not navigable structure.

### HyperMem — `external/everos/methods/HyperMem/`
- Fact = `Fact(content, episode_ids, temporal, spatial, keywords, query_patterns, confidence)` (`types.py:66-108`). Free-text claim + typed qualifier fields. Not a triplet.
- Qualifiers = first-class typed fields (temporal, spatial, keywords, query_patterns).
- Two-stage extraction: Pass 1 facts, Pass 2 role assignment (`extractors/fact_extractor.py:18-118`; `prompts/fact_prompts.py:50-71`). Not a single atomic JSON.
- `FactHyperedge` links facts to episodes via roles (`structure.py:108-129`); this is a **fact-to-episode hyperedge**, not a **fact-to-qualifier hyperedge**.
- Storage keeps typed qualifier fields but no separate qualifier node / edge per qualifier type.
- **Verdict**: closest prior system to HINGE, but invariants 6 and 7 still violated — typed qualifiers are payload, not topology.

### Hyper Triplet (ours) — `systems/hyper_triplet/`
- Fact = `Fact(subject, predicate, object)` (`types.py:37-47`). True `(h, r, t)`.
- Qualifiers = `Qualifiers(location, participants, activity_type, time_reference, mood, topic)` with explicit types (`types.py:50-88`).
- `NodeSet(fact, source_episode_ids, belief, qualifiers)` — HINGE's hyper-relational fact by construction (`types.py:91-108`).
- Atomic extraction ✅ — one prompt, one JSON response containing fact + qualifiers coupled (`prompts/node_set_generation.md`, `extractors.py:82-110`).
- MERGE on value identity ✅ — `HyperTripletGraph.merge_qualifier()` keyed by `(qualifier_type, value.strip().lower())` (`graph.py:90-117`).
- Typed edges per qualifier type ✅ — `EDGE_TYPE_BY_QUALIFIER` (AT_LOCATION / WITH_PARTICIPANT / ACTIVITY_TYPE / AT_TIME / IN_MOOD / ABOUT_TOPIC) in `graph.py:22-29`, materialised in `ltm_creator.py:46-81`.
- Retrieval: currently concatenates adjacent qualifier text into searchable document (`retrieval.py`). Graph propagation over typed edges (PPR-style) not yet implemented — this is the single partial point.
- **Verdict**: HINGE-faithful at data-model and storage layers. Retrieval layer is the work-in-progress that needs to traverse qualifier edges to fully satisfy invariant 7.

## Invariant 8 — LLM-as-classifier-only (grouping)

Added 2026-04-19 after [`grouping-node-principle.md`](./grouping-node-principle.md).
Distinguishes encoder-style grouping (reflection, summary of already-extracted facts) from classifier-style grouping (membership assignment to existing structure).

| System | Invariant 8 | Evidence |
|---|:---:|---|
| GAAMA | ❌ violated | `LLMReflectionExtractor` is an encoder over already-extracted facts — produces new compressed text stored as authoritative reflection nodes. The ceiling around 78.9% tracks exactly this violation. |
| HyperGraphRAG | ⚠ partial | Hyperedge `knowledge_segment` is an excerpt copied from source, not newly encoded from facts — closer to extraction than encoding, but not strictly classifier-pure either. |
| HippoRAG 2 | ✅ preserved | OpenIE triples are extraction (one-pass); no reflection / summary layer. |
| EverMemOS | ✅ preserved | `MemScene` clustering is geometric (centroid-based, no LLM encoding); `Foresight` is LLM-extracted from conversation text, not from facts; no reflection. |
| HyperMem | ✅ preserved | Topic / episode detection is LLM-as-classifier (membership yes/no); hypergraph embedding propagation is math, not LLM encoding. |
| Hyper Triplet (ours) | ✅ preserved | Single extraction pass produces node_sets; no reflection layer; community assignment (Phase D) will be classifier-style via Leiden + membership check. |

The ceiling discontinuity between GAAMA (78.9%) and HyperMem/EverMemOS (92.7-93.05%) is the clearest empirical signature of this invariant: encoder-style grouping caps at ~79%, classifier-style grouping reaches ~93%.

## Key takeaways

1. **No existing system fully satisfies all 7 HINGE invariants.** Every prior system trades off at least one.
2. **EverMemOS and HyperMem preserve typed qualifiers at the data-model level** but lose the HINGE correlation at storage (metadata vs first-class edges) and retrieval (vector / BM25 over flattened text vs qualifier graph traversal).
3. **Our Hyper Triplet is, at the code level, the most HINGE-faithful system among the six** — not by effort alone but because the user (영주) independently converged on the HINGE data model when designing the 4-layer approach.
4. **The research contribution is the measurement**: by running all six systems on LoCoMo under the same protocol, we can attribute specific accuracy points to specific HINGE invariants (e.g., "typed qualifiers preserved in storage" = +Npp on temporal questions).
5. **Our single partial ✅ (retrieval invariant 7) is the clearest next engineering target.** Adding qualifier-edge graph propagation (PPR or message-passing) to `retrieval.py` would bring our probe to full HINGE compliance and make the ablation interpretation cleaner.

## Downstream artifacts

- This matrix goes into the paper's Section "System comparison under HINGE invariants", replacing the classical Related Work table of accuracy numbers.
- The per-invariant attribution of accuracy delta is Phase 5 Ablation's core deliverable.
