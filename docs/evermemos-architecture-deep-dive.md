# EverMemOS Architecture Deep Dive

Source: subagent analysis of `external/everos/methods/evermemos/src/` on 2026-04-19.
Purpose: resolve the "unknown" cells in the HINGE compliance audit and define exact switch-points for Phase 5 decomposition ablation.

## 1. Request lifecycle

```
POST /api/v1/memories
  └─ MemoryController.add()                infra_layer/adapters/input/api/memory/memory_controller.py:100-157
     └─ DTO -> MemorizeRequest
        └─ biz_layer/mem_memorize.py:memorize()
           ├─ ConvMemCellExtractor.extract_memcells()          memcell_extractor/conv_memcell_extractor.py
           │     LLM: CONV_BATCH_BOUNDARY_DETECTION_PROMPT (1 call)
           ├─ [parallel] AtomicFactExtractor.extract_atomic_fact()   memory_extractor/atomic_fact_extractor.py:245-275
           │     LLM: ATOMIC_FACT_PROMPT (1 call, 5 retries)
           ├─ [parallel] ForesightExtractor.generate_foresights_for_conversation()   foresight_extractor.py:73-163
           │     LLM: FORESIGHT_GENERATION_PROMPT (1 call, 5 retries)
           ├─ [parallel] EpisodeMemoryExtractor.extract_episode()
           ├─ [optional] AgentCaseExtractor (only for AGENTCONVERSATION type)
           └─ _trigger_clustering()                         biz_layer/mem_memorize.py:110-150
                 ClusterManager.assign_to_cluster(episode_vector)
                 -> MemScene update in v1_mem_scenes
```

**LLM calls per MemCell: 3-4** (boundary + atomic_fact + foresight + episode, plus optional agent_case).

## 2. Data shapes vs Hyper Triplet

| EverMemOS (memory_types.py) | Our Hyper Triplet |
|---|---|
| `MemCell(user_id_list, original_data, timestamp, participants, sender_ids, type, event_id, group_id)` | `Chunk` of `EpisodeRef`s + session metadata |
| `AtomicFact(time, atomic_fact: List[str], fact_embeddings, parent_id)` | `Fact(subject, predicate, object)` + `NodeSet.source_episode_ids` |
| `Foresight(foresight, evidence, start_time, end_time, duration_days, parent_id)` | Partial: `Qualifiers.time_reference` only — **no validity interval** |
| `EpisodicMemory(summary, subject, episode, participants, vector, parent_id)` | Raw episode turns — **no narrative summary** |
| `MemScene.memcell_info / memscene_info` (cluster centroids, member counts) | `HyperTripletGraph` — **no clustering** |
| `ProfileMemory(explicit_info, implicit_traits)` | Not implemented |

**Key gaps in our Hyper Triplet:**
1. **Foresight** — time-bounded validity interval is a separate memory type in EverMemOS; ours compresses to a single `time_reference` string.
2. **Episode summary** — EverMemOS distinguishes the raw MemCell from an LLM-summarised EpisodicMemory; we treat raw turns as `Episode` nodes without summarisation.
3. **MemScene** — thematic clustering absent from our graph; typed qualifiers partially substitute.
4. **ProfileMemory** — user profile evolution absent.

## 3. Storage backend mapping

| Backend | Collections | Purpose |
|---|---|---|
| MongoDB | `v1_memcells`, `v1_atomic_fact_records`, `v1_foresight_records`, `v1_episodic_memories`, `v1_mem_scenes`, `v1_user_profiles` | source of truth |
| Elasticsearch | same collection names | BM25 full-text search |
| Milvus | same collection names | vector KNN |
| Redis | — | session cache |

**Correlation pattern**: linked via `parent_id` document fields (MemCell.event_id ↔ AtomicFact.parent_id ↔ Foresight.parent_id). **No explicit edge / join table** — retrieval is always 2-hop via indexes.

**Per HINGE invariant 6**: qualifiers are stored as document fields, NOT as first-class graph edges. This is a clear HINGE violation at the storage layer.

## 4. Retrieval paths

`agentic_layer/search_mem_service.py` exposes 5 methods:

| Method | Pipeline |
|---|---|
| `keyword` | ES BM25 only |
| `vector` | Milvus KNN only |
| `hybrid` (default) | ES + Milvus in parallel + rerank service |
| `rrf` | ES + Milvus → RRF fusion |
| `agentic` | multi-round LLM-guided: query rewriting → hybrid → sufficiency check → iterate |

**Foresight validity intervals are NOT used as retrieval gates.** Queries can filter on `timestamp: {gte, lte}` as a SearchMemoriesRequest.filters parameter, but Foresight.start_time / end_time are not consulted automatically.

**Reconstructive Recollection is mostly the `agentic` retrieval method** — coarse-to-fine expressed through multi-round query rewriting and sufficiency checks, not through a topic→episode→fact hyperedge walk.

## 5. MERGE semantics — resolved

Earlier audit marked invariant 4 (MERGE on value identity) as "unknown". Deep read resolves it to **❌ violated**:

- `ClusterManager.assign_to_cluster()` does **geometric centroid clustering** on episode vectors — NOT value-identity deduplication on facts.
- Same atomic fact appearing in two sessions produces two separate `AtomicFactRecord` documents with different IDs. No content-hash or semantic-equality dedup.
- Deduplication happens at **retrieval time** (reranker downweights near-duplicates), not at **storage time**.

Our Hyper Triplet's `merge_qualifier()` keyed by `(qualifier_type, normalized_value)` IS a fact-level MERGE, which EverMemOS does not have.

## 6. HINGE invariant compliance — refined

| Invariant | Prior audit | Refined | Evidence |
|---|---|---|---|
| 1. Atomicity of extraction | ⚠ partial | ❌ violated | AtomicFact + Foresight in two separate LLM calls; boundary detection is a third |
| 2. No flat-concept reduction | ✅ | ✅ | typed fields preserved (temporal separate from topical) |
| 3. Qualifier typing is load-bearing | ✅ | ⚠ partial | Foresight typed; location / participants / mood are **not separately extracted** — buried in episode summary text |
| 4. MERGE on value identity | ? | ❌ violated | centroid clustering, not fact-identity dedup |
| 5. Eval exposes correlation | ⚠ | ✅ partial | per-category LoCoMo numbers reported; no per-invariant attribution |
| 6. Storage keeps qualifiers first-class | ❌ | ❌ | parent_id document links, not typed edges |
| 7. Retrieval traverses qualifier edges | ❌ | ❌ | BM25 + vector + RRF + rerank; no qualifier-edge walk |

**Net**: EverMemOS is significantly **less HINGE-compliant at the storage + retrieval layers** than suggested by paper-level reading. It compensates via sophisticated operational engineering (parallel extraction, scene clustering, agentic retrieval) rather than hyper-relational data-model discipline.

## 7. Phase 5 ablation switch-board

Each ablation toggles specific EverMemOS components. Mapping to exact code locations:

### D-gaama (flat concept baseline)
- **Disable** `ForesightExtractor` — remove instantiation in `biz_layer/mem_memorize.py`
- **Simplify** `atomic_fact_prompts.py:ATOMIC_FACT_PROMPT` — drop time field, output plain fact list
- **Disable** scene clustering — `_trigger_clustering()` no-op
- **Expected**: accuracy drop proportional to HINGE invariants 2+3+4 being violated together

### D-mc-noscene (MemCell structure without scene consolidation)
- **Keep** AtomicFact + Foresight extraction
- **Disable** `_trigger_clustering()` in `biz_layer/mem_memorize.py:110-150`
- **Disable** scene-guided expansion in `search_mem_service.py`
- **Expected**: isolates MemScene's contribution; shows what typed qualifiers alone buy

### D-mc+scene (our baseline near-v4 target)
- Default EverMemOS behaviour, **disable only agentic retrieval** (use hybrid instead)

### D-mc+foresight (Foresight gating)
- Default behaviour, **plus** modify query path to filter by Foresight validity interval when query contains temporal terms ("when", "how long", "until")
- Not a simple toggle — requires adding a retrieval pre-filter

### D-recollection (full agentic)
- Default with `retrieve_method=AGENTIC` — multi-round LLM-guided recollection

### D-full (EverMemOS unmodified)
- Native everything

**Knob file**: introduce `biz_layer/memorize_config.py:MemorizeConfig` with boolean flags per component, conditionally skipping in the pipeline. This would be a fork of upstream, not a runtime switch.

## 8. For our Hyper Triplet refactor

Based on the deep read, the highest-value additions to `systems/hyper_triplet/` that bring us closer to EverMemOS-class scoring while staying HINGE-faithful:

1. **Foresight node type** — split `time_reference` qualifier into `(start_time, end_time, duration_days, foresight_text, evidence_text)`. One extra LLM call, but adds a distinct edge type.
2. **Episode summary** — summarise each chunk into one narrative string, store as a separate node. Cheap (reuses extraction prompt) and enables coarse retrieval.
3. **Community detection** — run Leiden on qualifier-nodes graph as a background step, assign community_id to each fact, use as a broad retrieval stage-1 filter. Implements the user-spec's "L3 auxiliary community" without touching EverMemOS.
4. **Validity-interval retrieval gate** — when query starts with "when"/"how long"/etc., prefer facts whose Foresight.valid_from ≤ query_date ≤ valid_until.

These are additions within the HINGE principle (invariant 3: typed qualifiers expand) rather than architectural pivots.
