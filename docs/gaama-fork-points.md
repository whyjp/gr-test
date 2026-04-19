# GAAMA → Hyper Triplet Fork Points

Line-referenced map of what Phase 3 **replaces**, **extends**, and **reuses**
in the upstream GAAMA codebase at `external/gaama/` @ HEAD `d9987ea`.

---

## Architecture at a glance

```
AgenticMemoryOrchestrator (services/orchestrator.py)
  └── ingest(trace_events) → buffer
      └── create(options) → chunks → LTMCreator.create_from_events(chunk)
          ├── Step 1: Episode nodes (NO LLM)       ltm_creator.py:92-152
          ├── Step 2: Fact+Concept  (1 LLM call)   ltm_creator.py:154-170, 209-409
          │    └── LLMFactExtractor.extract_facts()  llm_extractors.py:160-237
          │       └── prompt: prompts/fact_generation.md
          └── Step 3: Reflections   (1 LLM call)   ltm_creator.py:172-183, 411-518
               └── LLMReflectionExtractor           llm_extractors.py:240-end
                  └── prompt: prompts/reflection_generation.md
```

**Total per chunk: 2 LLM extraction calls** (not 3 as the original plan claims).

Storage: **single SQLite file** with 2 tables (`nodes`, `edges`) + 2 FTS5 virtuals
(`node_fts`, `edge_fts`). Node payload is JSON; `node_class` and `edge_type` are
strings, so new kinds/types **require no schema migration**.

---

## Core types — `external/gaama/core/types.py`

| Symbol | Line | Role in Hyper Triplet |
|---|---|---|
| `MemoryNode` dataclass | 72-131 | REUSE. Has generic `tags: Dict[str,str]` (L83) and optional kind-specific fields. Hyper Triplet typed qualifiers can ride on `concept_label` + `tags["qualifier_type"]` without adding fields. |
| `ALLOWED_NODE_KINDS` | 20 | EXTEND or reuse "concept". Options: (a) add 6 new kinds (`location`, `participant`, `activity_type`, `time_ref`, `mood`, `topic`); (b) keep one kind (`qualifier`) with `tags["qualifier_type"]` discriminator. Option (b) minimises churn. |
| `ALLOWED_EDGE_TYPES` | 22-25 | EXTEND. Current: `{NEXT, DERIVED_FROM, DERIVED_FROM_FACT, HAS_CONCEPT, ABOUT_CONCEPT}`. Add typed edges: `AT_LOCATION, WITH_PARTICIPANT, ACTIVITY_TYPE, AT_TIME, IN_MOOD, ABOUT_TOPIC` (or reuse `ABOUT_CONCEPT` with `Edge.label` as qualifier name). |
| `RetrievalBudget` | 163-174 | REUSE. Slots are `max_facts / max_reflections / max_skills / max_episodes`. **Qualifier nodes do NOT need a new slot** — they behave like concepts (graph anchors only, not pulled into the MemoryPack). |
| `Edge` | 138-145 | REUSE. `edge_type: str` + `label: str` = enough room for typed qualifiers. |
| `MemoryPack` | 182-326 | REUSE. Categories facts/reflections/skills/episodes already cover retrieval output; qualifier values surface via fact context strings. |

---

## What Phase 3 REPLACES

### 3.1 Prompt — `prompts/fact_generation.md` → `prompts/node_set_generation.md`

Current (fact_generation.md):
- Part 1 Facts: `{fact_text, belief, source_episode_ids, concepts[]}`
- Part 2 Concepts: `{concept_label, episode_ids[]}` — flat snake_case labels

Replace with node-set prompt producing hyper-relational structure per fact:

```json
{
  "fact": {"subject": "...", "predicate": "...", "object": "..."},
  "source_episode_ids": ["..."],
  "belief": 0.9,
  "qualifiers": {
    "location":       "...",
    "participants":   ["..."],
    "activity_type":  "...",
    "time_reference": "...",
    "mood":           "...",
    "topic":          "..."
  }
}
```

### 3.2 Extractor — `services/llm_extractors.py`

| Current class | Lines | Replacement |
|---|---|---|
| `LLMFactExtractor.extract_facts()` | 160-220 | `LLMNodeSetExtractor.extract_node_sets()` — same budget-truncate logic (`_budget_truncate` at L108-158 is reusable), new prompt path, new return shape `list[NodeSet]`. |
| `_parse_response()` | 222-237 | Adapt expected_keys to `["node_sets"]`. |

Keep: `_strip_json_block` (L22), `_retry_llm_for_json` (L32), `LLMReflectionExtractor` (L240+) — reused verbatim.

### 3.3 LTMCreator Step 2 — `services/ltm_creator.py:209-409`

Replace `_step2_generate_facts_and_concepts` with `_step2_generate_node_sets`:

| Keep (L235-297) | Change (L299-404) |
|---|---|
| Context retrieval (related older episodes, existing facts, existing concepts via vector search) | Call new `LLMNodeSetExtractor` instead |
| Lines 300-303 use `LLMFactExtractor.extract_facts(episode_nodes, related_older_episodes, existing_facts, existing_concepts)` | Replace with node-set extractor; existing_concepts parameter becomes `existing_qualifiers` (grouped by type) |
| | Replace concept-node building (L311-352) with **typed qualifier MERGE**: for each non-null qualifier value, `find_or_create` a node keyed by `(qualifier_type, value.lower())` and add typed edge |
| | Fact construction (L358-391) stays but `ABOUT_CONCEPT` edges (L393-401) replaced with typed edges per qualifier |

Step 1 (episode creation, L92-152) and Step 3 (reflections, L411-518) are untouched.

### 3.4 Orchestrator — `services/orchestrator.py`

**No replacement needed.** `AgenticMemoryOrchestrator` is a thin routing layer that
delegates to `LTMCreator` via composition (L93-100). Simply inject a
`HyperTripletLTMCreator` subclass (or replace the `self._ltm_creator` field).

Exception: the `_derive_budget_from_llm` path (L246-312) may want a new prompt
that mentions qualifier categories — but for fair comparison keep this identical
to GAAMA.

---

## What Phase 3 EXTENDS (no replace, just adds)

| File | Lines | Addition |
|---|---|---|
| `core/types.py` | 20 | Add `"qualifier"` to `ALLOWED_NODE_KINDS` (single-kind design) OR 6 new kinds |
| `core/types.py` | 22-25 | Add 6 edge types: `AT_LOCATION, WITH_PARTICIPANT, ACTIVITY_TYPE, AT_TIME, IN_MOOD, ABOUT_TOPIC` |
| `core/types.py` | optional | If strict typing wanted, add `qualifier_type: str = ""` and `qualifier_value: str = ""` to `MemoryNode`. Otherwise reuse `concept_label` + `tags`. |
| `prompts/` | new file | `node_set_generation.md` |
| `services/llm_extractors.py` | append | `class LLMNodeSetExtractor` |
| `services/ltm_creator.py` | override | Subclass `LTMCreator` as `HyperTripletLTMCreator`; override only `_step2_*` |

---

## What Phase 3 REUSES WITHOUT CHANGE

| Path | Reason |
|---|---|
| `adapters/sqlite_memory.py` (619 lines) | Schema is `node_class TEXT` + `payload TEXT (JSON)`; arbitrary kinds and edge types work. FTS5 indexes `node_to_embed_text()` which reads any node content. |
| `adapters/sqlite_vector.py` | Vector store is kind-agnostic. |
| `adapters/openai_llm.py`, `openai_embedding.py` | LLM/embedding adapters. |
| `services/ltm_retriever.py` | Node KNN → PPR → bucket by {fact, reflection, skill, episode}. Qualifiers serve as graph anchors for PPR but don't enter the output pack — matches how concepts already work. |
| `services/pagerank.py` (161 lines) | Hub-dampened PPR is edge-type-agnostic; optional per-edge-type weights (`edge_type_weights`) let us tune qualifier influence if needed. |
| `services/answer_from_memory.py` | Pure prompt-based answer generation from MemoryPack. |
| `services/graph_edges.py` | `make_edge(source, target, edge_type)` works for any edge_type string. |
| `services/orchestrator.py` | Composition-based; inject replacement `ltm_creator`. |
| `evals/locomo/*.py` (4 run scripts + locomo_eval.py) | Benchmark drivers are system-agnostic — they call through the `AgenticMemorySDK`. |
| `infra/prompt_loader.py` | Loads any prompt by name from `prompts/` dir. |

---

## Retrieval-time budget implication

`RetrievalBudget` has **four kind slots**: facts, reflections, skills, episodes.
`ltm_retriever.py:27-32` (`NODE_KIND_BUDGET_CONFIG`) does not include concept.
That means:

- Concept nodes are never directly pulled into the MemoryPack.
- They participate in PPR traversal but their text is not surfaced.
- The memory pack text comes from facts + reflections + episodes; qualifiers
  enrich the graph topology that PPR explores.

→ **Hyper Triplet gets the same deal for free.** Typed qualifiers increase graph
density (facts → qualifiers ← other facts about same location/participant/etc),
which boosts PPR signal without changing retrieval output shape.

---

## Minimal delta estimate

| Change type | Files | ~LOC |
|---|---|---|
| New prompt | 1 | ~60 |
| New extractor class | 1 (append to existing) | ~120 |
| New step2 override | 1 (subclass) | ~180 |
| Constants extension | `core/types.py` | ~10 |
| Plumbing (SDK factory to use new creator) | `api/` or config | ~30 |
| **Total new code** | 4-5 files | **~400 LOC** |

Everything else reused as-is.

---

## Open questions for Phase 2 run (deferred until API key active)

1. What's the actual OpenAI spend for `run_create_ltm.py` on 10 convs?
2. Does `run_semantic_eval.py` vs `run_ppr_eval.py` baseline match the 78.9%
   headline from the paper? (plan calls for ±2%p reproduction)
3. Does GAAMA's `MAX_TOKENS_PER_CHUNK` config affect accuracy meaningfully?
   (Needed to pick a fair value for Hyper Triplet comparison.)

These are answered in Phase 2 execution, not now.
