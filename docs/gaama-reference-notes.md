# GAAMA Reference — Offline Code Notes

Clone: `external/gaama/` @ HEAD `d9987ea` ("logic change")
Upstream: https://github.com/swarna-kpaul/gaama

## Key files mapped against the plan

| Plan's table (Phase 2.4) | Actual file | Notes |
|---|---|---|
| `services/ltm_creator.py` | ✅ `services/ltm_creator.py` | Replace target in Phase 3 |
| `services/llm_extractors.py` | ✅ `services/llm_extractors.py` | Replace target in Phase 3 |
| `services/orchestrator.py` | ✅ `services/orchestrator.py` | Edit target in Phase 3 |
| `services/ltm_retriever.py` | ✅ `services/ltm_retriever.py` | Reuse |
| `services/pagerank.py` | ✅ `services/pagerank.py` | Reuse |
| `services/answer_from_memory.py` | ✅ `services/answer_from_memory.py` | Reuse |
| `adapters/sqlite_memory.py` | ✅ `adapters/sqlite_memory.py` | Reuse, extend schema |
| `evals/locomo/run_*_eval.py` | ✅ 4 scripts present | Reuse |

Also present: `adapters/sqlite_vector.py`, `adapters/openai_llm.py`, `adapters/openai_embedding.py`, `adapters/llm_factory.py`, `adapters/local_blob.py`, `core/` (nodes, events), `prompts/` (3 markdown files), `infra/`.

## Extraction pipeline — CORRECTION TO THE PLAN

The plan states GAAMA makes **3 LLM calls** per chunk (fact + concept + reflection).
**Actual code: 2 calls per chunk.**

```
Chunk (Episode) → LLMFactExtractor.extract_facts()     # 1 LLM call — returns {facts, concepts}
                → LLMReflectionExtractor.extract_reflections()  # 1 LLM call
```

Prompt files:
- `prompts/fact_generation.md` — **Part 1: Facts + Part 2: Concepts** combined
- `prompts/reflection_generation.md`
- `prompts/answer_from_memory.md`

`extract_facts()` returns `(facts, concepts)` parsed from a single JSON response.

### Implication for Hyper Triplet contribution

- ❌ "2 vs 3 LLM calls, ~33% cheaper" — invalid; both systems do 2 calls.
- ✅ **Structural differentiation stands**: Hyper Triplet replaces GAAMA's flat concept labels with typed context nodes (Location, Participant, ActivityType, TimeRef, Mood).
- ✅ **Stronger MERGE** on typed node values still a claim.

### Reframed Phase 3 contribution (proposal)

1. **Structure** — typed context vs flat concept label
2. **Atomicity** — fact + typed context in one JSON shape (GAAMA's fact-concept binding is already atomic per call, so this overlaps)
3. **MERGE semantics** — value-equality MERGE on typed values vs string-label MERGE on concepts

Phase 6 cost analysis should focus on **tokens per call** and **extraction quality** (not call count).

## Next questions for Phase 2 execution

1. What LLM/model does the official `run_*_eval.py` use? (plan assumes gpt-4o-mini)
2. What's the expected runtime for `run_create_ltm.py` over 10 conversations?
3. Does GAAMA ship its own `locomo10.json` (yes — at `evals/locomo/data/locomo10.json`); diff against our downloaded copy?

These are deferred until Phase 2 actually runs (API key required).
