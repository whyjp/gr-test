# HyperGraphRAG Reference Notes

Clone: `external/hypergraph_rag/` @ HEAD `a804827`
Upstream: https://github.com/LHRLAB/HyperGraphRAG
Paper: Luo et al., NeurIPS 2025 ([arxiv 2503.21322](https://arxiv.org/abs/2503.21322))

## Repo layout

```
external/hypergraph_rag/
├── hypergraphrag/          core package (~4,000 LOC)
│   ├── base.py             Storage abstract base classes
│   ├── hypergraphrag.py    Main pipeline (549 LOC)
│   ├── operate.py          Extraction / retrieval (1,123 LOC) — heaviest file
│   ├── prompt.py           Extraction prompts (350 LOC)
│   ├── storage.py          JsonKVStorage, NanoVectorDBStorage, NetworkX graph
│   ├── llm.py              LLM adapters
│   ├── utils.py
│   └── kg/                 (subdir)
├── evaluation/             eval harness (domain benchmarks, not LoCoMo)
├── script_construct.py     CLI entry: build hypergraph
├── script_query.py         CLI entry: query
├── example_contexts.json
└── requirements.txt
```

## Extraction format — how a hyper-relation is represented

Key insight: HyperGraphRAG's hyperedge is **not** a structured `(s, p, o) + qualifier_dict`. It is a **continuous piece of text describing a complex n-ary relationship**, plus a bag of entities extracted within that text.

From `prompt.py:13-65`:

```
Each hyper-relation is output as tuple-delimited:
  ("hyper-relation"<|>knowledge_segment_text<|>completeness_score)

Each entity inside it:
  ("entity"<|>name<|>type<|>description<|>key_score)

Records separated by ##
```

Example (from the prompt's few-shot):
```
("hyper-relation"<|>"Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty."<|>7)##
("entity"<|>"Alex"<|>"person"<|>"Alex is a person who clenched his jaw, showing frustration..."<|>95)##
("entity"<|>"Taylor"<|>"person"<|>"Taylor is a person who has an authoritarian certainty."<|>90)##
```

So the hyper-relation `knowledge_segment_text` is a *sentence-ish excerpt* capturing the complex relationship; the entities enumerate who/what participates.

**Comparison vs our Hyper Triplet design:**

| Axis | HyperGraphRAG | Our Hyper Triplet |
|---|---|---|
| Hyperedge content | Free-text sentence + entity list + completeness score | Structured `(subject, predicate, object)` + typed qualifier dict |
| Qualifier typing | None — just "entity type" (person/object/etc.) | Explicit types: location / participant / activity_type / time_reference / mood / topic |
| Fact subject anchor | Implicit (any entity) | Explicit (fact.subject) |
| MERGE semantics | Entity-name merge | `(qualifier_type, normalized_value)` merge |

This means our Hyper Triplet is a more structured / type-strict variant of HyperGraphRAG's hyperedge. Phase 2A's LoCoMo adaptation work will need to decide whether to use their text-sentence format (easier to adapt from docs) or our typed dict (matches episodic memory prompts better).

## Storage (storage.py)

- **JsonKVStorage**: file-backed key-value with `asyncio` interface. Namespaces produce separate json files.
- **NanoVectorDBStorage**: uses `nano_vectordb` package. File-backed embeddings with cosine similarity, threshold filter.
- **NetworkXGraphStorage**: in-memory `networkx.Graph` (not shown in snippet; referenced via `BaseGraphStorage`).

No SQLite, no heavyweight DB. Whole graph lives in JSON + numpy. This is light-weight compared to GAAMA's SQLite+FTS5.

## How to adapt for LoCoMo (Phase 2A plan)

Upstream `script_construct.py` takes a list of `unique_contexts` (narrative documents). To feed LoCoMo:

1. Concatenate each LoCoMo session into one "document" or keep turns as separate contexts.
2. Write a LoCoMo loader that emits the same list-of-strings format.
3. Run the existing extraction pipeline — no code change to the core.
4. For evaluation, write a wrapper that iterates QA pairs, calls `rag.query(qa.question)`, and scores via our `LLMJudge`.

**Open question**: HyperGraphRAG's default prompts target factual documents (medical/legal). Dialogue turns may need prompt tuning. Keep this as a Phase 2A experiment variable — first run with default prompts, then iterate if accuracy is poor.

## Required dependencies (requirements.txt — check before installing)

- `networkx`, `nano-vectordb`, `numpy`, `openai`, `tqdm`
- Plus `tiktoken` and async libs

None are blockers for the current uv-managed environment; can add as an optional `[baselines]` extra when Phase 2A kicks in.

## Deferred for Phase 2A execution

- Run on 1 LoCoMo conv → measure per-conv cost
- Evaluate on full LoCoMo-10 → new number (not reported upstream)
- Compare extraction JSON quality vs our node_set gold fixture
