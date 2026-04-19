# LoCoMo-10 Baseline Numbers

Curated reference table of reported LoCoMo scores across systems. Used to set
Phase 2 reproduction targets and frame paper decomposition.

## Overall accuracy (LLM-as-judge)

| Method | Source | Overall |
|---|---|---|
| **EverMemOS** | press 2026-02-03 | **93.05%** |
| **HyperMem** | paper README (ACL 2026) | **92.73%** |
| HyperGraphRAG | HyperMem paper Table | 86.49% |
| MIRIX | HyperMem paper Table | 85.38% |
| MemMachine | blog | 84.87% |
| HippoRAG 2 | HyperMem paper Table | 81.62% |
| LightRAG | HyperMem paper Table | 79.87% |
| GAAMA | paper | 78.9% |
| MemOS | HyperMem paper Table | 75.80% |
| Tuned RAG (baseline) | GAAMA paper | 75.0% |
| Memobase | HyperMem paper Table | 72.01% |
| HippoRAG 1 | GAAMA paper | 69.9% |
| Mem0 (graph variant) | HyperMem paper Table | 68.44% |
| GraphRAG | HyperMem paper Table | 67.60% |
| Mem0 | HyperMem paper Table | 66.88% |
| Zep | HyperMem paper Table | 65.99% |
| LangMem | HyperMem paper Table | 58.10% |
| MemU | HyperMem paper Table | 56.55% |
| OpenAI | HyperMem paper Table | 52.90% |
| Nemori | prior | 52.1% |
| A-Mem | HyperMem paper Table | 48.38% |

## Per-category breakdown (from HyperMem README)

Percentages; LLM-as-judge via GPT-4o-mini over 3 rounds averaged.

| Method | Single-hop | Multi-hop | Temporal | Open Domain | Overall |
|---|---:|---:|---:|---:|---:|
| GraphRAG | 79.55 | 54.96 | 50.16 | 58.33 | 67.60 |
| LightRAG | 86.68 | 84.04 | 60.75 | 71.88 | 79.87 |
| HippoRAG 2 | 86.44 | 75.89 | 78.50 | 66.67 | 81.62 |
| HyperGraphRAG | 90.61 | 80.85 | 85.36 | 70.83 | 86.49 |
| OpenAI | 63.79 | 42.92 | 21.71 | 63.22 | 52.90 |
| LangMem | 62.23 | 47.92 | 23.43 | 72.20 | 58.10 |
| Zep | 61.70 | 41.35 | 49.31 | 76.60 | 65.99 |
| A-Mem | 39.79 | 18.85 | 49.91 | 54.05 | 48.38 |
| Mem0 | 67.13 | 51.15 | 55.51 | 72.93 | 66.88 |
| Mem0 (graph) | 65.71 | 47.19 | 58.13 | 75.71 | 68.44 |
| MIRIX | 85.11 | 83.70 | 88.39 | 65.62 | 85.38 |
| Memobase | 73.12 | 64.65 | 81.20 | 53.12 | 72.01 |
| MemU | 66.34 | 63.12 | 27.10 | 50.56 | 56.55 |
| MemOS | 81.09 | 67.49 | 75.18 | 55.90 | 75.80 |
| **HyperMem** | **96.08** | **93.62** | **89.72** | 70.83 | **92.73** |

(EverMemOS per-category numbers not yet extracted from its press materials; to
be confirmed during Phase 2B reproduction.)

## Category takeaways (for v4 decomposition story)

- **Single-hop** is near-saturated by HyperGraphRAG (90.61%) and HyperMem (96.08%) — little headroom.
- **Multi-hop**: HyperMem 93.62% vs GAAMA family ~80% — largest Lineage-A delta.
- **Temporal**: HyperMem 89.72% vs MIRIX 88.39% — hypergraph structure is not uniquely responsible here; MIRIX (non-hypergraph) matches.
- **Open domain**: **everyone below 77%, HyperMem at 70.83%**. This is the category where structure doesn't help and may actually hurt vs Zep's 76.60% / Memu's 72.93% / LangMem's 72.20%. Interesting signal.

Matches v4's per-category hypothesis: structural gains concentrate on temporal + multi-hop, not open-domain. The fact that even the SOTA drops to ~71% on open-domain suggests the ceiling there is driven by something other than memory structure (prompt engineering? answer-generation style?).

## What Phase 4 adds

Current numbers come from different papers with different LLMs / prompts / judges. Phase 4 runs the 6-system roster under:
- Same extract + answer LLM (gpt-4o-mini per GAAMA convention)
- Same judge (gpt-4o)
- Same embedding + rerank (OpenAI or consistent alternative)
- Same retrieval budget (1,000 words)
- Same QA categories (1-4, exclude adversarial 5)

Expected effect: the gap between the best and the worst compresses slightly, and
per-category patterns become comparable across lineages for the first time.
