# HippoRAG 2 Reference Notes

Clone: `external/hipporag/` @ HEAD `d437bfb`
Upstream: https://github.com/OSU-NLP-Group/HippoRAG
Papers:
- HippoRAG 1: Gutierrez et al., NeurIPS 2024 ([arxiv 2405.14831](https://arxiv.org/abs/2405.14831))
- HippoRAG 2: "From RAG to Memory", ICML 2025 ([arxiv 2502.14802](https://arxiv.org/abs/2502.14802))

The cloned repo is HippoRAG 2. HippoRAG 1 lives in the `legacy` branch.

## Repo layout

```
external/hipporag/
├── src/hipporag/           core package
├── main.py                 OpenAI demo entry
├── main_dpr.py             DPR retrieval variant
├── main_azure.py           Azure OpenAI variant
├── demo_*.py               per-provider demos (openai, azure, bedrock, local)
├── reproduce/              paper-reproduction scripts
├── test_*.py, tests_*.py   tests per provider
└── requirements.txt
```

## Important caveat vs our v3 plan

The HippoRAG 2 paper evaluates on **MuSiQue, 2Wiki, HotpotQA, LV-Eval, NaturalQuestions, PopQA, NarrativeQA** — NOT LoCoMo.

The 69.9% LoCoMo score cited in our v3 plan (and in GAAMA's paper) refers to **HippoRAG 1** (arxiv 2405.14831), not HippoRAG 2. GAAMA's paper ported HippoRAG's methodology to LoCoMo for comparison; the original authors never ran it themselves.

**Implications for Phase 1A:**
- Running HippoRAG 2 on LoCoMo = a new data point, not a reproduction of 69.9%.
- If we want to match the 69.9% headline, we need HippoRAG 1 (legacy branch) OR GAAMA's own HippoRAG port.
- Cleaner option: use GAAMA's bundled implementation as the "HippoRAG on LoCoMo" reference, and treat our HippoRAG 2 clone as a separate Lineage B datapoint on its native benchmarks.

## Headline positioning from README

HippoRAG 2 focus:
- **Associativity** (multi-hop retrieval)
- **Sense-making** (integrating large/complex contexts)
- **Continual learning** framing

Methodology (from `images/methodology.png` referenced in README): OpenIE extraction + passage nodes + query-to-triple matching + Personalized PageRank. Adds the passage node vs v1.

Public HF dataset: https://huggingface.co/datasets/osunlp/HippoRAG_2/tree/main

## Deferred to Phase 1A

- Check if GAAMA's code already includes an inline HippoRAG baseline (probably yes per `external/gaama/evals/locomo/run_rag_baseline.py`).
- Decide: reproduce HippoRAG 2 on LoCoMo ourselves, or adopt GAAMA's HippoRAG port, or skip Lineage-B floor entirely and report only GAAMA.
- Score it on the same judge/protocol as our other systems.
