# HyperMem Reference Notes

Paper: "HyperMem: Hypergraph Memory for Long-Term Conversations" ([arxiv 2604.08256](https://arxiv.org/abs/2604.08256))
Authors: Juwei Yue, Chuanrui Hu, Jiawei Sheng, Zuyi Zhou, Wenyuan Zhang, Tingwen Liu, Li Guo, Yafeng Deng
Published: 2026-04

## Status update (2026-04-19 PM): HyperMem is a sub-module of EverMemOS

Original arxiv landing page appeared not to link code. That was incomplete. HyperMem is shipped inside the EverMemOS monorepo at:

```
external/everos/methods/HyperMem/
├── hypermem/
├── README.md
├── requirements.txt
└── scripts/
```

See [`evermemos-reference-notes.md`](./evermemos-reference-notes.md) for the full picture. Key corrections vs earlier notes:

- HyperMem (92.73% LoCoMo) is a RESEARCH PROTOTYPE of the conversation-memory layer inside EverMemOS.
- The commercial / primary system is **EverMemOS** (93.05% LoCoMo, arxiv 2601.02163, 2026-01).
- Reimplementation from paper spec is no longer necessary — run the upstream `methods/HyperMem/` or the fuller `methods/evermemos/` stack.

The v3 plan's "Phase 2B — HyperMem reimpl" is retired. v4 replaces it with "Phase 2B — execute EverMemOS on LoCoMo via docker-compose".

## Architecture (from abstract + lineage doc)

**Three-level hierarchy:**
1. **Topics** — most abstract
2. **Episodes** — dialogue-session boundary units
3. **Facts** — atomic units

**Hyperedge construction:** related episodes **and** their facts are grouped into a single hyperedge. This creates a coherent unit rather than pairwise edges between fragmented entities.

**Retrieval:** coarse-to-fine (topic → episode → fact) with hypergraph-embedding propagation for lexical-semantic dual indexing. Early pruning for efficiency.

## LoCoMo result (from lineage doc §4)

**92.73%** LLM-as-judge accuracy — current SOTA.

Delta:
- vs GAAMA (78.9%): **+13.83pp**
- vs HippoRAG (69.9%): **+22.83pp**

## Known limitations (authors self-reported per lineage doc §5)

1. Single-user assumption — multi-user / multi-agent out of scope.
2. Open-domain questions needing out-of-dialogue knowledge → weak.
3. **Not streaming** — assumes structured dialogue turns arrive pre-parsed.

These are directly relevant axes for future differentiation but are orthogonal to the *this* project (LoCoMo decomposition).

## Relationship to our Hyper Triplet

Our Hyper Triplet ≈ HyperMem Level 3 (facts + hyperedges) WITHOUT Level 1 (topics) and Level 2 (episodes as coarse retrieval anchors).

The per-axis decomposition outlined in v3 Phase 5 lets us measure:
- Jump from GAAMA → our Hyper Triplet (typed qualifiers only)
- Jump from our Hyper Triplet → "+ episode hierarchy"
- Jump from "+ episode hierarchy" → HyperMem full (+ topic hierarchy)

So our Hyper Triplet + staged extensions should reproduce HyperMem's architecture incrementally, testing each level's contribution.

## Phase 2B plan (deferred)

1. Retry code-availability check monthly.
2. If no code by Phase 2B start: reimplement from paper, using our existing `HyperTripletLTMCreator` as the Level 3 starting point.
3. Topic layer: cluster/summarise concept/qualifier topics into coarser Topic nodes. Candidate approach: BERTopic or LLM-summarisation over existing topic qualifiers.
4. Episode layer: aggregate session-boundary metadata into Episode hyperedges.
5. Retrieval: coarse-to-fine propagation. Needs hypergraph-embedding module.

No implementation work in v3 until Phase 1B (GAAMA reproduction) is green.
