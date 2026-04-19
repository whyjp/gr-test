# HyperMem Reference Notes

Paper: "HyperMem: Hypergraph Memory for Long-Term Conversations" ([arxiv 2604.08256](https://arxiv.org/abs/2604.08256))
Authors: Juwei Yue, Chuanrui Hu, Jiawei Sheng, Zuyi Zhou, Wenyuan Zhang, Tingwen Liu, Li Guo, Yafeng Deng
Published: 2026-04 (~1 week before this note)

## Status: paper-only, no public code yet

WebFetch of the arxiv abstract page (2026-04-19) returned:
> The provided content does not include links to source code, GitHub repositories, or supplementary materials beyond the paper PDF and HTML versions available through arXiv.

Action for Phase 2B: re-check the arxiv page, OpenReview, and authors' personal pages periodically; if no code appears, reimplement from paper spec.

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
