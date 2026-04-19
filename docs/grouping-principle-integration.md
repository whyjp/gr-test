# Grouping-Node Principle → Project Integration

> Source: [`grouping-node-principle.md`](./grouping-node-principle.md) (user-supplied, 2026-04-19).
> Purpose: integrate the new principle into our existing direction, reconcile with HINGE north-star, update ablation design, document deltas.

---

## 1. The new principle in one sentence

> **Use LLM only to classify membership in existing structure, never to generate new compressed content.**

This generalizes HINGE's "transformation loss" claim from link-prediction to the memory domain. Reflection / summary / community-summary style grouping is **lossy**; topic-detection / scene-clustering style grouping is **lossless**.

Visual:

```
✅ classifier: fact → (topic_A? yes/no)
✅ classifier: fact → (temporal_scope_5? yes/no)
✅ classifier: fact → (ontology_category_Item? yes/no)

❌ encoder: facts → "these facts together mean..." (new summary text)
❌ encoder: community → "this community is about..." (compressed description)
```

## 2. Where the principle aligns with our existing design (already good)

- **Atomic node_set extraction** in `LLMNodeSetExtractor` — one LLM call produces fact + typed qualifiers in one JSON shape. No downstream reflection/summary pass. ✅
- **Typed qualifier values as first-class nodes** (via `HyperTripletGraph.merge_qualifier`) — qualifier values are classified category labels, not LLM-generated summaries. ✅
- **No reflection layer** — unlike GAAMA, our pipeline emits zero LLM-generated derived text. ✅
- **Per-category ablation target** — v5's ablation design already includes `no_community` / `no_hyper_edge` / `no_layer_separation` which map onto the grouping-node principle. ✅

## 3. What the principle reveals is MISSING in our existing direction

### 3.1 We conflate extraction and grouping semantically

Our `NodeSet` treats all LLM output as "extracted content". The new principle makes us distinguish:
- **Extraction (encoder, one-shot)**: the initial atomic fact+qualifier JSON. This IS encoder-style but it's the minimal loss possible (one pass, original text available as source).
- **Grouping (classifier, ongoing)**: assigning membership to topics/communities/scopes. This must NEVER re-encode.

We currently conflate both into the single extraction prompt. That's fine because we do no post-hoc grouping — but we should make the distinction explicit in prompts and documentation so future additions (community assignment, scope detection) don't accidentally slip into encoder mode.

### 3.2 L3 Auxiliary lacks an ontology axis

User's design wants **3-axis grouping**: L1 temporal + L2 context + L3 ontology (Palantir-style Object/Property/Link). Our current `L3Auxiliary` has only `topic / community_id / embedding_ref / source_ref` — no ontology slot.

**Fix**: extend `L3Auxiliary` with `ontology_type: str | None` and `ontology_properties: tuple[str, ...]`. Keep it optional so legacy NodeSets still work.

### 3.3 HINGE north-star invariants need an 8th entry

Our existing 7 HINGE invariants (atomicity, no flat-concept reduction, qualifier typing, MERGE on value, eval exposes correlation, storage keeps qualifiers first-class, retrieval traverses qualifier edges) don't explicitly forbid encoder-style grouping. Add:

> **Invariant 8 — LLM-as-classifier-only for grouping**
>
> All grouping nodes (community, scope, topic cluster, reflection-like) MUST be assigned via membership classification. An LLM MUST NOT generate new compressed/summarized content to store as a grouping node. If any grouping layer requires summarization, that summarization is a derived view for display only, never persisted as authoritative memory.

### 3.4 Compliance audit needs a column

Our `docs/hinge-compliance-audit.md` table has 7 invariant columns. Add column for Invariant 8.

Based on the grouping-principle doc's own claims:
| System | Inv 8 (classifier-only) | Evidence |
|---|---|---|
| GAAMA | ❌ violated | Reflections are LLM-encoded compressed content, not membership labels |
| HyperGraphRAG | ⚠ partial | Hyperedge "knowledge segment" is a free-text excerpt — copied from source, not newly encoded, so not strictly encoder-lossy, but also not classifier-pure |
| HippoRAG 2 | ✅ preserved | Triples are extraction (one-pass); no reflection |
| EverMemOS | ✅ preserved | MemScene = geometric centroid clustering; Foresight = LLM extracts FROM conversation, not FROM facts |
| HyperMem | ✅ preserved | Topic detection is membership classification |
| Hyper Triplet (ours) | ✅ preserved | Single extraction pass, no reflection, no post-hoc encoded summaries |

This sharpens our earlier audit: GAAMA's reflection violates invariant 8 even more clearly than invariant 2.

## 4. What the principle does not change in our plan

- v5 module name stays `hyper_triplet/`
- v5 phase order (A0 / A1 / B / C / D / E / F) unchanged
- 6-system baseline roster unchanged
- OpenAI LLM choice unchanged
- LoCoMo first, LongMemEval / MuSiQue deferred

## 5. Recommended concrete updates

### 5.1 Code changes (small, offline)

1. **Extend `L3Auxiliary`** in `systems/hyper_triplet/types.py` with:
   ```python
   ontology_type: str | None = None           # e.g. "Player", "Item", "Event"
   ontology_properties: tuple[str, ...] = ()  # e.g. ("level:42", "class:wizard")
   ```
2. **Add docstring header to `prompts/node_set_generation.md`** flagging it as extraction-only; forbid any derivative summary generation.
3. **Update `docs/hinge-north-star.md`** with invariant 8.
4. **Update `docs/hinge-compliance-audit.md`** with the classifier-only column.

### 5.2 Ablation additions

Add two new ablations that isolate the encoder-vs-classifier decision:

| Ablation | Semantics | Invariant tested |
|---|---|---|
| `gaama_style_reflection_on` | add a reflection-generating LLM call after node_set extraction; store as a new node type | violates Inv 8 — measures reflection-caused ceiling drop vs keeping classifier-only |
| `no_ontology_axis` | disable L3 ontology_type / properties; use only temporal + context axes | isolates the 3rd axis contribution (user's claim of 2-axis → 3-axis advantage over HyperMem/EverMemOS) |

These become additional rows in v5 Phase E's ablation runner.

### 5.3 Paper framing

Add a subsection "Classifier vs Encoder: the HINGE principle in the memory domain" citing:
- HINGE (Rosso 2020) for transformation loss in KG embedding
- User's `grouping-node-principle.md` for the memory-domain generalization
- Our audit as empirical evidence that SOTA systems already follow the principle implicitly

Claim: the ceiling discontinuity between GAAMA (78.9%) and HyperMem/EverMemOS (92-93%) is best explained by the encoder-to-classifier shift, not by "hypergraph vs triple" at the surface syntax level.

## 6. Minor corrections to the new doc itself

(Raised in the spirit of §10.2 "모든 실험마다 반드시 정확한 수치" — honest surface-up, consistent with our v4-v5 culture of flagging issues.)

1. §2 table reports GAAMA → HyperMem delta as "+14pp" — the actual number is +13.83pp (92.73 - 78.9). Minor rounding.
2. §2 row for HyperGraphRAG's LoCoMo number is not in the new table but was 86.49% per HyperMem's Table. Worth adding as a datapoint between GAAMA and HyperMem.
3. §4.3 "개념 레벨에서는 EverMemOS 추종" — reads as a contradiction with §3's "원리의 더 강한 실현". I think the intent is "개념 아키텍처는 공유, 원리는 더 일관되게" but the first reading may mislead. Purely stylistic.
4. §5.5 claims 3-axis vs HyperMem's 2-axis. HyperMem's axes are topic × episode (semantic hierarchy). Whether episode = time is debatable — episode is semantic segmentation within a temporal slice, so "2-axis topic + time" is approximately right but not exactly HyperMem's framing.

None of these change the core principle. All are annotation-level.

## 7. Immediate next steps (queued after user confirmation)

Phase B was scheduled next in v5 (boundary_detector + importance_scorer). Inserting before Phase B:

**Phase A2 — integrate grouping principle (offline)**
1. Add invariant 8 to hinge-north-star.md
2. Update compliance audit with the new column + verdicts
3. Extend L3Auxiliary with ontology_type / ontology_properties fields
4. Add classifier-only header to node_set_generation.md
5. Add 2 new ablation labels to v5 (gaama_style_reflection_on, no_ontology_axis)
6. Tests: verify L3Auxiliary ontology fields round-trip through NodeSet views
7. Commit + push

Then proceed to Phase B with the sharpened principle in hand.
