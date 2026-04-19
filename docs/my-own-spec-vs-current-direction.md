# my_own Spec vs Current Direction — Reconciliation & Gap Analysis

> User directive (2026-04-19, evening): compare `docs/my-own-test-design-spec.md` against the existing plan and find errors or holes on either side, then propose the next plan / modifications.
>
> This doc treats the new spec as **authoritative over v4** where they conflict, but also surfaces inconsistencies in the new spec itself so nothing gets swept under the rug.

---

## 1. At-a-glance compatibility

| Area | my_own spec (new) | v4 plan + code (existing) | Compatibility |
|---|---|---|---|
| Module name | `my_own/` | `systems/hyper_triplet/` | ⚠ conflict — **needs decision** |
| Atomic ingestion unit | node_set = fact + context + environment | `NodeSet(fact, qualifiers, source_episode_ids, belief)` | ✅ identical concept |
| Layer separation | **explicit L0/L1/L2/L3** (fact / temporal-importance / context / auxiliary) | flat `Qualifiers(location, participants, activity_type, time_reference, mood, topic)` | ❌ missing layer discipline |
| Hyper-relational edges | edge = `(src, type, dst, {qualifier_1..k})` | typed edges by qualifier type (AT_LOCATION, WITH_PARTICIPANT, ...) | ⚠ shape different — edges themselves aren't qualified |
| Storage model | **star-native** — one subgraph per node_set, KV-friendly O(k) access | generic adjacency graph (`HyperTripletGraph`), no star shape | ❌ missing |
| Retrieval pipeline | **3-stage** (broad-context → rank-importance → exact-fact) | 2 implementations: keyword+IDF, PPR over qualifier edges | ❌ different shape |
| Benchmarks | LoCoMo + LongMemEval + MuSiQue/HotpotQA | LoCoMo only | ⚠ scope narrower |
| Ablations | 7 principle-level (no_node_set, no_layer_separation, no_hyper_edge, no_star_storage, no_stage1, no_hybrid_index, no_community) | D-* decomposition against EverMemOS | ⚠ different axis |
| Baselines | HippoRAG2 + GAAMA + HyperMem | HippoRAG + GAAMA + HyperGraphRAG + HyperMem + EverMemOS (6 total) | ⚠ new spec drops EverMemOS |
| Seeds | `[42, 1337, 2024]`; must beat all 3 | `n_runs=10` + paired bootstrap | ⚠ different statistical protocol |
| LLM provider | OpenRouter / Bedrock / OpenAI (open question §11.2) | OpenAI (per plan) | ⚠ open question |
| Result layout | `eval/results/{dataset}_{system}_{seed}.json` | `results/{system}/` (empty placeholder) | ⚠ needs directory spec |
| Importance scoring | ACT-R / Bayesian surprise on L1 | not implemented | ❌ missing |
| Community detection | Leiden / incremental LPA on L3 | not implemented | ❌ missing |
| Boundary detection | explicit module before extraction | fixed `turns_per_chunk` chunking | ❌ missing |
| Edge qualifier scoring | qualifier features (confidence / source_type / temporal_validity) re-rank | retrieval uses qualifier **text**; not per-property | ❌ missing |

---

## 2. Areas where the new spec **extends** what we already have

These are additions, not corrections — our existing code stays valid but grows.

### 2.1 Explicit L0-L3 layer separation

The new spec (§3.2) mandates 4 distinct layers with **functional** roles:

| Layer | Role | Example |
|---|---|---|
| L0 | Core atomic fact (s, p, o) | `(user_42, equipped, legendary_sword)` |
| L1 | Temporal + importance metadata | `ts=2024-03-15`, `importance=0.87` |
| L2 | Context entities + environment | `location=dungeon_5`, `party_size=4` |
| L3 | Auxiliary (community, embedding ref) | `community_id=c_017` |

Our current `NodeSet` collapses L1 + L2 into `Qualifiers` and leaves L3 empty. Refactor:

```python
# before
class NodeSet(BaseModel):
    fact: Fact
    source_episode_ids: tuple[str, ...]
    belief: float
    qualifiers: Qualifiers   # mixes time + location + participants + mood + topic

# after
class NodeSet(BaseModel):
    ns_id: str
    fact: L0Fact              # (subject, predicate, object, edge_qualifiers)
    temporal: L1TemporalImportance   # timestamp, valid_from/until, importance
    context: L2Context        # context_entities, environment, participants
    aux: L3Auxiliary          # community_id, embedding_ref, source_ref, derived_metrics
```

**Backwards compatibility**: keep `Qualifiers` as a convenience adapter; provide `NodeSet.from_legacy(fact, qualifiers)` for the fixture-replay tests.

### 2.2 Star-native storage

Current `HyperTripletGraph` is a generic adjacency; every access is a graph walk. The new spec (§3.4) requires that each node_set be retrievable via one KV lookup — star = center (L0) + all L1/L2/L3 leaves in a single payload.

Concretely: add `StarStore` abstraction with `get(ns_id) -> NodeSet` in O(1) + a `qualifier_index` for inter-star joins.

Implementation choice: the in-memory Python graph already supports O(1) per-node-id lookup; the "star" is a logical grouping we materialize on ingest. The missing piece is **persisting each node_set as a single serialisable unit** (not per-node) so reads don't require joining across nodes. For the bench, a `dict[ns_id, NodeSet]` satisfies this.

### 2.3 Three-stage retrieval

Our current `retrieve()` is a single-pass token-scored retriever, and `retrieve_ppr()` is single-pass propagation. The new spec (§6) requires three stages with independent modules that can be ablated individually:

```
Stage 1 (Broad, context-level):
    query → entity/keyword extraction → L2 context lexical+semantic match
    → L3 community expansion
    → candidate pool (100-500)

Stage 2 (Rank, temporal/importance):
    L1 importance score + query-temporal alignment
    → top-K (10-30)

Stage 3 (Exact, fact + qualifier re-rank):
    L0 fact subject/predicate/object match
    + edge qualifier features (confidence, source_type, temporal_validity)
    → final context for answer generation
```

This is **incompatible with our current retrieval modules as primary path** — it replaces them. The current PPR and keyword retrievers become *implementations inside* stage 3's re-rank, or get retired.

### 2.4 Boundary detection

Our pipeline splits turns by `turns_per_chunk` (fixed count). The new spec (§4.1-C) requires a boundary detector that uses signal burst / temporal coherence / entity overlap. For LoCoMo dialogue this can reuse HyperMem's approach.

Small module (`boundary_detector.py`), stateless, decouples extraction granularity from raw input shape.

### 2.5 Community detection

New spec (§4.1-D) wants Leiden / incremental LPA on the graph → community_id stored in L3 Auxiliary. Community id becomes a broad retrieval stage-1 filter.

Straightforward with `networkx` + `python-louvain` (or `networkx.community.louvain_communities`). Background worker — not on ingest path.

### 2.6 Importance scoring on L1

Beyond `belief` (confidence), the new spec wants:
- **Base activation** (ACT-R style): `log(frequency) + recency_decay`
- **Bayesian surprise**: how unexpected is this fact given prior memory?
- Used in stage 2 rank.

Neither implemented. Non-trivial but self-contained in a background module.

---

## 3. Errors / inconsistencies in the new spec itself

Critical review — the user asked for honest surface-up of issues.

### 3.1 HINGE citation mismatch (important)

New spec §12 references HINGE as "CIKM 2023, arXiv:2208.14322 — Hyper-Relational Knowledge Graphs for Multi-hop QA using LLMs". **This is wrong**:

- `arXiv:2208.14322` is actually **HoLmES** (different paper line).
- The canonical HINGE is **Rosso, Yang, Cudré-Mauroux, WWW 2020**, DOI `10.1145/3366423.3380257` — the paper that coined hyper-relational fact = `(h, r, t) + {(k_i, v_i)}`.
- Our [`docs/hinge-north-star.md`](./hinge-north-star.md) and [`docs/hinge-technical-notes.md`](./hinge-technical-notes.md) cite HINGE correctly.

**Impact**: the new spec's design references ("HINGE 계보의 주장 수용") is correct in spirit — the author clearly knows the hyper-relational direction — but the specific citation points at a different paper. If the paper draft follows the new spec literally, reviewers will catch it.

**Recommendation**: keep our HINGE citation (Rosso 2020). If the author specifically wants to cite the 2023/CIKM paper (Di et al.), that's a **distinct related-work row**, not a replacement for HINGE.

### 3.2 EverMemOS absent from baselines

The new spec (§2, §8.3) lists baselines as HippoRAG2 / GAAMA / HyperMem. It does not mention **EverMemOS**, which a prior session established as the true LoCoMo SOTA (93.05%, arxiv 2601.02163, 2026-01).

HyperMem (92.73%) is literally a sub-module of EverMemOS per `docs/evermemos-reference-notes.md`. Skipping EverMemOS means the primary baseline is set below the actual SOTA.

**Question for user**: is EverMemOS deliberately excluded, or was the new spec drafted before the EverMemOS finding?

- If excluded because "commercial / too heavy to run": document this explicitly.
- If newer than EverMemOS awareness: add EverMemOS to the baseline list.

### 3.3 "Hyperedge not first-class" vs star-native storage

Spec §4.2 excludes hyperedge-as-first-class-object, arguing it conflicts with star-native storage (§3.4).

This reading confuses two things:
- **HINGE's hyper-relational fact** = base triple + k-v qualifier set = ONE atomic unit. This IS the star (center + leaves). HINGE does not require a separate "hyperedge" entity beyond the fact itself.
- A separate explicit hyperedge object table (HyperMem's `FactHyperedge` / HyperGraphRAG's `hyperedge` entity bag) is a different architectural choice.

So rejecting "hyperedge as first-class object" while keeping node_set = star is **internally consistent** — but the wording conflates the two. Paper writing should say "we reject an explicit hyperedge table; the node_set star IS our hyper-relational fact realisation" rather than "we reject hyper-relational".

### 3.4 Temporal qualifier in "edge_qualifiers" OR "L1 temporal"?

§3.1 puts temporal inside the core_fact as `edge_qualifiers`, but §3.2 puts temporal in L1 as its own layer. §5.3 has both `edge_qualifiers` on `L0Fact` AND a separate `L1TemporalImportance`.

This is ambiguous. My reading:
- **Per-relation temporal validity** (e.g., "valid_from":"2024-01-01" on the relation edge itself) → `L0Fact.edge_qualifiers`
- **Per-fact temporal context** (when the user said it; importance) → `L1TemporalImportance`

But the spec doesn't distinguish these. Either reading is defensible; pick one and document.

**Recommendation**: L0 carries relation-level qualifiers (HINGE-style); L1 carries observation-level timestamp + importance. Two distinct temporal concepts.

### 3.5 "Test leak 방지" concern

§10.1: "LoCoMo 데이터셋을 sanity로 쓰고 튜닝 대상으로 쓰지 말 것". Reasonable but currently our `tests/fixtures/locomo_conv26_session1_gold.json` uses conv-26 turns to drive offline regression tests. That IS a form of sanity, not tuning, and is bounded to 16 of 5,882 turns. But if v5 evaluation trains on LoCoMo splits we need to explicitly declare hold-out vs dev.

### 3.6 Ablation "no_stage1" assumes stages exist

Ablation §7 includes `no_stage1` (skip stage-1 broad search). This presumes the 3-stage pipeline is implemented; currently it isn't. Either the spec is aspirational (stages will exist in v5) or the ablation row is unrunnable today.

---

## 4. Errors / holes in our existing direction (v4)

Being fair to both sides.

### 4.1 `Qualifiers` lacks layer discipline

As noted in §2.1. Our current `NodeSet(fact, qualifiers)` is HINGE-compliant but mixes what the new spec wants separated. This is a real missing design decision that v4 never addressed — we emitted qualifiers as a flat bag without structuring them functionally.

### 4.2 No importance signal

v4 relies on `belief` (per-fact LLM-reported confidence) for ranking. We never defined an ACT-R-style activation score or Bayesian surprise. Our retrieval treats all facts as equally "salient" before ranking.

### 4.3 No community structure

Our graph has typed edges but no community / cluster structure above the fact level. HyperMem's topic layer, EverMemOS's MemScene, the new spec's L3 community — all three agree this is needed. v4 deferred it.

### 4.4 No boundary-aware ingest

Our `HyperTripletPipeline.ingest()` fixed-chunks by turn count. This wastes extraction opportunities when a session has 3 topic shifts and we chunk across them.

### 4.5 Single benchmark

v4 targets LoCoMo only. The new spec adds LongMemEval + MuSiQue/HotpotQA. Without those we over-fit our claims to LoCoMo's question distribution.

### 4.6 No `config.py`

New spec §10.2 requires all hyperparameters in a central `config.py`. Our retrieval modules have ad-hoc keyword args. Centralising them is straightforward and makes ablation sweeps cleaner.

### 4.7 Retrieval modules don't expose per-stage ablation

`retrieve()` and `retrieve_ppr()` are monolithic. To ablate stages we need explicit stage1/stage2/stage3 modules per the new spec.

---

## 5. Proposed next plan (v5 preview)

Scope: reconcile both, fix both sides' gaps. Offline work dominates; real LLM runs gated on API keys.

### 5.1 Decisions required from user (block on these)

1. **Module rename**: `systems/hyper_triplet/` → `systems/my_own/`? Or keep `hyper_triplet` as internal name, surface `my_own` as the public target? My recommendation: **alias** — keep `hyper_triplet/` code, add `systems/my_own/__init__.py` that re-exports everything. Zero churn, matches spec naming.
2. **EverMemOS in baselines**: include or exclude? My recommendation: **include** — it's the real SOTA. Add it with a note that execution requires docker-compose + API.
3. **HINGE citation**: use Rosso 2020 (our current) or also cite the 2023 variant? My recommendation: **Rosso 2020 primary**, 2023 paper as related work if relevant to specific techniques.
4. **LLM provider**: user open question §11.2 of the new spec. My recommendation: **OpenAI (gpt-4o-mini extract, gpt-4o judge)** per existing plan — cheapest path and matches HyperMem / GAAMA's evaluation protocols.
5. **LongMemEval + MuSiQue**: include now or Phase 6? My recommendation: **schedule as Phase 6** — wire LoCoMo first, add the other two only if LoCoMo signal is positive.

### 5.2 Scope of v5 (assuming defaults above)

**Phase A0 (alias + layer refactor)** — backwards-compat preserving refactor:
- Add `systems/my_own/` alias package
- Introduce `L0Fact / L1TemporalImportance / L2Context / L3Auxiliary` dataclasses
- Provide `NodeSet.from_legacy_qualifiers()` for existing tests
- All 117 existing tests continue to pass
- Commit as one atomic change

**Phase A1 (star storage + ns_id)** — deterministic hashing:
- Add `ns_id` deterministic hash from L0 content
- `StarStore` abstraction (backed by dict + qualifier index)
- O(1) full-star retrieval

**Phase B (boundary detection + importance scoring)**:
- `boundary_detector.py` — stateless detector
- `importance_scorer.py` — simple base activation + recency
- Tested on fixture conv-26

**Phase C (3-stage retrieval)**:
- `stage1_broad.py`, `stage2_rank.py`, `stage3_exact.py`, `pipeline.py`
- Existing `retrieve_ppr` becomes stage-1 broad option; existing `retrieve` becomes stage-3 fallback

**Phase D (community detection)**:
- `community_detector.py` — Leiden via networkx
- Populates L3.community_id as background step

**Phase E (central config + 7 ablations)**:
- `my_own/config.py` with all hyperparameters
- `ablation_runner.py` wiring each `no_*` toggle
- Extend existing `MultiSystemRunner` to be ablation-aware

**Phase F (benchmarks + seeds)**:
- 3-seed evaluation per spec §8.4
- `eval/results/{dataset}_{system}_{seed}.json` layout
- `summary.md` auto-generator

Phases A0+A1 are **today-level** work, entirely offline. Phase B-E need ~1 day each. Phase F needs API access.

### 5.3 Invariants preserved across the refactor

- HINGE's hyper-relational invariant (fact + qualifiers atomic)
- 117 existing tests stay green through every phase (regression harness)
- No external system code modified (HippoRAG / GAAMA / HyperGraphRAG / EverMemOS / HyperMem clones stay read-only)
- Commit after every phase's test suite goes green; push to `origin/main`

### 5.4 Deferred to v6+

- Kafka-scale streaming ingestion (belongs to graphdb-bench project, not this one per `project_evermemos_is_true_sota.md`)
- MMORPG domain adaptation
- Production latency optimisation
- Custom graph DB

---

## 6. Immediate proposal — next concrete steps

If the user agrees with the recommendations in §5.1, I can execute without blocking:

1. Write v5 plan doc fleshing §5.2 into a full spec (≈ the depth of v4)
2. Execute Phase A0 (alias + layer refactor) — offline, tests stay green, one commit
3. Execute Phase A1 (star storage + ns_id) — offline, one commit
4. Report back before starting Phase B for a checkpoint review

If the user wants different decisions on §5.1:

- Different module naming → adjust Phase A0
- Drop EverMemOS → simplify Phase 4 runner and baseline doc
- Include LongMemEval now → enlarge Phase F

**I'm blocking Phase B onwards on explicit confirmation of §5.1 decisions** because renaming modules and adding layers touches many tests — worth paying the cost of a quick confirmation round before committing to the refactor.
