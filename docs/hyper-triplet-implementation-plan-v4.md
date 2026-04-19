# Hyper Triplet: Implementation & Benchmark Plan — v4

> **⚠️ SUPERSEDED BY [v5](./hyper-triplet-implementation-plan-v5.md) — 2026-04-19 late evening**
>
> v5 integrates `docs/my-own-test-design-spec.md`: 4-layer functional separation,
> star-native storage, 3-stage retrieval, boundary + importance + community detection,
> and 7 additional principle-level ablations. User-confirmed decisions: module stays
> `hyper_triplet`, EverMemOS included as baseline, HINGE citation per Rosso 2020,
> OpenAI LLM, extra benchmarks deferred to Phase 6.

---

> **North star**: [`hinge-north-star.md`](./hinge-north-star.md) — HINGE (Rosso et al., WWW 2020) defines the hyper-relational data-model invariant. Every phase below is judged against it. HINGE is the principle; EverMemOS is an instantiation; GAAMA is a degenerate case.
>
> Replaces [v3](./hyper-triplet-implementation-plan-v3.md) (2026-04-19, evening).
> v4 integrates the user-provided EverMemOS analysis:
> - **EverMemOS** (arxiv 2601.02163, 2026-01) is LoCoMo SOTA at **93.05%**.
> - HyperMem (arxiv 2604.08256, 2026-04) is its conversation-memory sub-module — not a standalone system.
> - The EverOS monorepo (`github.com/EverMind-AI/EverMemOS`) ships both with docker-compose infrastructure.
> - Our Hyper Triplet design independently rediscovered the MemCell concept (atomic fact + Foresight + episodic trace). Architecture novelty claim is fully retired.

---

## What changed vs v3

| # | v3 | v4 |
|---|---|---|
| 1 | HyperMem is the LoCoMo SOTA at 92.73% | **EverMemOS** is the true SOTA at 93.05%; HyperMem is its sub-module. |
| 2 | Phase 2B = reimplement HyperMem from paper spec | **Phase 2B = run EverMemOS upstream via docker-compose** (MongoDB + Elasticsearch + Milvus + Redis). |
| 3 | Five-system roster | **Six-system roster** with EverMemOS as the Lineage-A+B endpoint. HyperMem stays, but as EverMemOS's inner conversation-memory layer. |
| 4 | Per-lineage framing | Same framing, but now EverMemOS spans the bridge explicitly: engram-inspired 3-stage lifecycle (Episodic Trace Formation → Semantic Consolidation → Reconstructive Recollection). |
| 5 | Hyper Triplet = ablation probe within Lineage A | **Same**, now mapped precisely to MemCell-level structure without MemScene consolidation and without Reconstructive Recollection. |

---

## User-provided design-convergence mapping (keep visible)

| User's prior 4-layer design | EverMemOS equivalent |
|---|---|
| Snowflake / Kafka signal input | Dialogue stream |
| Signal Registration | Episodic Trace Formation |
| Atomic node-set | **MemCell** (atomic fact + Foresight + episodic trace) |
| L0 facts | atomic fact (factoid triple) |
| L1 temporal · importance | **Foresight signal** (validity interval) |
| L2 context entity | episodic trace |
| L3 auxiliary / ontology labeling | **MemScene** (thematic consolidation) |
| Per-user memory | user-profile evolution |
| Memory Index serving | Reconstructive Recollection |
| Event Segmentation Theory | engram-inspired lifecycle |
| Temporal Context Model | MemScene-guided retrieval |

Interpretation: same problem + cognitive-science framing → same architecture. No architectural-layer novelty remains. Storage-layer engineering at Kafka-scale is the genuinely open axis, but **that belongs to a different project (graphdb-bench)**, not gr-test.

---

## LoCoMo-10 standings (as of 2026-04-19)

| System | Lineage | LoCoMo | Source |
|---|---|---|---|
| **EverMemOS** | A+B bridge | **93.05%** | press 2026-02-03, commercial |
| HyperMem (alone) | A (sub-module) | 92.73% | paper |
| MemMachine | B | 84.87% | MemMachine blog |
| GAAMA | B | 78.9% | paper |
| Tuned RAG | baseline | 75.0% | — |
| HippoRAG | B | 69.9% | as re-measured in GAAMA paper |
| Nemori | B | 52.1% | paper |
| A-Mem | B | 47.2% | paper |

---

## Research question (v4)

Same shape as v3, retargeted:

> **Decompose the ~15pp gap between GAAMA (78.9%) and EverMemOS (93.05%)**. How much comes from MemCell's atomic structure (vs GAAMA's fact-only), Foresight validity intervals (vs GAAMA's undifferentiated fact dates), MemScene consolidation (vs GAAMA's flat concept labels), and Reconstructive Recollection (vs GAAMA's semantic + PPR)?

Answerable by controlled intermediate design points.

---

## System roster (v4, six systems)

| Label | Role | Effect isolated | Status |
|---|---|---|---|
| **HippoRAG** | Lineage B baseline | flat triple floor | code via GAAMA inline or OSU-NLP-Group |
| **GAAMA** | Lineage B top | post-hoc concept + reflection | `external/gaama/` |
| **HyperGraphRAG** | Lineage A baseline | hyperedge without episodic memory | `external/hypergraph_rag/` |
| **Hyper Triplet (ours)** | Lineage A probe | typed qualifier hyperedges + episodic memory, NO MemScene | `systems/hyper_triplet/` |
| **HyperMem** | Lineage A conv-memory sub-module | MemCell + partial MemScene + retrieval | `external/everos/methods/HyperMem/` |
| **EverMemOS** | Lineage A+B endpoint / SOTA | full 3-stage lifecycle | `external/everos/methods/evermemos/` |

---

## Phase roster (v4)

### Phase 0 — Offline foundation (DONE)
- Repo scaffold, Python 3.11 uv, LoCoMo loader, eval skeleton, MockLLM, fixture-replay, gold regression fixture, graph + LTMCreator + retrieval + Pipeline adapter. CI. 98 tests green.
- Baselines cloned: `external/gaama/`, `external/hypergraph_rag/`, `external/hipporag/`, `external/everos/`.
- Reference notes for each under `docs/`.

### Phase 1 — Lineage B reproductions
- **1A** GAAMA 78.9% ± 2%p via `external/gaama/evals/locomo/run_*_eval.py` — mandatory.
- **1B** HippoRAG — optional; prefer GAAMA's inline HippoRAG port if present.

### Phase 2 — Lineage A baselines
- **2A** HyperGraphRAG on LoCoMo — adapt `external/hypergraph_rag/script_construct.py` for dialogue input. New measurement.
- **2B** EverMemOS via docker-compose:
    - `cd external/everos/methods/evermemos && docker-compose up -d` (MongoDB + ES + Milvus + Redis via WSL per user's feedback memory)
    - `uv sync && cp env.template .env && make run`
    - Health: `curl http://localhost:1995/health`
    - Ingest LoCoMo-10 through REST API (or SDK), record per-category accuracy
    - Target: reproduce 93.05% ± 2%p
- **2C** HyperMem standalone (inside EverOS monorepo) — sanity cross-check that 92.73% holds when run in isolation from the rest of EverMemOS.

### Phase 3 — Hyper Triplet probe
- Keep existing code (`systems/hyper_triplet/`), now labeled as the **MemCell-without-MemScene probe**.
- Wire it to GAAMA's SQLite store + PPR retrieval for a fair Lineage-A–at-triple-level datapoint.
- Do NOT add a MemScene or Reconstructive Recollection layer — that's the whole point of this probe (isolate what MemCell-level structure alone contributes).

### Phase 4 — Six-system comparison
Identical LoCoMo protocol. Per-category reports first; overall accuracy second. Paired bootstrap for significance.

### Phase 5 — Decomposition ablation
Isolate each EverMemOS component's contribution:

| Name | Config | Isolates |
|---|---|---|
| **D-base** | GAAMA | Lineage B reference |
| **D-mc** | Hyper Triplet | MemCell-like structure alone |
| **D-mc+scene** | Hyper Triplet + topic/episode clustering | add MemScene consolidation |
| **D-mc+foresight** | Hyper Triplet + validity-interval qualifier | add Foresight temporal layer |
| **D-recollection** | D-mc+scene + coarse-to-fine retrieval | add Reconstructive Recollection |
| **D-full** | EverMemOS | reference top |

If the delta concentrates on a specific stage, that's a publishable mechanistic finding even without a new system.

---

## Updated infrastructure requirements

Previous plan treated everything as SQLite + in-memory. v4 introduces Docker services for Phase 2B:

- **Docker services needed**: MongoDB 7.0, Elasticsearch 8.11, Milvus 2.5.2 (with etcd + MinIO backing), Redis 7.2.
- **Per user's feedback memory** ([`feedback_docker_db_wsl.md`](../../../.claude/projects/D--github-gr-test/memory/feedback_docker_db_wsl.md)), DB containers must be driven via WSL-backgrounded shell scripts.
- Scripts to add (Phase 2B prerequisite):
    - `scripts/evermemos-up.sh` — wraps `wsl -d Ubuntu-24.04 -- docker compose -f external/everos/methods/evermemos/docker-compose.yaml up -d` + readiness check
    - `scripts/evermemos-down.sh`
    - `scripts/evermemos-logs.sh`
    - `scripts/evermemos-status.sh`
- Ensure WSL is running first; user's environment has `Ubuntu-24.04` in Stopped state by default.

---

## Open questions for Phase 2B execution (all deferred until API keys active)

1. What does `external/everos/methods/evermemos/env.template` require exactly? (likely LLM + vectorise keys)
2. What's the expected wall-clock to ingest LoCoMo-10 through EverMemOS end-to-end?
3. Is there a benchmark entrypoint (`make eval-locomo`?) or must we iterate QA through the REST API?
4. Does HyperMem standalone in `methods/HyperMem/` depend on the surrounding evermemos services, or is it fully self-contained?
5. How much memory headroom does Milvus + ES + MongoDB + Redis need concurrently on a 3080 Ti host? May need resource limits.

These are all Phase 2B execution concerns, unblocked once OPENAI_API_KEY + vectorise credentials are wired.

---

## Paper positioning (v4)

Primary claim candidates, stronger than v3:

1. **"Decomposition of EverMemOS on LoCoMo"** — measure which stage (MemCell / MemScene / Reconstructive Recollection) contributes how many percentage points on each question category. Publishable even when EverMemOS itself isn't novel.
2. **"Lineage A vs B under identical protocol"** — prior work has not run HippoRAG, GAAMA, HyperGraphRAG, HyperMem, EverMemOS on the same harness with the same judge + same LLM + same budget. v4 produces that table.
3. **"Per-category specialisation"** — the mechanistic per-category shape (from the fact-vs-memory motivation) remains a defensible story: which stage helps which question type, and by how much.

No "we invented X" claim. Novelty is in the systematic comparison + decomposition.
