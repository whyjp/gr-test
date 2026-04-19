# EverMemOS Reference Notes

Clone: `external/everos/` @ HEAD `f06c303`
Upstream: https://github.com/EverMind-AI/EverMemOS (repo root is named `EverOS` inside)
Papers:
- EverMemOS: ["EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning"](https://arxiv.org/abs/2601.02163), 2026-01 — v1 2026-01-05, v2 2026-01-09, 11 authors from Chinese Academy of Sciences Institute of Information Engineering + EverMind AI + UCAS + Shanda
- HyperMem (sub-module): [arxiv 2604.08256](https://arxiv.org/abs/2604.08256), 2026-04
- EverMemBench: [arxiv 2602.01313](https://arxiv.org/abs/2602.01313)

Commercial launch: 2026-02-03 (EverMind AI). Cloud API is live. $80K Memory Genesis Competition 2026.

## Why this doc matters for gr-test

Previous plan versions treated HyperMem (92.73% LoCoMo) as SOTA. That was wrong:
- **EverMemOS scores 93.05% on LoCoMo** — the actual SOTA
- **HyperMem is EverMemOS's conversation-memory sub-module**, not a separate system
- The `external/hypergraph_rag/` and `external/hipporag/` clones are still valid as earlier-lineage baselines, but **EverMemOS is the real Lineage A+B endpoint** and must be the primary comparison target for v4.

## Repo layout (monorepo)

```
external/everos/
├── CLAUDE.md                  high-level quick-command overview
├── README.md                  main README
├── methods/
│   ├── evermemos/             EverCore — long-term memory OS (primary)
│   │   ├── docker-compose.yaml   full infrastructure
│   │   ├── Dockerfile
│   │   ├── Makefile
│   │   ├── config.json
│   │   ├── env.template
│   │   ├── pyproject.toml / uv.lock
│   │   └── src/
│   │       ├── run.py                  entry point
│   │       ├── agentic_layer/          memory_manager.py (core)
│   │       ├── memory_layer/
│   │       │   └── prompts/            EN + ZH prompt templates
│   │       ├── biz_layer/              business-rule layer
│   │       ├── core/                   domain types
│   │       ├── infra_layer/
│   │       │   └── adapters/input/api/ REST controllers (port 1995)
│   │       ├── service/
│   │       └── api_specs/              OpenAPI/schema definitions
│   └── HyperMem/              hypergraph-memory research prototype (thinner)
│       ├── hypermem/          package source
│       ├── README.md
│       ├── requirements.txt
│       └── scripts/
├── benchmarks/
│   ├── EverMemBench/          memory quality eval (arxiv 2602.01313)
│   └── EvoAgentBench/         longitudinal agent self-evolution
└── use-cases/
    ├── claude-code-plugin/
    └── game-of-throne-demo/
```

## Storage backends (confirmed from docker-compose.yaml)

| Service | Role |
|---|---|
| MongoDB 7.0 | primary document store (MemCell / MemScene payloads, tenants, user profiles) |
| Elasticsearch 8.11 | full-text index for lexical retrieval |
| Milvus 2.5.2 | vector DB (embeddings; uses etcd + MinIO as backing) |
| Redis 7.2 | cache layer |

**No graph DB.** This is structurally important: EverMemOS achieves SOTA via MongoDB + Milvus + ES + Redis, not a dedicated graph engine. Hyperedges live as document fields / Elasticsearch indexes / Milvus vectors, not as an explicit edge table.

REST API at port `1995`. Multi-tenant, fully async.

## Three-stage engram-inspired architecture (per paper §)

1. **Episodic Trace Formation** — dialogue stream is segmented into **MemCell**s. Each MemCell bundles:
    - atomic fact (factoid triple)
    - episodic trace (narrative/dialogue excerpt)
    - **Foresight signal** — a time-bounded plan/goal with an explicit validity interval
2. **Semantic Consolidation** — MemCells are online-clustered into thematic **MemScene**s. Scene-driven aggregation updates a per-user profile and distinguishes stable vs transient state.
3. **Reconstructive Recollection** — retrieval is MemScene-guided, coarse-to-fine (scene → cells → facts).

Retrieval per the HyperMem paper: RRF (BM25 + semantic) at each level, rerank, then expand via hyperedges to the next level. Embedding propagation with λ=0.5 between hyperedges and their incident nodes.

## Benchmark numbers (commercial press, 2026-02-03)

| Metric | EverMemOS | Baseline delta |
|---|---|---|
| LoCoMo (overall) | **93.05%** | vs HyperMem alone 92.73%, vs GAAMA 78.9% |
| Multi-hop reasoning | — | +19.7% vs strongest baseline |
| Temporal tasks | — | +16.1% |
| LongMemEval knowledge update | — | +20.6% |
| HaluMem recall | 90.04% | hallucination suppression |

## Running locally (from upstream CLAUDE.md)

```bash
cd external/everos/methods/evermemos
docker-compose up -d           # MongoDB + ES + Milvus + Redis
uv sync
cp env.template .env           # set LLM_API_KEY, VECTORIZE_API_KEY
make run                       # or: uv run python src/run.py
```

Health check: `curl http://localhost:1995/health`

**Not yet executed in this repo.** For Phase 2A/2B (v4 plan), we'll run this locally after API keys are provisioned.

## Mapping against user's prior design (from `hypergraph-memory-lineage.md` + user note)

| User's layer / concept | EverMemOS equivalent |
|---|---|
| Snowflake / Kafka signal input | Dialogue stream |
| Signal Registration | Episodic Trace Formation |
| Atomic node-set | **MemCell** (fact + Foresight + episodic trace bundle) |
| L0 facts | atomic fact (factoid triple) |
| L1 temporal · importance | **Foresight signal** (validity interval) |
| L2 context entity | episodic trace |
| L3 auxiliary / ontology labeling | **MemScene** (thematic consolidation) |
| Per-user memory | user-profile evolution |
| Memory Index serving | Reconstructive Recollection |
| Event Segmentation Theory | engram-inspired lifecycle |
| Temporal Context Model | MemScene-guided retrieval |

Interpretation: the architecture converged. Storage-layer engineering (Kafka-scale signal ingestion, write-heavy custom graph DB) remains the genuinely open axis.

## What this means for gr-test (Phase 2 pivot)

1. **Drop "implement HyperMem from paper spec"** — the code exists in `external/everos/methods/HyperMem/` and, more importantly, the full `methods/evermemos/` stack.
2. **EverMemOS becomes the primary Lineage-A+B endpoint.** Running it on LoCoMo-10 is a real pipeline with heavy deps (MongoDB/ES/Milvus/Redis) — **Phase 2 is a Docker-ful experiment**, not a pure Python reimplementation.
3. **Our Hyper Triplet stays as a mid-point probe** inside the lineage decomposition: MemCell-level structure (atomic fact + typed context) without MemScene consolidation and without Reconstructive Recollection.
4. **All upstream Docker infrastructure aligns with the user's prior preference** (docker via WSL-backgrounded scripts). Commands in scripts/ should wrap `docker compose up -d` with readiness checks.

## Open questions for Phase 2 execution (deferred until API keys active)

1. What does `env.template` require? (likely `LLM_API_KEY`, `VECTORIZE_API_KEY`, maybe reranker) — audit and record exactly.
2. How long does `make run` take to ingest LoCoMo-10 end-to-end? No published throughput.
3. Can evaluation be scripted against our `BenchmarkRunner` via the REST API (port 1995), or must we go through the SDK?
4. Does `methods/HyperMem/` run standalone, or does it depend on the evermemos services?
5. Is there a faster `make eval-locomo`-style command, or do we need to wire LoCoMo through the REST API per QA?
