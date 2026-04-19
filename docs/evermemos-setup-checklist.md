# EverMemOS — Setup Checklist

Pre-flight items before Phase 2B (EverMemOS reproduction run). Audit of
`external/everos/methods/evermemos/env.template` + `docker-compose.yaml`.

## Required API keys / endpoints

| Variable | Purpose | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) | LLM backend for extraction + answer generation | Paper default `openrouter` → `x-ai/grok-4-fast`. Commercial API works. Comma-separated keys for rate-limit distribution supported. |
| `VECTORIZE_API_KEY` / `VECTORIZE_BASE_URL` | Embedding service | Default model `Qwen/Qwen3-Embedding-4B`. Either self-hosted vLLM (`http://localhost:8000/v1`) or DeepInfra. Uses "EMPTY" for vLLM. |
| `VECTORIZE_FALLBACK_*` | Embedding fallback | Optional DeepInfra fallback when primary is vLLM. |
| `RERANK_API_KEY` / `RERANK_BASE_URL` | Reranker service | Default model `Qwen/Qwen3-Reranker-4B`. vLLM (`http://localhost:12000/v1/rerank`) or DeepInfra. |
| `RERANK_FALLBACK_*` | Reranker fallback | Optional. |
| `TENANT_SINGLE_TENANT_ID` | Dev tenant namespace | Set to `t_whyjp` or similar for local runs. Prefixes storage keys. |

## Docker-compose services (no API keys, just resources)

| Service | Image | Host port | Data volume |
|---|---|---|---|
| MongoDB | `mongo:7.0` | 27017 | `mongodb_data` |
| Elasticsearch | `docker.elastic.co/elasticsearch/elasticsearch:8.11.0` | 19200 (9200 internal) | `elasticsearch_data` |
| Milvus standalone | `milvusdb/milvus:v2.5.2` | 19530 (19530 internal), 9091 | `milvus_data` |
| Milvus etcd | `quay.io/coreos/etcd:v3.5.5` | (internal) | `milvus_etcd_data` |
| Milvus MinIO | `minio/minio:RELEASE.2023-03-20T20-16-18Z` | 9000, 9001 | `milvus_minio_data` |
| Redis | `redis:7.2-alpine` | 6379 | `redis_data` |

MongoDB credentials (from compose): `admin / memsys123`.
No auth configured for ES / Milvus (local dev).

## Resource estimate (rough)

- ES single node with `ES_JAVA_OPTS=-Xms1g -Xmx1g` → ~1.5 GB RAM
- Milvus + etcd + MinIO → ~1-2 GB RAM
- MongoDB → ~0.5-1 GB RAM
- Redis → ~100 MB
- **Total docker-side: ~4-5 GB RAM**, plus disk for data volumes (~few GB)

If the user also self-hosts `Qwen3-Embedding-4B` + `Qwen3-Reranker-4B` via vLLM:
- 3080 Ti has 12 GB VRAM. Qwen3-4B models ~8 GB each at FP16. Cannot run both on the single 3080 Ti without quantisation.
- **Recommended path**: use DeepInfra (or OpenRouter's embedding-capable endpoints) for embeddings+rerank to avoid local GPU constraints. Drop `VECTORIZE_PROVIDER=vllm` → `deepinfra` (or `openai` if supported).

## Host prerequisites (Windows 11 + WSL)

- WSL distro: `Ubuntu-24.04` (currently in Stopped state — must start first)
- Docker Desktop with WSL2 backend, OR Docker-in-WSL
- Per user's feedback memory: scripts must wrap `docker compose` via
  `wsl -d Ubuntu-24.04 -- ...` and never block the foreground session.

## Python environment

- Python 3.11+ (EverMemOS uses `uv` as well)
- `uv sync` inside `external/everos/methods/evermemos/`
- Will produce a separate venv at `external/everos/methods/evermemos/.venv/`
  (different from our root `.venv/`; that's fine).

## Evaluation entrypoint (to investigate in Phase 2B)

- `Makefile` has `make run`. Need to read the Makefile to find if there's a
  `make eval-locomo` or equivalent. If not, wire LoCoMo through the REST API
  at port `1995` using our existing `BenchmarkRunner` with a new `EverMemOSPipeline`.

## Not required

- No CUDA toolkit / nvcc needed if using DeepInfra fallback
- No special root / sudo (assuming Docker is accessible to WSL user)
- No extra Python on host — EverMemOS runs in its own uv venv

## Pre-flight commands (once API keys are ready)

```bash
# 1. Start WSL distro if stopped (one-time per host boot)
wsl -d Ubuntu-24.04 -- echo "WSL ready"

# 2. Bring up EverMemOS infrastructure
bash scripts/evermemos-up.sh             # to be added

# 3. Prepare env
cp external/everos/methods/evermemos/env.template \
   external/everos/methods/evermemos/.env
# Edit .env to fill keys

# 4. Install EverMemOS python deps
( cd external/everos/methods/evermemos && uv sync )

# 5. Start EverMemOS application
( cd external/everos/methods/evermemos && make run )

# 6. Health check
curl http://localhost:1995/health
```

Phase 2B is unblocked once steps 3+5 complete.
