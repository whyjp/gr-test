# Phase F Runbook

Step-by-step workflow to execute the 12-preset ablation sweep + 6-system
comparison on LoCoMo-10 once the API unblock lands.

Run this file top-to-bottom; each step has a pre-condition check so failures
surface early.

---

## 0. Pre-flight

### 0.1 Local environment

```bash
# Expect Python 3.11 via uv
uv run python --version    # -> 3.11.x

# Expect LoCoMo-10 dataset present (2.8 MB)
ls -la data/locomo10.json  # -> 2,805,274 bytes

# If missing:
bash scripts/fetch-locomo10.sh
```

### 0.2 API credentials

Edit `.env` at repo root (gitignored):

```
OPENAI_API_KEY=sk-...
# Optional: route via OpenRouter (paper defaults fine with OpenAI direct):
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
EXTRACT_MODEL=gpt-4o-mini
JUDGE_MODEL=gpt-4o
```

Verify account has credit:

```bash
uv run --extra llm python -c "
import os; from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
client = OpenAI()
r = client.chat.completions.create(model='gpt-4o-mini', messages=[{'role':'user','content':'ping'}], max_tokens=5)
print('OK:', r.choices[0].message.content)
"
```

If this 429s with `insufficient_quota`: top up `platform.openai.com/billing`
minimum $5 or swap to OpenRouter.

### 0.3 Offline dry-run

Sanity-check the orchestrator without spending credit:

```bash
uv run python scripts/run_phase_f.py \
  --dry-run \
  --sample-ids conv-26 \
  --presets baseline no_community \
  --seeds 42 \
  --results-dir /tmp/phase_f_dry
```

Expect: `[phase_f] total_elapsed≈0.1s` + 2 JSON files + `_flat/summary.md`.
If this fails, do not proceed — the code path is broken, not the API.

---

## 1. Smoke test (real API, ~$0.10)

One conversation, baseline preset only, one seed.

```bash
uv run --extra llm python scripts/smoke_test_openai.py
```

This uses `HyperTripletPipelineV5` with `OpenAIAdapter` on conv-26,
evaluates 5 cat-1/cat-2 QA via `KeywordMockJudge`, writes
`results/smoke_test.json`.

Success criteria:
- No tracebacks
- `ingest_time < 3 minutes`
- At least 1 QA judged CORRECT (otherwise inspect the generated answers in
  the JSON — template_answerer may need tuning for your prompt variant)

---

## 2. Small targeted run (real API, ~$1-2)

Verify Phase F orchestrator against real API on a small slice:

```bash
uv run --extra llm python scripts/run_phase_f.py \
  --sample-ids conv-26 \
  --presets baseline no_community gaama_style_reflection_on \
  --seeds 42 \
  --results-dir results/phase_f_small
```

Inspect:
- `results/phase_f_small/42/locomo10_baseline_42.json` — overall accuracy
  should be meaningful (not 0 or 1, not NaN)
- `results/phase_f_small/_flat/summary.md` — tabular summary with 3 rows

Expected delta pattern:
- `gaama_style_reflection_on` should underperform `baseline`
  (empirical signature of Invariant #8 violation per §2 of
  `grouping-node-principle.md`). Exact magnitude TBD.

---

## 3. Full Hyper Triplet × 12 ablations × 3 seeds

```bash
uv run --extra llm python scripts/run_phase_f.py \
  --results-dir results/phase_f
```

Cost estimate:
- Extract: 10 convs × ~27 sessions × ~8 turns/chunk × ~200 tokens/prompt
  ≈ 216 extractions × $0.0003 ≈ **$0.07 per seed per preset**
- Judge: 10 convs × ~154 QA × ~3 rounds × ~100 tokens ≈ 4,620 judge calls
  per seed per preset × $0.01 ≈ **$46 per seed per preset** (gpt-4o)
- Full sweep (12 presets × 3 seeds): sharing extractions where possible —
  realistic $30–80.

Wall time: 1-2 hours for 1 seed; 3-6 hours for full 3-seed sweep.

---

## 4. Baseline comparison (incremental)

Each baseline adapter in `systems/baselines/` has a `_readiness_hint()`
that cites the exact glue needed. Recommended order:

### 4.1 GAAMA (Python, SQLite, cheapest)

Estimated 1-2 hours to wire. Cost per run similar to our Hyper Triplet.

### 4.2 HyperGraphRAG (Python, NanoVectorDB)

Similar effort; different input format — concatenate LoCoMo sessions
into narrative text per conv or per session.

### 4.3 HippoRAG 2 (Python, igraph)

Decide: use GAAMA's inline HippoRAG port (to match the 69.9% headline in
`baseline-numbers.md`) or run HippoRAG 2 as a new datapoint.

### 4.4 HyperMem (Python + vLLM GPU)

3080 Ti may not fit both Qwen3-4B embedders + reranker at FP16. Use
DeepInfra fallback unless 8-bit quantisation is set up.

### 4.5 EverMemOS (docker-compose + REST)

```bash
bash scripts/evermemos-up.sh      # starts MongoDB + ES + Milvus + Redis
( cd external/everos/methods/evermemos && uv sync && make run )
curl http://localhost:1995/health  # expect 200
```

Then implement `EverMemOSAdapter.ingest` / `.retrieve` as REST calls.

Once each adapter is live, add it to `scripts/run_phase_f.py`'s
`pipeline_factory` dispatch.

---

## 5. Post-sweep: populate the paper

1. `results/phase_f/_flat/summary.md` — ready table; copy into
   `docs/paper-outline-draft.md` §7.1.
2. Per-category deltas — run `format_ablation_report(sweep)` from
   `htb.eval.ablation_runner` (Python REPL) against the final
   `AblationSweepResult`; copy into §7.2.
3. Paired-bootstrap p-values — `AblationSweepResult.paired_bootstrap_against_baseline(preset_name)`
   for each preset; flag p<0.05 in the paper's delta table.

---

## 6. Troubleshooting

### 6.1 `openai.RateLimitError: 429 insufficient_quota`

Account has no credit. Top up billing OR swap `OPENAI_BASE_URL` to an
OpenRouter endpoint with a funded OpenRouter key.

### 6.2 `openai.APIConnectionError`

Network or endpoint issue. Retry; the OpenAIAdapter has built-in retries
(`max_retries=3`, `timeout=60s`).

### 6.3 `PipelineNotReadyError: [<name>] cannot ingest() — adapter is not ready.`

Baseline adapter stub is still a stub. Read the readiness hint in the
exception message — it cites the exact upstream module / config needed.

### 6.4 EverMemOS services fail to become healthy

```bash
bash scripts/evermemos-status.sh   # which container is red?
bash scripts/evermemos-logs.sh milvus-standalone  # or whichever
```

Milvus typically takes 60-90s to pass health; patience required on
first-ever boot (MinIO + etcd bucket init).

### 6.5 Judge returns CORRECT too generously

The LoCoMo-standard prompt is "generous: touches the same topic". If you
want stricter judging, override `OpenAIJudge.prompt_template` (not yet a
public API — currently requires patching `llm_judge.py`).

### 6.6 Results JSON files are huge

`records` array holds every per-QA record; a 10-conv run is ~500 records
per preset. If disk is tight, drop records from the JSON by editing
`serialize_system_result` (keep only the aggregate).

---

## 7. Kill switch

If a run goes wrong halfway, you can:

```bash
# Cancel the process; partial JSONs are valid per-seed
rm -rf results/phase_f/_flat/   # regenerate after next run
```

The per-preset JSONs under `results/phase_f/{seed}/` are independent —
you can rerun individual presets via `--presets <names>`.
