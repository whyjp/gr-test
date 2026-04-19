# hyper-triplet-bench

HippoRAG2 vs GAAMA vs **Hyper Triplet** on LoCoMo-10.

See `docs/hyper-triplet-implementation-plan.md` for the full plan.

## Layout

```
data/                   LoCoMo-10 dataset (gitignored; re-download via scripts/)
systems/
  hipporag2/            Phase 1 baseline
  gaama/                Phase 2 baseline (forks/reuses external/gaama)
  hyper_triplet/        Phase 3 proposal
eval/
  judge.py              LLM-as-judge
  metrics.py            Accuracy, per-category, mean+std
  runner.py             3-system evaluation driver
results/                Per-system scores (gitignored)
ablation/               A1..A5 variants (gitignored)
scripts/                Dataset + DB + bench utilities
external/               3rd-party clones, gitignored (e.g. external/gaama)
src/htb/                Shared library code (data loaders, types, utils)
tests/                  pytest — offline, no API required
docs/                   Plans + design notes
```

## Quickstart

```bash
# 1. Install Python 3.11 and create venv (uv manages everything)
uv python install 3.11
uv venv --python 3.11
source .venv/Scripts/activate   # bash on Windows

# 2. Sync dependencies
uv sync                          # core only
uv sync --extra llm              # + OpenAI (needs OPENAI_API_KEY)
uv sync --extra llm --extra embed

# 3. Dataset
bash scripts/fetch-locomo10.sh

# 4. Offline tests
uv run pytest
```

## Status

- [x] Repo scaffold
- [ ] Python 3.11 venv
- [ ] LoCoMo-10 dataset + loader
- [ ] GAAMA reference clone
- [ ] Offline eval skeleton
- [ ] Phase 1 HippoRAG2
- [ ] Phase 2 GAAMA
- [ ] Phase 3 Hyper Triplet
- [ ] Phase 4 full eval
- [ ] Phase 5 ablation
- [ ] Phase 6 cost analysis
