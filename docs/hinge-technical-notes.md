# HINGE & sHINGE — Technical Notes

Mathematical and architectural detail recovered from the HINGE (WWW 2020) and
sHINGE (TKDE 2024) papers. Both preprints are hosted by the authors at
exascale.info and parsed by a subagent on 2026-04-19.

## HINGE (Rosso, Yang, Cudré-Mauroux — WWW 2020)

### Data model

Hyper-relational fact `F = ((h, r, t), {(k_i, v_i) : i = 1..n})`.

- `(h, r, t)` is the **base triplet** — head entity, relation, tail entity.
- `{(k_i, v_i)}` are **qualifier key–value pairs**, where k_i is a relation and v_i is an entity.
- **n is unbounded** (the paper's explicit Goal III — handle arbitrary qualifier counts).
- Qualifiers are **untyped** at the entity level — no ontology assumed. This changes in sHINGE.

A classical triple fact is just the degenerate case `n = 0`.

### Architecture — two parallel CNN pipelines

Embeddings `h̄, r̄, t̄, k̄, v̄ ∈ ℝ^K` with `K = 100`.

**Triple CNN** (captures triple-wise relatedness):
- Stack `[h̄; r̄; t̄]` into `T ∈ ℝ^{3×K}`.
- One 2D conv with `n_f = 400` filters of size **3×3**.
- Output: `n_f` feature maps of size `K − 2`, flattened to `α ∈ ℝ^{n_f · (K-2)}`.

**Quintuple CNN** (captures quintuple-wise relatedness — joint triple + qualifier):
- For each `(k_i, v_i)`, stack `[h̄; r̄; t̄; k̄_i; v̄_i]` into `H_i ∈ ℝ^{5×K}`.
- One 2D conv with `n_f = 400` filters of size **5×3**.
  - First dim = 5 = quintuple axis (ensures the filter sees the full (h, r, t, k_i, v_i) jointly)
  - Second dim = 3 matches the triple CNN's second dim so outputs can be merged
- Output per qualifier: `β_i ∈ ℝ^{n_f · (K-2)}`.

**Merge** — the critical HINGE operator:
- Stack `α` with all `β_i` into `(n + 1) × n_f · (K − 2)` matrix.
- Apply **column-wise minimum** across the `n + 1` rows.
- A fully connected layer projects the merged vector to scalar score `φ`.

The column-wise minimum is the mathematical realisation of HINGE's correlation invariant: **a high score requires BOTH the triplet AND every qualifier to be plausible individually**. One bad qualifier pulls the whole fact's score down.

### Training

- Loss: softplus negative log-likelihood
  `L = Σ_{τ ∈ pos} log(1 + e^{−φ(τ)}) + log(1 + e^{φ(τ′)})`
  where `τ′` is a corrupted negative sample (one element replaced).
- Per-element corruption: randomly replace one of `h, t, r, k_i, v_i`.
- Optimiser: Adam, lr = 1e-4, batch size 128.
- Non-linearity: ReLU; batch norm after each conv layer.

### Benchmarks and headline results

| Benchmark | Metric | HINGE | Best prior |
|---|---|---|---|
| **WikiPeople** Original | head/tail MRR | 0.4763 | +13.2% over NaLP-Fix |
| WikiPeople Original | head/tail Hits@10 | 0.5846 | — |
| WikiPeople Original | relation MRR | 0.9500 | +15.1% over NaLP-Fix |
| **JF17K** Original | head/tail MRR | 0.4489 | — |
| JF17K Original | relation MRR | 0.9367 | +84.1% over NaLP-Fix |
| Vs triple-only baselines (TransE/H/R/D, RESCAL, DistMult, ComplEx, Analogy, ConvE) | head/tail MRR | — | +0.81% to +41.45% |

WD50K is NOT evaluated in HINGE — that benchmark comes later with StarE.

### Reference implementation

Public code: https://github.com/eXascaleInfolab/HINGE_code (cloned at `external/hinge/`).

## sHINGE (Rosso, Yang, Cudré-Mauroux — TKDE 2024)

### What "schema-aware" means

sHINGE keeps HINGE's fact model and adds a **second, parallel module** that runs over **entity-typed** versions of the fact.

New Definition 3 (Schema):
- For each entity `e`, the KG ontology provides a **type** (class label: human, university, physics, …).
- A triple `(h, r, t)` has a **typed-triple** `(type(h), r, type(t))`.
- A qualifier `(k_i, v_i)` becomes `(k_i, type(v_i))`. Note: k_i (relation) is NOT retyped.
- Each entity can have multiple types; sHINGE learns the top-`s` types per entity (best `s = 1`).
- Types sourced externally: Freebase `/type/object` for JF17K, Wikidata P31 for WikiPeople. No ontology authoring — reuse what the KG already has.

### Architecture delta

Mirror of HINGE's two CNNs:
- **Typed triple CNN**: 3×3 filter over `[h̄_type; r̄; t̄_type]` → schema triple vector `γ`.
- **Typed quintuple CNN**: 5×3 filter over `[h̄_type; r̄; t̄_type; k̄_i; v̄_{i,type}]` → schema quintuple vectors `δ_i`.
- Min-merge across `γ` and all `δ_i` produces the **schema channel** `h_s`.
- Original fact-side merge produces `h_f`.
- **Final score**: `h_f` and `h_s` are **concatenated** and passed through the final FC layer.

If `s > 1` (multiple types per entity), up to `m_1 · m_2` triple-wise and `m_1 · m_2 · n_c` quintuple-wise schema vectors are produced before min-pooling.

### Results (Table 2)

| Benchmark | Metric | sHINGE | Delta over HINGE |
|---|---|---|---|
| WikiPeople | head/tail MRR | 0.4780 | +0.36% |
| WikiPeople | relation MRR | 0.9977 | +5.0% |
| JF17K | head/tail MRR | 0.4582 | +2.1% |
| JF17K | relation MRR | 0.9961 | +6.3% |

Overall gains reported vs different baseline classes: **+19.1%** over triple-based, **+12.9%** over schema-aware, **+1.8%** over hyper-relational. Tested against 21 baselines (m-TransH, NaLP / NaLP-Fix, NeuInfer, HypE, HyperMLN, GETD, SIC, RAM, tNaLP, and more).

Schema ablation: the schema channel helps more on JF17K (higher hyper-relational fact share) than on WikiPeople.

### Reference implementation

Public code: https://github.com/RyanLu32/sHINGE (cloned at `external/shinge/`).

## Implications for our project

### 1. The column-wise min merge is a design principle, not just a math detail

HINGE's column-wise minimum bakes in the hyper-relational correlation at the scoring layer. In our retrieval (graph-based, not embedding-based), the analogue is:

> A fact's retrieval score should go up **only when the query matches the base triplet AND its qualifiers jointly**, not when either matches alone.

Our current `retrieval.py` uses sum-of-IDF on merged (fact + qualifier) tokens. This is **additive, not multiplicative / minimum-like**. An ablation probe should test: what if we require qualifier match as a hard gate on certain query types (e.g. "when X?" queries gate on time_reference qualifier presence)?

### 2. Schema awareness (sHINGE) is a known-good extension axis

sHINGE types entity **values** but not qualifier **keys**. Our design already types qualifier keys (location vs participant vs time_reference), which is stronger. Further, typing qualifier values (e.g. location = geographic place, participant = person) is future work — an sHINGE-style ablation.

### 3. Scoring-function-level correlation is separate from structural-level correlation

HINGE achieves correlation via the CNN + min-merge. Our system achieves it via **typed graph edges + MERGE on value identity**. These are orthogonal routes to the same invariant:
- HINGE: correlation lives in the scoring neural network.
- Ours: correlation lives in the graph topology.

A future evaluation could combine both — a neural scorer that reads our typed qualifier graph instead of flat `(h, r, t, k_i, v_i)` tuples.

### 4. Qualifier count is unbounded

HINGE's Goal III (arbitrary n). Our `Qualifiers` model currently uses 6 fixed slots (location, participants, activity_type, time_reference, mood, topic). This is **more restrictive than HINGE** — a bounded schema. Ablation question: does bounded schema help or hurt? It trades flexibility for type safety. LoCoMo's dialogue domain seems to cluster around these 6 types, but the choice should be justified.

### 5. Non-fact data: relation and key prediction

HINGE evaluates not just head/tail prediction but also **relation** and **key** prediction (what relation or qualifier key goes between two entities). Our current setup only does QA retrieval. If LoCoMo QA pairs are reframed into link-prediction form, we could run HINGE-style evaluation as well — but that's out of scope for v4.
