# my_own Test — Enhanced Design Spec (HyperMem 흡수 기반)

> **⚙ 2026-04-19 반영 사항 (사용자 결정):**
> - **모듈명은 `hyper_triplet`로 통일** — 이 문서에서 `my_own`은 `systems/hyper_triplet/`로 읽는다.
> - **베이스라인에 EverMemOS (93.05%) 포함** — HyperMem은 EverMemOS의 conversation-memory 서브모듈. 실제 LoCoMo SOTA는 EverMemOS.
> - **HINGE 인용은 Rosso et al., WWW 2020** (`docs/hinge-north-star.md` 기준). §12의 `arxiv 2208.14322`는 HoLmES 논문이므로 HINGE 인용으로는 사용하지 않는다.
> - **LLM provider**: OpenAI (gpt-4o-mini extract, gpt-4o judge) — 기존 v4 계획 유지.
> - **LongMemEval / MuSiQee**: Phase 6로 미뤄둔다. v5는 LoCoMo 먼저 확정.
>
> 그 외 §3 설계 원칙 4개 (node-set / 4-layer / hyper-relational / star-native)와 §6 3-stage retrieval, §7 7-ablation, §8 3-seed 프로토콜은 v5 plan에 그대로 반영.
> 완전한 재작성된 계획은 [`hyper-triplet-implementation-plan-v5.md`](./hyper-triplet-implementation-plan-v5.md)에 있다.

---

> **이 문서의 목적**
> `gr-test` 벤치마크 레포의 `my_own` 테스트 구현을 개선하기 위한 설계 지시서다.
> Claude Code는 이 문서를 **single source of truth**로 삼아 my_own 구현을 갱신한다.
> HippoRAG2, GAAMA, HyperMem 세 베이스라인 대비 일관된 우위를 목표로 한다.

---

## 0. TL;DR (읽기 순서)

1. 벤치마크의 진짜 목적은 **"triple + α"가 triple-SOTA를 이긴다"는 실증**이다 — 논문 작성이 일차 목표가 아니다.
2. 현재 베이스라인: **HippoRAG2** (triple-SOTA), **GAAMA** (triple 확장형), **HyperMem** (hypergraph 3-level).
3. my_own의 설계 원칙 네 가지: **node-set atomic ingestion**, **4-layer functional separation**, **hyper-relational edges as first-class**, **star-native storage model**.
4. HyperMem은 우리 설계와 **방향은 같으나 지점이 다르다** — "post-hoc hypergraph grouping" vs "ingestion-time node-set separation". HyperMem의 3-level semantic hierarchy와 coarse-to-fine retrieval은 흡수 가능하다.
5. 이 문서의 §5~§8이 **Claude Code가 실제로 작업할 체크리스트**다.

---

## 1. Benchmark 목적과 제약

### 1.1 진짜 목적
- **증명할 명제**: "Triple-centric SOTA (HippoRAG2, GAAMA) 대비, triple + α (node-set + layer separation + hyper-relational + star storage) 조합이 retrieval 품질에서 일관된 우위를 낸다."
- 논문화는 이차적 목표다. 따라서 실험 설계는 **공정성보다 명제 검증 가능성**이 우선이지만, ablation은 리뷰 대비해 남겨둔다.

### 1.2 비목적 (명시적으로 하지 않는 것)
- 새로운 embedding 모델을 만들지 않는다. 기존 embedder를 재사용한다.
- LLM fine-tuning을 하지 않는다. Prompting과 structured output만 사용한다.
- Production-grade latency 최적화는 하지 않는다 (벤치마크는 functional 검증 우선).

### 1.3 작업 환경 가정
- Claude Code는 `gr-test` 레포 안에서 동작한다.
- 데이터는 점진적으로 주입되며, 병렬 벤치마크가 백그라운드에서 돈다.
- **주의**: my_own의 코드 변경은 기존 베이스라인 (HippoRAG2, GAAMA, HyperMem) 코드를 건드리지 않는다. my_own은 독립 모듈로 유지한다.

---

## 2. SOTA Landscape — 각 베이스라인의 정확한 위치

### 2.1 HippoRAG 2 (triple SOTA)
- **구조**: Schemaless KG + Personalized PageRank (PPR) 기반 retrieval.
- **추출**: OpenIE + NER로 triple 뽑고, passage와 연결.
- **강점**: Factual memory / multi-hop reasoning에서 GraphRAG, RAPTOR, LightRAG 대비 efficient.
- **약점**: Triple 이외의 정보(qualifier, context, temporal)는 구조적으로 표현 불가. Schemaless라 타입 제약이 없어 노이즈 많음.
- **내부 모델**: `(entity, relation, entity)` + passage link. End.

### 2.2 GAAMA (triple 확장형)
- **구조**: Triple을 후처리로 계층화 — base triple 위에 qualifier/metadata 얹음.
- **핵심 접근**: Post-hoc layering — ingest는 triple로 하고, layering은 별도 패스.
- **약점**: Ingestion과 layering이 분리되어 있어 context 손실 발생 (fact와 context가 따로 추출되면서 연결 정보 약화).
- **my_own과의 차별점 직격**: 우리는 **ingestion-time**에 fact+context+environment를 원자적으로 묶는다. Post-hoc 복원이 아니다.

### 2.3 HyperMem (hypergraph 3-level)
- **논문**: arXiv 2604.08256.
- **구조**: 3-level hierarchy — topics → episodes → facts, hyperedge로 episode와 fact를 그룹화.
- **Indexing**: 대화 스트림에서 episode boundary 감지 → topic hyperedge로 aggregation → fact 추출.
- **Retrieval**: Hybrid lexical-semantic index + coarse-to-fine (topics → episodes → facts).
- **벤치마크**: LoCoMo에서 Mem0, Zep, MemGPT 계열 대비 SOTA.
- **강점**: Hyperedge로 high-order association 표현 가능. Pairwise 한계 극복.
- **약점 (우리 목적 대비)**:
  - Hyperedge grouping이 **post-hoc** 스타일 (episode boundary 감지 후 grouping)
  - Topics/episodes/facts는 **semantic level** 분리이지, functional role 분리가 아님
  - Storage model은 일반 graph 위에 hyperedge reification (star-native 아님)
  - 대화형 long-term memory 대상이라, signal stream 관점의 설계는 아님

---

## 3. my_own의 설계 원칙 (절대 흔들리지 않는 4개)

### 3.1 Node-set Atomic Ingestion
하나의 signal/document에서 **fact triple + context entities + environment entities**를 **단일 원자 단위**로 추출한다. Post-hoc으로 재조립하지 않는다.

```
signal/doc → node_set {
  core_fact: (subject, predicate, object, edge_props)
  context: [ctx_entity_1, ctx_entity_2, ...]
  environment: [env_entity_1, env_entity_2, ...]
  participants: [participant_1, ...]
  temporal: {ts, valid_from, valid_until}
  importance: scalar
  source_ref: signal_id | doc_id | chunk_id
}
```

- **이게 HippoRAG2와의 차이**: HippoRAG2는 fact만 뽑고 context는 passage 링크로 약하게 연결. 우리는 context를 **구조적 노드**로 승격.
- **이게 GAAMA와의 차이**: GAAMA는 post-hoc으로 qualifier 붙임. 우리는 ingestion 시점에 atomic.
- **이게 HyperMem과의 차이**: HyperMem hyperedge는 episode들의 grouping(aggregation). 우리 node-set은 **단일 fact 단위의 원자 묶음**(separation).

### 3.2 4-Layer Functional Separation
Node-set이 ingest되면 즉시 네 개의 layer로 **functional role**에 따라 노드가 분배된다.

| Layer | 역할 | 예시 |
|---|---|---|
| **L0 — Fact** | Core atomic fact (subject, predicate, object) | `(user_42, equipped, legendary_sword)` |
| **L1 — Temporal/Importance** | 시간·중요도 메타 | `ts=2024-03-15T14:22`, `importance=0.87` |
| **L2 — Context** | Fact를 둘러싼 상황 | `location=dungeon_5`, `party_size=4` |
| **L3 — Auxiliary** | 파생 정보 (community assignment, embedding pointer 등) | `community_id=c_017`, `emb_ref=vec_4823` |

- L0~L3은 **동일한 node-set id**로 묶여 있어야 한다. Retrieval 시 layer를 선택적으로 포함/제외할 수 있어야 한다.
- **HyperMem topics/episodes/facts와의 차이**: HyperMem은 semantic granularity(큰 주제 → 작은 사실), 우리는 functional role(무엇(fact) / 언제·얼마나 중요(temporal) / 어디서·누구와(context) / 파생 계산(aux)).

### 3.3 Hyper-relational Edges as First-class
Edge는 `(src, type, dst)` triple이 아니라 `(src, type, dst, {qualifier_1, ..., qualifier_k})`로 저장한다.

- Qualifier는 edge property로 취급.
- HINGE 계보의 주장 수용: "edge properties are first-class, not second-class metadata."
- **Retrieval에서 qualifier가 score에 기여**해야 한다. 단순 필터가 아니라.

### 3.4 Star-native Storage Model
각 node-set은 **하나의 star subgraph**로 모델링 — center=L0 fact node, leaves=L1/L2/L3 nodes.
- Storage는 KV-friendly access pattern을 가정: `get(ns_id) → full star in O(k)`.
- Inter-star 연결은 shared entity(같은 user, 같은 item 등)를 통한 **indirect join**으로 처리.
- **이 원칙의 의의**: General graph DB(Neo4j, Memgraph)의 pointer/MVCC 오버헤드 없이, Redis Hash 한 번의 `HGETALL`로 star 전체가 나온다.

---

## 4. HyperMem에서 흡수할 것 / 흡수하지 않을 것

### 4.1 흡수할 것 (my_own에 통합)

#### (A) Coarse-to-fine Retrieval Strategy
HyperMem의 검색 전략 "topics → episodes → facts"의 발상을 **layer-aware retrieval**로 변환한다.

```
Stage 1: L2 context nodes에서 query semantic/lexical 매칭
         → 관련 node_set 후보 pool 확보
Stage 2: L1 temporal/importance로 후보 rank
         → top-K node_sets
Stage 3: L0 fact nodes 상세 retrieval + edge qualifier로 re-rank
         → final answer context
```

- HyperMem이 topics로 broad search 후 facts로 narrow down하듯, 우리는 **context로 broad → importance로 rank → fact로 exact**.
- **구현 포인트**: my_own의 retrieval 모듈을 3-stage pipeline으로 리팩터링.

#### (B) Hybrid Lexical-Semantic Index
HyperMem은 lexical (BM25 류)과 semantic (embedding) 인덱스를 병용한다. 우리도 그렇게 한다.

- **lexical index**: L0 fact의 subject/predicate/object entity 텍스트 + L2 context entity 텍스트에 BM25.
- **semantic index**: L0 fact의 자연어 verbalization + L2 context의 verbalization에 dense embedding.
- **두 스코어의 합성**: weighted sum (e.g., `0.4 * bm25 + 0.6 * cos`) — 튜닝 대상.

#### (C) Episode Boundary Detection (변형)
HyperMem은 대화 스트림에서 episode boundary를 감지한다. 우리는 **signal burst / temporal coherence**로 동일 패턴 적용.

- 입력이 document 모음이면: chunk의 temporal proximity와 entity overlap으로 node-set 경계 감지.
- 입력이 대화이면: HyperMem 방식을 그대로 차용하되, 결과물을 topic hyperedge가 아니라 **같은 context-cluster에 속하는 node-sets**로 묶는다.
- **구현 포인트**: `boundary_detector.py` 모듈 추가. 기존 my_own 파이프라인의 ingestion 단계 앞에 삽입.

#### (D) Topic-level Aggregation → Community Detection
HyperMem의 topic hyperedge aggregation은 우리의 **L3 community assignment**로 대체한다.

- Ingestion 직후 처리 대신 **background batch**로 community detection 실행 (Leiden / incremental LPA 등).
- 감지된 community id를 L3 auxiliary 노드에 기록.
- Retrieval stage 1에서 community id를 broad filter로 사용 가능 (HyperMem의 topic-level coarse search에 대응).

### 4.2 흡수하지 않을 것 (의도적 배제)

- **Hyperedge as first-class object**: HyperMem은 hyperedge를 직접 객체로 다룬다. 우리는 hyperedge를 **edge qualifier 세트**로 내재화한다 (§3.3). 이건 star-native storage 모델과 호환되지 않기 때문이다.
- **3-level semantic hierarchy를 그대로 가져오기**: Topics/episodes/facts 구분은 대화 도메인 특화. 우리는 L0~L3 functional separation을 유지한다.
- **Conversation-only assumption**: HyperMem은 대화 turn 기반. 우리는 document + signal + conversation 모두 처리 가능해야 한다.

---

## 5. my_own 구현 아키텍처 (Claude Code 작업 대상)

### 5.1 전체 파이프라인

```
[Input: doc / signal / conversation]
            ↓
  [Boundary Detector]        ← HyperMem 흡수 (4.1-C)
            ↓
  [Node-set Extractor]       ← §3.1, LLM + structured output
            ↓
  [Layer Assigner]           ← §3.2, rule-based + LLM verifier
            ↓
  [Edge Qualifier Builder]   ← §3.3, hyper-relational properties
            ↓
  [Star Storage Writer]      ← §3.4, KV-friendly layout
            ↓
  [Background Workers]
    ├─ Community Detector    ← HyperMem 흡수 (4.1-D)
    ├─ Importance Scorer     ← ACT-R / Bayesian surprise
    └─ Index Builder         ← 4.1-B, lexical + semantic
            ↓
  [Retrieval Engine]         ← HyperMem 흡수 (4.1-A), 3-stage
```

### 5.2 모듈별 파일 구조 (권장)

```
my_own/
├── extractor/
│   ├── boundary_detector.py       # 신규 (HyperMem 스타일)
│   ├── node_set_extractor.py      # 신규, LLM prompting
│   ├── layer_assigner.py          # 신규, L0~L3 분배
│   └── edge_qualifier_builder.py  # 신규, hyper-relational props
├── storage/
│   ├── star_store.py              # 신규, KV-native abstraction
│   ├── redis_backend.py           # Phase 2 (아직 FalkorDB면 falkor_backend.py)
│   └── schema.py                  # node-set / L0~L3 schema
├── index/
│   ├── lexical_index.py           # BM25 on L0/L2 text
│   ├── semantic_index.py          # embedding on verbalizations
│   └── hybrid_scorer.py           # 가중합 + re-rank
├── retrieval/
│   ├── stage1_broad.py            # L2 context 기반 pool
│   ├── stage2_rank.py             # L1 temporal/importance rank
│   ├── stage3_exact.py            # L0 fact + qualifier re-rank
│   └── pipeline.py                # 3-stage orchestration
├── background/
│   ├── community_detector.py      # Leiden / incremental LPA
│   └── importance_scorer.py       # activation-based
├── eval/
│   ├── adapter.py                 # LoCoMo / LongMemEval / 자체 어댑터
│   └── ablation_runner.py         # §7
└── config.py                      # 모든 하이퍼파라미터
```

### 5.3 Node-set 스키마 (예시)

```python
# my_own/storage/schema.py

from dataclasses import dataclass, field
from typing import Any

@dataclass
class L0Fact:
    subject: str
    predicate: str
    object: str
    edge_qualifiers: dict[str, Any] = field(default_factory=dict)
    # qualifier 예: {"confidence": 0.9, "valid_from": "2024-01-01", ...}

@dataclass
class L1TemporalImportance:
    timestamp: str
    valid_from: str | None = None
    valid_until: str | None = None
    importance: float = 0.0
    # activation_base: float, bayesian_surprise: float 등 서브 스코어도 여기

@dataclass
class L2Context:
    context_entities: list[dict]   # [{"type": "location", "id": "..."}, ...]
    environment: dict[str, Any]    # 자유 key-value
    participants: list[str] = field(default_factory=list)

@dataclass
class L3Auxiliary:
    community_id: str | None = None
    embedding_ref: str | None = None
    source_ref: str | None = None
    derived_metrics: dict[str, float] = field(default_factory=dict)

@dataclass
class NodeSet:
    ns_id: str                     # 원자 단위 식별자
    fact: L0Fact
    temporal: L1TemporalImportance
    context: L2Context
    aux: L3Auxiliary
```

### 5.4 Extractor Prompt Skeleton

```
You are extracting a single atomic node-set from the input.
Do NOT decompose the input into multiple triples.
Extract ONE core fact and all its surrounding context as ONE atomic unit.

Output JSON with the following structure:
{
  "core_fact": {"subject": "...", "predicate": "...", "object": "...", "qualifiers": {...}},
  "context_entities": [{"type": "...", "id": "...", "role": "..."}, ...],
  "environment": {"...": "..."},
  "participants": ["..."],
  "temporal": {"timestamp": "...", "valid_from": "...", "valid_until": "..."}
}

Rules:
- If the input contains multiple independent facts, return multiple node-sets (one per JSON object).
- If context is uncertain, output empty arrays rather than hallucinating.
- `qualifiers` MUST include edge metadata (source, confidence, temporal bound on the relation itself).
- Every entity referenced must have a stable `id` (use deterministic hash if none exists).
```

---

## 6. Retrieval Pipeline 상세 (Claude Code 작업 핵심)

### 6.1 Stage 1: Broad (context-level)
- Input: user query.
- 동작:
  1. Query에서 entity/keyword 추출 (LLM or NER).
  2. L2 context index (lexical + semantic)로 매칭되는 node-set 후보 pool 확보.
  3. L3 community_id로 같은 커뮤니티 속한 node-sets 확장 (HyperMem의 topic-level 발상).
- Output: 후보 node-set ids (top-N, N=100~500 default).

### 6.2 Stage 2: Rank (temporal/importance)
- Input: 후보 node-set ids.
- 동작:
  1. 각 node-set의 L1 importance + query와의 temporal alignment로 점수 계산.
  2. Temporal alignment: query가 "언제"를 함의하면 L1.timestamp와의 거리로 감쇠.
  3. Importance: base activation + frequency + recency 조합.
- Output: top-K node-sets (K=10~30 default).

### 6.3 Stage 3: Exact (fact + qualifier re-rank)
- Input: top-K node-sets.
- 동작:
  1. L0 fact의 subject/predicate/object와 query entity 매칭.
  2. Edge qualifier를 feature로 사용한 re-rank (e.g., confidence 낮은 건 강등).
  3. Answer generation context에 필요한 layer 조립 — 기본적으로 L0 + L2(context) 포함, L1/L3은 optional.
- Output: final context for LLM answer generation.

### 6.4 구현 주의사항
- 각 stage는 **독립 모듈**이어야 한다. Ablation에서 stage를 on/off 할 수 있어야 한다.
- Stage 2의 importance weight는 **하이퍼파라미터**. `config.py`에 노출.
- Stage 3의 qualifier feature는 기본적으로 {confidence, source_type, temporal_validity} 3개만 우선 구현.

---

## 7. Ablation 설계 (명제 검증용)

각 ablation은 my_own의 일부 기능을 꺼서 성능 변화를 측정한다. 이게 "어느 원칙이 실제로 기여했는가"를 증명한다.

| Ablation | 끄는 것 | 검증하려는 것 |
|---|---|---|
| `no_node_set` | Node-set 대신 독립 triple로 추출 | Node-set atomic ingestion의 가치 |
| `no_layer_separation` | L0~L3을 flat하게 합침 | 4-layer functional separation의 가치 |
| `no_hyper_edge` | Edge qualifier 제거, plain triple edge | Hyper-relational의 가치 |
| `no_star_storage` | General graph 저장 (e.g., NetworkX) | Star-native storage 효과 (latency) |
| `no_stage1` | Retrieval stage 1 스킵, stage 2에서 전체 pool | Context-level broad search의 가치 |
| `no_hybrid_index` | Lexical만 또는 semantic만 | Hybrid index의 가치 |
| `no_community` | L3 community_id 사용 안 함 | Community detection의 검색 기여도 |

**리포트 포맷**: 각 ablation 결과를 `eval/results/ablation_{name}.json`에 저장. 최종 비교표는 `eval/results/ablation_summary.md`로 자동 생성.

---

## 8. 평가 지표와 벤치마크 데이터

### 8.1 필수 벤치마크
- **LoCoMo**: HyperMem이 쓴 것과 동일. Single-hop / multi-hop / temporal / open-domain 4 카테고리.
- **LongMemEval**: Multi-session conversation. HippoRAG2와 비교용.
- **MuSiQue or HotpotQA**: Multi-hop QA. GAAMA와 비교용.

### 8.2 Metric
- **Primary**: LLM-as-a-judge accuracy (HyperMem과 동일 프로토콜). GPT-4o-mini or Claude Sonnet 사용.
- **Secondary**:
  - F1 (token-level)
  - Exact Match
  - Retrieval Recall@K (K=1, 3, 5, 10)
  - End-to-end latency (ms)
  - Ingestion throughput (node-sets/sec)

### 8.3 비교 대상 체크리스트
매 실험에서 다음 네 결과를 **반드시 같은 테이블에** 보고:

1. HippoRAG2 (triple SOTA baseline)
2. GAAMA (triple 확장형)
3. HyperMem (hypergraph 3-level)
4. **my_own (ours)**

+ §7의 ablation 결과들.

### 8.4 통계적 유의성
- 각 실험은 random seed 3개로 반복 (seed=[42, 1337, 2024]).
- 평균 + std 모두 리포트.
- my_own의 우위 주장은 **seed 3개 전부에서 베이스라인 대비 개선**이어야 한다.

---

## 9. 점진적 작업 순서 (Claude Code용 우선순위)

아래 순서대로 작업한다. 각 단계는 **다음 단계로 넘어가기 전에 테스트 통과**가 조건이다.

### Phase A — 스키마와 Storage (1~2일 작업 추정)
1. `my_own/storage/schema.py` 작성 (§5.3 기준)
2. `my_own/storage/star_store.py`: in-memory (dict 기반) star_store 먼저 구현. 나중에 FalkorDB backend 붙일 것.
3. Unit test: node-set CRUD (create, get full star, update, delete)

### Phase B — Extractor (2~3일)
1. `boundary_detector.py`: document/conversation 공통 인터페이스
2. `node_set_extractor.py`: §5.4 prompt 기반 LLM 호출
3. `layer_assigner.py`: 규칙 + LLM verifier로 L0~L3 분배
4. `edge_qualifier_builder.py`: qualifier 추출/정규화
5. Integration test: raw document → node-set 저장까지

### Phase C — Indexing과 Retrieval (3~5일)
1. `index/lexical_index.py`, `index/semantic_index.py`, `index/hybrid_scorer.py`
2. `retrieval/stage1_broad.py`, `retrieval/stage2_rank.py`, `retrieval/stage3_exact.py`, `retrieval/pipeline.py`
3. 간단한 self-made QA 5개로 smoke test

### Phase D — Background (1~2일)
1. `community_detector.py`: NetworkX 기반 Leiden으로 start. Production용 incremental은 나중.
2. `importance_scorer.py`: 단순 base activation으로 start (log(frequency) + recency decay).

### Phase E — Evaluation (지속)
1. `eval/adapter.py`: LoCoMo adapter 먼저 (HyperMem 레포 참고)
2. LongMemEval, MuSiQue adapter 순차 추가
3. `eval/ablation_runner.py`: §7의 ablation 자동화
4. 결과 비교표 자동 생성

### Phase F — HyperMem 흡수 검증 (Phase C 이후 언제든)
- §4.1의 네 가지 흡수 요소가 **각각 기여하는지** mini-ablation으로 확인.
- HyperMem 자체와 my_own의 head-to-head에서 우위가 있는지 체크.

---

## 10. 주의사항 (반드시 지킬 것)

### 10.1 하지 말 것
- **새 embedding 모델 학습 금지**. 기존 sentence-transformers / bge / openai embedding 중 선택.
- **HippoRAG2/GAAMA/HyperMem 코드 건드리지 말 것**. 공정 비교가 깨진다.
- **LoCoMo 데이터셋을 sanity로 쓰고 튜닝 대상으로 쓰지 말 것**. Test leak 방지.

### 10.2 반드시 할 것
- 모든 하이퍼파라미터는 `my_own/config.py`에서 관리. 매직 넘버 금지.
- 모든 LLM 호출은 `my_own/llm_client.py`를 거친다 (재시도, 로깅, 캐싱 일관화).
- Node-set 하나마다 `ns_id`는 **결정론적 해시**로 생성 (reproducibility).
- 각 Phase 종료 시 **결과 스냅샷을 git commit**. Ablation 재현 가능해야 한다.

### 10.3 보고 포맷
- 모든 실험 결과는 `eval/results/{dataset}_{system}_{seed}.json`.
- 요약은 `eval/results/summary.md`에 테이블로.
- **매 실험마다 반드시** my_own vs HippoRAG2 vs GAAMA vs HyperMem 4자 비교 포함.

---

## 11. 의사결정이 필요한 open question (영주에게 확인 필요)

Claude Code는 아래 항목이 모호하면 **진행을 멈추고 질문**한다:

1. **Storage backend 결정 시점**: Phase A에서 in-memory로 시작하지만, FalkorDB backend 붙이는 타이밍은?
2. **LLM provider**: Bedrock Claude (APAC) 쓰는가, OpenAI? 비용·latency 측면 어느 쪽 우선?
3. **데이터 주입 병렬성**: 현재 병렬 주입 중인데, my_own이 incremental ingest를 지원해야 하는가, batch로 일시 정지하고 주입하는가?
4. **Ablation 우선순위**: §7의 7개 전부 돌릴지, 핵심 3개(no_node_set / no_layer_separation / no_hyper_edge)만 먼저 돌릴지?
5. **HyperMem 흡수 강도**: §4.1의 네 요소 중 구현 비용이 큰 게 있으면(특히 community detector), 우선순위 어떻게?

---

## 12. 참고 문서 (이 문서를 읽기 전/후로 볼 것)

### 이 문서와 함께 읽어야 하는 프로젝트 내부 문서
- `docs/memory-graph-mmorpg.html` (v3.2) — 전체 아키텍처 rationale
- `docs/GAAMA-comparison.md` (있다면) — GAAMA와의 상세 대조
- 한국어 novelty assessment reports — 방법론적 SOTA positioning

### 외부 참고 (이 문서 작성의 근거)
- HippoRAG 2: "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models" (ICML 2025)
- HippoRAG 1: NeurIPS 2024
- HyperMem: arXiv:2604.08256 (3-level hypergraph, LoCoMo SOTA)
- HINGE: "Learning Representations for Hyper-Relational Knowledge Graphs" (CIKM 2023, arXiv:2208.14322)
- HOLMES: "Hyper-Relational Knowledge Graphs for Multi-hop QA using LLMs" (ACL 2024)
- HGMem: "Improving Multi-step RAG with Hypergraph-based Memory" (arXiv:2512.23959)
- Graph-based Agent Memory Survey: arXiv:2602.05665

### 내부 논리 일관성 체크리스트
이 문서를 실행에 옮기기 전에 Claude Code는 다음을 확인한다:
- [ ] §3 네 원칙 중 하나라도 코드에서 지켜지지 않으면 그 이유를 명시
- [ ] §4.1의 흡수 요소들이 모두 구현 되었는지, 안 되었다면 Phase 어디에 배치되어 있는지
- [ ] §7 ablation이 모두 실행 가능한 상태인지
- [ ] §8 비교 대상 4자(HippoRAG2, GAAMA, HyperMem, my_own)가 동일 프로토콜로 실행되는지

---

## 13. 최종 명제 (한 줄로)

> **"Triple-SOTA(HippoRAG2)와 triple-확장(GAAMA), hypergraph 확장(HyperMem)이 각자 답한 부분 문제들을, node-set atomic ingestion + 4-layer functional separation + hyper-relational edges + star-native storage 조합이 하나의 통합된 답으로 풀어 낸다. 이 조합은 context-preserving retrieval 품질에서 세 베이스라인을 일관되게 이긴다."**

이 명제가 실증될 때까지 my_own은 계속 개선된다. 이 문서는 그 개선의 single source of truth다.
