# Grouping Node 원리 — 메모리 시스템 아키텍처의 핵심 통찰

> 작성: 2026-04-19
> 목적: `graphdb-bench` 및 memory graph 프로젝트의 아키텍처 결정 원리 확정
> 선행 문서: `hypergraph-memory-lineage.md` (시스템 계보 정리)
>
> **핵심 명제**: 메모리 시스템의 성능 상한선은 "triplet vs hypergraph"라는 표면 구문이 아니라, **어떤 종류의 grouping node를 갖는가**로 결정된다.

---

## 1. 핵심 통찰

지난 6년간 메모리/KG 시스템 연구를 관통하는 실제 축은 다음과 같다.

**진짜 질문**: atomic fact들을 "어떻게 묶을 것인가"
**진짜 답**: Grouping node의 유형과 그 의미론적 속성이 전체 시스템의 ceiling을 결정한다

Triplet vs hypergraph는 수학적 형식의 차이일 뿐이고, 실제로는 **reification으로 triplet 안에서도 hypergraph의 표현력을 복원 가능**하다. 성능 차이를 만드는 것은:

1. Grouping node가 **존재하는가**
2. Grouping 기준이 **의미론적으로 올바른가** (topic + time vs source vs 요약)
3. Grouping 프로세스가 **무손실인가** (분류 vs 압축)

---

## 2. Grouping Node 유형 비교 (핵심 표)

| 시스템 | Grouping Node 유형 | 생성 방식 | 무손실 여부 | LoCoMo | Grouping 평가 |
|--------|-------------------|----------|-----------|--------|-------------|
| **HippoRAG** (NeurIPS'24) | 없음 (phrase 노드만, PPR로 확산) | — | — | 69.9% | Grouping 부재 |
| **HippoRAG 2** (ICML'25) | **Passage** (source 기반) | 문서 경계 = 자동 | 무손실 (단, source-based) | N/A* | Source 기준이라 semantic aggregation 실패 |
| **GAAMA** (2026-03) | **Concept** + **Reflection** | Concept=LLM 분류 / Reflection=LLM 요약 | 부분 손실 (reflection 때문) | 78.9% | Concept은 옳고 reflection이 ceiling 유발 |
| **HyperMem** (2026-04) | **Topic** + **Episode** | LLM boundary detection (분류) | 무손실 | 92.7% | Topic × Time 2축 reification |
| **EverMemOS** (2026-01) | **MemScene** + **Foresight** | Online thematic clustering (분류) | 무손실 | 93.0% | Semantic consolidation + temporal validity |
| **영주님 설계** | **L1 temporal** + **L2 context** + **L3 ontology** | 다축 동시 reification (분류) | 무손실 | TBD | **3축 reification** — 가장 표현력 높음 |

*HippoRAG 2 공식 LoCoMo 점수는 미보고. 저자들이 MuSiQue/2Wiki/HotpotQA 중심으로 평가.

### 표 해석의 핵심 패턴

**패턴 1 — Grouping node 유무로 인한 큰 점프**
- HippoRAG (69.9%) → GAAMA (78.9%): +9pp. Concept grouping 도입 효과.
- GAAMA (78.9%) → HyperMem/EverMemOS (93%대): +14pp. Reflection을 제거하고 순수 구조적 grouping만 남김.

**패턴 2 — Grouping 기준의 정확성이 ceiling을 결정**
- Source 기반 (HippoRAG 2): 문서 단위 grouping. "7개 세션에 걸친 tournament 수 집계" 같은 cross-session aggregation 실패.
- Topic + Time 2축 (HyperMem): 의미적 경계와 시간적 경계를 동시에 반영. 90%대 돌파.
- Semantic + Temporal validity (EverMemOS): MemScene (의미) + Foresight (시간 유효기간). 현재 SOTA.

**패턴 3 — Grouping 생성 방식의 손실성 여부**
- LLM이 **새로운 콘텐츠를 압축 생성**하는 방식(GAAMA의 reflection) → 손실, ceiling 유발
- LLM이 **기존 fact의 membership만 판정**하는 방식(HyperMem/EverMemOS의 topic detection) → 무손실

---

## 3. 지배 원리

위 세 패턴에서 하나의 원리가 도출된다.

> **LLM을 "새로운 압축된 표현을 생성"하는 데 쓰지 말고, "기존 fact가 어떤 구조에 속하는지 분류"하는 데만 써라.**

### 두 역할의 구조적 차이

| 역할 | LLM 사용 방식 | 정보 흐름 | 대표 사례 | 결과 |
|------|-------------|---------|---------|------|
| **Encoder** (새 콘텐츠 생성) | LLM이 fact들을 입력받아 새 요약 텍스트 생성 | fact → [압축] → 요약 노드 | GAAMA reflection, 일반적인 GraphRAG community summary | **손실**. 원본 nuance 유실, 불일치 누적 |
| **Classifier** (membership 판정) | LLM이 fact와 후보 grouping을 보고 소속만 결정 | fact → [분류] → "이 topic/episode에 속함" 라벨 | HyperMem topic detection, EverMemOS scene clustering | **무손실**. 원본 fact는 그대로 보존 |

### 이 원리가 HINGE 논문의 일반화

HINGE (WWW'20)가 link prediction에서 보인 결론 — "transformation(triplet화)으로 인한 정보 손실이 최대 29.3%" — 의 memory 도메인 버전이 바로 이 원리다:

- HINGE: 구조 변경(transformation)이 손실을 만든다 → hyperedge를 구조적으로 보존하라
- 이 원리: 콘텐츠 압축(encoding)이 손실을 만든다 → grouping은 membership 판정으로만 하라

**동일한 원리의 두 표현**이며, 둘 다 "lossy compensation" vs "structure preservation"의 구조적 우열을 증명한다.

---

## 4. 영주님 구현에의 적용

### 4.1. L0-L3 레이어를 원리에 맞게 재정의

기존 설계를 원리 관점에서 재해석하면:

```
L0 (facts):     atomic MemCell 노드 — 원본 그대로 보존
                LLM 역할: 추출(extraction)만

L1 (temporal):  validity_interval 가진 temporal scope 노드
                LLM 역할: fact의 시간 boundary 판정 (분류)
                생성 방식: signal burst detection + semantic drift threshold

L2 (context):   topic/entity cluster 노드
                LLM 역할: fact의 topic membership 판정 (분류)
                생성 방식: online topic clustering

L3 (ontology):  Palantir-style Object/Property/Link 스키마 노드
                LLM 역할: fact의 ontology category 판정 (분류)
                생성 방식: 사전 정의 스키마 + membership assignment
```

**모든 레이어에서 LLM은 classifier로만 작동**. Reflection/summary 생성은 어느 레이어에도 넣지 않는다.

### 4.2. Reified Super-Node 패턴으로 구현

Memgraph 같은 일반 graph DB에서도 hypergraph 표현력 확보 가능:

```
L0 fact 노드 ─[belongs_to]→ L1 temporal_scope 노드
L0 fact 노드 ─[belongs_to]→ L2 context 노드
L0 fact 노드 ─[belongs_to]→ L3 ontology 노드

L1 × L2 × L3 교집합 → 자동 생성되는 L4 MemScene 노드 (선택)
```

HyperMem/EverMemOS와 동일한 표현력을 triplet-friendly 구조로 확보. 별도의 hypergraph DB 불필요.

### 4.3. 기존 SOTA와의 포지션

| 축 | HyperMem/EverMemOS | 영주님 설계 | 차이 |
|----|-------------------|-----------|------|
| Grouping 축 수 | 2축 (topic + time) | 3축 (temporal + context + ontology) | **+1축, 표현력 높음** |
| Ontology 통합 | 없음 | Palantir-style schema | **정형 도메인 지식 활용** |
| 입력 rate | 대화 턴당 1회 | Kafka streaming (1k+ msg/sec) | **2-3 orders of magnitude 차이** |
| Storage | 기존 DB 조합 (API 레이어) | Custom graph DB (write-heavy 최적화) | **엔지니어링 우위** |
| 도메인 | 일반 dialogue | MMORPG player behavior | **고빈도 agent signal** |

**개념 레벨에서는 EverMemOS 추종, 구현 레벨에서는 차별화.** 이 포지션이 정확하다.

---

## 5. 추출 전략: Triplet + α 원자 추출

Section 3의 지배 원리("LLM-as-classifier")를 ingestion 파이프라인에 적용하면, 기존 시스템들과 다른 추출 패턴이 자연스럽게 도출된다. **본 프로젝트의 구현 방식이 이 지점에서 기존 SOTA보다 한 걸음 더 나아간다.**

### 5.1. 패턴의 정의

원본 log 또는 document 한 단위에서 **triplet과 그 맥락(α)을 원자적으로 동시 추출**하고, α를 각 레이어의 reified node로 즉시 분산한다.

```
입력: log line / document chunk 1개
  ↓ (단일 추출 pass)
출력: (base_triplet, α_temporal, α_context, α_ontology)
  ↓ (즉시 분산)
L0 ← base_triplet
L1 ← α_temporal  → temporal_scope 노드
L2 ← α_context   → context 노드
L3 ← α_ontology  → ontology 노드

모든 α 노드는 각 레이어 내에서 자체 triplet 구조를 가지며,
L0 fact와는 belongs_to edge로 연결된다.
```

**핵심 구현 관찰**: 각 레이어 내부는 triplet 구조로 유지하되, 레이어 간 reification edge로 hypergraph 표현력을 확보. **전용 hypergraph DB 없이 Memgraph 등 표준 graph DB로 구현 가능**하며, 이것이 custom graph DB 설계의 근거가 된다.

### 5.2. 기존 SOTA와의 비교

| 시스템 | 추출 단계 수 | 레이어 할당 시점 | Streaming 적합도 | 도메인 제약 |
|--------|-------------|-----------------|----------------|-----------|
| GAAMA | 3 (fact → concept → reflection) | Batch consolidation | 낮음 | 범용 |
| HyperMem | 2 (fact → topic/episode boundary) | Near-real-time clustering | 중간 | 범용 |
| EverMemOS | MemCell 번들 추출 + 별도 MemScene clustering | Hybrid | 중간 | 범용 |
| **본 프로젝트** | **1 (triplet + α 동시)** | **Ingestion time 즉시** | **높음** | **Ontology 사전 정의 필요** |

### 5.3. 구조적 이점

**Batch consolidation 단계 제거**
- HyperMem/GAAMA는 topic 경계 감지를 위해 fact를 축적한 뒤 clustering → 배치성 내재
- Triplet + α 즉시 분산은 각 log 입력이 self-contained → **streaming native**
- 이것이 Kafka-scale signal rate에서 결정적 차이를 만듦

**LLM call 감소**
- Extract-then-cluster: 추출 LLM + 분류 LLM + (GAAMA의 경우) reflection 생성 LLM
- Triplet + α 원자 추출: 단일 LLM call에서 triplet과 α 동시 추출
- MMORPG signal처럼 구조화된 입력에서는 α의 일부(timestamp, arena_id, character_id)가 **파싱만으로 획득 가능** → LLM 부담 추가 감소

**원본 맥락 손실 최소화**
- Extract-then-cluster: fact를 원본 맥락에서 분리한 후 grouping 시도 → 맥락 복구 과정에서 손실 위험
- Triplet + α 원자 추출: 맥락이 fact와 함께 한 번에 추출됨 → **원본 맥락을 손실 없이 레이어로 분해**
- 이것이 section 3 지배 원리의 "무손실" 속성을 ingestion time에 확보하는 방법

### 5.4. 지배 원리와의 정합성

왜 이 접근이 구조적으로 옳은가:

1. 원리 (section 3): "LLM은 classifier로만 써라, 새 콘텐츠 생성 금지"
2. 직접 귀결: classifier 작업(레이어 할당)은 추출 시점에 동시 수행 가능
3. 동시 수행하지 않을 이유가 없다: 추가 consolidation 단계는 classifier 작업을 지연시키는 것 외에는 기능적 의미가 없음

따라서 **triplet + α 원자 추출은 section 3 지배 원리의 가장 직접적이고 순수한 구현 형태**이며, 별도 consolidation을 요구하는 기존 시스템들은 **원리를 불완전하게 적용한 과도기적 아키텍처**로 해석된다.

### 5.5. HyperMem/GAAMA와의 관계

**공통 지향**
- fact와 맥락을 별도 레이어 노드로 reification
- 무손실 classifier 방식 grouping
- Topic × Time 축으로 triplet-only 구조의 ceiling 돌파

**본 프로젝트의 구체화 방향**
- **추출 시점의 즉시성**: ingestion time에 모든 레이어 할당 확정 → batch/stream 경계 제거
- **3축 분산**: HyperMem 2축(topic × time), GAAMA 2축(concept + reflection) 대비 **temporal + context + ontology 3축**
- **Ontology의 명시적 활용**: Palantir-style schema로 α_ontology 축을 1급 레이어로 지원
- **Structured signal 최적화**: 자유 텍스트 dialogue가 아닌 구조화 signal에서 α 추출 비용 절감

**포지션의 정확한 표현**

HyperMem/GAAMA가 지향한 "fact-맥락 레이어 분리"라는 방향은 같지만, 본 프로젝트는 그 실현을:
- **더 이른 시점에** (consolidation → ingestion)
- **더 많은 축으로** (2축 → 3축)
- **더 도메인-친화적으로** (범용 dialogue → structured signal)

적용한다. **동일 원리의 더 강한 실현**이며, 단순히 "디테일이 더 있는" 수준이 아니라 **원리를 더 일관되게 적용한 결과**라고 정리할 수 있다.

### 5.6. 적용 조건 및 한계

이 패턴이 유효하려면 도메인이 다음 조건을 만족해야 한다.

**필요 조건**
- **Ontology 사전 정의**: α_ontology 축에 할당할 카테고리가 명시되어 있어야 함
  - MMORPG의 경우: Player / Character / Item / Event / Location / Quest 등 명확 → 충족
- **맥락의 명시성**: 원본 log에 α가 암시적이 아닌 명시적으로 존재해야 함
  - MMORPG signal: timestamp, entity_id, location_id가 구조화되어 있음 → 충족

**적용 한계**
- **Open-domain 자유 대화**: α가 텍스트 안에 녹아있어 즉시 추출이 어려움 → HyperMem/EverMemOS 스타일 consolidation이 더 적합
- **Ontology가 빠르게 진화하는 도메인**: 새 카테고리 출현 빈도가 높으면 batch reclassification이 여전히 필요

**결론**: MMORPG player behavior 도메인의 구조적 특성이 이 패턴을 **구조적으로 최적으로** 만든다. 본 프로젝트의 도메인 선택 자체가 이 추출 전략의 정당성을 확보한다.

---

## 6. graphdb-bench Test Invariants

원리를 프로젝트 벤치마크에 반영하기 위한 불변 조건들. AIDE tree search나 LLM Wiki가 이 제약을 위반하지 못하도록 명시.

### Invariant 1: LLM-as-Classifier Only
> Grouping 노드를 생성하는 LLM call은 반드시 membership 판정 task여야 한다. 새로운 콘텐츠를 압축/요약하여 노드에 저장하는 방식은 채택하지 말 것.

위반 예:
- ❌ "이 세션을 요약해서 summary 노드에 저장"
- ❌ "관련 fact들로부터 reflection을 생성"
- ❌ "Community detection 결과를 LLM으로 자연어 요약"

허용 예:
- ✅ "이 fact가 topic A에 속하는지 판정 (yes/no)"
- ✅ "이 fact의 temporal scope가 기존 scope와 연속되는지 판정"
- ✅ "이 fact의 ontology category를 기존 스키마 중에서 선택"

### Invariant 2: Multi-Axis Grouping 강제
> Fact는 최소 2개 축(temporal + semantic)으로 grouping되어야 한다. 단축 grouping(source-only, time-only, topic-only)은 baseline 비교용으로만 유지.

### Invariant 3: No Decomposition Penalty
> HKG→KG decomposition 기법(arxiv 2508.03280 계열)을 primary 비교 대상으로 설정하지 말 것. 해당 방식은 static KG + link prediction용이며, streaming memory 도메인에서는 비교 가치가 낮음.

### Invariant 4: Write-Path가 일급 측정 대상
> Retrieval accuracy뿐 아니라 write throughput + grouping 생성 latency를 필수 지표로 포함. EverMemOS 대비 최소 100x write throughput이 primary success criterion.

---

## 7. 실행 체크리스트

### 7.1. 즉시 중단할 것
- ❌ Triplet-only 아키텍처로 계속 진행
- ❌ Reflection/summary 기반 grouping 노드 설계
- ❌ HINGE/HyperGraphRAG 계보 무시하고 HippoRAG 계보만 참조
- ❌ Extract-then-cluster 방식 채택 (streaming rate에서 작동 불가)

### 7.2. 즉시 시작할 것
- ✅ EverMemOS 코드 clone 및 MemCell/MemScene 내부 storage 구조 분석
- ✅ HyperMem 논문 정독 후 Topic×Episode와 영주님 L1×L2×L3 매핑 명확화
- ✅ graphdb-bench에 EverMemOS를 primary baseline으로 등록
- ✅ LLM-as-classifier prompt engineering 실험 (topic boundary detection, temporal scope detection)
- ✅ **Triplet + α 원자 추출 prompt 설계** — 단일 LLM call에서 base_triplet, α_temporal, α_context, α_ontology를 동시 추출하는 JSON schema 확정
- ✅ **MMORPG signal에서 α 파싱 가능 부분 식별** — LLM 호출 없이 structured field에서 획득 가능한 α 컴포넌트 목록화 (cost 절감)
- ✅ **즉시 분산 write path 설계** — 추출된 α를 L1/L2/L3에 병렬 insert하는 transaction 단위 정의

### 7.3. 장기 방향
- ⚙️ Custom graph DB: L0-L3 multi-axis reification을 native로 지원
- ⚙️ 즉시 분산 ingest 파이프라인: Kafka → LLM extraction → multi-layer parallel write
- ⚙️ Kafka-scale ingest: EverMemOS 대비 2-3 orders of magnitude 처리량 달성
- ⚙️ MMORPG 도메인에서 4-layer reification의 downstream 효과 측정

---

## 8. 한 줄 요약

> **"요약하지 말고 분류하라. 압축하지 말고 층을 쌓아라. 새 노드를 만들되 새 콘텐츠는 만들지 말라. 그리고 모든 할당을 추출 시점에 동시에 끝내라."**

이 원리가 HINGE(2020)에서 시작해 EverMemOS(2026)까지 이어지는 6년간 연구의 증류된 결론이며, 영주님 4-layer 설계 + triplet + α 원자 추출 패턴은 이 원리의 가장 완성된 표현이다. 구현 목표는 명확하다 — **SOTA의 개념 아키텍처를 받되, ingestion time에 모든 레이어 할당을 확정하는 streaming-native 파이프라인과 자신만의 W/R 특성에 맞는 custom graph DB로 재구현**. 이것이 유일하게 의미 있는 contribution 지점이다.

---

## Appendix: 참조 문서

- **선행 문서**: `hypergraph-memory-lineage.md` — HINGE→HyperGraphRAG→HyperMem 계보 전체
- **핵심 논문**:
  - HINGE: https://dl.acm.org/doi/10.1145/3366423.3380257
  - HyperMem: https://arxiv.org/abs/2604.08256
  - EverMemOS: https://arxiv.org/abs/2601.02163
  - HyperGraphRAG: https://arxiv.org/abs/2503.21322
  - HippoRAG 2: https://arxiv.org/abs/2502.14802
  - GAAMA: https://arxiv.org/abs/2603.27910
- **구현 참조**:
  - EverMemOS: https://github.com/EverMind-AI/EverMemOS
  - HippoRAG: https://github.com/OSU-NLP-Group/HippoRAG
  - HyperGraphRAG: https://github.com/LHRLAB/HyperGraphRAG
  - GAAMA: https://github.com/swarna-kpaul/gaama
