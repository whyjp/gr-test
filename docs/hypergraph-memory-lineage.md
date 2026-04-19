# Hypergraph Memory Lineage — 벤치마크 반영용 정리

> 작성: 2026-04-19
> 목적: `graphdb-bench` 및 memory graph 프로젝트 구현 방향 결정을 위한 레퍼런스
> **핵심**: hypergraph 기반 메모리는 이미 존재하며 일주일 전 LoCoMo SOTA(92.73%)를 찍었음. 구현 novelty는 "hypergraph 자체"가 아니라 "어떤 축에서 확장하느냐"에 있음.

---

## 1. TL;DR

1. **Beyond Triplets (HINGE, WWW'20)** → **HyperGraphRAG (NeurIPS'25)** → **HyperMem (2026-04, ~1주 전)** 으로 이어지는 **단일 계보**가 이미 "fact triple의 한계를 구조적으로 극복" 하는 문제의 표준 해법이 됨.
2. **HippoRAG / GAAMA / Mem0 / Zep 계열은 이 계보와 다른 커뮤니티**에서 나왔으며, 여전히 (h, r, t) triple 기반. 요약·반성(reflection)·community detection 등 "post-hoc compensation" 전략에 의존.
3. **HyperMem은 두 계보가 최초로 만난 지점** — hypergraph 구조 + episodic memory + LoCoMo 벤치마크. 현재 SOTA.
4. 따라서 "hypergraph memory 구현" 자체는 더 이상 novelty가 아님. 내 구현의 기여는 **streaming event signal / 4-layer / production-grade write-heavy graph DB / MMORPG 도메인**에 있어야 함.

---

## 2. 두 계보의 대비

### Lineage A: Hyper-relational KG Embedding → HypergraphRAG (구조 보존파)

| 연도 | 논문 | 핵심 기여 |
|------|------|----------|
| 2020 WWW | **HINGE** (Rosso et al.) | n-ary fact를 base triple + k-v pair로 직접 학습, transformation 기반 방식 대비 최대 41% MRR 개선 |
| 2020 EMNLP | **StarE** (Galkin et al.) | Message passing GNN, qualifier 개수 제한 제거, WD50K 데이터셋 공개 |
| 2021 | Hy-Transformer | Transformer self-attention for hyper-relational |
| 2022 | QUAD | Base triple + qualifier 이중 aggregator |
| 2023 | NeuInfer, Shomer et al. | FC neural network / 표현 학습 개선 |
| 2024 TKDE | **sHINGE** (원저자 직접 후속) | Schema awareness 추가, transformation 방식별 최대 29.3% 성능 저하 재확인 |
| 2025 NeurIPS | **HyperGraphRAG** (Luo et al.) | Hypergraph 기반 RAG 전체 파이프라인, medical/legal/agriculture 도메인 |
| 2026 WWW | **HyperRAG** | HyperRetriever + HyperMemory 이중 paradigm |
| 2026-04 | **HyperMem** | 3-level 계층 hypergraph memory, LoCoMo SOTA 92.73% |

**공통 철학**: "fact의 complex n-ary 관계를 decomposition 없이 **구조적으로** 보존한다. 요약·추론은 표현력을 이미 확보한 후의 선택적 도구이다."

### Lineage B: Triple-based Memory Agent (보상파)

| 연도 | 시스템 | 핵심 아키텍처 |
|------|--------|-------------|
| 2023 | **Mem0** | Fact extraction + vector retrieval + ADD/UPDATE/DELETE 명령 |
| 2024 | **Zep / Graphiti** | Temporal KG, bi-temporal edge (valid_at / recorded_at) |
| 2024 NeurIPS | **HippoRAG** | OpenIE triple + synonym edge + Personalized PageRank |
| 2025 ICML | **HippoRAG 2** | 위에 passage node + context edge 추가, query-to-triple 매칭 |
| 2025 | **A-Mem, Nemori** | Narrative/agentic memory 변형 |
| 2026-03 | **GAAMA** | 4-node(episode/fact/reflection/concept) + 5 edge-type + PPR |

**공통 철학**: "triple 구조 유지하되, reflection / concept node / community summary / PPR 확산 등으로 **pairwise 한계를 post-hoc 보상**한다."

---

## 3. 왜 Lineage B는 Lineage A를 인지하지 못했는가

연구 커뮤니티 측면의 분리가 명확하다.

**데이터셋 분리**
- Lineage A: JF17K, WikiPeople, WD50K (Wikidata qualifier 기반)
- Lineage B: LoCoMo, LongMemEval, MSC (multi-session dialogue)
- **교집합이 없었음**. LoCoMo는 2024년 공개된 dialogue 벤치마크, WD50K는 2020년부터 KG embedding 벤치마크.

**용어 분리**
- A: "hyper-relational fact", "qualifier", "n-ary", "hyperedge"
- B: "episodic memory", "reflection", "associative retrieval", "hippocampal indexing"
- 키워드가 겹치지 않아서 literature search에서 서로 안 걸림.

**평가 지표 분리**
- A: MRR, Hits@10 (link prediction)
- B: F1, LLM-as-judge reward (QA accuracy)

**타이밍**
- HippoRAG: 2024 NeurIPS (HyperGraphRAG 보다 1년 앞섬)
- GAAMA: 2026-03 (HyperGraphRAG과 동시대인데도 **인용 없음**) — GAAMA reference를 보면 HippoRAG / A-Mem / Nemori만 언급.
- HyperMem이 2026-04에 나와서 비로소 양쪽을 명시적으로 비교하기 시작.

**결론**: 이것은 기술적 모순이 아니라 **커뮤니티 경계**의 문제였다. 실제로 HyperMem 논문의 case study를 보면 HyperGraphRAG이 "aerial yoga"를 시간대 잘못 잡아서 실패하고 GraphRAG이 pairwise edge 분절로 multi-session aggregation 실패하는 구체적 사례가 있음. 즉 Lineage A(정적 지식 hypergraph)도 Lineage B(episodic memory)도 단독으로는 LoCoMo를 완전히 풀지 못하며, **둘의 결합이 정답**이었음이 실증됨.

---

## 4. LoCoMo 벤치마크 결과 비교 (최신)

| 시스템 | 아키텍처 카테고리 | LoCoMo 점수 | 비고 |
|--------|-----------------|------------|------|
| **HyperMem** (2026-04) | Hypergraph + episodic | **92.73%** | SOTA, 3-level 계층 |
| MemMachine | Episodic + profile memory | 84.87% | 상용 제품 |
| **GAAMA** (2026-03) | 4-node KG + PPR | 78.9% | concept/reflection 추가 |
| Tuned RAG baseline | Flat RAG | 75.0% | — |
| **HippoRAG** (2024) | OpenIE triple + PPR | 69.9% | 초기 hippocampal memory |
| Nemori | Narrative memory | 52.1% | — |
| A-Mem | Agentic memory | 47.2% | — |

주: MemMachine 블로그 자료, GAAMA 논문 Table 1, HyperMem 논문 Table 기준. 동일 prompt/LLM은 아니므로 절대 비교는 주의.

**패턴**: hypergraph 도입(HyperMem)으로 +14pp 급 점프. 같은 triple 기반 안에서는 GAAMA가 HippoRAG 대비 +9pp(concept/reflection 효과). → **구조 표현력 자체의 ceiling이 존재**하며 hypergraph가 그 ceiling을 깬다는 가설이 경험적으로 지지됨.

---

## 5. HyperMem 상세 분해 (내 구현과 가장 가까운 기존 시스템)

**구조**
- Level 1: Topics (가장 추상)
- Level 2: Episodes (dialogue session boundary 단위)
- Level 3: Facts (atomic 단위)
- Hyperedge: 관련 episode들과 그 fact들을 하나의 hyperedge로 묶음

**검색**
- Coarse-to-fine: topic → episode → fact
- Hypergraph embedding propagation으로 lexical-semantic 이중 인덱싱
- 조기 pruning으로 효율성 확보

**한계 (논문에서 저자가 직접 언급)**
- Single-user 가정 (multi-user/multi-agent 미고려)
- Open-domain 질문에서 대화 외부 지식 필요시 약함
- **Streaming signal 아님** — dialogue turn이 이미 구조화된 형태로 들어오는 것을 전제

---

## 6. 내 구현의 포지셔닝 (HyperMem 대비 차이점)

| 축 | HyperMem | 내 구현 | 차이 |
|----|----------|---------|------|
| 입력 소스 | Dialogue turn (구조화됨) | Kafka streaming + Snowflake signal | **실시간 비정형 event stream** |
| 계층 수 | 3 (topic/episode/fact) | 4 (L0 facts / L1 temporal·importance / L2 context / L3 auxiliary) | **temporal·importance 별도 레이어** |
| 시간 모델링 | episode boundary만 | Event Segmentation Theory + Temporal Context Model 기반 | **인지과학적 시간 레이어** |
| Storage | 논문 prototype (미상) | Memgraph (MVCC, O(1) pointer mutation) | **production write-heavy 운영** |
| 도메인 | 일반 dialogue | MMORPG player behavior | **high-frequency agent signal** |
| 쓰기 빈도 | 대화 턴당 1회 | 초당 수천 signal | **처리량 2-3 orders of magnitude 차이** |

**novelty가 확실히 남아있는 축**
1. Streaming event signal에서 **hyperedge를 on-the-fly 생성·병합·쇠퇴**시키는 dynamic hypergraph 운영. 현 학계 문헌에 선례 희박.
2. **L1 temporal·importance**를 별도 레이어로 분리하는 구조. HyperMem/HyperGraphRAG은 모두 평평하거나 단순 계층.
3. **Production graph DB 위에 hyperedge를 first-class로** 올리는 실측 벤치마크. TypeDB/PERA 이론 레퍼런스만 존재하고 실측 데이터 부족.

**novelty가 아닌 것 (인정해야 할 부분)**
1. "Atomic node-set ingestion" 개념 자체 — HyperGraphRAG의 hyperedge construction과 동형.
2. Temporal/importance 레이어링 개념 — OKH-RAG, HyperMem에 부분적으로 이미 존재.
3. Context entity 별도 노드화 — StarE qualifier, HINGE key-value와 동일.

---

## 7. graphdb-bench 반영 포인트

### 7.1. 비교 대상 시스템 재구성

벤치마크 harness의 비교 대상을 **Lineage A / B 양쪽에서 대표 시스템을 각각 뽑아 공정 비교**하도록 구성.

**Lineage A 대표**
- HyperGraphRAG (github.com/LHRLAB/HyperGraphRAG) — static hypergraph RAG
- HyperMem (논문 공개됨, 코드 확인 필요) — episodic hypergraph

**Lineage B 대표**
- HippoRAG 2 (github.com/OSU-NLP-Group/HippoRAG) — OpenIE + PPR
- GAAMA (github.com/swarna-kpaul/gaama) — concept/reflection
- Graphiti / Zep — bi-temporal

### 7.2. 벤치마크 차원 추가

기존 graphdb-bench이 DB write throughput에 초점이라면, 다음 축을 추가 고려:

1. **Write-heavy hyperedge insertion TPS**: Kafka streaming 시뮬레이션으로 초당 hyperedge 생성 속도. Memgraph vs FalkorDB vs Neo4j 비교의 실질 의미.
2. **Dynamic hyperedge merge latency**: 기존 hyperedge와 새 signal 병합 시 점유 시간.
3. **Multi-layer propagation cost**: L0→L1→L2→L3 전파시 각 레이어 업데이트 latency.
4. **Retrieval accuracy under write load**: 쓰기 부하가 걸린 상태에서 LoCoMo-유사 질문에 대한 정확도 (현재 대부분 벤치마크는 static graph 가정).

### 7.3. AIDE tree search / LLM Wiki 반영

- HINGE → HyperGraphRAG → HyperMem 계보를 LLM Wiki의 **한 node**로 등록하고, 실험 중 "hypergraph 관련 실패" 발생시 해당 node를 참조하도록 설정.
- 이 문서 자체를 Wiki seed로 사용. 특히 "Lineage A/B 커뮤니티 분리" 맥락은 LLM이 놓치기 쉬운 부분이므로 명시.
- test invariant에 **"hyperedge construction은 post-hoc summarization으로 대체되어서는 안 된다"** 추가 — 아니면 AIDE가 tree search 중에 Lineage B 해법으로 회귀할 위험 있음.

---

## 8. 실행 우선순위 (개인 판단)

1. **HyperMem 논문 PDF 정독** (arxiv 2604.08256) — 내 구조와 동일점/차이점 1:1 대조표 작성. 이게 제일 먼저.
2. **HyperGraphRAG 코드 분석** (github.com/LHRLAB/HyperGraphRAG) — hyperedge construction prompt, storage schema 확인. LightRAG, Text2NKG, HAHE 기반이므로 이 세 개도 병행.
3. **graphdb-bench에 Lineage A/B 대표 시스템 integration** — 최소 HyperGraphRAG, HippoRAG 2, GAAMA 세 개는 같은 harness 안에 넣어서 write throughput + retrieval accuracy 동시 측정.
4. **내 4-layer 구조가 LoCoMo에서 HyperMem 대비 어떤 질문 유형에 강한지** 확인. 특히 temporal reasoning / multi-session aggregation 축에서 L1 temporal layer의 효과 입증 여부가 관건.

---

## 9. 참고 (핵심 소스만)

- **HINGE**: Rosso et al., Beyond Triplets, WWW 2020. https://dl.acm.org/doi/10.1145/3366423.3380257
- **StarE**: Galkin et al., EMNLP 2020. https://arxiv.org/abs/2009.10847
- **sHINGE**: Rosso/Yang/Cudré-Mauroux, TKDE 2024. https://exascale.info/assets/pdf/KnowledgeGraphEmbeddings_TKDE2024.pdf
- **HyperGraphRAG**: Luo et al., NeurIPS 2025. https://arxiv.org/abs/2503.21322 / github.com/LHRLAB/HyperGraphRAG
- **HyperRAG**: WWW 2026. https://arxiv.org/abs/2602.14470
- **HyperMem**: 2026-04. https://arxiv.org/abs/2604.08256
- **OKH-RAG**: 2026-04-14. https://arxiv.org/abs/2604.12185
- **HGMem**: 2025-12. https://arxiv.org/abs/2512.23959
- **HippoRAG**: Gutierrez et al., NeurIPS 2024. https://arxiv.org/abs/2405.14831 / github.com/OSU-NLP-Group/HippoRAG
- **GAAMA**: Paul et al., 2026-03. https://arxiv.org/abs/2603.27910 / github.com/swarna-kpaul/gaama
- **LoCoMo**: Maharana et al., 2024. https://arxiv.org/abs/2402.17753
- **Understanding HKGE (반론)**: 2025-08. https://arxiv.org/abs/2508.03280

---

## Appendix: 빠른 의사결정 가이드

**Q. "hypergraph 기반 memory" 를 내가 처음 만든다고 주장해도 되는가?**
A. 아니다. HINGE(2020) → HyperGraphRAG(2025) → HyperMem(2026-04)로 이미 확립됨.

**Q. 그럼 내 구현이 의미가 없는가?**
A. 있다. 단 "hypergraph 도입" 자체가 아니라 **streaming event / 4-layer / production write-heavy / MMORPG** 라는 조합에 의미가 있다.

**Q. 가장 빨리 비교 가능한 baseline은?**
A. HyperGraphRAG (코드 있음, medical/legal 벤치마크 존재) + HippoRAG 2 (코드 있음, LoCoMo 가능). 이 둘을 graphdb-bench에 먼저 integration.

**Q. 내 구현의 "proof" 를 어디서 받아야 하는가?**
A. LoCoMo 정확도 + Kafka 1k+ msg/sec 상태에서의 hyperedge insertion TPS. 전자는 Lineage B 커뮤니티, 후자는 production DB 커뮤니티가 각각 평가 가능.
