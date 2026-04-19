# Hyper Triplet: 구현 및 벤치마크 계획서

> **⚠️ SUPERSEDED BY [v2](./hyper-triplet-implementation-plan-v2.md) — 2026-04-19**
> v1은 다음 사항이 잘못되어 있음:
> - LoCoMo-10 스키마 (list vs dict)
> - GAAMA 실제 LLM 호출 수 (3이 아닌 2)
> - Phase 3 기여 프레이밍 (비용 효율이 아닌 hyper-relational 구조)
>
> 구현과 연구는 v2를 따를 것. 아래 v1 원문은 히스토리 보존용.

---

> HippoRAG2 → GAAMA → Hyper Triplet 3단계 진화 비교
> LoCoMo-10 동일 벤치마크 · 동일 평가 프로토콜 · IE 출력 단위만 변경
> 논문 작성 전 구현·실험 가이드

---

## 목표

LoCoMo-10 벤치마크에서 세 시스템을 동일 조건으로 비교한다:

```
HippoRAG2  (flat triple KG)           → 69.9% (보고값)
GAAMA      (triple → 사후 계층 KG)     → 78.9% (보고값)
제안        (node-set → 즉시 계층 KG)   → ???%
```

증명할 것: IE 출력 단위를 트리플렛에서 노드셋으로 바꾸면, 같은 정보에서 더 나은 검색 성능을 얻는다.

---

## Phase 0: 환경 및 데이터 준비

### 0.1 공통 인프라

```
hyper-triplet-bench/
├── data/
│   └── locomo10.json              ← snap-research/locomo에서 다운로드
├── systems/
│   ├── hipporag2/                 ← HippoRAG2 재현
│   ├── gaama/                     ← GAAMA 재현
│   └── hyper_triplet/             ← 신규 구현
├── eval/
│   ├── judge.py                   ← 공통 LLM-as-judge
│   ├── metrics.py                 ← 점수 계산 (accuracy, per-category)
│   └── runner.py                  ← 3개 시스템 동일 프로토콜 실행
├── results/
│   ├── hipporag2/
│   ├── gaama/
│   └── hyper_triplet/
├── ablation/
│   ├── triple_only/               ← 노드셋에서 맥락 제거 (L0만)
│   ├── sequential_nodeset/        ← 노드셋이지만 3단계 순차 적재
│   └── no_environment/            ← 맥락은 있되 환경 노드 제거
└── README.md
```

### 0.2 LoCoMo-10 데이터셋

- 소스: `https://github.com/snap-research/locomo`
- 파일: `data/locomo10.json`
- 구조: 10개 대화, 대화당 ~600턴/~16K토큰/~32세션
- QA: ~1,540개, 4개 카테고리 (single-hop, multi-hop, temporal, open-domain)
- 카테고리 5(adversarial)는 평가에서 제외 (GAAMA, Mem0 등 모든 논문이 제외)

```python
# data/locomo10.json 구조
{
  "conv-XX": {
    "speaker_a": "...",
    "speaker_b": "...",
    "session_1": [...],           # 대화 턴 리스트
    "session_1_date_time": "...",  # 세션 타임스탬프
    "session_1_summary": "...",   # 세션 요약
    "events_session_1": [...],    # 이벤트 요약 (ground truth)
    "qa": [                       # QA 쌍
      {
        "question": "...",
        "answer": "...",
        "category": 1,            # 1=single-hop, 2=multi-hop, 3=temporal, 4=open-domain
        "evidence": ["dia_XX"]    # 근거 대화 ID
      }
    ]
  }
}
```

### 0.3 통제 변수

| 변수 | 값 | 근거 |
|------|-----|------|
| LLM (추출+답변) | `gpt-4o-mini` | GAAMA 논문과 동일 |
| LLM (judge) | `gpt-4o` | 공정 평가를 위해 상위 모델 사용 |
| 임베딩 | `text-embedding-3-small` | GAAMA와 동일 |
| 검색 예산 | 1,000 words | GAAMA와 동일 |
| 실행 횟수 | 10회 | 평균 + 표준편차. Mem0 논문 프로토콜 |
| 온도 | 0.0 (추출), 0.1 (judge) | 재현성 |
| QA 카테고리 | 1~4만 (5 제외) | 전 논문 공통 |

---

## Phase 1: HippoRAG2 재현

### 1.1 목표

보고값 69.9%를 재현한다. GAAMA 논문에서 보고한 HippoRAG 점수이므로, GAAMA의 평가 프로토콜로 실행.

### 1.2 파이프라인

```
입력: LoCoMo 대화 텍스트

1. OpenIE 추출
   - 각 대화 턴에서 (S, P, O) 트리플렛 추출
   - LLM 기반 Named Entity Recognition

2. Flat KG 구축
   - 엔티티 노드 + 관계 엣지
   - 동의어/유사어 엣지 추가
   - 맥락 계층 없음

3. 검색: Personalized PageRank
   - 질의에서 시드 엔티티 추출
   - 시드에서 PPR 실행
   - 상위 passage 반환

4. 답변 생성
   - 검색된 context + 질문 → LLM → 답변
```

### 1.3 구현 선택

```python
# systems/hipporag2/pipeline.py

class HippoRAG2Pipeline:
    def ingest(self, conversations: List[Conversation]):
        """대화를 flat KG로 변환"""
        for conv in conversations:
            for session in conv.sessions:
                for turn in session.turns:
                    triples = self.extract_triples(turn.text)  # OpenIE
                    self.kg.add_triples(triples)               # flat 적재

    def retrieve(self, query: str, budget: int = 1000) -> str:
        """PPR 기반 검색"""
        seed_entities = self.extract_entities(query)
        ppr_scores = self.personalized_pagerank(seed_entities)
        passages = self.rank_passages(ppr_scores, budget)
        return passages

    def answer(self, query: str, context: str) -> str:
        """답변 생성"""
        return self.llm.generate(query=query, context=context)
```

### 1.4 검증 기준

- 보고값 69.9% ± 3%p 이내 재현
- 카테고리별 점수 분포가 GAAMA 논문의 HippoRAG 행과 유사
- 실패 시: 평가 프로토콜 차이 확인 (프롬프트, context budget 등)

---

## Phase 2: GAAMA 재현

### 2.1 목표

보고값 78.9%를 재현한다. 공식 코드(swarna-kpaul/gaama)를 사용.

### 2.2 파이프라인

```
입력: LoCoMo 대화 텍스트

1. 에피소드 보존 (비LLM)
   - 각 대화 턴을 Episode 노드로 저장 (원본 그대로)
   - Episode 간 :NEXT 체이닝
   - 타임스탬프 메타데이터

2. Fact + Concept 추출 (LLM)
   - Episode → LLM → atomic fact 추출
   - Episode → LLM → topic concept 추출
   - Fact → Episode: :DERIVED_FROM 엣지
   - Episode → Concept: :HAS_CONCEPT 엣지
   - Fact → Concept: :ABOUT_CONCEPT 엣지

3. Reflection 생성 (LLM)
   - 여러 Fact를 종합 → 고차 통찰
   - Reflection → Fact: :SYNTHESIZED_FROM 엣지

4. 검색: Semantic + Hub-dampened PPR
   - 질의 → 임베딩 → KNN → 시드 노드
   - PPR (edge-type-aware, hub dampening)
   - 가중 합산 스코어 (PPR weight 0.1, semantic weight 1.0)

5. 답변 생성
   - 검색된 memory pack → LLM → 답변
```

### 2.3 구현: GAAMA 공식 코드 사용

```bash
# GAAMA 설치 및 실행
git clone https://github.com/swarna-kpaul/gaama.git
cd gaama

# Step 1: LTM 생성
cd gaama/evals/locomo
python run_create_ltm.py                    # 10개 대화 전부

# Step 2: 평가
python run_semantic_eval.py                 # semantic only
python run_ppr_eval.py --ppr-weight 0.1     # semantic + PPR

# Step 3: RAG baseline (비교용)
python run_rag_baseline.py --step all
```

### 2.4 핵심 코드 파악 대상

| 파일 | 역할 | 변경 대상 여부 |
|------|------|-------------|
| `services/ltm_creator.py` | 3단계 LTM 파이프라인 | **Phase 3에서 교체** |
| `services/llm_extractors.py` | fact/concept/reflection 추출 프롬프트 | **Phase 3에서 교체** |
| `services/orchestrator.py` | ingest → create → retrieve 오케스트레이션 | **Phase 3에서 수정** |
| `services/ltm_retriever.py` | semantic + PPR 검색 | 재사용 |
| `services/pagerank.py` | hub-dampened PPR | 재사용 |
| `services/answer_from_memory.py` | 답변 생성 | 재사용 |
| `adapters/sqlite_memory.py` | 노드/엣지 저장소 | 재사용 (스키마 확장) |
| `evals/locomo/run_*_eval.py` | 평가 스크립트 | 재사용 |

### 2.5 검증 기준

- 보고값 78.9% ± 2%p 이내 재현
- Ablation: semantic only vs semantic+PPR 차이가 ~1%p (논문과 일치)
- 노드 통계: 대화당 평균 Episode/Fact/Concept/Reflection 노드 수 기록

---

## Phase 3: Hyper Triplet 구현

### 3.1 핵심 변경: IE 출력 단위의 재정의

GAAMA의 코드를 포크하여, **추출 파이프라인만** 교체한다.
검색과 답변 생성은 GAAMA와 동일하게 유지 — **변수를 IE 출력 단위 하나로 격리**.

```
GAAMA:
  Episode → [LLM call 1: facts] → [LLM call 2: concepts] → [LLM call 3: reflections]
  3번의 LLM 호출, 각각 독립적

Hyper Triplet:
  Episode → [LLM call 1: node-set (fact + context + environment)] → [LLM call 2: reflections]
  2번의 LLM 호출, 1번째에서 fact과 context가 동시에 바인딩
```

### 3.2 노드셋 추출 프롬프트 설계

```python
# systems/hyper_triplet/extractors.py

NODESET_EXTRACTION_PROMPT = """
You are an information extractor. From the given conversation excerpt,
extract FACTS with their CONTEXT and ENVIRONMENT as a structured node-set.

For each distinct fact/event mentioned:
1. FACT: (subject, predicate, object) — the core assertion
2. CONTEXT (if identifiable):
   - location: where this happened
   - participants: who was involved
   - activity_type: what kind of activity (solo/group, casual/formal, etc.)
   - time_reference: any temporal reference mentioned
3. ENVIRONMENT (if identifiable):
   - mood/atmosphere: emotional tone
   - circumstances: surrounding conditions
   - topic_category: broad topic this falls under

Output as JSON array of node-sets. Each node-set is ONE atomic unit.
Do NOT extract facts without checking for available context.
If context is not mentioned, omit that field (do not hallucinate).

Conversation excerpt:
{text}

Output format:
[
  {
    "fact": {"subject": "...", "predicate": "...", "object": "..."},
    "context": {
      "location": "..." or null,
      "participants": ["..."] or null,
      "activity_type": "..." or null,
      "time_reference": "..." or null
    },
    "environment": {
      "mood": "..." or null,
      "circumstances": "..." or null,
      "topic_category": "..." or null
    }
  }
]
"""
```

### 3.3 노드셋 적재: 원자적 MERGE

```python
# systems/hyper_triplet/ltm_creator.py

class HyperTripletLTMCreator:
    def create_ltm(self, episode: Episode) -> List[str]:
        """에피소드에서 노드셋 추출 → 원자적 적재"""

        # Step 1: 노드셋 추출 (LLM 1회 호출)
        node_sets = self.extract_node_sets(episode.text)

        created_ids = []
        for ns in node_sets:
            # Step 2: 원자적 적재 — fact + context + environment를 한 트랜잭션에서
            fact_id = self._create_fact_node(ns["fact"], episode)

            if ns.get("context"):
                ctx = ns["context"]
                if ctx.get("location"):
                    loc_id = self._merge_context_node("Location", ctx["location"])
                    self._create_edge(fact_id, loc_id, "AT_LOCATION")
                if ctx.get("participants"):
                    for p in ctx["participants"]:
                        p_id = self._merge_context_node("Participant", p)
                        self._create_edge(fact_id, p_id, "WITH_PARTICIPANT")
                if ctx.get("activity_type"):
                    act_id = self._merge_context_node("ActivityType", ctx["activity_type"])
                    self._create_edge(fact_id, act_id, "ACTIVITY_TYPE")
                if ctx.get("time_reference"):
                    time_id = self._merge_context_node("TimeRef", ctx["time_reference"])
                    self._create_edge(fact_id, time_id, "AT_TIME")

            if ns.get("environment"):
                env = ns["environment"]
                if env.get("topic_category"):
                    # topic_category = GAAMA의 concept에 해당
                    topic_id = self._merge_context_node("Topic", env["topic_category"])
                    self._create_edge(fact_id, topic_id, "ABOUT_TOPIC")
                if env.get("mood"):
                    mood_id = self._merge_context_node("Mood", env["mood"])
                    self._create_edge(fact_id, mood_id, "IN_MOOD")

            created_ids.append(fact_id)

        # Step 3: Reflection (GAAMA와 동일 — 여러 fact 종합)
        if len(created_ids) >= 3:
            reflections = self.generate_reflections(created_ids)
            created_ids.extend(reflections)

        return created_ids

    def _merge_context_node(self, node_type: str, value: str) -> str:
        """MERGE 패턴 — 동일 값이면 재사용, 없으면 생성"""
        existing = self.store.find_node(type=node_type, value=value)
        if existing:
            return existing.id
        return self.store.create_node(type=node_type, value=value)
```

### 3.4 GAAMA 대비 변경점 요약

| 구성요소 | GAAMA | Hyper Triplet | 변경 이유 |
|---------|-------|---------------|----------|
| 추출 프롬프트 | fact/concept 별도 프롬프트 | 노드셋 통합 프롬프트 | fact-context 바인딩을 추출 시점에 확정 |
| LLM 호출 횟수 | 3회 (fact, concept, reflection) | 2회 (node-set, reflection) | 1회 줄어 비용 절감 + 바인딩 정확도 |
| 적재 방식 | 순차 (fact → concept 별도 생성) | 원자적 (node-set 한 트랜잭션) | 고아 fact/orphan concept 방지 |
| 맥락 노드 | Concept (topic label) | Context + Environment (타입드) | 더 풍부한 구조 |
| MERGE 패턴 | concept 중복 체크 (약함) | 전 맥락 노드 MERGE (강함) | 동일 장소/참여자 재사용 |
| 검색 | semantic + PPR (동일) | semantic + PPR (동일) | **통제 변수 — 변경 없음** |
| 답변 생성 | 동일 | 동일 | **통제 변수 — 변경 없음** |
| 평가 | 동일 | 동일 | **통제 변수 — 변경 없음** |

### 3.5 노드 유형 매핑

```
GAAMA 노드 유형          Hyper Triplet 노드 유형
────────────────────────────────────────────────
Episode                  Episode (동일)
Fact                     Fact (동일)
Concept (topic label)    Topic (environment.topic_category)
                         + Location (context.location)     ← 신규
                         + Participant (context.participants) ← 신규
                         + ActivityType (context.activity_type) ← 신규
                         + TimeRef (context.time_reference) ← 신규
                         + Mood (environment.mood)          ← 신규
Reflection               Reflection (동일)
```

GAAMA의 Concept 1종 → Hyper Triplet의 6종 타입드 맥락 노드로 분화.
검색 시 타입별 edge traversal이 가능해짐.

---

## Phase 4: 평가 실행

### 4.1 공통 평가 프로토콜

```python
# eval/runner.py

class BenchmarkRunner:
    def run(self, system_name: str, system: Pipeline, n_runs: int = 10):
        dataset = load_locomo10("data/locomo10.json")
        all_scores = []

        for run_id in range(n_runs):
            scores = []
            for conv in dataset.conversations:
                # 1. 메모리 적재
                system.reset()
                system.ingest(conv)

                # 2. QA 평가
                for qa in conv.qa_pairs:
                    if qa.category == 5:  # adversarial 제외
                        continue
                    context = system.retrieve(qa.question, budget=1000)
                    answer = system.answer(qa.question, context)
                    judgment = self.judge(qa.question, qa.gold_answer, answer)
                    scores.append({
                        "conv_id": conv.id,
                        "category": qa.category,
                        "correct": judgment == "CORRECT",
                        "run_id": run_id
                    })

            all_scores.append(scores)

        self.save_results(system_name, all_scores)
        self.print_summary(system_name, all_scores)
```

### 4.2 LLM-as-Judge

```python
# eval/judge.py

JUDGE_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Be generous: as long as the generated answer touches on the same topic
as the gold answer, count it as CORRECT.
For time-related questions, the gold answer will be a specific date/month/year.

Output ONLY 'CORRECT' or 'WRONG'.
"""

class LLMJudge:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature

    def judge(self, question: str, gold: str, generated: str) -> str:
        response = self.llm.generate(
            JUDGE_PROMPT.format(
                question=question,
                gold_answer=gold,
                generated_answer=generated
            )
        )
        return response.strip()  # "CORRECT" or "WRONG"
```

### 4.3 리포트 형식

```
================================================================
                    LoCoMo-10 Benchmark Results
                    10 runs, mean ± std
================================================================

System              Overall   Single   Multi   Temporal   Open
─────────────────────────────────────────────────────────────────
HippoRAG2           69.9%     72.1%   65.3%    68.4%    71.2%
                    ±1.2%    ±1.5%   ±2.1%   ±1.8%    ±3.4%

GAAMA               78.9%     81.2%   74.6%    77.3%    80.1%
                    ±0.9%    ±1.1%   ±1.7%   ±1.4%    ±2.8%

Hyper Triplet        ??. ?%    ??.?%   ??.?%    ??.?%    ??.?%
                    ±?.?%    ±?.?%   ±?.?%   ±?.?%    ±?.?%
================================================================

Improvement over GAAMA: +?.?%p (p-value: 0.XXX)
================================================================
```

---

## Phase 5: Ablation 실험

### 5.1 변수 격리

| 실험 | 설명 | 증명할 것 |
|------|------|----------|
| **A1: Triple-only** | 노드셋 프롬프트에서 context/environment 제거. fact만 추출 | 맥락 노드 추가의 가치 |
| **A2: Sequential node-set** | 노드셋을 뽑되, GAAMA처럼 3단계 순차 적재 | 원자적 vs 순차의 차이 |
| **A3: No environment** | context만 추출, environment 제거 | 환경 노드의 추가 가치 |
| **A4: GAAMA concepts as node-set** | GAAMA의 concept를 노드셋 프롬프트로 추출 | 프롬프트 변경만의 효과 |
| **A5: LLM call count control** | Hyper Triplet의 LLM 호출 2회를 GAAMA와 동일 3회로 맞춤 | 호출 횟수 vs 추출 방식 격리 |

### 5.2 기대 결과 패턴

```
Triple-only (A1)        < GAAMA          < Hyper Triplet
  맥락 없음                사후 맥락          즉시 맥락
  → 이러면 "맥락 노드의 가치"를 입증

GAAMA                   ≈ Sequential (A2) < Hyper Triplet
  사후 순차                사후 순차(노드셋)    원자적
  → 이러면 "원자적 적재의 가치"를 입증

Hyper Triplet w/o env (A3) < Hyper Triplet
  context만                context + environment
  → 이러면 "환경 노드의 추가 가치"를 입증
```

---

## Phase 6: 비용 분석

### 6.1 LLM 호출 비용 비교

| 시스템 | 호출 횟수/대화 | 용도 | 예상 비용 (gpt-4o-mini) |
|--------|-------------|------|----------------------|
| HippoRAG2 | ~N (턴 수) | OpenIE 추출 | 높음 (턴별 호출) |
| GAAMA | 3 × chunk 수 | fact + concept + reflection | 중간 |
| Hyper Triplet | 2 × chunk 수 | node-set + reflection | **GAAMA의 ~67%** |

노드셋 추출이 GAAMA보다 LLM 호출이 1/3 적으면서 성능이 같거나 높다면, 비용 효율성도 논문 기여.

### 6.2 추적할 메트릭

```python
# 모든 시스템에서 기록
{
    "system": "hyper_triplet",
    "conv_id": "conv-26",
    "ingestion": {
        "llm_calls": 42,
        "total_input_tokens": 128000,
        "total_output_tokens": 32000,
        "wall_time_sec": 120,
        "nodes_created": 245,
        "edges_created": 530,
        "node_type_counts": {
            "Episode": 32, "Fact": 89, "Topic": 15,
            "Location": 12, "Participant": 28, ...
        }
    },
    "retrieval": {
        "avg_latency_ms": 45,
        "avg_context_tokens": 800
    },
    "evaluation": {
        "overall_accuracy": 0.82,
        "per_category": { "1": 0.85, "2": 0.78, ... }
    }
}
```

---

## Phase 7: 논문 연결

### 7.1 실험 결과 → 논문 구조

| 실험 | 논문 섹션 | 주장 |
|------|---------|------|
| Phase 1~3 비교 | Main Results | 노드셋 추출이 사후 계층화보다 성능이 높다 |
| Phase 5 A1 | Ablation | 맥락 노드가 검색 성능을 유의하게 개선한다 |
| Phase 5 A2 | Ablation | 원자적 적재가 순차 적재보다 fact-context 정합성이 높다 |
| Phase 5 A3 | Ablation | 환경 노드가 temporal/open-domain 질의에 기여한다 |
| Phase 6 | Efficiency | 노드셋 추출이 GAAMA보다 LLM 호출 33% 적다 |

### 7.2 실패 시나리오 대응

| 시나리오 | 대응 |
|---------|------|
| GAAMA 재현 실패 (78.9% 미달) | 평가 프로토콜 차이 확인. EasyLocomo로 크로스체크 |
| Hyper Triplet이 GAAMA보다 낮음 | 프롬프트 개선. 노드셋 구조 단순화. 그래도 낮으면 ablation에서 "왜 낮은지" 분석이 논문 |
| 차이가 유의하지 않음 (±2%p 이내) | 비용 효율성(LLM 호출 33% 감소)으로 기여 전환. "동일 성능, 적은 비용" |
| 카테고리별 편차가 큼 | temporal에서 높고 single-hop에서 같다면, "맥락이 필요한 질의에서 이점" 주장 |

### 7.3 타이틀 후보

```
Option A: "From Triples to Node-Sets:
           Redefining IE Output Units for Graph-Based Agent Memory"

Option B: "Hyper Triplet: Atomic Context Extraction
           for Associative Memory Graphs"

Option C: "Beyond Sequential Layering:
           Atomic Node-Set Ingestion for Long-Term Memory Graphs"
```

---

## 체크리스트

### Phase 0
- [ ] LoCoMo-10 데이터셋 다운로드
- [ ] OpenAI API 키 설정
- [ ] Python 환경 구성 (venv, 의존성)
- [ ] 공통 eval 코드 작성 (judge.py, metrics.py, runner.py)

### Phase 1: HippoRAG2
- [ ] HippoRAG2 코드 clone/구현
- [ ] LoCoMo-10에서 실행
- [ ] 보고값 69.9% 재현 확인
- [ ] 카테고리별 점수 기록

### Phase 2: GAAMA
- [ ] GAAMA 코드 clone
- [ ] LoCoMo-10에서 LTM 생성
- [ ] semantic eval + PPR eval 실행
- [ ] 보고값 78.9% 재현 확인
- [ ] 노드 통계 기록 (타입별 수, 엣지 수)
- [ ] 핵심 코드 파악 (ltm_creator.py, llm_extractors.py)

### Phase 3: Hyper Triplet
- [ ] GAAMA 포크
- [ ] 노드셋 추출 프롬프트 설계 및 테스트
- [ ] 원자적 적재 구현 (MERGE 패턴)
- [ ] 타입드 맥락 노드 스키마 정의
- [ ] LoCoMo-10에서 실행
- [ ] 점수 기록

### Phase 4: 평가
- [ ] 3개 시스템 10회 실행
- [ ] 평균/표준편차 계산
- [ ] 카테고리별 비교
- [ ] 통계적 유의성 검정 (paired t-test or bootstrap)

### Phase 5: Ablation
- [ ] A1: Triple-only 실행
- [ ] A2: Sequential node-set 실행
- [ ] A3: No environment 실행
- [ ] A4: GAAMA concepts as node-set 실행
- [ ] A5: LLM call count control 실행

### Phase 6: 비용 분석
- [ ] LLM 호출 수/토큰 수 비교
- [ ] 노드/엣지 수 비교
- [ ] 레이턴시 비교

### Phase 7: 논문
- [ ] 결과 표 작성
- [ ] ablation 분석
- [ ] 비용 효율성 분석
- [ ] 논문 초고 작성

---

## 참고 코드/데이터 링크

| 리소스 | URL |
|--------|-----|
| LoCoMo-10 데이터셋 | github.com/snap-research/locomo |
| GAAMA 코드 | github.com/swarna-kpaul/gaama |
| EasyLocomo (간소화 평가) | github.com/playeriv65/EasyLocomo |
| LoCoMo 감사 | github.com/dial481/locomo-audit |
| Backboard LoCoMo 벤치 | github.com/Backboard-io/Backboard-Locomo-Benchmark |
| GAAMA 논문 | arxiv.org/abs/2603.27910 |
| HippoRAG2 논문 | arxiv.org/abs/2502.14802 |
| LoCoMo 논문 | arxiv.org/abs/2402.17753 |
