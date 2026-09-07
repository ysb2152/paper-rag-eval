# DEVELOPMENT_JOURNEY

> 개발 일지. 사실·정량 로그만 담는다. 홍보/어필 서술은 [README](README.md), 상세 지표 표는 [EVALUATION](EVALUATION.md).

## 상태표 (최신 우선)

| 날짜(ISO) | 분류 | 내용 | 관련ID |
|---|---|---|---|
| 2026-09-08 | decision | P0-3 개정 — 채점 무료 로컬/Qasper-F1/NLI, Claude 는 비교·검증용(하드 의존 아님), 생성 로컬7B↔Claude ablation축 | P0-3 |
| 2026-09-08 | decision | 스택·평가셋·주차계획 확정, 리포 스캐폴딩 | P0-1 ~ P0-6 |
| 2026-09-08 | docs | DEVELOPMENT_JOURNEY / README / EVALUATION 3분리 골격 생성 | — |

---

# Part 0 — 기획 · 기술 결정

## 프로젝트 개요

**평가 주도 RAG 어시스턴트 (도메인: AI/ML 논문 / arXiv·NLP)**

대부분의 RAG 데모가 빠뜨리는 **평가 하네스**를 1급 시민으로 삼는 프로젝트. "동작하는 데모"가 아니라 **측정 → 진단 → ablation → 회귀 테스트** 루프로 검색·생성 품질을 정량 관리하는 것이 목표. 직전 프로젝트(saturi-translator)에서 보인 "데이터로 진단·held-out 측정·ablation 비교" 강점을 LLM 애플리케이션 축으로 확장.

## 기술 결정 (트레이드오프·기각 대안 명시)

### P0-1. 언어·런타임 = Python
- **결정:** Python. 로컬 GPU 임베딩/리랭커 생태계(sentence-transformers, FlagEmbedding)와 RAG 툴링이 Python 중심.
- **기각:** TS/Node — 로컬 임베딩 추론 생태계 빈약.

### P0-2. 임베딩·리랭커 = 로컬 GPU (RTX 3080, 10GB)
- **결정:** 임베딩 `BAAI/bge-m3`, 리랭커 `BAAI/bge-reranker-v2-m3` 를 로컬 GPU에서 추론.
- **근거:** (1) 비용 0 — 관리형 임베딩 API의 반복 인덱싱·재실험 비용 제거. (2) GPU 활용·온디바이스 추론 역량 어필(saturi 연장선). (3) bge-m3는 다국어·긴 컨텍스트(8k) 지원으로 논문 문단에 적합.
- **제약/미지수:** 10GB VRAM — bge-m3(560M) + 리랭커 동시 상주 시 배치 크기 조정 필요. full-text 임베딩 시 인덱싱 시간 측정 예정.
- **기각:** Voyage/OpenAI 임베딩 — 실험 반복마다 과금, 포트폴리오상 "로컬 ML" 어필 손실.

### P0-3. 채점은 무료 로컬/표준지표, Claude 는 하드 의존이 아니라 비교·검증용 (2026-09-08 개정)
- **초기 가정(폐기):** 생성·채점을 모두 Claude API 로. → 유일 과금이 채점(LLM-judge)에 몰려 예산 리스크. 또한 LLM-judge 는 채점 기준 자체라 재현성·객관성 약점.
- **개정 결정:**
  - **정량 채점 백본 = 전부 무료·재현 가능.** 검색은 gold evidence 자동채점(recall@k/MRR/nDCG). 답변 정확도는 **Qasper 공식 Answer-F1**(이 벤치마크의 표준 지표). Faithfulness/환각은 **로컬 NLI 모델**(evidence ⊨ answer 함의)로 근사.
  - **답변 생성 = 로컬 7B + Claude 를 ablation 축으로.** `Qwen2.5-7B-Instruct`(4-bit, 10GB 상주) vs `claude-sonnet-5` vs `claude-opus-5`. 생성 모델 품질을 측정 대상으로 삼음(컨셉 정합).
  - **LLM-as-judge = 선택적 검증 도구.** 소량만 돌려 "judge 점수 vs F1/NLI 상관" 을 보여 judge 신뢰도 검증. Claude Haiku + Batch API 로 ~$10–20. 미실행해도 프로젝트 성립.
- **근거:** (1) 정량 백본이 무료라 예산 리스크 제거·완전 로컬 재현 가능(포트폴리오 강점). (2) Qasper 는 원래 F1 채점 벤치마크 → 표준 지표가 임시 LLM-judge보다 객관적. (3) Claude 를 "프리미엄 비교 대상"으로 두면 로컬 vs API 트레이드오프를 **측정으로** 보여줄 수 있음.
- **기각:** 생성·채점 전면 Claude — 예산 리스크 + 채점 객관성 약화. / 채점 전면 로컬 LLM-judge — 7B judge 신뢰도 부족, 표준 F1 대비 이점 없음.
- **미지수:** 로컬 NLI 모델 선정(예: DeBERTa-v3 MNLI/ANLI)·faithfulness 근사 정확도, Qasper unanswerable/abstractive 답변의 F1 처리, 7B 생성 VRAM(임베딩·리랭커와 동시 상주 시 스케줄링).

### P0-4. 벡터 스토어 = FAISS (로컬), pgvector 는 ablation 축
- **결정:** FAISS(flat/IVF)로 시작. pgvector 는 "운영형 스토어" 비교 축으로 후순위.
- **근거:** 로컬·의존성 최소·재현 쉬움. 벡터 스토어 자체는 이 프로젝트의 핵심 변수가 아님(핵심은 평가 하네스).

### P0-5. 평가셋 = Qasper 백본 + 합성 보강
- **결정:** **Qasper**(NLP 논문 QA, gold evidence paragraph + reference answer 라벨) 를 평가 백본으로. Qasper 논문 = 코퍼스, Qasper 질문 = eval셋.
- **근거:** (1) retrieval 채점을 gold evidence 로 **자동화**(recall@k/MRR/nDCG). (2) answer faithfulness/correctness 를 reference answer 로 채점. (3) 평가셋 직접 구축 노동 대폭 절감 + "평가 주도" 컨셉 정합.
- **보강:** 커버리지 부족 유형은 Claude 로 합성 QA(질문+gold passage+reference) 생성 후 일부 수동 검증.
- **미지수:** Qasper full-text 파싱 품질, unanswerable 질문 처리 방침.

### P0-6. 서빙 = FastAPI + 최소 UI (후순위)
- **결정:** 평가 하네스·수치 확보가 먼저. 서빙(스트리밍 답변+인용 표시)은 Week 4.

## 평가 지표 (설계) — 상세는 [EVALUATION](EVALUATION.md)

- **검색(retrieval):** Recall@k, MRR@k, nDCG@k, Hit@k — gold evidence 대비.
- **답변(answer):** Answer Correctness = **Qasper Answer-F1**(무료·표준). Faithfulness/Hallucination = **로컬 NLI**(evidence⊨answer, 무료). LLM-as-judge 는 이들과의 **상관 검증용 선택 지표**.
- **회귀(CI):** 고정 eval셋 + 임계값, 변경마다 재실행·표 기록.

## Ablation 축 (예정)

청킹(크기/overlap, fixed vs semantic) · 임베딩 모델 · 검색(dense vs hybrid BM25+dense) · 리랭커(none vs bge-reranker) · top-k · 프롬프트 변형 · **생성 모델(로컬 Qwen2.5-7B vs claude-sonnet-5 vs claude-opus-5)**.

## 주차별 로드맵

- [ ] **Week 1 — 데이터 + 베이스라인 검색.** Qasper 적재·청킹·bge-m3 임베딩·FAISS 인덱스·dense 검색. **검색 평가 하네스**(recall@k/MRR/nDCG) → 첫 수치.
- [ ] **Week 2 — 생성 + 답변 평가.** Claude 인용 기반 생성 · LLM-judge(faithfulness/correctness/hallucination). end-to-end 베이스라인 수치.
- [ ] **Week 3 — Ablation + 회귀.** 청킹·임베딩·hybrid·리랭커·top-k·프롬프트 비교표 → best config. **회귀 하네스**(임계값 + CI).
- [ ] **Week 4 — 서빙 + 마감.** FastAPI + 최소 UI(스트리밍+인용) · 데모 GIF · 문서 마감.

---

# Part A — 개요 · 설계

_(Week 1 착수 시 작성: 문제 정의 · 유저 플로우 · 파이프라인 다이어그램)_

---

# Part B — 개발 로그

_(B-1 부터 시간순. 제목=질문/행동 · 문제/제약 문맥 · 해결 서술 · 결과/결정 볼드 · 일반 원칙 1줄)_

---

# Part C — 회고

## C-1. 배운 것
_(주제별 원칙 + 구체 사례, 초기 가정 vs 현실)_

## C-2. 남은 로드맵
_(위 체크박스 로드맵과 연동)_
