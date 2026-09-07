# (제목 미정) — 평가 주도 RAG 어시스턴트 for AI/ML 논문

> AI/ML 논문(arXiv·NLP)에 대해 질문하면 근거 문단을 인용해 답한다. 핵심은 **답변 그 자체가 아니라, 검색·답변 품질을 정량으로 측정·개선하는 평가 하네스**다.

_이 README는 포트폴리오 어필용 개요다. 개발 일지는 [DEVELOPMENT_JOURNEY](DEVELOPMENT_JOURNEY.md), 상세 지표는 [EVALUATION](EVALUATION.md) 참고._

## 한눈에

- **문제:** 대부분의 RAG 데모는 "그럴듯한 답"을 보여줄 뿐, 검색이 맞는 문단을 가져왔는지·답이 근거에 충실한지를 **측정하지 않는다**.
- **접근:** 검색 평가(recall@k·MRR·nDCG) + 답변 평가(Qasper Answer-F1 · 로컬 NLI 기반 faithfulness/환각률) + 청킹/임베딩/리랭커/프롬프트/생성모델 **ablation** + **회귀 테스트(CI식)** 를 1급 시민으로.
- **채점은 전부 무료·재현 가능**(gold evidence + 표준 F1 + 로컬 NLI). Claude 는 하드 의존이 아니라 로컬 vs API 를 측정으로 비교하는 대상.
- **스택:** 로컬 GPU 임베딩·리랭커·NLI·생성(bge-m3 / bge-reranker / Qwen2.5-7B, RTX 3080) + FAISS + (선택) Claude API + FastAPI.

## 핵심 성과표

_(수치 확보 후 채움 — 예: 베이스라인 대비 best config Recall@5, 환각률 개선 %)_

| 지표 | 베이스라인 | 개선 후 | Δ |
|---|---|---|---|
| Recall@5 | — | — | — |
| Faithfulness | — | — | — |
| Hallucination rate | — | — | — |

## 데모

_(Week 4: GIF)_

## 아키텍처

_(다이어그램 예정: ingest → chunk → embed(로컬 GPU) → FAISS → retrieve → rerank → Claude 생성(인용) → LLM-judge 평가)_

## 문서

- [DEVELOPMENT_JOURNEY.md](DEVELOPMENT_JOURNEY.md) — 개발 일지 (기술 결정·정량 로그)
- [EVALUATION.md](EVALUATION.md) — 평가 방법·상세 지표 표

## 데이터·라이선스

평가·코퍼스: **Qasper** (Allen Institute for AI, NLP 논문 QA). arXiv 논문 라이선스는 논문별 상이 — 사용 시 명시.
