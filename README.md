# PaperRAG — 평가 주도 RAG 어시스턴트 for AI/ML 논문

> AI/ML 논문(arXiv·NLP)에 대해 질문하면 근거 문단을 인용해 답한다. 핵심은 **답변 그 자체가 아니라, 검색·답변 품질을 정량으로 측정·개선하는 평가 하네스**다.

_이 README는 포트폴리오 어필용 개요다. 개발 일지는 [DEVELOPMENT_JOURNEY](DEVELOPMENT_JOURNEY.md), 상세 지표는 [EVALUATION](EVALUATION.md) 참고._

## 한눈에

- **문제:** 대부분의 RAG 데모는 "그럴듯한 답"을 보여줄 뿐, 검색이 맞는 문단을 가져왔는지·답이 근거에 충실한지를 **측정하지 않는다**.
- **접근:** 검색 평가(recall@k·MRR·nDCG) + 답변 평가(Qasper Answer-F1 · 로컬 NLI 기반 faithfulness/환각률) + 청킹/임베딩/리랭커/프롬프트/생성모델 **ablation** + **회귀 테스트(CI식)** 를 1급 시민으로.
- **채점은 전부 무료·재현 가능**(gold evidence + 표준 F1 + 로컬 NLI). Claude 는 하드 의존이 아니라 로컬 vs API 를 측정으로 비교하는 대상.
- **스택:** 로컬 GPU 임베딩·리랭커·NLI·생성(bge-m3 / bge-reranker / Qwen2.5-7B, RTX 3080) + FAISS + (선택) Claude API + FastAPI.

## 구현 범위 (계획)

첫 버전은 **논문 하나를 선택하고 질문하는 QA**다. 선택한 논문 안에서 근거를 검색하고, 답변과 인용 원문을 함께 제공한다. 근거가 부족하면 답변을 보류한다. 검색·답변 평가 결과와 실패 사례를 재실행할 수 있는 상태까지 완성한다.

이후 **관련 논문 검색 → 논문 선택 → 해당 논문에 질문**하는 흐름으로 확장한다. 여러 논문을 종합하는 답변은 별도 후속 후보로 둔다. 결정 근거와 완료 기준은 DEVELOPMENT_JOURNEY의 P0-7에 기록한다.

## 핵심 성과표

_10편 개발용 부분집합(micro 평균) — 검색 축은 보류 규칙 통과 26문항, 답변 축은 전체 51문항. 검색은 BM25 베이스라인 대비 best config(dense→리랭커), 답변 축은 oracle 근거(천장) 대비 검색 근거(실제)의 진단. 상세·조건은 [EVALUATION](EVALUATION.md)._

**검색 (BM25 → dense→리랭커)**

| 지표 | BM25 | dense→리랭커 | Δ |
|---|---|---|---|
| Hit@5 | 0.385 | 0.692 | +0.308 |
| Recall@5 | 0.346 | 0.577 | +0.231 |

**답변 축 진단 (oracle 근거 → 검색 근거)**

| 지표 | oracle(천장) | 검색(실제) |
|---|---|---|
| Answer-F1 | 0.533 | 0.416 |
| 충실율(NLI) | 0.778 | 0.615 |
| 환각율 | 0.222 | 0.385 |

검색 품질이 답 품질의 천장이다 — 검색이 gold 문단을 top-5에 못 올리면 정확도·충실성이 함께 떨어진다(B-8·B-9). 작동점은 근거 수 k=5다: Answer-F1이 k=5에서 최대이고 k=10은 추가 근거의 노이즈로 오히려 하락한다(0.416→0.366, B-15).

## 데모

현재는 단일 논문 JSON 로더, BM25 문단 검색, 명확한 본문 근거가 있는 질문의 검색 평가를 실행할 수 있다. GPU 패키지 없이 확인하려면 다음과 같이 설치한다. 입력 JSON은 Hugging Face Qasper의 논문 한 행을 저장한 형식이며 별도로 준비해야 한다. 로컬 GPU로는 dense(bge-m3)·리랭커(bge-reranker) 검색과 BM25·dense·hybrid(RRF) 비교도 실행하며, 10편 26문항 결과는 [EVALUATION의 B-6](EVALUATION.md)에 기록한다. 또한 Qwen2.5-7B로 근거에서 답을 생성해 Answer-F1로 채점하고(oracle 근거 대비 검색 근거의 손실 측정), 결과는 [EVALUATION의 B-8](EVALUATION.md)에 기록한다. W3에서는 골든(`baselines/retrieval.json`) 대비 검색 지표 하락을 회귀로 막고(CPU는 GitHub Actions 자동, GPU는 로컬 수동), config·top-k별 ablation 표를 재생성한다(B-11~B-13).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install rank-bm25==0.2.2 numpy==2.2.6
.\.venv\Scripts\python.exe -m src.retrieve.bm25 data/qasper-train-first-paper.json "How big is seed lexicon used for training?" --top-k 5
.\.venv\Scripts\python.exe -m src.eval.retrieval data/qasper-train-first-paper.json --top-k 5
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

평가 명령은 질문별 검색 ID·점수·Hit·Recall·RR, 질문 평균, 보류 사유를 JSON으로 출력한다. 첫 샘플에서는 9개 중 4개를 평가하고 5개를 보류한다. 전체 벤치마크 성능이 아닌 개발용 실행 결과이며, 포함 조건과 수치는 [EVALUATION의 B-3](EVALUATION.md#b-3-bm25-4개-질문-평가-2026-09-18)에 기록한다.

**W4 — FastAPI `/ask` 서빙 (Swagger `/docs`)**

![/ask 데모](docs/demo.gif)

논문 한 편과 질문을 주면 검색(dense→리랭커)으로 근거 문단을 모아 로컬 7B가 답하고, 인용 문단과 함께 반환한다. 근거가 답을 담지 못하면 보류한다(`abstained`). 세 모델은 서버 기동 시 1회 로딩해 상주한다 — 10GB VRAM 제약 때문에 경량 모델(임베더·리랭커)은 CPU, 7B만 GPU 4bit에 둔다. 그래서 7B 로드(~28초)는 기동 때 한 번뿐이고 이후 요청은 즉답이다(B-14).

```powershell
.\.venv\Scripts\python.exe -m pip install fastapi==0.115.6 "uvicorn[standard]==0.34.0"
.\.venv\Scripts\python.exe -m uvicorn src.serve.app:app
# http://127.0.0.1:8000/docs 에서 /ask 호출 (paper_path, question)
```

## 아키텍처

![아키텍처 — 런타임 파이프라인과 평가 하네스](docs/architecture.svg)

런타임은 논문·질문을 받아 검색(dense bge-m3) → 리랭커(bge-reranker-v2-m3) → 생성(Qwen2.5-7B)으로 답·인용을 내고 근거 부족 시 보류한다. 평가 하네스가 각 단계를 gold evidence·reference answer·NLI로 측정하고, ablation·회귀 골든(CI)으로 지표 하락을 막는다.

## 문서

- [DEVELOPMENT_JOURNEY.md](DEVELOPMENT_JOURNEY.md) — 개발 일지 (기술 결정·정량 로그)
- [EVALUATION.md](EVALUATION.md) — 평가 방법·상세 지표 표

## 데이터·라이선스

평가·코퍼스: **Qasper** (Allen Institute for AI, NLP 논문 QA). arXiv 논문 라이선스는 논문별 상이 — 사용 시 명시.
