# paper-rag-eval

평가 주도 RAG 어시스턴트 (AI/ML 논문 QA, arXiv/NLP 도메인). 핵심은 답변 자체가 아니라 **검색·답변 품질을 정량으로 측정·개선하는 평가 하네스**.

## 문서 맵

- [README.md](README.md) — 포트폴리오 개요(어필용)
- [DEVELOPMENT_JOURNEY.md](DEVELOPMENT_JOURNEY.md) — 개발 일지(사실·정량, Part 0 결정 P0-1~6)
- [EVALUATION.md](EVALUATION.md) — 상세 지표·방법론

문서를 새로 쓰거나 진행 기록을 남길 때는 `devlog-convention` skill을 따른다(Part 0/A/B/C 구조·상태표·체크박스 로드맵·ID 의사결정). 구조를 이 파일에 다시 적지 않는다.

## 확정 스펙 (요약 — 상세 근거는 DEVELOPMENT_JOURNEY.md Part 0)

- **도메인**: AI/ML 논문(NLP). **평가셋**: Qasper (gold evidence + reference answer).
- **P0-7 — 구현 범위**: 단일 논문 QA(답변·인용·답변 보류·평가)를 먼저 완성한 뒤 관련 논문 검색 → 논문 선택 → QA로 확장한다. 여러 논문 종합 답변은 별도 후속 후보다. 완료 기준은 DEVELOPMENT_JOURNEY.md의 P0-7을 따른다.
- **스택**: Python + 로컬 GPU(RTX 3080, 10GB) `bge-m3` 임베딩 · `bge-reranker-v2-m3` 리랭커 · 로컬 NLI · `Qwen2.5-7B-Instruct`(4-bit) 생성 + FAISS + FastAPI(Week 4).
- **P0-3(2026-09-08 개정)**: 채점 백본은 전면 무료·재현 가능(Qasper Answer-F1 + 로컬 NLI). Claude API는 하드 의존이 아니라 (1) 생성 모델 ablation 축, (2) LLM-judge 상관 검증(선택, ~$10–20)으로만 사용.
- **로드맵**: W1 검색 하네스(recall@k/MRR/nDCG) → W2 답변 하네스(F1/NLI) → W3 ablation+회귀(CI) → W4 서빙+데모.

## 코드 규칙

- 첫 로더는 Python 3.10 이상에서 표준 라이브러리만 사용한다. 확인한 환경은 Python 3.10.7이다.
- BM25 검색은 프로젝트 `.venv`의 rank-bm25 0.2.2를 사용한다. 최소 설치 명령은 README.md 참고.
- 테스트: `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- 포맷터·린터는 아직 선정하지 않았다.

## 커밋

의미 단위로 커밋, 서명 규칙은 세션 지침을 따른다.
