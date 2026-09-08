# paper-rag-eval

평가 주도 RAG 어시스턴트 (AI/ML 논문 QA, arXiv/NLP 도메인). 핵심은 답변 자체가 아니라 **검색·답변 품질을 정량으로 측정·개선하는 평가 하네스**.

## 문서 맵

- [README.md](README.md) — 포트폴리오 개요(어필용)
- [DEVELOPMENT_JOURNEY.md](DEVELOPMENT_JOURNEY.md) — 개발 일지(사실·정량, Part 0 결정 P0-1~6)
- [EVALUATION.md](EVALUATION.md) — 상세 지표·방법론

문서를 새로 쓰거나 진행 기록을 남길 때는 `devlog-convention` skill을 따른다(Part 0/A/B/C 구조·상태표·체크박스 로드맵·ID 의사결정). 구조를 이 파일에 다시 적지 않는다.

## 확정 스펙 (요약 — 상세 근거는 DEVELOPMENT_JOURNEY.md Part 0)

- **도메인**: AI/ML 논문(NLP). **평가셋**: Qasper (gold evidence + reference answer).
- **스택**: Python + 로컬 GPU(RTX 3080, 10GB) `bge-m3` 임베딩 · `bge-reranker-v2-m3` 리랭커 · 로컬 NLI · `Qwen2.5-7B-Instruct`(4-bit) 생성 + FAISS + FastAPI(Week 4).
- **P0-3(2026-09-08 개정)**: 채점 백본은 전면 무료·재현 가능(Qasper Answer-F1 + 로컬 NLI). Claude API는 하드 의존이 아니라 (1) 생성 모델 ablation 축, (2) LLM-judge 상관 검증(선택, ~$10–20)으로만 사용.
- **로드맵**: W1 검색 하네스(recall@k/MRR/nDCG) → W2 답변 하네스(F1/NLI) → W3 ablation+회귀(CI) → W4 서빙+데모.

## 코드 규칙

_(아직 코드 없음 — 첫 모듈 작성 시 언어 버전/포맷터/린터/테스트 명령을 여기에 추가)_

## 커밋

의미 단위로 커밋, 서명 규칙은 세션 지침을 따른다.
