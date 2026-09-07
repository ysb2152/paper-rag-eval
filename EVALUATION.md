# EVALUATION

> 평가 방법과 정량 지표. 측정 방법(테스트셋 크기·하드웨어·시간)을 함께 기록해 재현 가능하게 한다.

## 평가셋

- **백본:** Qasper (NLP 논문 QA). 질문마다 **gold evidence paragraph** 와 **reference answer** 라벨 보유.
- **dev / test 분리:** dev(소형, 반복 실험·비용 절감) / held-out test(최종 보고). 크기·분리 방법은 착수 시 확정·기록.
- **하드웨어:** 로컬 Windows + RTX 3080 (10GB). 임베딩·리랭커 로컬 추론. 생성·judge = Claude API.

## 검색 평가 (Retrieval)

gold evidence paragraph 를 정답 집합으로 자동 채점.

| 지표 | 정의 |
|---|---|
| Recall@k | 상위 k 안에 gold evidence 포함 비율 |
| MRR@k | 첫 gold 문단의 역순위 평균 |
| nDCG@k | 순위 가중 정확도 |
| Hit@k | 최소 1개 gold 포함 여부 |

_(결과 표: config × 지표. before/after + % 개선, k 값·eval셋 크기·인덱싱 시간 명시.)_

## 답변 평가 (Answer) — LLM-as-judge

| 지표 | 정의 | 채점 |
|---|---|---|
| Faithfulness | 답이 검색된 근거에만 기반하는가(환각 없음) | LLM-judge |
| Answer Correctness | reference answer 대비 정확도 | LLM-judge |
| Citation Accuracy | 인용 문단이 실제로 주장을 뒷받침하는가 | LLM-judge |
| Hallucination Rate | 근거 없는 주장 비율 | LLM-judge |

_(judge 모델·프롬프트·자기일관성(반복/온도) 방침, 인간 검증 서브셋 상관 기록 예정.)_

## Ablation

| 축 | 후보 |
|---|---|
| 청킹 | 크기/overlap, fixed vs semantic |
| 임베딩 | bge-m3 vs 대안 |
| 검색 | dense vs hybrid(BM25+dense) |
| 리랭커 | none vs bge-reranker-v2-m3 |
| top-k | 3 / 5 / 10 |
| 프롬프트 | 변형 A/B |

_(각 축별 표 + 채택 결정 ID.)_

## 회귀 (CI)

고정 test셋 + 지표 임계값. 변경마다 재실행, 하락 시 실패. 결과 로그를 상태표/여기 표로 누적.

## 비용 로그

| 항목 | 호출 수 | 모델 | 실측 비용 |
|---|---|---|---|
| 생성 | — | — | — |
| judge | — | — | — |

_(Batch API 사용 여부·절감액 포함.)_
