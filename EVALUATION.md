# EVALUATION

> 평가 방법과 정량 지표. 측정 방법(테스트셋 크기·하드웨어·시간)을 함께 기록해 재현 가능하게 한다.

## 평가셋

- **백본:** Qasper (NLP 논문 QA). 질문마다 **gold evidence paragraph** 와 **reference answer** 라벨 보유.
- **dev / test 분리:** dev(소형, 반복 실험·비용 절감) / held-out test(최종 보고). 크기·분리 방법은 착수 시 확정·기록.
- **하드웨어:** 로컬 Windows + RTX 3080 (10GB). 임베딩·리랭커·NLI·로컬 생성(7B) 로컬 추론. 채점 백본 무료. Claude 생성/judge 는 비교·검증용(선택).

## 검색 평가 (Retrieval)

gold evidence paragraph 를 정답 집합으로 자동 채점.

| 지표 | 정의 |
|---|---|
| Recall@k | 상위 k 안에 gold evidence 포함 비율 |
| MRR@k | 첫 gold 문단의 역순위 평균 |
| nDCG@k | 순위 가중 정확도 |
| Hit@k | 최소 1개 gold 포함 여부 |

_(결과 표: config × 지표. before/after + % 개선, k 값·eval셋 크기·인덱싱 시간 명시.)_

## 답변 평가 (Answer)

기본 채점은 **무료·재현 가능**(표준지표 + 로컬 모델). LLM-judge 는 이들과의 상관을 보는 **선택적 검증 지표**.

| 지표 | 정의 | 채점 방법 | 비용 |
|---|---|---|---|
| Answer Correctness | reference answer 대비 정확도 | **Qasper Answer-F1** (벤치마크 표준) | 무료 |
| Faithfulness | 답이 검색된 근거에 함의되는가(환각 없음) | **로컬 NLI**(evidence ⊨ answer, 예: DeBERTa-MNLI/ANLI) | 무료 |
| Hallucination Rate | 근거로 함의되지 않는 문장 비율 | 로컬 NLI (문장 단위 non-entailment) | 무료 |
| (검증) Judge 상관 | 위 지표 vs LLM-judge 일치도 | Claude judge 소량 + 상관계수 | ~$10–20 (선택) |

_(로컬 NLI 모델 선정·faithfulness 근사 정확도, LLM-judge 프롬프트·자기일관성 방침, 인간 검증 서브셋 상관은 착수 시 기록.)_

## Ablation

| 축 | 후보 |
|---|---|
| 청킹 | 크기/overlap, fixed vs semantic |
| 임베딩 | bge-m3 vs 대안 |
| 검색 | dense vs hybrid(BM25+dense) |
| 리랭커 | none vs bge-reranker-v2-m3 |
| top-k | 3 / 5 / 10 |
| 프롬프트 | 변형 A/B |
| 생성 모델 | 로컬 Qwen2.5-7B vs claude-sonnet-5 vs claude-opus-5 |

_(각 축별 표 + 채택 결정 ID.)_

## 회귀 (CI)

고정 test셋 + 지표 임계값. 변경마다 재실행, 하락 시 실패. 결과 로그를 상태표/여기 표로 누적.

## 비용 로그

| 항목 | 호출 수 | 모델 | 실측 비용 |
|---|---|---|---|
| 생성 | — | — | — |
| judge | — | — | — |

_(Batch API 사용 여부·절감액 포함.)_
