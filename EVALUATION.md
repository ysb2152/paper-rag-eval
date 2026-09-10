# EVALUATION

> 평가 방법과 정량 지표. 측정 방법(테스트셋 크기·하드웨어·시간)을 함께 기록해 재현 가능하게 한다.

## 평가셋

- **검색 범위(P0-7):** 첫 버전에서는 질문 대상 논문을 입력으로 제공하고 해당 논문의 본문만 검색한다. 관련 논문을 찾는 기능은 첫 버전 완료 후 추가하며, 전체 코퍼스 검색 결과는 별도 평가 조건으로 보고한다.
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

### B-2. BM25 단일 질문 실행 결과 (2026-09-10)

- 대상: Qasper train 논문 `1909.00694`, 질문 `d0bc782961567dc1dd7e074b621a6d6be44bb5b4` 한 개. 전체 벤치마크 결과가 아니다.
- 질문: `How big is seed lexicon used for training?`
- 검색 범위: 해당 논문의 본문 문단 72개 중 토큰이 있는 56개. 제목·초록·섹션명·정답 주석은 검색 점수에 사용하지 않는다.
- 조건: 소문자 변환 + `\w+` 토큰화, 불용어 제거·어간 추출 없음. BM25Okapi(`k1=1.5`, `b=0.75`, `epsilon=0.25`), 동점은 원문 순서.
- 환경: Windows, Python 3.10.7, rank-bm25 0.2.2, NumPy 2.2.6. CPU 실행이며 검색 시간은 미측정.

| 순위 | 문단 ID | BM25 점수 | 주석의 정답 근거 |
|---|---|---|---|
| 1 | `1909.00694:s5:p0` | 7.6175 | 아니오 |
| 2 | `1909.00694:s14:p1` | 6.6708 | 아니오 |
| 3 | `1909.00694:s12:p0` | 6.1692 | 아니오 |
| 4 | `1909.00694:s12:p6` | 5.9757 | 아니오 |
| 5 | `1909.00694:s4:p1` | 5.1750 | 아니오 |
| 11 | `1909.00694:s11:p4` | 3.3243 | 예 |

정답 근거는 전체 순위를 조회한 뒤 주석과 대조했다. BM25 점수는 답변 정확도나 확률이 아니다. 비교할 개선 실험이 아직 없으므로 개선율은 산출하지 않았다.

로컬에 저장한 단일 논문 JSON을 입력으로 사용한다. 원본 데이터 파일은 Git에 포함하지 않는다.

```powershell
.\.venv\Scripts\python.exe -m src.retrieve.bm25 data/qasper-train-first-paper.json "How big is seed lexicon used for training?" --top-k 5
```

정답 근거의 전체 순위를 확인하려면 같은 명령의 `--top-k`를 `72`로 지정한다.

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
