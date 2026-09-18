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

### B-3. BM25 4개 질문 평가 (2026-09-18)

- 입력: B-2와 같은 Qasper train 논문 `1909.00694`의 질문 9개. 개발용 부분집합이며 held-out 최종 평가가 아니다.
- 포함 조건: 정답 주석 1개, `unanswerable=False`, 근거 1개 이상, 모든 근거가 각각 정확히 하나의 본문 문단 ID에 연결됨. 같은 ID의 반복 주석은 집합으로 중복 제거한다.
- 복수 정답은 동일 근거를 공유해도 일괄 보류한다. 일부 근거만 연결된 경우도 질문 전체를 보류한다. 표·그림을 포함한 전체 QA의 지원 범위는 이 결과로 평가하지 않는다.
- 검색 조건: B-2와 동일한 본문 문단 56개 및 BM25 설정, `top_k=5`. 질문마다 기존 검색 함수를 호출하며 색인을 재생성한다.
- 환경: Windows, Python 3.10.7, rank-bm25 0.2.2, NumPy 2.2.6, CPU. 검색·인덱싱 시간은 미측정. 단위 테스트 15개 통과(테스트 러너 보고 시간 0.038초이며 검색 지연시간이 아님).
- Hit@k는 근거를 하나 이상 찾으면 1, Recall@k는 찾은 고유 정답 ID 수 / 전체 고유 정답 ID 수, RR@k는 첫 정답의 1-based 순위 역수다. k 안에서 못 찾으면 셋 모두 0이며 `first_gold_rank`는 `null`이다.
- 평균은 평가한 4개 질문의 단순 평균이다. 보류 5개는 분모에 넣지 않는다. 평가 가능 질문이 0개면 평균도 `null`이다. 출력의 `mean.mrr`은 질문별 `rr`의 평균이다.

| 질문 ID | 질문 | 정답 근거 ID (논문 ID 접두사 생략) | 전체 검색에서 근거 순위 | Hit@5 | Recall@5 | RR@5 |
|---|---|---|---|---|---|---|
| `86abeff85f3db79cf87a8c993e5e5aa61226dc98` | What are labels available in dataset for supervision? | `s0:p0` | 9 | 0 | 0 | 0 |
| `39f8db10d949c6b477fa4b51e7c184016505884f` | How does their model learn using mostly raw data? | `s0:p2` | 30 | 0 | 0 | 0 |
| `d0bc782961567dc1dd7e074b621a6d6be44bb5b4` | How big is seed lexicon used for training? | `s11:p4` | 11 | 0 | 0 | 0 |
| `a592498ba2fac994cd6fad7372836f0adb37e22a` | How large is raw corpus used for training? | `s11:p0`, `s11:p4` | 6, 20 | 0 | 0 | 0 |
| **평균 (4개)** | | | | **0** | **0** | **0 (MRR@5)** |

전체 근거 순위는 별도의 전체 문단 검색으로 진단한 값이다. 평가 JSON의 `first_gold_rank`는 요청한 k 범위만 뜻하므로 이번 k=5 출력에서는 모두 `null`이다. 비교할 개선 실험이 없어 개선율은 산출하지 않는다.

| 보류 질문 ID | 이유 코드 |
|---|---|
| `753990d0b621d390ed58f20c4d9e4f065f0dc672` | `answer_count_not_one` |
| `9d578ddccc27dd849244d632dd0f6bf27348ad81` | `unmapped_evidence` |
| `02e4bf719b1a504e385c35c6186742e720bcb281` | `answer_count_not_one` |
| `44c4bd6decc86f1091b5fc0728873d9324cdde4e` | `answer_count_not_one`, `unmapped_evidence` |
| `c029deb7f99756d2669abad0a349d917428e9c12` | `unmapped_evidence` |

복수 정답은 3개 질문, 미연결 근거는 3개 질문에서 나타나며 1개가 겹친다. 따라서 보류 질문 수는 5개다. 이 외 입력에는 `unanswerable`(답변 불가), `no_evidence`(근거 없음), `ambiguous_evidence`(동일 텍스트가 복수 위치에 연결됨)도 기록한다. `answer_count_not_one`에는 정답 주석 0개도 포함된다.

```powershell
.\.venv\Scripts\python.exe -m src.eval.retrieval data/qasper-train-first-paper.json --top-k 5
```

[평가 코드](src/eval/retrieval.py)는 포함·보류 질문, 정답 ID, 검색 ID와 점수, 지표를 JSON으로 출력한다. 로컬 실행 결과는 `runs/bm25-first-paper-k5.json`에 보관했으며 원시 데이터와 실행 결과 파일은 Git에서 제외한다. 선택된 4개는 표·복수 정답 문제를 제외한 제한된 사례로, 논문 전체 9개나 전체 Qasper의 대표 성능으로 보고하지 않는다.

### B-4. 반환 개수만 5에서 10으로 늘린 비교 (2026-09-18)

B-3와 동일한 논문·4개 질문·본문 문단·BM25 설정·평가 정책을 사용해 두 조건을 다시 실행했다. 변경한 것은 `top_k`뿐이다. 평가 질문 ID와 보류 목록이 동일하고, k=10 결과의 앞 5개 ID와 점수가 k=5 결과와 정확히 같음을 확인했다. 검색 알고리즘이나 순위는 변경하지 않았다.

| 질문 요약 (B-3와 같은 순서) | Hit@5 → Hit@10 | Recall@5 → Recall@10 | RR@5 → RR@10 |
|---|---|---|---|
| 감독 학습 라벨 | 0 → 1 | 0 → 1 | 0 → 0.111111 (1/9) |
| 원시 데이터로 학습하는 방법 | 0 → 0 | 0 → 0 | 0 → 0 |
| seed lexicon 크기 | 0 → 0 | 0 → 0 | 0 → 0 |
| 원시 말뭉치 크기 | 0 → 1 | 0 → 0.5 | 0 → 0.166667 (1/6) |

| 평균 지표 | k=5 | k=10 | 절대 변화 |
|---|---|---|---|
| Hit | 0% | 50% | +50%p |
| Recall (질문별 평균) | 0% | 37.5% | +37.5%p |
| MRR | 0 | 0.069444 | +0.069444 |

기준값이 0이므로 상대 개선율 `(변경 후 - 변경 전) / 변경 전`은 정의하지 않는다. Recall은 `(1+0+0+0.5)/4=0.375`이며 질문마다 같은 가중치를 준다. 정답 문단 수를 모두 합쳐 계산하는 비율과 다르다. 보류 5개는 두 조건 모두 평균에서 제외했다.

9위와 6위 근거가 추가로 포함됐지만 11위·30위 근거는 여전히 범위 밖이다. 원시 말뭉치 질문도 20위 근거는 놓쳤다. **반환 범위를 넓힌 효과이며 순위 개선의 증거는 아니다.** 각 질문의 반환 문단 수는 5개에서 10개로 늘었지만 생성 입력 토큰 수·답변 품질·지연시간·비용은 측정하지 않았다. 기본 k=5는 유지하며 k=10을 최종 설정으로 채택한 것은 아니다.

환경과 평가 대상은 B-3와 같다. 다음 명령으로 재실행할 수 있다. 원시 출력은 로컬 `runs/bm25-first-paper-k5.json`과 `runs/bm25-first-paper-k10.json`에 저장하고 Git에서 제외했다.

```powershell
.\.venv\Scripts\python.exe -m src.eval.retrieval data/qasper-train-first-paper.json --top-k 5
.\.venv\Scripts\python.exe -m src.eval.retrieval data/qasper-train-first-paper.json --top-k 10
```

### B-5. BM25 후보 20개 재정렬 (리랭커) 비교 (2026-09-18)

- 대상: B-3와 같은 논문 `1909.00694`의 평가 4개 질문. 개발용 부분집합이며 held-out 최종 평가가 아니다.
- 방법: 각 질문에서 BM25 상위 20개를 후보로 고정하고, ⓐ 기존 BM25 순서 top-5와 ⓑ 리랭킹 후 top-5를 같은 후보에서 비교. 정답 근거는 리랭커 입력에서 제외.
- 모델: `BAAI/bge-reranker-v2-m3` 교차 인코더. (질문, 문단) 관련도 logit 내림차순, 동점은 BM25 순서 유지. max_length 512, batch 8.
- 환경: Windows, Python 3.10.7, torch 2.5.1+cu121, transformers 5.17.0, RTX 3080 GPU 추론. 지연시간 미측정. 후보 20개 중 잘린 것 0개.

| 질문 요약 | gold(접두사 생략) | BM25 후보 순위 | 기존 top-5 Hit | 리랭킹 top-5 Hit | 리랭킹 후 gold 순위(20중) |
|---|---|---|---|---|---|
| 감독 학습 라벨 | s0:p0 | 9 | 0 | 0 | 14 |
| 원시 데이터 학습 방법 | s0:p2 | 30 | 0 | 0 | 후보 밖 |
| seed lexicon 크기 | s11:p4 | 11 | 0 | 1 | 2 |
| 원시 말뭉치 크기 | s11:p0, s11:p4 | 6, 20 | 0 | 0 | 6, 7 |

| 평균 지표 (4개) | BM25 top-5 | 리랭킹 top-5 |
|---|---|---|
| Hit@5 | 0 | 0.25 |
| Recall@5 | 0 | 0.25 |
| MRR@5 | 0 | 0.125 |

리랭킹이 회복한 1개는 seed lexicon 질문(BM25 11위 → 리랭킹 2위)이다. 감독 학습 라벨(hard miss, gold 14위)·원시 말뭉치(near miss, gold 6·7위)는 후보 안에 있었지만 top-5에 못 들었고, 원시 데이터 학습 질문은 gold가 후보 20 밖(BM25 30위)이라 리랭킹으로 회복 불가다. 기준값이 0이라 상대 개선율은 산출하지 않는다. 단일 논문·4문항 개발용 결과로 전체 Qasper 대표 성능이 아니다.

```powershell
.\.venv\Scripts\python.exe -m scripts.rerank_experiment data/qasper-train-first-paper.json --candidate-k 20 --top-k 5 --out runs/rerank-first-paper-k20.json
```

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
