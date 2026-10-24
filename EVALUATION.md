# EVALUATION

> 평가 방법과 정량 지표. 측정 방법(테스트셋 크기·하드웨어·시간)을 함께 기록해 재현 가능하게 한다.

## 평가셋

- **검색 범위(P0-7):** 첫 버전에서는 질문 대상 논문을 입력으로 제공하고 해당 논문의 본문만 검색한다(아래 검색·답변 평가). 관련 논문을 찾는 기능은 첫 버전 완료 후 추가했고, 전체 코퍼스에서 논문을 고르는 조건은 별도로 보고한다([코퍼스 라우팅 평가](#코퍼스-라우팅-평가-corpus-routing)).
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

### B-6. BM25 · dense(bge-m3) · hybrid(RRF) 비교 (2026-09-18)

- 대상: Qasper train 앞 10편에서 보류 규칙(B-3)을 통과한 평가 질문 26개. 체리피킹 없이 원본 등장 순서대로 추출. 개발용 부분집합이며 held-out 최종 평가가 아니다.
- 방법: 세 방식 모두 논문의 전체 문단을 순위 매기고 top-5로 Hit/Recall/MRR을 집계(마이크로 평균, 질문 26개 균등 가중). dense는 `BAAI/bge-m3`의 CLS 벡터를 L2 정규화해 코사인 유사도로 정렬. hybrid는 BM25·dense 순위를 RRF(1/(k+순위) 합, k=60)로 융합.
- 환경: Windows, Python 3.10.7, torch 2.5.1+cu121, transformers 5.17.0, RTX 3080 GPU 추론. max_length 512. 지연시간 미측정.

| 평균 지표 (26문항) | BM25 | dense | hybrid |
|---|---|---|---|
| Hit@5 | 0.385 | **0.615** | 0.423 |
| Recall@5 | 0.346 | **0.558** | 0.404 |
| MRR@5 | 0.237 | **0.404** | 0.304 |
| Hit 개수(26중) | 10 | 16 | 11 |

- dense가 세 지표 모두 최선이고, 단순 RRF hybrid는 BM25와 dense 사이에 머물러 dense를 넘지 못한다. BM25가 dense보다 약해 동등 가중 융합이 강한 신호를 희석하기 때문이다.
- 표본 크기 주의: 같은 실험을 첫 논문 1편(평가 4문항)만으로 하면 BM25 Hit@5 0, dense 0.5, hybrid 0.25로, 그 논문이 BM25에 유난히 불리해 오해를 준다. 26문항으로 늘리자 BM25가 0.385로 올라왔다. 단일 논문 수치는 결론으로 삼지 않는다.
- 10편·26문항도 Qasper 전체 대표 성능은 아니며, held-out 최종 평가는 이후 단계다.

```powershell
# Qasper 원본을 data/qasper-raw/ 에 두고 10편 추출 후 실행
.\.venv\Scripts\python.exe -m scripts.extract_papers --count 10
.\.venv\Scripts\python.exe -m scripts.hybrid_experiment data/papers/*.json --top-k 5 --out runs/hybrid-10papers.json
```

### B-7. 1단계 검색기별 리랭킹 파이프라인 비교 (2026-09-18)

- 대상: B-6과 같은 10편 26문항. 개발용 부분집합이며 held-out 최종 평가가 아니다.
- 방법: 세 방식의 top-5를 마이크로 평균 비교. ⓐ dense 단독, ⓑ BM25 후보 20 → 리랭킹, ⓒ dense 후보 20 → 리랭킹. 정답 근거는 리랭커 입력에서 제외.
- 모델: 검색 `BAAI/bge-m3`, 리랭커 `BAAI/bge-reranker-v2-m3`를 함께 로드. candidate_k 20, max_length 512.
- 환경: Windows, Python 3.10.7, torch 2.5.1+cu121, transformers 5.17.0, RTX 3080 GPU 추론. 지연시간·VRAM 미측정.

| 평균 지표 (26문항) | dense 단독 | BM25→리랭커 | dense→리랭커 |
|---|---|---|---|
| Hit@5 | 0.615 | 0.462 | **0.692** |
| Recall@5 | 0.558 | 0.404 | **0.577** |
| MRR@5 | 0.404 | 0.335 | **0.461** |

- dense→리랭커가 전 지표 최선으로, 리랭킹이 dense 위에서 추가 이득을 낸다(Hit 0.615→0.692).
- BM25→리랭커(0.462)는 dense 단독(0.615)보다도 낮다. 같은 리랭커라도 후보가 약하면 좋은 후보(dense)의 무리랭킹보다 못하다. 후보만 dense로 바꾸면 0.462→0.692.
- 해석: 리랭커는 후보 집합 안에서만 재정렬하므로 gold가 후보 밖이면 회복 불가. 1단계 검색 recall이 리랭커 성능의 천장이다. W1 전체 그림은 BM25 0.385(B-6) → dense 0.615 → dense→리랭커 0.692.

```powershell
.\.venv\Scripts\python.exe -m scripts.pipeline_experiment data/papers/*.json --candidate-k 20 --top-k 5 --out runs/pipeline-10papers.json
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

### B-8. Answer-F1: 추출 베이스라인 · 7B 생성 · 검색 근거 (2026-09-18)

- 대상: 10편의 전체 질문 51개(보류 필터 없이 채점). 개발용 부분집합이며 held-out 최종 평가가 아니다.
- 채점: Qasper 공식 Answer-F1(정규화 후 토큰 F1, 주석자 최대). 유형별 개수는 예측이 가장 잘 맞는 reference로 분류돼 실행마다 흔들리므로 전체 평균을 헤드라인으로 본다.
- 근거 조건: oracle = gold evidence, 검색 = dense 후보 20 → 리랭커 top-5.
- 생성: `Qwen2.5-7B-Instruct` bitsandbytes 4-bit(NF4), 그리디, max_new_tokens 64, 프롬프트 v3(간결·yes/no·근거 없으면 Unanswerable). 환경 torch 2.5.1+cu121, RTX 3080.

| 방식(근거) | 전체 Answer-F1 |
|---|---|
| 추출 베이스라인(oracle) | 0.137 |
| 7B 생성(oracle) | 0.506 |
| 7B 생성 + 프롬프트 v3(oracle) | **0.533** |
| 7B 생성 v3(검색 근거) | 0.416 |

| 유형(v3) | 추출(oracle) | 생성(oracle) | 생성(검색) |
|---|---|---|---|
| extractive | 0.166 | 0.541 | 0.443 |
| abstractive | 0.190 | 0.351 | 0.363 |
| boolean | 0.000 | 0.400 | 0.250 |
| unanswerable | 0.000 | 1.000 | 0.500 |

- 추출은 완벽한 근거로도 0.137로 낮다. 예측(문단 전체)이 정답 스팬보다 길어 정밀도가 무너진다. 생성이 간결히 답해 0.533으로 3.9배.
- 검색 근거로 바꾸면 0.533→0.416(−0.116). 검색이 gold를 top-5에 못 올리면 답할 수 없고(extractive 하락), unanswerable은 검색이 노이즈 top-5를 줘 보류가 깨진다(1.000→0.500). 검색 품질이 답 품질의 천장이다.
- 프롬프트는 yes/no 지시 하나만 더한 v3가 최선이었다(여러 지시를 섞으면 이득·손해 상쇄). 7B 양자화는 AWQ가 torch를 다운그레이드해 bitsandbytes 4-bit로 폴백. 상세는 [DEVELOPMENT_JOURNEY B-8](DEVELOPMENT_JOURNEY.md).

```powershell
.\.venv\Scripts\python.exe -m scripts.answer_experiment data/papers/*.json --out runs/answer-oracle-extractive.json
.\.venv\Scripts\python.exe -m scripts.generate_experiment data/papers/*.json --out runs/answer-oracle-qwen.json
.\.venv\Scripts\python.exe -m scripts.answer_retrieved_experiment data/papers/*.json --out runs/answer-retrieved-qwen.json
```

### B-9. Faithfulness(NLI 충실성·환각율) (2026-09-18)

- 대상: B-8과 같은 51문항의 생성 답(oracle 근거 vs 검색 근거). Answer-F1과 독립된 축.
- 채점: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`로 근거를 전제, 답을 가설로 두고 함의 판정. entailment=충실, neutral/contradiction=환각. 보류(Unanswerable)는 주장을 안 한 것이라 제외. 라벨은 config.id2label로 읽는다.

| 근거 조건 | 채점된 답 | 보류 제외 | 충실율(entail) | 환각율 | 모순율 |
|---|---|---|---|---|---|
| oracle | 36 | 15 | **0.778** | 0.222 | 0.056 |
| 검색(dense→리랭커 top-5) | 39 | 12 | 0.615 | **0.385** | 0.103 |

- 검색 노이즈는 정확도(Answer-F1 0.533→0.416, B-8)뿐 아니라 충실성(0.778→0.615, 환각 0.222→0.385)도 깎는다. 검색이 무관한 문단을 주면 모델이 근거에 없는 말을 답에 넣어 함의가 깨진다.
- oracle도 환각 0.222로 완전하지 않다(사전지식 혼입·서술형/수치 답에 대한 NLI 한계·재구성 표현). NLI는 근사 렌즈이므로 절대값보다 oracle 대비 검색의 상대 격차를 신뢰 신호로 본다.
- 정확도(F1)와 충실성(NLI)은 직교하는 두 축이라 함께 봐야 "근거 없이 맞힘"과 "근거대로인데 표현 달라 손해"를 구분한다.

```powershell
.\.venv\Scripts\python.exe -m scripts.nli_experiment runs/answer-oracle-qwen-v3.json --out runs/nli-oracle-qwen-v3.json
.\.venv\Scripts\python.exe -m scripts.nli_experiment runs/answer-retrieved-qwen.json --out runs/nli-retrieved-qwen.json
```

## 코퍼스 라우팅 평가 (Corpus routing)

P0-7 2단계(코퍼스에서 논문 선택 → 선택 논문 내 QA)의 평가다. 앞의 검색·답변 평가가 "논문이 정해진" 가정이었다면, 여기서는 단일 논문 가정 없이 50편 코퍼스에서 질문 → 논문 → 문단 → 답으로 이어지는 흐름을 측정한다.

### 코퍼스 문단 인덱스 (B-18, 2026-10-22)

- 50편 3011문단(빈 문단 제외)을 `BAAI/bge-m3`로 한 번 임베딩해 디스크에 영속화한다. 벡터 행렬(`vectors.npy`, (3011,1024) float32)과 행↔문단 메타(`meta.json`, id·text·section)로 나누고 행 순서를 맞춘다. 호출마다 재임베딩하던 B-16의 ~18s 지연을 빌드 타임 1회로 옮긴 것.
- 검색은 정규화 벡터의 전수 내적(FAISS IndexFlatIP와 같은 의미)에 numpy argsort로 top-k. 3011벡터 규모는 정확 탐색이 ms급이라 근사 인덱스(IVF/HNSW)는 아직 불필요하다.
- sanity: "How big is the seed lexicon used for training?"에 50편 전역에서 정답 문단 `1909.00694:s11:p4`가 1위(내적 0.595). 단일 논문 가정 없이 코퍼스에서 바로 근거를 찾는다.
- 환경: Windows, Python 3.10.7, RTX 3080 GPU 임베딩. 재생성 `python -m src.index.corpus`. 저장물은 gitignore(`/data/*`).

### B-19. 논문 선택 — 50편에서 top-k 라우팅 recall·MRR (2026-10-23)

- 대상: 50편 전체 질문 196개. gold는 "질문이 속한 원 논문" 1편(Qasper 질문-논문 결속)이다. 문단 검색과 달리 보류(deferral) 필터는 적용하지 않는다 — 논문 선택의 gold는 항상 단일·명확해 모호성 필터가 불필요하다.
- 방법: B-18 코퍼스 인덱스에서 문단 점수를 **논문별 max로 집계**(D2)해 논문을 순위 매긴다([src/index/corpus.py](src/index/corpus.py)의 `rank_papers`). 정답 논문이 1개뿐이라 recall@k가 "정답 논문 순위 ≤ k"와 동치여서, `score_retrieval`을 질문당 1번(k=전체) 호출해 `first_gold_rank`를 얻고 recall@1/3/5·MRR을 그 순위로 직접 판정했다(중복 검색 없음).
- 환경: Windows, Python 3.10.7, RTX 3080 GPU(bge-m3 임베딩). 지연 미측정.

| recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|
| 0.444 | 0.602 | 0.643 | 0.549 |

- recall@3 0.602가 D6에서 사전 등록한 목표(≥0.6)를 막 넘겼다.
- recall@1 0.444는 top-1 라우팅 작동점의 1단계 성공률로, 아래 3단계 분해에서 ①단계의 천장이 된다 — 질문의 56%가 틀린 논문으로 가면 top-1에서는 그 아래 문단검색·답변이 복구되지 않는다.
- 원칙: 정답이 단일이면 recall@k는 순위 하나로 환원된다. 순위를 한 번 구해 k별로 비교하면 중복 검색 없이 recall·MRR을 함께 얻는다.

재현은 [paper_selection.py](src/eval/paper_selection.py)의 `evaluate_paper_selection()`로 한다(모듈 `main`은 아래 분해를 출력하므로 recall은 함수를 직접 호출).

### B-20. end-to-end 3단계 조건부 분해 — 어디서 새는가 (2026-10-23)

- 대상: 196문항을 관문마다 조건부로 좁힌다. ① 전체 196 → top-1 라우팅이 gold 논문과 일치하고 채점 가능한(deferral 없는) 47편이 ②로 → ② Hit@5 성공(gold 근거를 top-5에 확보)한 31문항이 ③으로.
- 방법(방안 1 — 조건부): 각 관문을 **앞 관문이 성공한 부분집합에서만** 측정해 "논문을 틀린 것"과 "논문은 맞췄는데 문단을 못 잡은 것"이 섞이지 않게 한다([src/eval/paper_selection.py](src/eval/paper_selection.py)의 `evaluate_decomposition`).
  - ② 논문 내 검색: B-18 코퍼스 인덱스를 라우팅된 논문의 문단으로 필터해 그 안에서 dense top-5를 뽑는다([src/index/corpus.py](src/index/corpus.py)의 `within_papers`). 리랭커 없음(dense-only). Hit@5로 집계.
  - ③ 답변: ②가 잡은 top-5 문단 텍스트를 `Qwen2.5-7B-Instruct`(4-bit NF4, 프롬프트 v3, 그리디)에 넣어 답을 생성하고 Qasper Answer-F1로 잰다(답 축 B-8과 동일 채점).
- 환경: Windows, Python 3.10.7, RTX 3080. 임베더(bge-m3)와 7B(GPU 4bit)를 함께 상주시켜도 OOM 없음. 지연 미측정.

| 관문 | 측정 | 값 | 표본 |
|---|---|---|---|
| ① 라우팅+채점가능 | top-1 = gold 논문 & 보류 없음 | 47/196 (0.240) | 196 |
| ②\|① 논문 내 검색 | Hit@5 | 0.660 | 47 |
| ③\|② 답변 | Answer-F1 | 0.483 | 31 |

- ①의 47은 B-19 recall@1(0.444 ≈ 87문항)에서 보류 질문을 뺀 채점 가능분이다. 라우팅은 맞혔지만 gold 근거가 없거나 보류 대상인 질문은 ②③의 분모로 쓸 수 없어 제외한다.
- ③ Answer-F1 0.483은 라우팅도 검색도 성공한 뒤의 **순수 생성 품질**이다. 근거를 top-5에 확보하고도 ~0.48에 머문다는 건 이 작동점의 병목이 검색이 아니라 생성 자체임을 뜻한다. B-8의 단일 논문 답변 격자(oracle 0.533 → 검색 근거 0.416)를 코퍼스 라우팅까지 확장한 셈이다.
- ②는 리랭커 없는 dense-only다(단일 논문 모드는 dense→리랭커로 Hit@5 0.692, B-7). 여기에 리랭커를 넣어 봤으나 이득은 확인되지 않았다(아래 B-22).
- 표본이 작아(②47·③31) Answer-F1은 한 문제가 ~3%p를 움직인다. 소표본·dense-only 조건으로 읽는다.
- 원칙: 다단계 파이프라인의 손실은 각 관문을 앞 관문 성공 부분집합에 조건부로 재야 병목이 분리된다. 전체에서 뭉뚱그리면 어느 단계를 고쳐야 할지 보이지 않는다.

```powershell
.\.venv\Scripts\python.exe -m src.eval.paper_selection
```

### B-22. ② 리랭커 ablation — 논문 내 검색에 리랭커가 이득인가 (2026-10-24)

- 질문: B-20의 ②(dense-only Hit@5 0.660)에 리랭커를 붙이면 오르는가. 단일 논문 모드에선 이득이었다(B-7: dense 0.615 → dense→리랭커 0.692).
- 대상·조건: B-20과 **동일한 라우팅 통과 47문항**·같은 코퍼스 인덱스(vectors.npy). dense 단계를 고정하고 리랭커만 더해 격리한다 — 논문 내 dense top-20 후보를 `bge-reranker-v2-m3`로 재정렬해 top-5(서빙 `/ask`와 같은 candidate_k=20). dense top-5는 같은 후보의 앞 5개라 B-20의 0.660을 그대로 재현한다.
- 측정: [src/eval/paper_selection.py](src/eval/paper_selection.py)의 `evaluate_within_reranker`. 환경 B-20과 동일(임베더+리랭커 GPU 상주).

| 논문 내 검색 (n=47) | Hit@5 | 적중 |
|---|---|---|
| dense top-5 (B-20 재현) | 0.660 | 31/47 |
| dense top-20 → 리랭커 top-5 | 0.596 | 28/47 |

쌍대 2×2 (같은 질문의 dense·리랭커 적중):

| | 리랭커 hit | 리랭커 miss |
|---|---|---|
| dense hit | 25 | 6 |
| dense miss | 3 | 13 |

- 불일치 쌍은 6(dense만) 대 3(리랭커만), 합 9개다. McNemar 정확검정 양측 p ≈ 0.51(= 2·Σₖ₌₀³ C(9,k)/2⁹)로 이 차이는 노이즈와 구별되지 않는다.
- 결론: 이 세팅·표본에서 리랭커의 이득은 확인되지 않았다. 점추정은 −0.064지만 유의하지 않아 "손해"라고도 단정할 수 없다. B-7의 이득(단일 논문·논문 native 문단·26문항)이 코퍼스 인덱스 청크·47문항 세팅에선 재현되지 않은 것으로, 리랭커 이득은 세팅 의존이다.
- 실무 결론: ②에 리랭커를 둘 근거가 없고, 병목은 ③ 생성(Answer-F1 ~0.48)이라 ②는 레버가 아니다.
- 원칙: 소표본의 지표 차이는 쌍대 불일치(McNemar)로 유의성을 확인하기 전까지 방향조차 단정하지 않는다.

```powershell
.\.venv\Scripts\python.exe -c "import json; from src.eval.paper_selection import evaluate_within_reranker; print(json.dumps(evaluate_within_reranker()))"
```

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

축별 측정 결과·채택 결정은 아래. 측정된 축만 기록하고(검색·top-k·프롬프트·생성), 청킹·임베딩 대안·생성 모델(Claude) 축은 후속이다.

### 검색 축 — 회귀 골든 기준 (2026-10-13)

`baselines/retrieval.json`에 커밋된 회귀 골든에서 config별 검색 지표를 뽑은 표. 수치 출처가 회귀 기준선이라 재실행마다 동일하며 `scripts/ablation_table.py`로 재생성한다. 10편 26문항 마이크로 평균(dense·리랭커는 RTX 3080 GPU 티어, 로컬 수동).

| config | Hit@5 | Recall@5 | MRR@5 |
|---|---|---|---|
| bm25 | 0.385 | 0.346 | 0.237 |
| dense | 0.615 | 0.558 | 0.404 |
| dense_rerank | 0.692 | 0.577 | 0.461 |

검색이 BM25→dense에서 크게(Hit@5 +0.231), dense→리랭커에서 한 번 더(+0.077) 오른다. 질문별 진단은 B-6·B-7이고, 이 표는 그 수치를 회귀 골든으로 고정한 것이다.

```powershell
.\.venv\Scripts\python.exe -m scripts.ablation_table
```

### top-k 축 — 검색 지표의 k 의존성 (2026-10-16)

candidate_k(리랭커 후보 풀)를 20으로 고정하고 top_k만 3·5·10으로 바꿔 [파이프라인 실험](scripts/pipeline_experiment.py)을 돌린 마이크로 평균(10편 26문항, RTX 3080 GPU 로컬). `scripts/topk_sweep.py`로 재생성한다. k=5 행은 회귀 골든과 동일(dense 0.615/0.558/0.404, dense_rerank 0.692/0.577/0.461)해 스윕 경로의 일관성을 확인했다.

| k | config | Hit@k | Recall@k | MRR |
|---|---|---|---|---|
| 3 | dense | 0.462 | 0.404 | 0.372 |
| 3 | bm25_rerank | 0.346 | 0.327 | 0.308 |
| 3 | dense_rerank | 0.500 | 0.462 | 0.417 |
| 5 | dense | 0.615 | 0.558 | 0.404 |
| 5 | bm25_rerank | 0.462 | 0.404 | 0.335 |
| 5 | dense_rerank | 0.692 | 0.577 | 0.461 |
| 10 | dense | 0.692 | 0.654 | 0.414 |
| 10 | bm25_rerank | 0.615 | 0.577 | 0.358 |
| 10 | dense_rerank | 0.769 | 0.750 | 0.471 |

- Hit·Recall은 k에 단조 증가하고 모든 k에서 dense_rerank > dense > bm25_rerank 순서가 유지된다 — 약한 1단계 검색(bm25)이 리랭커 천장을 낮춘다는 B-7이 다시 확인된다.
- **MRR은 k에 거의 둔감하다**(dense 0.372→0.404→0.414). 첫 gold 문단이 대개 상위에 있어, k를 키우면 하위 순위의 gold가 recall로 더 잡힐 뿐 첫 정답 순위(MRR)는 거의 그대로다. k는 recall을 사지 순위 품질을 사지 않는다.
- **작동점 k는 검색 축만으로 결정하지 않는다(B-13).** recall은 k=10이 가장 높지만, 생성기에 문단을 더 주면 노이즈로 Answer-F1·충실성이 떨어질 수 있다(B-8·B-9). 어느 k가 답 품질을 최대화하는지는 답 축의 k 의존성을 측정해야 정해진다(아래 "답 축" 절에서 측정 완료 — Answer-F1은 k=5에서 최대, k=5 확정).

```powershell
.\.venv\Scripts\python.exe -m scripts.topk_sweep
```

### 답 축 — 검색 근거 수(evidence_k)의 F1 의존성 (2026-10-19)

위 top-k 축은 검색 지표만 봤다. 여기서는 candidate_k=20을 고정하고 생성기에 넣는 근거 수 evidence_k만 3·5·10으로 바꿔, 검색 근거로 답을 생성한 Answer-F1을 비교했다(51문항, RTX 3080 GPU 로컬). [answer_retrieved_experiment](scripts/answer_retrieved_experiment.py)의 `run`을 세 번 호출하는 `scripts/answer_topk_sweep.py`로 재생성한다.

| evidence_k | mean Answer-F1 |
|---|---|
| 3 | 0.358 |
| 5 | 0.416 |
| 10 | 0.366 |

- **F1은 비단조로 k=5에서 최대다.** k=3은 근거가 모자라고, k=10은 근거를 더 주는데도 0.416→0.366으로 떨어진다 — 늘어난 문단의 노이즈가 생성을 흐린다.
- 검색 축에서 recall은 k=10이 최선이었지만(위 표), 답 축에서는 그 추가 recall이 F1을 깎는다. "검색 recall을 더 사도 답 품질은 공짜가 아니다"(B-8·B-9)가 실측으로 확인됐다.
- **작동점 k=5는 B-13의 보류성 결정에서 측정된 결정으로 확정된다(B-15).** k=5의 0.416은 B-8의 검색 근거 F1과 일치해 스윕 경로의 일관성도 확인된다.

```powershell
.\.venv\Scripts\python.exe -m scripts.answer_topk_sweep
```

### 프롬프트 축 — 생성 지시 변형 (B-8)

Qwen2.5-7B·oracle 근거·그리디에서 생성 지시(프롬프트)만 바꿔 Answer-F1을 비교했다. yes/no 지시 하나를 더한 v3가 최선이었고, 여러 지시를 한꺼번에 섞으면 이득과 손해가 상쇄됐다. **채택: 프롬프트 v3.** 변형별 수치는 별도로 기록하지 않았고(정성 관측), 조건·분석은 B-8을 따른다.

### 생성 모델 축 — 추출 vs 로컬 7B (B-8)

같은 oracle 근거에서 추출 베이스라인(근거 문단을 그대로 반환)과 Qwen2.5-7B 생성의 Answer-F1을 비교했다.

| 방식(oracle 근거) | Answer-F1 |
|---|---|
| 추출 베이스라인 | 0.137 |
| 7B 생성 + 프롬프트 v3 | **0.533** |

추출은 예측(문단 전체)이 정답 스팬보다 길어 정밀도가 무너진다(0.137). 7B가 간결히 답해 3.9배. **채택: 로컬 7B 생성.** 로컬 모델 크기·계열 비교는 아래 스윕, `claude-sonnet-5`/`claude-opus-5` 생성 축(P0-3의 선택)은 미실행이다. 상세는 B-8.

### 생성 모델 축 — 크기·계열 스윕 (2026-10-21)

같은 oracle 근거·v3 프롬프트에서 생성 모델만 바꿔 Answer-F1을 비교했다(51문항, 4bit, RTX 3080). [생성 모델 스윕](scripts/generation_model_sweep.py)으로 재생성한다. 10GB VRAM·로컬 디스크 제약으로 모델을 하나씩 올려 돌리고(다음 모델 전 VRAM 정리), 7B급 가중치(~15GB/모델)는 측정 후 캐시에서 지웠다(결과값은 보관 여부와 무관).

| 생성 모델 | 크기 | Answer-F1(oracle) |
|---|---|---|
| Qwen2.5-3B-Instruct | 3B | 0.380 |
| **Qwen2.5-7B-Instruct** | 7B | **0.533** |
| Mistral-7B-Instruct-v0.3 | 7B | 0.332 |
| Llama-3.1-8B-Instruct | 8B | 0.499 |

- **크기**: 3B→7B에서 +0.153. 7B 작동점이 타당하며, B-8에서 7B를 채택한 근거를 크기 축으로 뒷받침한다.
- **계열**(비슷한 크기): Qwen-7B > Llama-3.1-8B(0.499) > Mistral-7B(0.332).
- **주의 — 프롬프트 공정성**: v3 프롬프트는 Qwen에 맞춰 튜닝돼 Qwen에 유리하다. 이 표는 "Qwen이 본질적으로 우월"이 아니라 "이 프롬프트·조건에서 Qwen 최적, Llama 경쟁력, Mistral 격차"로 읽어야 한다. 특히 Mistral은 챗 템플릿이 별도 system 역할을 안 받아 지시를 `[INST]`에 접어 넣으므로 간결·보류(Unanswerable) 지시를 덜 따랐을 수 있다. 교훈: 한 모델에 튜닝한 프롬프트로 계열을 비교하면 그 모델에 치우친다.

```powershell
.\.venv\Scripts\python.exe -m scripts.generation_model_sweep
```

## 회귀 (CI)

고정 test셋 + 지표 임계값. 변경마다 재실행, 하락 시 실패. 결과 로그를 상태표/여기 표로 누적.

회귀 하네스는 골든(`baselines/retrieval.json`)에 config별 `{hit@5, recall, mrr}` + `n_questions`를 저장하고, 재실행 결과가 허용치(절대 0.01)를 넘게 하락하면 실패(exit 1)시킨다. 티어를 나눠, CPU(bm25)는 GitHub Actions에서 push·PR마다 자동(unittest 30개 + bm25 골든 비교), GPU(dense·dense_rerank)는 RTX 3080에서 `--tier gpu`로 로컬 수동 비교한다. 골든 갱신은 `--update-baseline`로만 한다(손으로 적지 않음). 설계 근거는 DEVELOPMENT_JOURNEY B-11·B-12.

| 티어 | config | 실행 | 골든 Hit@5 |
|---|---|---|---|
| CPU | bm25 | CI 자동 | 0.385 |
| GPU | dense | 로컬 수동 | 0.615 |
| GPU | dense_rerank | 로컬 수동 | 0.692 |

```powershell
.\.venv\Scripts\python.exe -m scripts.regression_runner --tier all
```

## held-out 최종 평가 (train 미사용 논문)

본 평가의 모든 수치는 튜닝·측정에 쓴 논문 기준이다(단일 논문 작동점 앞 10편, 코퍼스 앞 50편). 작동점이 한 번도 안 본 논문에 일반화되는지 확인한다.

- **데이터**: Qasper train split에서 튜닝·측정에 쓰지 않은 51~60편(10편). 정석 held-out인 dev/test split은 질문당 주석이 여럿이라 현재 단일 주석 하네스가 전부 보류(`answer_count_not_one`)해 쓰지 못한다(다중 주석 채점은 후속). train 미사용분은 같은 단일 주석이라 위 dev 수치와 같은 척도로 직접 비교된다.
- **측정**: 코드 변경 없이 기존 스크립트에 held-out 경로만 전달. 검색 [pipeline_experiment](scripts/pipeline_experiment.py), oracle 생성 [generate_experiment](scripts/generate_experiment.py), 검색근거 생성 [answer_retrieved_experiment](scripts/answer_retrieved_experiment.py). 작동점 candidate_k=20·k=5·Qwen2.5-7B(4bit) 동일.

| 축 | dev (참조) | held-out | Δ |
|---|---|---|---|
| 검색 dense Hit@5 | 0.615 | 0.591 | −0.024 |
| 검색 dense→리랭커 Hit@5 | 0.692 | 0.682 | −0.010 |
| oracle-7B Answer-F1 | 0.533 | 0.528 | −0.005 |
| 검색근거-7B Answer-F1 (k=5) | 0.416 | 0.368 | −0.048 |

표본: 검색 22문항, 답변 44문항(dev는 26·51).

- 검색(dense→리랭커)과 oracle 생성이 dev와 거의 동일 → 핵심 작동점이 미사용 논문에 유지된다(과적합 신호 없음).
- 검색근거 F1만 −0.048 낮다. oracle이 거의 같으므로 생성기 일반화가 아니라 held-out 검색이 소폭 덜 잡힌 몫 + 소표본 탓으로 읽는다.
- 소표본 주의: held-out 22문항에서 bm25→리랭커(Hit@5 0.727)가 dense→리랭커(0.682)를 앞섰으나, dev의 dense_rerank 우위(B-7)를 뒤집을 표본은 아니다.

```powershell
# held-out 10편 추출(train 51~60) 후 data/papers-heldout/*.json 에 대해:
.\.venv\Scripts\python.exe -m scripts.pipeline_experiment data/papers-heldout/*.json
.\.venv\Scripts\python.exe -m scripts.generate_experiment data/papers-heldout/*.json
.\.venv\Scripts\python.exe -m scripts.answer_retrieved_experiment data/papers-heldout/*.json
```

## 비용 로그

| 항목 | 호출 수 | 모델 | 실측 비용 |
|---|---|---|---|
| 생성 | — | — | — |
| judge | — | — | — |

_(Batch API 사용 여부·절감액 포함.)_
