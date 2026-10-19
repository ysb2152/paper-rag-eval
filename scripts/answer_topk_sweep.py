"""검색 근거 개수(evidence_k)를 바꾸면 답 품질(Answer-F1)이 어떻게 변하는가.

B-13의 top-k 스윕은 검색 지표(Recall·MRR)만 봤다. 여기서는 candidate_k를 20으로
고정하고 생성기에 넣는 근거 수 evidence_k만 3/5/10으로 바꿔, 검색 근거로 답을
생성했을 때의 Answer-F1을 비교한다. 근거를 더 주면(큰 k) 놓친 근거를 담을 확률은
오르지만 노이즈도 함께 늘어, 답 품질이 공짜로 오르지 않는다. 이 축을 재면 B-13에서
보류했던 작동점 k 결정을 검색 축이 아니라 답 축의 실측으로 확정할 수 있다.
"""

from pathlib import Path

from scripts.answer_retrieved_experiment import run

paths = [path for path in Path("data/papers").glob("*.json")]

evidence_k = [3, 5, 10]

print("| evidence_k | mean Answer-F1 | 평가 문항 |")
print("|---|---|---|")

for k in evidence_k:
    report = run(paths, 20, k)
    print(f"| {k} | {report['mean_f1']:.3f} | {report['questions']} |")
