# 검색 근거 개수(evidence_k)를 바꾸면 답 품질(Answer-F1)이 어떻게 변하는가. candidate_k는 20
# 고정, evidence_k만 3/5/10으로 바꿔 Answer-F1 비교. 근거를 더 주면(큰 k) 놓친 근거를 담을
# 확률은 오르지만 노이즈도 함께 늘어 답 품질이 공짜로 오르지 않는다. B-13에서 보류한 작동점 k를
# 검색 축이 아니라 답 축의 실측으로 확정하려는 것.

from pathlib import Path

from scripts.answer_retrieved_experiment import run

paths = [path for path in Path("data/papers").glob("*.json")]

evidence_k = [3, 5, 10]

print("| evidence_k | mean Answer-F1 | 평가 문항 |")
print("|---|---|---|")

for k in evidence_k:
    report = run(paths, 20, k)
    print(f"| {k} | {report['mean_f1']:.3f} | {report['questions']} |")
