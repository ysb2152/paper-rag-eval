"""같은 BM25 후보에서 기존 순서 top-k와 리랭킹 후 top-k를 비교한다.

후보 수를 늘리는 효과와 순서를 바꾸는 효과를 섞지 않으려고, 두 방식 모두
동일한 BM25 후보 집합(candidate_k개)만 본다. 정답 근거는 리랭커 입력에 넣지 않는다.
후보 밖에 있는 근거는 어느 쪽도 회복할 수 없으므로, 이 실험이 보여 주는 건
'후보 안에서의 재정렬 효과'뿐이다.
"""

import argparse
import json
from pathlib import Path

from src.eval.retrieval import deferral_reasons, gold_paragraph_ids, score_retrieval
from src.ingest.qasper import load_paper
from src.retrieve.bm25 import search_bm25
from src.retrieve.rerank import MODEL_NAME, load_reranker, rerank


def count_truncated(tokenizer, query: str, texts: list[str], max_length: int) -> int:
    """max_length 때문에 잘리는 후보가 몇 개인지 센다. 잘림은 리랭커가 문단 뒷부분을
    보지 못했다는 뜻이라 결과 해석에 필요하다."""
    truncated = 0
    for text in texts:
        length = len(tokenizer(query, text, truncation=False)["input_ids"])
        if length > max_length:
            truncated += 1
    return truncated


def run(path: str, candidate_k: int = 20, top_k: int = 5, max_length: int = 512) -> dict:
    paper = load_paper(path)
    model, tokenizer, device = load_reranker()

    rows = []
    for question in paper.questions:
        if deferral_reasons(question):
            continue

        gold_ids = gold_paragraph_ids(question)
        bm25_results = search_bm25(paper.paragraphs, question.text, candidate_k)
        candidates = [result.paragraph for result in bm25_results]

        baseline = bm25_results[:top_k]
        baseline_ids = [result.paragraph.id for result in baseline]
        reranked = rerank(
            question.text, candidates, model, tokenizer, device, top_k, max_length
        )
        reranked_ids = [result.paragraph.id for result in reranked]

        rows.append({
            "question_id": question.id,
            "question": question.text,
            "gold_ids": sorted(gold_ids),
            "candidate_count": len(candidates),
            "truncated_candidates": count_truncated(
                tokenizer, question.text, [p.text for p in candidates], max_length
            ),
            "baseline_top": baseline_ids,
            "baseline_scores": [result.score for result in baseline],
            "baseline": score_retrieval(baseline_ids, gold_ids, top_k),
            "reranked_top": reranked_ids,
            "reranked_scores": [result.score for result in reranked],
            "reranked": score_retrieval(reranked_ids, gold_ids, top_k),
        })

    count = len(rows)

    def mean(method: str, metric: str):
        return sum(row[method][metric] for row in rows) / count if count else None

    return {
        "paper_id": paper.id,
        "model": MODEL_NAME,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "max_length": max_length,
        "evaluated_questions": count,
        "mean": {
            method: {metric: mean(method, metric) for metric in ("hit", "recall", "rr")}
            for method in ("baseline", "reranked")
        },
        "questions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Qasper 논문 한 편의 JSON 파일")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    report = run(args.path, args.candidate_k, args.top_k, args.max_length)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
