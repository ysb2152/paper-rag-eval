"""정답 근거가 명확히 연결된 질문으로 단일 논문의 BM25 검색을 평가한다."""

import argparse
import json

from src.ingest.qasper import Paper, load_paper
from src.retrieve.bm25 import search_bm25


def score_retrieval(retrieved_ids: list[str], gold_ids: set[str], top_k: int) -> dict:
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")
    if not gold_ids:
        raise ValueError("채점할 정답 근거가 필요합니다.")

    selected = retrieved_ids[:top_k]
    matched = gold_ids.intersection(selected)
    first_rank = next(
        (rank for rank, paragraph_id in enumerate(selected, start=1)
         if paragraph_id in gold_ids),
        None,
    )
    return {
        "hit": int(bool(matched)),
        "recall": len(matched) / len(gold_ids),
        "rr": 1 / first_rank if first_rank is not None else 0.0,
        "first_gold_rank": first_rank,
    }


def deferral_reasons(question) -> list[str]:
    """정답 근거가 하나로 명확히 연결되지 않아 채점에서 보류할 이유를 모은다."""
    reasons = []
    if len(question.answers) != 1:
        reasons.append("answer_count_not_one")
    if any(answer.unanswerable for answer in question.answers):
        reasons.append("unanswerable")
    if any(not answer.evidence for answer in question.answers):
        reasons.append("no_evidence")
    evidence = [item for answer in question.answers for item in answer.evidence]
    if any(not item.paragraph_ids for item in evidence):
        reasons.append("unmapped_evidence")
    if any(len(item.paragraph_ids) > 1 for item in evidence):
        reasons.append("ambiguous_evidence")
    return reasons


def gold_paragraph_ids(question) -> set[str]:
    """보류 사유가 없는 질문에서 각 근거가 가리키는 본문 문단 ID를 모은다."""
    evidence = [item for answer in question.answers for item in answer.evidence]
    return {item.paragraph_ids[0] for item in evidence}


def evaluate_paper(paper: Paper, top_k: int = 5) -> dict:
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    evaluated = []
    deferred = []
    for question in paper.questions:
        reasons = deferral_reasons(question)
        if reasons:
            deferred.append({
                "question_id": question.id,
                "question": question.text,
                "reasons": reasons,
            })
            continue

        # 연결된 근거만 남겨 분모를 줄이지 않고, 모든 근거가 명확한 질문만 채점한다.
        gold_ids = gold_paragraph_ids(question)
        results = search_bm25(paper.paragraphs, question.text, top_k)
        retrieved_ids = [result.paragraph.id for result in results]
        evaluated.append({
            "question_id": question.id,
            "question": question.text,
            "gold_ids": sorted(gold_ids),
            "retrieved_ids": retrieved_ids,
            "scores": [result.score for result in results],
            **score_retrieval(retrieved_ids, gold_ids, top_k),
        })

    count = len(evaluated)
    return {
        "paper_id": paper.id,
        "top_k": top_k,
        "total_questions": len(paper.questions),
        "evaluated_questions": count,
        "deferred_questions": len(deferred),
        "mean": {
            "hit": sum(row["hit"] for row in evaluated) / count if count else None,
            "recall": sum(row["recall"] for row in evaluated) / count if count else None,
            "mrr": sum(row["rr"] for row in evaluated) / count if count else None,
        },
        "evaluated": evaluated,
        "deferred": deferred,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Qasper 논문 한 편의 JSON 파일")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    report = evaluate_paper(load_paper(args.path), args.top_k)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
