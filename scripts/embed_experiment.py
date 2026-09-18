"""같은 논문·같은 질문에서 BM25와 dense(bge-m3) 검색을 비교한다.

리랭킹 실험은 BM25 후보 안에서의 재정렬만 봤다면, 이 실험은 검색 자체를 바꾼다.
그래서 후보를 고정하지 않고 두 방식 모두 전체 문단을 순위 매긴다. top_k(기본 5)로
Hit/Recall/MRR을 재고, 전체 순위에서 gold가 몇 위인지도 함께 기록해 'BM25가 아래로
묻어 둔 근거를 dense가 위로 끌어올리는가'를 직접 본다.
"""

import argparse
import json
from pathlib import Path

from src.eval.retrieval import deferral_reasons, gold_paragraph_ids, score_retrieval
from src.ingest.qasper import load_paper
from src.retrieve.bm25 import search_bm25
from src.retrieve.embed import MODEL_NAME, load_embedder, search_embed


def full_gold_rank(retrieved_ids: list[str], gold_ids: set[str]) -> int | None:
    """전체 순위에서 첫 gold 근거가 몇 위인지 돌려준다(없으면 None)."""
    if not retrieved_ids:
        return None
    return score_retrieval(retrieved_ids, gold_ids, len(retrieved_ids))["first_gold_rank"]


def run(path: str, top_k: int = 5, max_length: int = 512) -> dict:
    paper = load_paper(path)
    model, tokenizer, device = load_embedder()
    total_paragraphs = len(paper.paragraphs)

    rows = []
    for question in paper.questions:
        if deferral_reasons(question):
            continue

        gold_ids = gold_paragraph_ids(question)
        bm25_ids = [
            r.paragraph.id
            for r in search_bm25(paper.paragraphs, question.text, total_paragraphs)
        ]
        dense_ids = [
            r.paragraph.id
            for r in search_embed(
                paper.paragraphs, question.text, model, tokenizer, device,
                total_paragraphs, max_length,
            )
        ]

        rows.append({
            "question_id": question.id,
            "question": question.text,
            "gold_ids": sorted(gold_ids),
            "bm25_top": bm25_ids[:top_k],
            "dense_top": dense_ids[:top_k],
            "bm25_gold_rank": full_gold_rank(bm25_ids, gold_ids),
            "dense_gold_rank": full_gold_rank(dense_ids, gold_ids),
            "bm25": score_retrieval(bm25_ids, gold_ids, top_k),
            "dense": score_retrieval(dense_ids, gold_ids, top_k),
        })

    count = len(rows)

    def mean(method: str, metric: str):
        return sum(row[method][metric] for row in rows) / count if count else None

    return {
        "paper_id": paper.id,
        "model": MODEL_NAME,
        "top_k": top_k,
        "max_length": max_length,
        "total_paragraphs": total_paragraphs,
        "evaluated_questions": count,
        "mean": {
            method: {metric: mean(method, metric) for metric in ("hit", "recall", "rr")}
            for method in ("bm25", "dense")
        },
        "questions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Qasper 논문 한 편의 JSON 파일")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    report = run(args.path, args.top_k, args.max_length)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
