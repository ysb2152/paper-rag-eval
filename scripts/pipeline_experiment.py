"""1단계 검색기를 바꾸면 리랭커가 더 살아나는지 본다.

리랭커는 1단계 검색이 후보로 올린 문단만 재정렬하므로, 후보의 질(검색 recall)이
리랭커 성능의 천장을 정한다. 같은 평가 질문에서 세 가지를 비교한다.
  1) dense 단독 top-k (리랭킹 없음, 기준선)
  2) BM25 후보 candidate_k → 리랭킹 top-k
  3) dense 후보 candidate_k → 리랭킹 top-k
정답 근거는 리랭커 입력에 넣지 않고 채점에만 쓴다. 여러 논문의 평가 질문을 모아
마이크로 평균한다.
"""

import argparse
import json
from pathlib import Path

from src.eval.retrieval import deferral_reasons, gold_paragraph_ids, score_retrieval
from src.ingest.qasper import load_paper
from src.retrieve.bm25 import search_bm25
from src.retrieve.embed import load_embedder, search_embed
from src.retrieve.rerank import load_reranker, rerank

METHODS = ("dense", "bm25_rerank", "dense_rerank")


def build_row(paper, question, embedder, reranker, candidate_k, top_k, max_length) -> dict:
    emb_model, emb_tok, emb_dev = embedder
    rr_model, rr_tok, rr_dev = reranker
    total = len(paper.paragraphs)
    gold_ids = gold_paragraph_ids(question)

    dense_all = search_embed(paper.paragraphs, question.text, emb_model, emb_tok, emb_dev, total, max_length)
    dense_ids = [r.paragraph.id for r in dense_all]
    dense_candidates = [r.paragraph for r in dense_all[:candidate_k]]

    bm25_candidates = [
        r.paragraph for r in search_bm25(paper.paragraphs, question.text, candidate_k)
    ]

    bm25_rerank_ids = [
        r.paragraph.id
        for r in rerank(question.text, bm25_candidates, rr_model, rr_tok, rr_dev, top_k, max_length)
    ]
    dense_rerank_ids = [
        r.paragraph.id
        for r in rerank(question.text, dense_candidates, rr_model, rr_tok, rr_dev, top_k, max_length)
    ]

    ranked = {"dense": dense_ids, "bm25_rerank": bm25_rerank_ids, "dense_rerank": dense_rerank_ids}
    return {
        "paper_id": paper.id,
        "question_id": question.id,
        "question": question.text,
        "gold_ids": sorted(gold_ids),
        **{f"{name}_top": ids[:top_k] for name, ids in ranked.items()},
        **{name: score_retrieval(ids, gold_ids, top_k) for name, ids in ranked.items()},
    }


def run(paths: list[str], candidate_k: int = 20, top_k: int = 5, max_length: int = 512) -> dict:
    embedder = load_embedder()
    reranker = load_reranker()

    rows = []
    for path in paths:
        paper = load_paper(path)
        for question in paper.questions:
            if deferral_reasons(question):
                continue
            rows.append(build_row(paper, question, embedder, reranker, candidate_k, top_k, max_length))

    count = len(rows)

    def mean(method: str, metric: str):
        return sum(row[method][metric] for row in rows) / count if count else None

    return {
        "papers": len(paths),
        "candidate_k": candidate_k,
        "top_k": top_k,
        "max_length": max_length,
        "evaluated_questions": count,
        "mean": {
            method: {metric: mean(method, metric) for metric in ("hit", "recall", "rr")}
            for method in METHODS
        },
        "questions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Qasper 단일 논문 JSON 파일들")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    report = run(args.paths, args.candidate_k, args.top_k, args.max_length)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
