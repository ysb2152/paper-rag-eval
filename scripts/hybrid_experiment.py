"""여러 논문·여러 질문에서 BM25·dense(bge-m3)·hybrid(RRF) 세 검색을 비교한다.

dense 실험에서 BM25(어휘)와 dense(의미)가 서로 다른 질문에서 강한 것을 봤다. hybrid는
두 순위를 RRF로 합쳐 강점을 한 순위로 모은다. 세 방식 모두 전체 문단을 순위 매기고,
top-k로 Hit/Recall/MRR을 재며, 전체 순위에서 gold가 몇 위인지도 기록한다. 단일 논문은
표본이 작아 결론을 못 내므로 여러 논문의 평가 질문을 모아 마이크로 평균한다.
"""

import argparse
import json
from pathlib import Path

from src.eval.retrieval import deferral_reasons, gold_paragraph_ids, score_retrieval
from src.ingest.qasper import load_paper
from src.retrieve.bm25 import search_bm25
from src.retrieve.embed import MODEL_NAME, load_embedder, search_embed
from src.retrieve.hybrid import reciprocal_rank_fusion

METHODS = ("bm25", "dense", "hybrid")


def full_gold_rank(retrieved_ids: list[str], gold_ids: set[str]) -> int | None:
    """전체 순위에서 첫 gold 근거가 몇 위인지 돌려준다(없으면 None)."""
    if not retrieved_ids:
        return None
    return score_retrieval(retrieved_ids, gold_ids, len(retrieved_ids))["first_gold_rank"]


def build_row(paper, question, model, tokenizer, device, top_k, max_length, rrf_k) -> dict:
    """한 질문에 대해 세 방식의 top-k·gold 순위·지표를 담은 행을 만든다."""
    total = len(paper.paragraphs)
    gold_ids = gold_paragraph_ids(question)
    bm25_ids = [r.paragraph.id for r in search_bm25(paper.paragraphs, question.text, total)]
    dense_ids = [
        r.paragraph.id
        for r in search_embed(
            paper.paragraphs, question.text, model, tokenizer, device, total, max_length
        )
    ]
    hybrid_ids = [pid for pid, _ in reciprocal_rank_fusion([bm25_ids, dense_ids], rrf_k)]

    ranked = {"bm25": bm25_ids, "dense": dense_ids, "hybrid": hybrid_ids}
    return {
        "paper_id": paper.id,
        "question_id": question.id,
        "question": question.text,
        "gold_ids": sorted(gold_ids),
        **{f"{name}_top": ids[:top_k] for name, ids in ranked.items()},
        **{f"{name}_gold_rank": full_gold_rank(ids, gold_ids) for name, ids in ranked.items()},
        **{name: score_retrieval(ids, gold_ids, top_k) for name, ids in ranked.items()},
    }


def run(paths: list[str], top_k: int = 5, max_length: int = 512, rrf_k: int = 60) -> dict:
    model, tokenizer, device = load_embedder()

    rows = []
    for path in paths:
        paper = load_paper(path)
        for question in paper.questions:
            if deferral_reasons(question):
                continue
            rows.append(
                build_row(paper, question, model, tokenizer, device, top_k, max_length, rrf_k)
            )

    count = len(rows)

    def mean(method: str, metric: str):
        return sum(row[method][metric] for row in rows) / count if count else None

    return {
        "model": MODEL_NAME,
        "papers": len(paths),
        "top_k": top_k,
        "max_length": max_length,
        "rrf_k": rrf_k,
        "evaluated_questions": count,
        "mean": {
            method: {metric: mean(method, metric) for metric in ("hit", "recall", "rr")}
            for method in METHODS
        },
        "questions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Qasper 단일 논문 JSON 파일들(또는 글롭 확장)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    report = run(args.paths, args.top_k, args.max_length, args.rrf_k)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
