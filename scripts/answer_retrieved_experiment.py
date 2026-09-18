"""검색 근거로 답을 생성해 oracle 대비 Answer-F1 손실을 잰다.

oracle 실험은 gold 근거를 줬지만, 실제 시스템은 검색으로 근거를 모은다. 여기서는 W1
최선 파이프라인(dense 후보 candidate_k → 리랭커 top evidence_k)으로 근거를 뽑아 같은
7B 생성기에 넣는다. oracle 결과와의 차이가 검색 오류가 답 품질에 주는 손실이다. 채점
질문 집합은 oracle 실험과 같게 전체 질문을 쓴다(검색은 gold 없이도 수행).
"""

import argparse
import gc
import json
from pathlib import Path

import torch

from src.eval.answer import answer_f1_detail
from src.generate.qwen import MODEL_NAME, generate_answer, load_generator
from src.ingest.qasper import load_paper
from src.retrieve.embed import load_embedder, search_embed
from src.retrieve.rerank import load_reranker, rerank


def retrieve_evidence(paths, candidate_k, evidence_k, max_length):
    """검색기로 질문마다 dense→리랭커 top evidence_k 근거 문단을 모은다.

    검색기와 7B를 동시에 올리면 10GB VRAM에 안 맞아, 검색을 먼저 끝내고 모델을 내린다.
    """
    emb_model, emb_tok, emb_dev = load_embedder()
    rr_model, rr_tok, rr_dev = load_reranker()

    items = []
    for path in paths:
        paper = load_paper(path)
        total = len(paper.paragraphs)
        for question in paper.questions:
            dense_all = search_embed(
                paper.paragraphs, question.text, emb_model, emb_tok, emb_dev, total, max_length
            )
            candidates = [r.paragraph for r in dense_all[:candidate_k]]
            reranked = rerank(
                question.text, candidates, rr_model, rr_tok, rr_dev, evidence_k, max_length
            )
            items.append((paper.id, question, [r.paragraph.text for r in reranked]))

    del emb_model, rr_model
    gc.collect()
    torch.cuda.empty_cache()
    return items


def run(paths: list[str], candidate_k: int = 20, evidence_k: int = 5,
        max_length: int = 512, max_new_tokens: int = 64) -> dict:
    items = retrieve_evidence(paths, candidate_k, evidence_k, max_length)

    gen_model, gen_tok = load_generator()
    rows = []
    for paper_id, question, evidence in items:
        prediction = generate_answer(question.text, evidence, gen_model, gen_tok, max_new_tokens)
        f1, matched_type = answer_f1_detail(prediction, question)
        rows.append({
            "paper_id": paper_id,
            "question_id": question.id,
            "question": question.text,
            "evidence": evidence,
            "prediction": prediction,
            "matched_type": matched_type,
            "has_evidence": bool(evidence),
            "f1": f1,
        })

    count = len(rows)
    types = sorted({row["matched_type"] for row in rows})
    by_type = {
        answer_type: {
            "count": sum(1 for row in rows if row["matched_type"] == answer_type),
            "mean_f1": sum(row["f1"] for row in rows if row["matched_type"] == answer_type)
            / sum(1 for row in rows if row["matched_type"] == answer_type),
        }
        for answer_type in types
    }

    return {
        "model": MODEL_NAME,
        "papers": len(paths),
        "questions": count,
        "candidate_k": candidate_k,
        "evidence_k": evidence_k,
        "mean_f1": sum(row["f1"] for row in rows) / count if count else None,
        "by_type": by_type,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Qasper 단일 논문 JSON 파일들")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--evidence-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    report = run(args.paths, args.candidate_k, args.evidence_k, args.max_length, args.max_new_tokens)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
