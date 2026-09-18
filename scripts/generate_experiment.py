"""oracle 근거로 7B 생성기의 Answer-F1을 재고 추출 베이스라인과 비교한다.

검색 오류를 배제하려고 gold evidence(oracle)를 근거로 주고 Qwen2.5-7B로 답을 생성한다.
추출 베이스라인(근거 문단을 그대로 답으로)은 예측이 너무 길어 정밀도가 낮았다. 생성기가
간결한 답으로 그 바닥을 얼마나 올리는지, 답 유형별로 어디서 이득이 큰지 본다.
"""

import argparse
import json
from pathlib import Path

from scripts.answer_experiment import oracle_evidence_paragraphs
from src.eval.answer import answer_f1_detail
from src.generate.qwen import MODEL_NAME, generate_answer, load_generator
from src.ingest.qasper import load_paper


def run(paths: list[str], max_new_tokens: int = 64) -> dict:
    model, tokenizer = load_generator()

    rows = []
    for path in paths:
        paper = load_paper(path)
        for question in paper.questions:
            evidence = oracle_evidence_paragraphs(question)
            prediction = generate_answer(question.text, evidence, model, tokenizer, max_new_tokens)
            f1, matched_type = answer_f1_detail(prediction, question)
            rows.append({
                "paper_id": paper.id,
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
        "max_new_tokens": max_new_tokens,
        "mean_f1": sum(row["f1"] for row in rows) / count if count else None,
        "by_type": by_type,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Qasper 단일 논문 JSON 파일들")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    report = run(args.paths, args.max_new_tokens)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
