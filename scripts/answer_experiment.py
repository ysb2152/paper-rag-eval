"""oracle 근거로 추출 베이스라인의 Answer-F1을 잰다.

W2의 첫 단위다. 검색 오류를 배제하려고 gold evidence(oracle)를 그대로 답으로 내놓고
reference answer와의 토큰 F1을 잰다. 이 값은 "완벽한 근거를 그대로 뱉었을 때"의 바닥값
이며, 답 유형별로 나눠 추출이 어디서 통하고 어디서 막히는지 본다. 생성 모델은 아직
쓰지 않는다(다음 단위에서 7B 생성으로 이 바닥 위를 올린다).
"""

import argparse
import json
from pathlib import Path

from src.eval.answer import answer_f1_detail
from src.ingest.qasper import Question, load_paper


def oracle_evidence_paragraphs(question: Question) -> list[str]:
    """모든 주석자 답의 gold 근거 텍스트를 중복 없이 모은다."""
    seen = set()
    parts = []
    for answer in question.answers:
        for evidence in answer.evidence:
            text = evidence.text.strip()
            if text and text not in seen:
                seen.add(text)
                parts.append(text)
    return parts


def oracle_evidence_text(question: Question) -> str:
    """gold 근거 문단을 하나의 문자열로 이어 붙인다(추출 베이스라인 답)."""
    return " ".join(oracle_evidence_paragraphs(question))


def run(paths: list[str]) -> dict:
    rows = []
    for path in paths:
        paper = load_paper(path)
        for question in paper.questions:
            prediction = oracle_evidence_text(question)
            f1, matched_type = answer_f1_detail(prediction, question)
            rows.append({
                "paper_id": paper.id,
                "question_id": question.id,
                "question": question.text,
                "matched_type": matched_type,
                "has_evidence": bool(prediction),
                "f1": f1,
            })

    count = len(rows)
    types = sorted({row["matched_type"] for row in rows})
    by_type = {}
    for answer_type in types:
        group = [row for row in rows if row["matched_type"] == answer_type]
        by_type[answer_type] = {
            "count": len(group),
            "mean_f1": sum(row["f1"] for row in group) / len(group),
        }

    return {
        "papers": len(paths),
        "questions": count,
        "mean_f1": sum(row["f1"] for row in rows) / count if count else None,
        "by_type": by_type,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Qasper 단일 논문 JSON 파일들")
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    report = run(args.paths)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
