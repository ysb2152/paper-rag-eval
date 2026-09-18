"""Qasper 전체 배포본에서 논문을 한 편씩 단일 논문 JSON으로 추출한다.

Qasper 원본(qasper-train-v0.3.json)은 {논문ID: {title, abstract, full_text, qas,
figures_and_tables}} 형태이고, full_text와 qas가 객체들의 리스트다. 반면 우리 로더는
HuggingFace datasets가 쓰는 컬럼형(full_text.section_name·paragraphs가 병렬 리스트,
qas.question_id·question·answers가 병렬 리스트)을 기대한다. 그래서 각 논문을 컬럼형으로
변환하고 id를 넣어 개별 파일로 저장한다. 체리피킹을 피해 원본 등장 순서대로 앞에서부터
count편을 뽑는다.

원본은 저장소에 올리지 않는다(용량·라이선스). 재현하려면 공식 배포본을 받아
data/qasper-raw/ 에 두고 이 스크립트를 실행한다:
  https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
"""

import argparse
import json
from pathlib import Path


def to_single_paper(paper_id: str, paper: dict) -> dict:
    """리스트형 원본 논문을 로더가 읽는 컬럼형 단일 논문으로 바꾼다."""
    full_text = {
        "section_name": [section["section_name"] for section in paper["full_text"]],
        "paragraphs": [section["paragraphs"] for section in paper["full_text"]],
    }
    # 원본은 답변을 {answer, annotation_id, worker_id}로 감싸지만, 로더는 answer 본문만
    # 쓰므로 answer 목록을 꺼내 컬럼형 answers.answer 아래로 모은다.
    qas = {
        "question_id": [qa["question_id"] for qa in paper["qas"]],
        "question": [qa["question"] for qa in paper["qas"]],
        "answers": [
            {"answer": [item["answer"] for item in qa["answers"]]}
            for qa in paper["qas"]
        ],
    }
    return {
        "id": paper_id,
        "title": paper["title"],
        "abstract": paper["abstract"],
        "full_text": full_text,
        "qas": qas,
        "figures_and_tables": paper.get("figures_and_tables"),
    }


def extract(source: Path, out_dir: Path, count: int) -> list[str]:
    papers = json.loads(source.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for paper_id in list(papers.keys())[:count]:
        single = to_single_paper(paper_id, papers[paper_id])
        path = out_dir / f"{paper_id}.json"
        path.write_text(json.dumps(single, ensure_ascii=False), encoding="utf-8")
        saved.append(paper_id)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/qasper-raw/qasper-train-v0.3.json"),
        help="Qasper 전체 JSON 경로",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    saved = extract(args.source, args.out_dir, args.count)
    print(f"{len(saved)}편 추출 → {args.out_dir}")
    for paper_id in saved:
        print(" ", paper_id)


if __name__ == "__main__":
    main()
