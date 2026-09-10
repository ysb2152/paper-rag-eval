"""Hugging Face 형식으로 저장한 Qasper 논문 한 편을 읽는다."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paragraph:
    id: str
    section: str
    section_index: int
    paragraph_index: int
    text: str


@dataclass
class Evidence:
    text: str
    paragraph_ids: list[str]


@dataclass
class Answer:
    unanswerable: bool
    extractive_spans: list[str]
    free_form_answer: str
    yes_no: bool | None
    evidence: list[Evidence]


@dataclass
class Question:
    id: str
    text: str
    answers: list[Answer]


@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    paragraphs: list[Paragraph]
    questions: list[Question]


def load_paper(path: str | Path) -> Paper:
    # Windows에서 저장한 JSON에 BOM이 있어도 같은 방식으로 읽는다.
    with Path(path).open(encoding="utf-8-sig") as source:
        data = json.load(source)

    paragraphs = []
    paragraph_ids_by_text: dict[str, list[str]] = {}
    sections = zip(
        data["full_text"]["section_name"],
        data["full_text"]["paragraphs"],
        strict=True,
    )
    for section_index, (section, texts) in enumerate(sections):
        for paragraph_index, text in enumerate(texts):
            paragraph_id = f"{data['id']}:s{section_index}:p{paragraph_index}"
            paragraphs.append(
                Paragraph(paragraph_id, section, section_index, paragraph_index, text)
            )
            paragraph_ids_by_text.setdefault(text, []).append(paragraph_id)

    questions = []
    rows = zip(
        data["qas"]["question_id"],
        data["qas"]["question"],
        data["qas"]["answers"],
        strict=True,
    )
    for question_id, text, annotations in rows:
        answers = []
        for annotation in annotations["answer"]:
            # 같은 문단이 여러 위치에 있으면 후보를 모두 남긴다. 표·그림이나
            # 본문과 일치하지 않는 근거도 버리지 않고 빈 위치 목록으로 보존한다.
            evidence = [
                Evidence(item, list(paragraph_ids_by_text.get(item, [])))
                for item in annotation["evidence"]
            ]
            answers.append(
                Answer(
                    unanswerable=annotation["unanswerable"],
                    extractive_spans=annotation["extractive_spans"],
                    free_form_answer=annotation["free_form_answer"],
                    yes_no=annotation["yes_no"],
                    evidence=evidence,
                )
            )
        questions.append(Question(question_id, text, answers))

    return Paper(data["id"], data["title"], data["abstract"], paragraphs, questions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Qasper 논문 한 편의 JSON 파일")
    args = parser.parse_args()
    paper = load_paper(args.path)
    evidence = [
        item
        for question in paper.questions
        for answer in question.answers
        for item in answer.evidence
    ]
    print(json.dumps({
        "paper_id": paper.id,
        "title": paper.title,
        "paragraphs": len(paper.paragraphs),
        "questions": len(paper.questions),
        "answers": sum(len(question.answers) for question in paper.questions),
        "evidence_annotations": len(evidence),
        "matched_evidence": sum(bool(item.paragraph_ids) for item in evidence),
        "unmatched_text_evidence": sum(
            not item.paragraph_ids and not item.text.startswith("FLOAT SELECTED")
            for item in evidence
        ),
        "figure_table_evidence": sum(
            item.text.startswith("FLOAT SELECTED") for item in evidence
        ),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
