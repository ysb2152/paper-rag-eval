"""Hugging Face 형식으로 저장한 Qasper 논문 한 편을 읽는다"""
from __future__ import annotations
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
class Paper:
    id: str
    title: str
    abstract: str
    paragraphs: list[Paragraph]
    questions: list[Question]


@dataclass
class Question:
    id: str
    text: str
    answers: list[Answer]


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


def load_paper(path: str | Path) -> Paper:
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)

    paragraphs_list = []
    section_names = data["full_text"]["section_name"]
    section_texts = data["full_text"]["paragraphs"]
    id_by_text = {}

    for section_index, (section, texts) in enumerate(
        zip(section_names, section_texts, strict=True)
    ):
        for paragraph_index, text in enumerate(texts):
            paragraph_id = f"{data['id']}:s{section_index}:p{paragraph_index}"
            paragraphs_list.append(
                Paragraph(paragraph_id, section, section_index, paragraph_index, text)
            )
            id_by_text.setdefault(text, []).append(paragraph_id)

    questions_list = []
    question_id = data["qas"]["question_id"]
    question_text = data["qas"]["question"]
    question_answers = data["qas"]["answers"]

    for qid, q_text, q_ans in zip(
        question_id, question_text, question_answers, strict=True
    ):
        answer_list = []
        for i in q_ans["answer"]:
            evidence_list = []
            for text in i["evidence"]:
                evidence_list.append(Evidence(text, id_by_text.get(text, [])))
            answer_list.append(
                Answer(
                    i["unanswerable"],
                    i["extractive_spans"],
                    i["free_form_answer"],
                    i["yes_no"],
                    evidence_list,
                )
            )
        questions_list.append(Question(qid, q_text, answer_list))

    return Paper(
        data["id"], data["title"], data["abstract"], paragraphs_list, questions_list
    )


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
