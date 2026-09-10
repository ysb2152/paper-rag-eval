import json
import tempfile
import unittest
from pathlib import Path

from src.ingest.qasper import load_paper


class LoadPaperTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "paper.json"
        self.answer = {
            "unanswerable": False,
            "extractive_spans": [],
            "free_form_answer": "",
            "yes_no": False,
            "evidence": ["same paragraph", "missing text", "FLOAT SELECTED: Table 1"],
        }
        self.data = {
            "id": "sample",
            "title": "Sample paper",
            "abstract": "Abstract",
            "full_text": {
                "section_name": ["Intro", "Results"],
                "paragraphs": [["same paragraph", ""], ["same paragraph"]],
            },
            "qas": {
                "question_id": ["q1"],
                "question": ["Does it work?"],
                "answers": [{"answer": [self.answer]}],
            },
        }

    def load(self):
        self.path.write_text(json.dumps(self.data), encoding="utf-8-sig")
        return load_paper(self.path)

    def test_preserves_positions_and_all_evidence_candidates(self):
        paper = self.load()
        self.assertEqual(len(paper.paragraphs), 3)
        self.assertEqual(paper.paragraphs[1].text, "")
        self.assertEqual(paper.paragraphs[2].section_index, 1)
        self.assertEqual(paper.paragraphs[2].paragraph_index, 0)
        evidence = paper.questions[0].answers[0].evidence
        self.assertEqual(evidence[0].paragraph_ids, ["sample:s0:p0", "sample:s1:p0"])
        self.assertEqual(evidence[1].text, "missing text")
        self.assertEqual(evidence[1].paragraph_ids, [])
        self.assertEqual(evidence[2].text, "FLOAT SELECTED: Table 1")

    def test_preserves_multiple_answers_and_false_vs_null(self):
        self.data["qas"]["answers"][0]["answer"].append({
            **self.answer, "yes_no": None, "unanswerable": True, "evidence": [],
        })
        answers = self.load().questions[0].answers
        self.assertEqual(len(answers), 2)
        self.assertIs(answers[0].yes_no, False)
        self.assertIsNone(answers[1].yes_no)
        self.assertTrue(answers[1].unanswerable)
        self.assertEqual(answers[1].evidence, [])

    def test_rejects_misaligned_questions(self):
        self.data["qas"]["question"].append("Unpaired question")
        with self.assertRaises(ValueError):
            self.load()

    def test_rejects_misaligned_sections(self):
        self.data["full_text"]["section_name"].append("Unpaired section")
        with self.assertRaises(ValueError):
            self.load()


if __name__ == "__main__":
    unittest.main()
