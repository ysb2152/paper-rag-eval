import unittest

from src.eval.answer import answer_f1, answer_string, normalize_answer, token_f1
from src.ingest.qasper import Answer, Question


def make_answer(unanswerable=False, extractive_spans=None, free_form_answer="", yes_no=None):
    return Answer(unanswerable, extractive_spans or [], free_form_answer, yes_no, [])


class NormalizeTests(unittest.TestCase):
    def test_lowercases_removes_punctuation_and_articles(self):
        self.assertEqual(normalize_answer("The Cat, a Dog!"), "cat dog")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_answer("  two   words  "), "two words")


class TokenF1Tests(unittest.TestCase):
    def test_identical_after_normalization_is_one(self):
        self.assertEqual(token_f1("The seed lexicon", "seed lexicon"), 1.0)

    def test_no_overlap_is_zero(self):
        self.assertEqual(token_f1("apple", "orange"), 0.0)

    def test_partial_overlap(self):
        # 예측 2토큰, 정답 3토큰, 공통 2토큰 → p=1.0, r=2/3, f1=0.8
        self.assertAlmostEqual(token_f1("seed lexicon", "seed lexicon size"), 0.8)

    def test_empty_prediction_is_zero(self):
        self.assertEqual(token_f1("", "answer"), 0.0)


class AnswerStringTests(unittest.TestCase):
    def test_unanswerable_wins(self):
        answer = make_answer(unanswerable=True, extractive_spans=["ignored"])
        self.assertEqual(answer_string(answer), "Unanswerable")

    def test_extractive_spans_before_free_form(self):
        answer = make_answer(extractive_spans=["a", "b"], free_form_answer="ignored")
        self.assertEqual(answer_string(answer), "a, b")

    def test_free_form_when_no_spans(self):
        self.assertEqual(answer_string(make_answer(free_form_answer="a summary")), "a summary")

    def test_yes_and_no(self):
        self.assertEqual(answer_string(make_answer(yes_no=True)), "Yes")
        self.assertEqual(answer_string(make_answer(yes_no=False)), "No")


class AnswerF1Tests(unittest.TestCase):
    def test_max_over_annotators(self):
        question = Question("q", "text", [
            make_answer(free_form_answer="completely different"),
            make_answer(free_form_answer="seed lexicon"),
        ])
        self.assertEqual(answer_f1("seed lexicon", question), 1.0)

    def test_requires_references(self):
        with self.assertRaises(ValueError):
            answer_f1("x", Question("q", "text", []))


if __name__ == "__main__":
    unittest.main()
