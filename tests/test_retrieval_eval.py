import unittest
from unittest.mock import patch

from src.eval.retrieval import evaluate_paper, score_retrieval
from src.ingest.qasper import Answer, Evidence, Paper, Paragraph, Question
from src.retrieve.bm25 import SearchResult


class RetrievalMetricsTests(unittest.TestCase):
    def test_partial_recall_and_one_based_rank(self):
        metrics = score_retrieval(["B", "A", "C", "E"], {"A", "C", "E"}, 3)
        self.assertEqual(metrics, {
            "hit": 1, "recall": 2 / 3, "rr": 0.5, "first_gold_rank": 2,
        })

    def test_gold_outside_cutoff_and_empty_results(self):
        for ids in [["B", "C", "A"], []]:
            with self.subTest(ids=ids):
                self.assertEqual(score_retrieval(ids, {"A"}, 2), {
                    "hit": 0, "recall": 0.0, "rr": 0.0, "first_gold_rank": None,
                })

    def test_duplicate_hits_do_not_inflate_recall(self):
        self.assertEqual(score_retrieval(["A", "A"], {"A", "C"}, 5)["recall"], 0.5)

    def test_invalid_cutoff_and_empty_gold_are_not_scores(self):
        for gold, top_k in [({"A"}, 0), ({"A"}, -1), (set(), 5)]:
            with self.subTest(gold=gold, top_k=top_k), self.assertRaises(ValueError):
                score_retrieval([], gold, top_k)


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.paragraphs = [Paragraph("A", "Section", 0, 0, "seed lexicon")]
        self.answer = Answer(False, [], "reference", None, [Evidence("seed lexicon", ["A"])])

    @patch("src.eval.retrieval.search_bm25")
    def test_macro_average_excludes_deferred_and_search_gets_no_gold(self, search):
        partial = Answer(False, [], "", None, [Evidence("seed lexicon", ["A"]), Evidence("table", [])])
        paper = Paper("paper", "Title", "", self.paragraphs, [
            Question("q1", "first question", [self.answer]),
            Question("q2", "second question", [self.answer]),
            Question("q3", "table question", [partial]),
            Question("q4", "multiple answers", [self.answer, partial]),
        ])
        search.side_effect = [[SearchResult(self.paragraphs[0], 2.0)], []]
        report = evaluate_paper(paper)
        self.assertEqual(report["evaluated_questions"], 2)
        self.assertEqual(report["deferred_questions"], 2)
        self.assertEqual(report["mean"], {"hit": 0.5, "recall": 0.5, "mrr": 0.5})
        self.assertEqual(report["deferred"][0]["reasons"], ["unmapped_evidence"])
        self.assertEqual(report["deferred"][1]["reasons"], ["answer_count_not_one", "unmapped_evidence"])
        self.assertEqual(search.call_count, 2)
        self.assertEqual(search.call_args_list[0].args, (self.paragraphs, "first question", 5))

    @patch("src.eval.retrieval.search_bm25")
    def test_no_eligible_questions_produce_null_means(self, search):
        cases = [
            ([], "answer_count_not_one"),
            ([Answer(True, [], "", None, [Evidence("seed lexicon", ["A"])])], "unanswerable"),
            ([Answer(False, [], "", None, [])], "no_evidence"),
            ([Answer(False, [], "", None, [Evidence("repeated", ["A", "B"])])], "ambiguous_evidence"),
        ]
        for answers, reason in cases:
            with self.subTest(reason=reason):
                paper = Paper("p", "", "", self.paragraphs, [Question("q", "seed", answers)])
                report = evaluate_paper(paper)
                self.assertEqual(report["mean"], {"hit": None, "recall": None, "mrr": None})
                self.assertIn(reason, report["deferred"][0]["reasons"])
        search.assert_not_called()

    def test_invalid_cutoff_even_with_no_questions(self):
        with self.assertRaises(ValueError):
            evaluate_paper(Paper("p", "", "", [], []), 0)


if __name__ == "__main__":
    unittest.main()
