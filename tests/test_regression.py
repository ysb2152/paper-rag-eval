import unittest

from src.eval.regression import find_regression_list


class RegressionCompareTests(unittest.TestCase):
    def test_only_drops_are_reported(self):
        gold = {"hit@5": 0.60, "recall": 0.40, "mrr": 0.50}
        current = {"hit@5": 0.50, "recall": 0.41, "mrr": 0.50}
        regressions = find_regression_list("bm25", gold, current, 0.01)
        self.assertEqual(len(regressions), 1)
        row = regressions[0]
        self.assertEqual(row["config"], "bm25")
        self.assertEqual(row["metric"], "hit@5")
        self.assertEqual(row["baseline"], 0.60)
        self.assertEqual(row["current"], 0.50)
        self.assertAlmostEqual(row["drop"], 0.10)

    def test_no_drop_returns_empty(self):
        gold = {"hit@5": 0.60, "recall": 0.40, "mrr": 0.50}
        for current in [dict(gold), {"hit@5": 0.70, "recall": 0.45, "mrr": 0.55}]:
            with self.subTest(current=current):
                self.assertEqual(find_regression_list("dense", gold, current, 0.01), [])

    def test_tolerance_is_strict_greater_than(self):
        gold = {"m": 3.0}
        current = {"m": 2.0}
        self.assertEqual(find_regression_list("c", gold, current, 1.0), [])
        self.assertEqual(len(find_regression_list("c", gold, current, 0.5)), 1)


if __name__ == "__main__":
    unittest.main()
