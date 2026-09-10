import unittest

from src.ingest.qasper import Paragraph
from src.retrieve.bm25 import search_bm25


class BM25Tests(unittest.TestCase):
    def setUp(self):
        texts = ["", "seed lexicon", "neural network", "weather forecast", "sports news"]
        self.paragraphs = [
            Paragraph(f"paper:s0:p{i}", "Section", 0, i, text)
            for i, text in enumerate(texts)
        ]

    def test_search_preserves_source_after_skipping_empty_paragraph(self):
        results = search_bm25(self.paragraphs, "SEED lexicon?", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIs(results[0].paragraph, self.paragraphs[1])
        self.assertGreater(results[0].score, 0)

    def test_no_tokens_or_unknown_query_returns_no_results(self):
        for paragraphs, query in [
            ([], "seed"), (self.paragraphs[:1], "seed"),
            (self.paragraphs, "?!"), (self.paragraphs, "unseenword"),
        ]:
            with self.subTest(query=query, count=len(paragraphs)):
                self.assertEqual(search_bm25(paragraphs, query), [])

    def test_ties_keep_source_order_and_large_k_is_bounded(self):
        results = search_bm25(self.paragraphs, "seed", top_k=10)
        self.assertEqual([item.paragraph.id for item in results], [
            paragraph.id for paragraph in self.paragraphs[1:]
        ])

    def test_rejects_nonpositive_k(self):
        for top_k in [0, -1]:
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                search_bm25(self.paragraphs, "seed", top_k)


if __name__ == "__main__":
    unittest.main()
