import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from pydantic import ValidationError

import src.serve.app as app_module
from src.serve.app import (
    AskRequest,
    CorpusRequest,
    ask,
    ask_corpus,
    is_abstain,
    load_corpus,
    safe_paper,
)


# 실모델·검색·생성을 가짜로 대체해 서버의 검증·보류 판정·경로 제한·인덱스 지연 로딩만 확인한다.
# GPU나 모델 로딩 없이 라우트 glue를 검증하는 구조 테스트다.

def fake_result(text):
    return SimpleNamespace(paragraph=SimpleNamespace(text=text))


# is_abstain — 대소문자·끝 마침표가 흔들려도 'Unanswerable'을 보류로 본다

class AbstainNormalizeTests(unittest.TestCase):
    def test_variants_counted_as_abstain(self):
        for ans in ["Unanswerable", "unanswerable.", " UNANSWERABLE ", "Unanswerable."]:
            self.assertTrue(is_abstain(ans), ans)

    def test_real_answer_not_abstain(self):
        for ans in ["Yes", "The seed lexicon has 15 words."]:
            self.assertFalse(is_abstain(ans), ans)


# evidence_k는 1~20만 허용한다(요청 모델 단계에서 검증)

class EvidenceKValidationTests(unittest.TestCase):
    def test_out_of_range_rejected(self):
        for k in (0, -1, 21):
            with self.subTest(k=k), self.assertRaises(ValidationError):
                AskRequest(paper_path="data/papers/x.json", question="q", evidence_k=k)
            with self.subTest(model="corpus", k=k), self.assertRaises(ValidationError):
                CorpusRequest(question="q", evidence_k=k)

    def test_in_range_ok(self):
        self.assertEqual(AskRequest(paper_path="data/papers/x.json", question="q", evidence_k=20).evidence_k, 20)


# safe_paper — data/papers 밖 경로는 읽지 않고 400으로 막는다

class SafePaperTests(unittest.TestCase):
    @mock.patch.object(app_module, "load_paper")
    def test_path_outside_papers_dir_rejected(self, load_paper):
        for bad in ["data/secret.json", "../etc/passwd", "data/papers/../secret.json"]:
            with self.subTest(bad=bad), self.assertRaises(HTTPException) as cm:
                safe_paper(bad)
            self.assertEqual(cm.exception.status_code, 400)
        load_paper.assert_not_called()

    @mock.patch.object(app_module, "load_paper", return_value=SimpleNamespace(paragraphs=[]))
    def test_path_inside_papers_dir_loads(self, load_paper):
        safe_paper("data/papers/1909.00694.json")
        load_paper.assert_called_once()


# load_corpus — 인덱스가 없으면 생성 명령을 담아 503, 이미 로딩됐으면 디스크 안 읽고 반환

class LazyCorpusTests(unittest.TestCase):
    def setUp(self):
        app_module.app.state.corpus = None
        self.addCleanup(setattr, app_module.app.state, "corpus", None)

    def test_missing_index_raises_503(self):
        with mock.patch.object(app_module, "VECTORS_PATH", "data/index/__nope__.npy"):
            with self.assertRaises(HTTPException) as cm:
                load_corpus()
        self.assertEqual(cm.exception.status_code, 503)

    def test_cached_corpus_returned_without_disk(self):
        sentinel = (object(), [{"id": "x:0"}])
        app_module.app.state.corpus = sentinel
        self.assertIs(load_corpus(), sentinel)


# /ask·/ask/corpus 라우트 glue — 가짜 파이프라인으로 응답 형태와 라우팅을 확인

class RouteGlueTests(unittest.TestCase):
    def setUp(self):
        app_module.app.state.embedder = ("m", "t", "cpu")
        app_module.app.state.reranker = ("m", "t", "cpu")
        app_module.app.state.generator = ("m", "t")

    @mock.patch.object(app_module, "search_embed", return_value=[fake_result("c1")])
    @mock.patch.object(app_module, "rerank", return_value=[fake_result("ev1")])
    @mock.patch.object(app_module, "load_paper", return_value=SimpleNamespace(paragraphs=[]))
    @mock.patch.object(app_module, "generate_answer", return_value="Yes")
    def test_ask_returns_answer_and_evidence(self, *_):
        resp = ask(AskRequest(paper_path="data/papers/x.json", question="q"))
        self.assertEqual(resp.answer, "Yes")
        self.assertFalse(resp.abstained)
        self.assertEqual(resp.evidence, ["ev1"])

    @mock.patch.object(app_module, "search_embed", return_value=[fake_result("c1")])
    @mock.patch.object(app_module, "rerank", return_value=[fake_result("ev1")])
    @mock.patch.object(app_module, "load_paper", return_value=SimpleNamespace(paragraphs=[]))
    @mock.patch.object(app_module, "generate_answer", return_value="unanswerable.")
    def test_ask_abstains_on_normalized_unanswerable(self, *_):
        resp = ask(AskRequest(paper_path="data/papers/x.json", question="q"))
        self.assertTrue(resp.abstained)

    @mock.patch.object(app_module, "search_embed", return_value=[fake_result("c1")])
    @mock.patch.object(app_module, "rerank", return_value=[fake_result("ev1")])
    @mock.patch.object(app_module, "load_paper", return_value=SimpleNamespace(paragraphs=[]))
    @mock.patch.object(app_module, "generate_answer", return_value="A")
    @mock.patch.object(app_module, "rank_papers", return_value=[("1909.00694", 0.9)])
    def test_corpus_routes_to_top1_paper(self, *_):
        app_module.app.state.corpus = (object(), [{"id": "1909.00694:0"}])
        self.addCleanup(setattr, app_module.app.state, "corpus", None)
        resp = ask_corpus(CorpusRequest(question="q"))
        self.assertEqual(resp.routed_paper_id, "1909.00694")
        self.assertEqual(resp.answer, "A")


if __name__ == "__main__":
    unittest.main()
