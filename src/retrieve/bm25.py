"""논문 본문을 문단 단위로 BM25 검색한다."""

import argparse
import json
import re
from dataclasses import asdict, dataclass

from rank_bm25 import BM25Okapi

from src.ingest.qasper import Paragraph, load_paper


@dataclass
class SearchResult:
    paragraph: Paragraph
    score: float


def tokenize(text:str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def search_bm25(paragraphs:list[Paragraph], query:str, top_k: int = 5)-> list[SearchResult]:
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    documents = []
    tokens = []
    SearchResult_lists = []

    for paragraph in paragraphs:
        words = tokenize(paragraph.text)
        if words:
            documents.append(paragraph)
            tokens.append(words)

    query_words = tokenize(query)

    # 예외처리 1: 검색할 문단이 없거나 질문에 단어가 없는 경우
    if not tokens or not query_words:
        return []

    bm25_calculator = BM25Okapi(tokens, k1=1.5, b=0.75, epsilon=0.25)

    # 예외처리 2: 질문 단어가 본문 어디에도 없으면 모두 0점이므로 빈 결과를 낸다
    if not any(word in bm25_calculator.idf for word in query_words):
        return []

    scores = bm25_calculator.get_scores(query_words)
    orders = sorted(range(len(documents)), key=lambda x: -scores[x])

    # top_k보다 문단 수가 적으면 있는 만큼만 담는다
    if len(documents) < top_k:
        for i in range(len(documents)):
            SearchResult_lists.append(
                SearchResult(documents[orders[i]], float(scores[orders[i]]))
            )
    else:
        for i in range(top_k):
            SearchResult_lists.append(
                SearchResult(documents[orders[i]], float(scores[orders[i]]))
            )

    return SearchResult_lists


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Qasper 논문 한 편의 JSON 파일")
    parser.add_argument("query", help="검색할 질문")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    paper = load_paper(args.path)
    results = search_bm25(paper.paragraphs, args.query, args.top_k)
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
