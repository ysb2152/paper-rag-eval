"""논문 본문을 문단 단위로 BM25 검색한다."""

import argparse
import json
import re
from dataclasses import asdict, dataclass

from rank_bm25 import BM25Okapi

from src.ingest.qasper import Paragraph, load_paper


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


@dataclass
class SearchResult:
    paragraph: Paragraph
    score: float


def search_bm25(
    paragraphs: list[Paragraph], query: str, top_k: int = 5
) -> list[SearchResult]:
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    documents = []
    tokens = []
    for paragraph in paragraphs:
        words = tokenize(paragraph.text)
        if words:
            documents.append(paragraph)
            tokens.append(words)

    query_words = tokenize(query)
    if not tokens or not query_words:
        return []

    index = BM25Okapi(tokens, k1=1.5, b=0.75, epsilon=0.25)
    # 질문의 단어가 본문에 하나도 없으면 임의의 문단을 검색 결과로 내놓지 않는다.
    if not any(word in index.idf for word in query_words):
        return []

    scores = index.get_scores(query_words)
    # 점수가 같으면 원문 순서를 유지하여 재실행 시 순위가 바뀌지 않게 한다.
    order = sorted(range(len(documents)), key=lambda i: -scores[i])
    return [SearchResult(documents[i], float(scores[i])) for i in order[:top_k]]


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
