"""BM25가 고른 후보 문단을 교차 인코더 리랭커로 다시 정렬한다.

BM25는 질문과 문단을 따로 벡터로 보고 점수를 매기지만, 리랭커는 (질문, 문단)을
한 입력으로 함께 넣어 관련도를 직접 계산한다. 이미 학습된 가중치로 점수만 얻는
추론이며, 이 논문으로 모델을 다시 학습(미세조정)하지는 않는다.
"""

import argparse
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.ingest.qasper import Paragraph, load_paper
from src.retrieve.bm25 import SearchResult, search_bm25

MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def load_reranker(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device


def rerank(
    query: str,
    candidates: list[Paragraph],
    model,
    tokenizer,
    device: str,
    top_k: int = 5,
    max_length: int = 512,
    batch_size: int = 8,
) -> list[SearchResult]:
    if not candidates:
        return []

    pairs = [(query, paragraph.text) for paragraph in candidates]
    scores: list[float] = []
    # 후보 전체를 배치로 나눠 추론한다. 배치 크기는 한 번에 GPU로 보내는 쌍의 수일 뿐,
    # 검토하는 후보 수(len(candidates))와는 다르다.
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits.view(-1).float()
            scores.extend(logits.tolist())

    # 점수가 같으면 후보로 들어온 순서(BM25 순위)를 유지한다.
    order = sorted(range(len(candidates)), key=lambda i: -scores[i])
    return [SearchResult(candidates[i], scores[i]) for i in order[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Qasper 논문 한 편의 JSON 파일")
    parser.add_argument("query", help="검색할 질문")
    parser.add_argument("--candidate-k", type=int, default=20, help="BM25 후보 수")
    parser.add_argument("--top-k", type=int, default=5, help="리랭킹 후 남길 수")
    args = parser.parse_args()

    paper = load_paper(args.path)
    candidates = [
        result.paragraph
        for result in search_bm25(paper.paragraphs, args.query, args.candidate_k)
    ]
    model, tokenizer, device = load_reranker()
    reranked = rerank(args.query, candidates, model, tokenizer, device, args.top_k)
    print(json.dumps(
        [{"id": result.paragraph.id, "score": result.score} for result in reranked],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
