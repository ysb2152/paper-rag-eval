"""논문 본문을 문단 단위로 dense 임베딩 검색한다.

BM25는 질문과 문단에서 겹치는 단어를 세어 점수를 매기므로, 같은 뜻을 다른 단어로
말한 근거는 놓치기 쉽다. dense 검색은 문장을 의미 벡터로 바꿔 겹치는 단어가 없어도
의미가 비슷하면 가깝게 잡는다. 여기서는 bge-m3로 문단과 질문을 각각 임베딩한 뒤
코사인 유사도가 높은 순으로 문단을 고른다. 이미 학습된 가중치로 벡터만 뽑는
추론이며, 이 논문으로 모델을 다시 학습하지는 않는다.

bge-m3의 dense 벡터는 마지막 은닉 상태의 CLS 토큰(맨 앞 토큰) 하나를 뽑아 L2로
정규화해 얻는다. 정규화하면 두 벡터의 내적이 그대로 코사인 유사도가 된다.
"""

import argparse
import json

import torch
from transformers import AutoModel, AutoTokenizer

from src.ingest.qasper import Paragraph, load_paper
from src.retrieve.bm25 import SearchResult

MODEL_NAME = "BAAI/bge-m3"


def load_embedder(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # torch 2.5.1은 보안 이슈로 pytorch_model.bin(torch.load)을 거부하므로 safetensors를 쓴다.
    model = AutoModel.from_pretrained(model_name, use_safetensors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device


@torch.no_grad()
def encode(
    texts: list[str],
    model,
    tokenizer,
    device: str,
    max_length: int = 512,
    batch_size: int = 8,
) -> torch.Tensor:
    """문장 목록을 정규화된 dense 벡터로 바꾼다. 반환은 (문장 수, 차원) 텐서."""
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        cls = model(**inputs).last_hidden_state[:, 0]
        cls = torch.nn.functional.normalize(cls, p=2, dim=1)
        vectors.append(cls.cpu())
    return torch.cat(vectors, dim=0)


def search_embed(
    paragraphs: list[Paragraph],
    query: str,
    model,
    tokenizer,
    device: str,
    top_k: int = 5,
    max_length: int = 512,
    batch_size: int = 8,
) -> list[SearchResult]:
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    documents = [paragraph for paragraph in paragraphs if paragraph.text.strip()]
    if not documents:
        return []

    doc_vectors = encode(
        [paragraph.text for paragraph in documents],
        model,
        tokenizer,
        device,
        max_length,
        batch_size,
    )
    query_vector = encode([query], model, tokenizer, device, max_length, batch_size)[0]

    scores = doc_vectors @ query_vector
    # 점수가 같으면 원문 순서를 유지해 재실행 시 순위가 바뀌지 않게 한다.
    order = sorted(range(len(documents)), key=lambda i: -scores[i])
    return [SearchResult(documents[i], float(scores[i])) for i in order[:top_k]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Qasper 논문 한 편의 JSON 파일")
    parser.add_argument("query", help="검색할 질문")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    paper = load_paper(args.path)
    model, tokenizer, device = load_embedder()
    results = search_embed(
        paper.paragraphs,
        args.query,
        model,
        tokenizer,
        device,
        args.top_k,
        args.max_length,
    )
    from dataclasses import asdict

    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
