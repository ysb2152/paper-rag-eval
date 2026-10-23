# 생성 모델 축 ablation — 같은 oracle 근거·프롬프트에서 생성 모델만 바꿔 Answer-F1을 비교한다.
# 검색 오류를 배제하려 gold evidence를 근거로 고정, 프롬프트도 v3로 고정하고 모델만 바꾼다. 요인
# 하나만 바꿔 "어느 생성 모델이 근거에서 답을 더 잘 뽑나"를 격리해서 본다. 10GB VRAM이라 모델을
# 하나씩 올려 돌리고 다음 전에 내린다. v3 프롬프트가 Qwen에 맞춰 튜닝된 것이라 타 계열엔 불리할
# 수 있다 — 통제된 비교를 위해 고정하되 결과 해석의 주의로 남긴다.

import gc
from pathlib import Path

import torch

from scripts.answer_experiment import oracle_evidence_paragraphs
from src.eval.answer import answer_f1_detail
from src.generate.qwen import generate_answer, load_generator
from src.ingest.qasper import load_paper

# 비교하는 생성 모델들. 10GB VRAM이라 run_model이 한 번에 하나만 올리고 다음 모델 전에
# VRAM을 비운다. 디스크가 빠듯하면 모델 하나를 받아 측정하고 HF 캐시에서 지운 뒤 다음으로
# 넘어간다(7B급 가중치가 모델당 ~15GB). Llama는 게이트 모델이라 받으려면 HF 로그인이 필요하다.
MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
]


# 모델 하나를 올려 전 질문을 oracle 근거로 생성·채점, 평균 F1과 문항 수를 돌려주고 VRAM을 비운다

def run_model(model_name: str, papers: list[str], max_new_tokens: int = 64):
    model, tokenizer = load_generator(model_name)
    f1s = []
    for path in papers:
        paper = load_paper(path)
        for question in paper.questions:
            evidence = oracle_evidence_paragraphs(question)
            prediction = generate_answer(question.text, evidence, model, tokenizer, max_new_tokens)
            f1, _ = answer_f1_detail(prediction, question)
            f1s.append(f1)
    # 다음 모델을 올리기 전에 GPU 메모리를 비운다.
    del model
    gc.collect()
    torch.cuda.empty_cache()
    mean = sum(f1s) / len(f1s) if f1s else None
    return mean, len(f1s)


# 모델들을 차례로 돌려 Answer-F1 비교 표를 출력

def main() -> None:
    papers = [str(path) for path in Path("data/papers").glob("*.json")]
    print("| 생성 모델 | Answer-F1(oracle) | 문항 |")
    print("|---|---|---|")
    for model_name in MODELS:
        mean, count = run_model(model_name, papers)
        print(f"| {model_name} | {mean:.3f} | {count} |")


if __name__ == "__main__":
    main()
