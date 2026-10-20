"""Qwen2.5-7B-Instruct로 근거에 기반해 질문에 답한다.

10GB VRAM에 7B를 올리려고 bitsandbytes 4-bit(NF4)로 양자화해 로드한다. 근거 문단만
근거로 삼아 간결히 답하고, 근거가 답을 담지 못하면 보류('Unanswerable')하도록
지시한다. 평가 재현을 위해 표본추출 없이 그리디로 디코딩한다. 이미 학습된 가중치로
답을 생성하는 추론이며 이 데이터로 미세조정하지 않는다.
"""

from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You answer questions about a scientific paper using only the provided evidence. "
    "Answer concisely with a short phrase or a single sentence, not a paragraph. "
    "For yes/no questions, answer 'Yes' or 'No'. "
    "If the evidence does not contain the answer, reply exactly with 'Unanswerable'."
)


def load_generator(model_name: str = MODEL_NAME):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization, device_map="auto"
    )
    model.eval()
    return model, tokenizer


def build_messages(question: str, evidence: list[str]) -> list[dict]:
    joined = "\n\n".join(evidence) if evidence else "(근거 없음)"
    user = f"Evidence:\n{joined}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


@torch.no_grad()
def generate_answer(
    question: str,
    evidence: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 64,
) -> str:
    prompt = tokenizer.apply_chat_template(
        build_messages(question, evidence), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = generated[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def stream_answer(
    question: str,
    evidence: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 64,
):
    """generate_answer와 같은 답을 생성하되, 완성될 때까지 기다리지 않고 토큰을
    만들어지는 대로 조각 문자열로 내보낸다. 서빙에서 답을 실시간으로 흘려보내기 위함이다.

    transformers의 generate는 블로킹이라, TextIteratorStreamer에 생성 토큰을 밀어넣게
    하고 generate 자체는 별도 스레드에서 돌린다. 이 함수는 스트리머를 순회하며 조각을
    yield한다. torch.no_grad()는 스레드 로컬이라 generate를 돌리는 워커 스레드 안에서
    다시 감싸야 추론 그래프가 쌓이지 않는다.
    """
    prompt = tokenizer.apply_chat_template(
        build_messages(question, evidence), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    def run():
        with torch.no_grad():
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

    thread = Thread(target=run)
    thread.start()
    for piece in streamer:
        yield piece
    thread.join()
