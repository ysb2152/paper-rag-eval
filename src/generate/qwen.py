# Qwen2.5-7B-Instruct로 근거에 기반해 질문에 답한다. 10GB VRAM에 7B를 올리려고 bitsandbytes
# 4-bit(NF4)로 양자화해 로드. 근거 문단만 써서 간결히 답하고, 근거가 답을 못 담으면
# 'Unanswerable'로 보류하도록 지시한다. 평가 재현을 위해 표본추출 없이 그리디로 디코딩한다.

from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "You answer questions about a scientific paper using only the provided evidence. "
    "Answer concisely with a short phrase or a single sentence, not a paragraph. "
    "For yes/no questions, answer 'Yes' or 'No'. "
    "If the evidence does not contain the answer, reply exactly with 'Unanswerable'."
)


# Qwen2.5-7B를 4-bit(NF4)로 양자화해 로드

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


# 근거 문단과 질문을 system/user 메시지로 조립. 근거 없으면 '(근거 없음)'

def build_messages(question: str, evidence: list[str]) -> list[dict]:
    joined = "\n\n".join(evidence) if evidence else "(근거 없음)"
    user = f"Evidence:\n{joined}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# 근거로 답을 생성해 문자열로 돌려준다(그리디 디코딩)

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


# generate_answer와 같은 답을 생성하되 완성까지 기다리지 않고 토큰을 만들어지는 대로
# 조각 문자열로 yield한다(서빙 실시간 전송용). transformers의 generate는 블로킹이라
# TextIteratorStreamer에 토큰을 밀어넣고 generate는 별도 스레드에서 돌린다. torch.no_grad()는
# 스레드 로컬이라 generate를 돌리는 워커 스레드 안에서 다시 감싸야 추론 그래프가 안 쌓인다.
# stop_event를 주면 클라이언트가 중간에 끊겼을 때 호출부가 이 이벤트를 세워 생성을 조기 종료시킨다

def stream_answer(
    question: str,
    evidence: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 64,
    stop_event=None,
):
    prompt = tokenizer.apply_chat_template(
        build_messages(question, evidence), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    stopping_criteria = None
    if stop_event is not None:
        class StopOnEvent(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop_event.is_set()
        stopping_criteria = StoppingCriteriaList([StopOnEvent()])

    def run():
        with torch.no_grad():
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                stopping_criteria=stopping_criteria,
            )

    thread = Thread(target=run)
    thread.start()
    for piece in streamer:
        yield piece
    thread.join()
