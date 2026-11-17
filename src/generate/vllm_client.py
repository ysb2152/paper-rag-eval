from src.generate.prompt import build_messages
import openai
import os
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"


def generate_answer(
    question: str,
    evidence: list[str],
    base_url: str,
    max_new_tokens: int = 64,

) -> str:

    return_dict=generate_with_usage(question,evidence,base_url,max_new_tokens)

    return return_dict["answer"]

def generate_with_usage(question,evidence,base_url,max_new_tokens=256):

    model_name = os.environ.get("LLM_MODEL")
    reasoning_effort=os.environ.get("LLM_REASONING_EFFORT")
    extra=dict()
    if reasoning_effort:
        extra["reasoning_effort"]=reasoning_effort
    if not model_name:
        model_name = MODEL_NAME

    messages = build_messages(question, evidence)
    client = openai.OpenAI(base_url=base_url, api_key=os.environ["LLM_API_KEY"])
    response = client.chat.completions.create(model=model_name, messages=messages, temperature=0,max_tokens=max_new_tokens,**extra)

    usage= (response.usage).model_dump()
    choice=response.choices[0]

    return {"answer": (choice.message.content or "").strip(), "finish_reason":choice.finish_reason, "usage":usage}

def stream_answer(question,evidence,base_url,max_new_tokens=64):

    model_name = os.environ.get("LLM_MODEL")
    if not model_name:
        model_name = MODEL_NAME
    messages=build_messages(question,evidence)
    client = openai.OpenAI(base_url=base_url, api_key=os.environ["LLM_API_KEY"])
    response=client.chat.completions.create(messages=messages,model=model_name,temperature=0,max_tokens=max_new_tokens,stream=True)
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
