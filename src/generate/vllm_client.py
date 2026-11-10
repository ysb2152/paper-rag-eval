from src.generate.prompt import build_messages
import openai
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"



def generate_answer(
    question: str,
    evidence: list[str],
    base_url: str,
    max_new_tokens: int = 64,
) -> str:
    messages=build_messages(question,evidence)
    client=openai.OpenAI(base_url=base_url,api_key="EMPTY")
    response=client.chat.completions.create(model=MODEL_NAME,messages=messages,temperature=0,max_tokens=max_new_tokens)
    return (response.choices[0].message.content).strip()



