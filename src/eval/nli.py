"""로컬 NLI로 생성한 답이 근거에 함의되는지(faithfulness) 판정한다.

Answer-F1은 reference와의 겹침만 보므로, 답이 근거에서 실제로 나온 말인지(환각이
아닌지)는 따로 재야 한다. NLI(자연어 추론)는 전제→가설 관계를 entailment/neutral/
contradiction으로 분류한다. 충실성은 근거를 전제, 답을 가설로 두고 판정한다. 근거가
답을 함의하면 충실, neutral이면 근거 밖에서 지어낸 것(환각), contradiction이면 근거와
반대다. 방향이 중요하다: 근거가 답보다 정보가 많으므로 반대로(답→근거) 보면 충실한
답도 neutral로 오판된다. 라벨 순서는 모델마다 다르므로 config.id2label을 읽어 쓴다.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def load_nli(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer, device


@torch.no_grad()
def entailment(
    premise: str,
    hypothesis: str,
    model,
    tokenizer,
    device: str,
    max_length: int = 512,
) -> dict:
    """전제(근거)가 가설(답)을 함의하는지 판정한다. 라벨별 확률과 최상위 라벨을 돌려준다."""
    inputs = tokenizer(
        premise, hypothesis, truncation=True, max_length=max_length, return_tensors="pt"
    ).to(device)
    probs = torch.softmax(model(**inputs).logits[0], dim=-1)
    label_probs = {
        model.config.id2label[i].lower(): float(probs[i]) for i in range(probs.shape[0])
    }
    top_label = max(label_probs, key=label_probs.get)
    return {
        "label": top_label,
        "entailment": label_probs.get("entailment", 0.0),
        "probs": label_probs,
    }
