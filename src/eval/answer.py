"""생성한 답과 Qasper reference answer의 토큰 F1을 잰다.

Qasper 공식 evaluator의 채점 방식을 그대로 따른다. 답을 정규화(소문자화·문장부호
제거·관사 제거·공백 정리)한 뒤 토큰 단위 F1을 계산하고, 한 질문에 주석자 답이 여러
개면 각 reference와의 F1 중 최댓값을 쓴다. reference 문자열은 답 유형에 따라 정해지며
우선순위가 있다: 답 보류 → 추출 스팬 → 서술형 → 예/아니오.
"""

import re
import string
from collections import Counter

from src.ingest.qasper import Answer, Question


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    lowered = text.lower()
    return " ".join(remove_articles(remove_punctuation(lowered)).split())


def token_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_string(answer: Answer) -> str:
    """한 주석자 답을 채점용 문자열로 바꾼다. Qasper 우선순위를 따른다."""
    if answer.unanswerable:
        return "Unanswerable"
    if answer.extractive_spans:
        return ", ".join(answer.extractive_spans)
    if answer.free_form_answer:
        return answer.free_form_answer
    if answer.yes_no:
        return "Yes"
    if answer.yes_no is not None:
        return "No"
    return ""


def answer_type(answer: Answer) -> str:
    """답 유형을 분류한다. 유형별로 추출이 통하는 정도가 달라 진단에 쓴다."""
    if answer.unanswerable:
        return "unanswerable"
    if answer.extractive_spans:
        return "extractive"
    if answer.free_form_answer:
        return "abstractive"
    if answer.yes_no is not None:
        return "boolean"
    return "none"


def answer_f1_detail(prediction: str, question: Question) -> tuple[float, str]:
    """최대 F1과 그 F1을 만든 reference의 답 유형을 함께 돌려준다."""
    if not question.answers:
        raise ValueError("채점할 reference 답이 필요합니다.")
    scored = [
        (token_f1(prediction, answer_string(answer)), answer_type(answer))
        for answer in question.answers
    ]
    return max(scored, key=lambda item: item[0])


def answer_f1(prediction: str, question: Question) -> float:
    """생성 답과 질문의 모든 주석자 답을 비교해 최대 토큰 F1을 돌려준다."""
    return answer_f1_detail(prediction, question)[0]
