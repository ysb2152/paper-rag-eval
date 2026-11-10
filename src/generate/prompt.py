SYSTEM_PROMPT = (
    "You answer questions about a scientific paper using only the provided evidence. "
    "Answer concisely with a short phrase or a single sentence, not a paragraph. "
    "For yes/no questions, answer 'Yes' or 'No'. "
    "If the evidence does not contain the answer, reply exactly with 'Unanswerable'."
)


def build_messages(question: str, evidence: list[str]) -> list[dict]:
    joined = "\n\n".join(evidence) if evidence else "(근거 없음)"
    user = f"Evidence:\n{joined}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
