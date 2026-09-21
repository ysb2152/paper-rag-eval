import argparse
import json

from src.ingest.qasper import Paper, load_paper
from src.retrieve.bm25 import search_bm25

def score_retrieval(retrieved_ids, gold_ids, top_k) -> dict:

    if top_k<1:
        raise ValueError(" top_k 는 1 이상이어야 함")
    if not gold_ids:
        raise ValueError(" gold_ids is empty")

    selected=retrieved_ids[:top_k]

    # 중복 있어도 recall 부풀지 않도록

    matched=gold_ids.intersection(selected)
    #first_rank 초기화
    first_rank= None
    for rank,paragraph_id in enumerate(selected, start=1):
        if paragraph_id in gold_ids:
            first_rank=rank
            break

    return {"hit": int(bool(matched)),
            "recall": len(matched)/len(gold_ids),
            "rr": 1/first_rank if first_rank is not None else 0.0,
            "first_gold_rank": first_rank
            }

def gold_paragraph_ids(question) -> set[str]:
    evidence=[item for answer in question.answers for item in answer.evidence]


    return {item.paragraph_ids[0] for item in evidence}

def deferral_reasons(question) -> list[str]:
    reasons=[]
    # 답이 정확히 1개가 아닌 경우

    if  len(question.answers)!=1:
        reasons.append("answer_count_not_one")
    # 답이 unanswerable 하나라도 있을 경우

    if any(answer.unanswerable for answer in question.answers):
        reasons.append("unanswerable")
    # 답에 하나라도 근거없을 경우

    if any(not answer.evidence for answer in question.answers):
        reasons.append("no_evidence")

    evidence=[item for answer in question.answers for item in answer.evidence]

    # 근거중 하나라도 매핑 비었을 경우
    if any(not item.paragraph_ids for item in evidence):
        reasons.append("unmapped_evidence")

    # 근거중 하나라도 매핑 2개 이상
    if any(len(item.paragraph_ids)>1 for item in evidence):
        reasons.append("ambiguous_evidence")

    return reasons

def evaluate_paper(paper: Paper,top_k=5)->dict:

    if top_k<1:
        raise ValueError("top_k 1보다 작음.")

    scored_q=[]
    amb_q=[]
    for question in paper.questions:
        deferral = deferral_reasons(question)
        if deferral:
            amb_q.append({"question_id":question.id,"question":question.text,"reasons":deferral})
            continue
        result=search_bm25(paper.paragraphs,question.text,top_k)
        gold=gold_paragraph_ids(question)
        retrived_ids=[r.paragraph.id for r in result ]
        score=score_retrieval(retrived_ids,gold,top_k)
        scores=[s.score for s in result]
        scored_q.append({"question_id":question.id,"question":question.text,"gold_ids":sorted(gold),"retrieved_ids":retrived_ids,"scores":scores,**score})

    count=len(scored_q)

    hit_total=sum(row["hit"] for row in scored_q)
    recall_total = sum(row["recall"] for row in scored_q)
    mrr_total = sum(row["rr"] for row in scored_q)

    mean={"hit":hit_total/count if count else None,"recall":recall_total/count if count else None,"mrr":mrr_total/count if count else None}

    return {"paper_id": paper.id,
            "top_k": top_k,
            "total_questions": len(paper.questions),
            "evaluated_questions": count,
            "deferred_questions": len(amb_q),
            "mean": mean,
            "evaluated": scored_q,
            "deferred": amb_q}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Qasper 논문 한 편의 JSON 파일")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    report = evaluate_paper(load_paper(args.path), args.top_k)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
