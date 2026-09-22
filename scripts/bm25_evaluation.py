from pathlib import Path

from src.ingest.qasper import load_paper

from src.eval.retrieval import evaluate_paper
# data/papers 폴더의 논문 JSON을 전부 불러온다.

papers = [load_paper(path) for path in Path("data/papers").glob("*.json")]


def compute_bm25_metrics():
    evaluated_paper=[]

    for paper in papers:

        report=evaluate_paper(paper,5)
        # dictionary 펼치기
        evaluated_paper.extend(report["evaluated"])

    all_hit,all_recall,all_mrr=0,0,0
    # 모든 hit@5,recall,mrr 에 대한 마이크로 평균
    for ele in evaluated_paper:

        all_hit+=ele["hit"]

        all_recall+=ele["recall"]

        all_mrr+=ele["rr"]

    # 질문 개수
    n_questions=len(evaluated_paper)
    #hit,recall,mrr은 딕셔너리로, n_questions는 분리하여 제공
    return_dict={"hit@5":all_hit/n_questions,"recall":all_recall/n_questions,"mrr":all_mrr/n_questions}


    return (return_dict,n_questions)
    