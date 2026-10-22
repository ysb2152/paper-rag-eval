from pathlib import Path
from src.ingest.qasper import load_paper
import json
import numpy as np
from src.index.corpus import rank_papers, VECTORS_PATH, META_PATH
from src.retrieve.embed import load_embedder
from src.eval.retrieval import score_retrieval

def build_dataset():

    papers = [load_paper(path) for path in Path("data/papers").glob("*.json")]
    question_list = []

    for paper in papers:
        for question in paper.questions:
            question_list.append((question.id,question.text,paper.id))

    return question_list,papers

def evaluate_paper_selection():

    vectors=np.load(VECTORS_PATH)
    meta=json.load((open(META_PATH,encoding="utf-8")))
    model,tokenizer,device=load_embedder()
    recall={"1":0,"3":0,"5":0}
    rr_sum=0
    question_list,corpus_papers=build_dataset()


    for qid,text,gold in question_list:
        ranked=rank_papers(text,vectors,meta,model,tokenizer,device)
        retrieved_ids=[pid for pid, _ in ranked]
        score=score_retrieval(retrieved_ids,{gold},len(retrieved_ids))
        rank=score["first_gold_rank"]
        rr=score["rr"]
        for k in recall:
            if rank is not None and rank <= int(k):
                recall[k]+=1
        rr_sum+=rr

    n=len(question_list)

    for k in recall:
        recall[k]=recall[k]/n

    result={"corpus_papers":len(corpus_papers),"n_questions":n,"recall":recall,"mrr":rr_sum/n}

    return result

def main():
    print(json.dumps(evaluate_paper_selection(), ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
    