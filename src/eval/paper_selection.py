from pathlib import Path
from src.ingest.qasper import load_paper
import json
import numpy as np
from src.index.corpus import rank_papers, VECTORS_PATH, META_PATH, within_papers
from src.retrieve.embed import load_embedder
from src.eval.retrieval import score_retrieval,deferral_reasons,gold_paragraph_ids
from src.generate.qwen import load_generator,generate_answer
from src.eval.answer import answer_f1

# 코퍼스 전 논문을 읽어 (질문id, 질문텍스트, 논문id, 질문객체) 목록과 논문들을 돌려준다

def build_dataset():

    papers = [load_paper(path) for path in Path("data/papers").glob("*.json")]
    question_list = []

    for paper in papers:
        for question in paper.questions:
            question_list.append((question.id,question.text,paper.id,question))

    return question_list,papers

# ① 논문 선택 — 196문항 top-1 라우팅의 recall@1/3/5·MRR

def evaluate_paper_selection():

    vectors=np.load(VECTORS_PATH)
    meta=json.load((open(META_PATH,encoding="utf-8")))
    model,tokenizer,device=load_embedder()
    recall={"1":0,"3":0,"5":0}
    rr_sum=0
    question_list,corpus_papers=build_dataset()


    for qid,text,gold_paper_id,question in question_list:
        ranked=rank_papers(text,vectors,meta,model,tokenizer,device)
        retrieved_ids=[pid for pid, _ in ranked]
        score=score_retrieval(retrieved_ids,{gold_paper_id},len(retrieved_ids))
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

# 라우팅·채점 통과 판정 — 보류 없고 top-1이 gold 논문이면 ②③ 재료를 돌려준다

def routed_item(question,gold_paper_id,vectors,meta,model,tokenizer,device):

    if deferral_reasons(question) :
        return None

    ranked = rank_papers(question.text, vectors, meta, model, tokenizer, device)

    if ranked[0][0] != gold_paper_id:
        return None

    return question.text,gold_paper_id,gold_paragraph_ids(question),question

# 라우팅 통과한 질문들만 모아 ②③ 입력 부분집합으로

def build_routed_subset(vectors,meta,model,tokenizer,device):

    question_list,papers=build_dataset()
    passed=[]

    for qid,text,gold_paper_id,question in question_list:

        gold=routed_item(question,gold_paper_id,vectors,meta,model,tokenizer,device)

        if gold is not None:

            passed.append(gold)

    return passed

# ②③ 조건부 분해 — 통과분의 논문 내 Hit@5, ②성공분의 Answer-F1

def evaluate_decomposition():

    vectors = np.load(VECTORS_PATH)
    meta = json.load((open(META_PATH, encoding="utf-8")))
    model, tokenizer, device = load_embedder()
    subset=build_routed_subset(vectors,meta,model,tokenizer,device)
    hit_sum=0
    gen_model,gen_tokenizer=load_generator()
    id_to_text= {m["id"]: m["text"] for m in meta}
    f1_sum,f1_count=0,0

    for text,gold_paper_id,gold_paragraph_id,question in  subset:

        ranked=within_papers(text,vectors,gold_paper_id,meta,model,tokenizer,device,5)
        retrieved_ids=[pid for pid, _ in ranked]
        score=score_retrieval(retrieved_ids,gold_paragraph_id,5)
        hit_sum+=score["hit"]

        if score["hit"]==1:

            evidence_texts=[id_to_text[pid] for pid in retrieved_ids]
            pred=generate_answer(text,evidence_texts,gen_model,gen_tokenizer)
            f1=answer_f1(pred,question)
            f1_sum+=f1
            f1_count+=1



    return {"n": len(subset),
            "hit@5": hit_sum/len(subset),
            "generated": f1_count,
            "answer_f1": f1_sum/f1_count if f1_count else None
            }

# CLI: 분해 결과를 JSON으로 출력

def main():

    print(json.dumps(evaluate_decomposition(), ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
