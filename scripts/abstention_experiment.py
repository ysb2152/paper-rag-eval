"""검색이 자신 없으면(리랭커 top 점수가 낮으면) 보류할 수 있는지 진단한다.

"""

from pathlib import Path

from src.ingest.qasper import load_paper
from src.eval.retrieval import deferral_reasons
from src.retrieve.bm25 import search_bm25
from src.retrieve.rerank import load_reranker,rerank
from src.eval.retrieval import gold_paragraph_ids,score_retrieval

# data/papers 폴더의 논문 JSON을 전부 불러온다.

papers = [load_paper(path) for path in Path("data/papers").glob("*.json")]
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
model,tokenizer,device=load_reranker(MODEL_NAME)

# 질문 id, 신뢰도, 히트유무 리스트

ids_score_hit=[]

for paper in papers:
    for question in paper.questions:
        if deferral_reasons(question):
            continue
        SearchResultList=search_bm25(paper.paragraphs,question.text,20)

        # 후보 문단 리스트

        CandidatePara=[ele.paragraph for ele in SearchResultList]

        # trust=신뢰도 top score

        Reranked_Top_5=rerank(question.text,CandidatePara,model,tokenizer,device,5)
        trust=Reranked_Top_5[0].score
        gold=gold_paragraph_ids(question)
        retrieved_ids=[r.paragraph.id for r in Reranked_Top_5]
        hit=score_retrieval(retrieved_ids,gold,5)["hit"]
        print(trust,hit)
        ids_score_hit.append({"question.id": question.id,"trust":trust,"hit":hit})


# 점수 낮은 구간일수록 hit 비율이 낮을까?

# 임계값 List

thresholds_list=[-3,-2,-1,0,1,2,3,4]

for ele in thresholds_list:
    top_score_over_th = [row for row in ids_score_hit if row["trust"] >= ele]
    top_score_under_th = [row for row in ids_score_hit if row["trust"] < ele]
    print(f"cutoff:{ele} over",len(top_score_over_th),sum(row["hit"] for row in top_score_over_th)/len(top_score_over_th))
    print(f"cutoff:{ele} under",len(top_score_under_th),sum(row["hit"] for row in top_score_under_th)/len(top_score_under_th))