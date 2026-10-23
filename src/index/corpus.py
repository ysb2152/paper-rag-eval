from src.ingest.qasper import load_paper
from pathlib import Path
from src.retrieve.embed import load_embedder, encode

import numpy as np
import json

VECTORS_PATH = "data/index/vectors.npy"
META_PATH = "data/index/meta.json"


# data/papers 전 논문의 비어있지 않은 문단을 임베딩해 vectors.npy·meta.json으로 영속화

def build():

    # text 평탄화

    papers = [load_paper(path) for path in Path("data/papers").glob("*.json")]
    corpus=[]
    for paper in papers:
        for paragraph in paper.paragraphs:
            if paragraph.text.strip():
                corpus.append(paragraph)


    # vector 저장

    model,tokenizer,device=load_embedder()
    vectors=encode([p.text for p in corpus],model,tokenizer,device,512,8)
    npVectors=vectors.numpy()
    Path("data/index").mkdir(parents=True, exist_ok=True)
    np.save(VECTORS_PATH,npVectors)

    # meta 저장

    corpusdict=[{"id":p.id,"text":p.text,"section":p.section} for p in corpus]
    with open(META_PATH,"w",encoding="utf-8") as f:
        json.dump(corpusdict,f,ensure_ascii=False)

# 질문을 임베딩해 코퍼스 전역에서 top-k 문단을 (meta, score)로 돌려준다

def search(query,k=5):

    vectors=np.load(VECTORS_PATH)
    meta=json.load(open(META_PATH,encoding="utf-8"))
    model, tokenizer, device = load_embedder()
    query_vector=(encode([query],model,tokenizer,device,512,8)[0]).numpy()

    scores=vectors @ query_vector
    top_k_index=np.argsort(-scores)[:k]
    sortedindex=[]

    for i in range(len(top_k_index)):
        sortedindex.append((meta[top_k_index[i]],float(scores[top_k_index[i]])))


    return sortedindex

# 문단 점수를 논문별 max로 집계해 논문을 (논문id, score) 내림차순으로 돌려준다

def rank_papers(query,vectors,meta,model,tokenizer,device):

    query_vector = (encode([query], model, tokenizer, device, 512, 8)[0]).numpy()
    scores = vectors @ query_vector
    scoredict=dict()
    for i in range(len(meta)):
        paper_id=meta[i]["id"].split(":")[0]
        scoredict[paper_id]=max(float(scores[i]),scoredict.get(paper_id,float(scores[i])))

    sortedict=sorted(scoredict.items(),key=lambda x: -x[1])

    return sortedict


# 한 논문의 문단만 추려 질문과의 유사도 top_k를 (문단id, score)로 돌려준다

def within_papers(query,vectors,paper_id,meta,model,tokenizer,device,k=5):

    query_vector = (encode([query], model, tokenizer, device, 512, 8)[0]).numpy()
    scores = vectors @ query_vector
    paragraph_list=[]

    for i in range(len(meta)):

        if paper_id==meta[i]["id"].split(":")[0]:
            paragraph_list.append((meta[i]["id"], float(scores[i])))


    paragraph_list=sorted(paragraph_list,key=lambda x: -x[1])

    return paragraph_list[:k]

if __name__ == "__main__": build()