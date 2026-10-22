from src.ingest.qasper import load_paper
from pathlib import Path
from src.retrieve.embed import load_embedder, encode

import numpy as np
import json

VECTORS_PATH = "data/index/vectors.npy"
META_PATH = "data/index/meta.json"


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



if __name__ == "__main__": build()