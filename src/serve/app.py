import json
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from src.retrieve.embed import load_embedder,search_embed
from src.retrieve.rerank import load_reranker,rerank
from src.generate.qwen import load_generator,generate_answer,stream_answer
from src.ingest.qasper import load_paper
import numpy as np
from src.index.corpus import rank_papers,VECTORS_PATH,META_PATH

STATIC_DIR = Path(__file__).parent / "static"

candidate_k=20
max_length=512

# 서버 기동 때 임베더·리랭커(CPU)와 생성기(GPU)를 한 번 로딩해 상주시킨다

@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.embedder=load_embedder(device="cpu")
    app.state.reranker=load_reranker(device="cpu")
    app.state.generator=load_generator()
    app.state.corpus=(np.load(VECTORS_PATH),json.load(open(META_PATH,encoding="utf-8")))
    yield

app = FastAPI(lifespan=lifespan)

# /ask 요청 바디 — 논문 경로·질문·근거 수

class AskRequest(BaseModel):
    paper_path: str
    question: str
    evidence_k: int = 5
# /ask 응답 — 답·근거 문단·보류 여부

class AskResponse(BaseModel):
    answer: str
    evidence: list[str]
    abstained: bool

class CorpusRequest(BaseModel):
    question: str
    evidence_k: int = 5

class CorpusResponse(BaseModel):
    answer : str
    evidence: list[str]
    abstained: bool
    routed_paper_id : str

# 검색→리랭크→생성으로 답과 근거를 돌려준다. 'Unanswerable'이면 보류



def run_pipeline(paper,question,evidence_k):

    embed_model, embed_tok, embed_device = app.state.embedder
    rr_model, rr_tok, rr_dev = app.state.reranker
    gen_model, gen_tok = app.state.generator
    candidate_list = search_embed(paper.paragraphs, question, embed_model, embed_tok, embed_device, candidate_k,
                                  max_length)
    can_list = []
    for ele in candidate_list:
        can_list.append(ele.paragraph)
    evidence = rerank(question, can_list, rr_model, rr_tok, rr_dev, evidence_k, max_length)
    evidence_list = []
    for ele in evidence:
        evidence_list.append(ele.paragraph.text)
    answer = generate_answer(question, evidence_list, gen_model, gen_tok)
    if answer == "Unanswerable":
        abstained = True
    else:
        abstained = False

    return (answer,evidence_list,abstained)

@app.post("/ask", response_model=AskResponse)

def ask(req: AskRequest):

    paper = load_paper(req.paper_path)
    answer,evidence_list,abstained=run_pipeline(paper,req.question,req.evidence_k)

    return AskResponse(answer=answer, evidence=evidence_list,abstained=abstained)

# payload를 SSE data 한 줄로 포맷

def sse(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# /ask와 같은 파이프라인이되 근거·생성 토큰을 SSE로 실시간 전송

@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    embed_model, embed_tok, embed_device = app.state.embedder
    rr_model,rr_tok,rr_dev=app.state.reranker
    gen_model, gen_tok = app.state.generator
    paper=load_paper(req.paper_path)
    candidate_list=search_embed(paper.paragraphs,req.question,embed_model,embed_tok,embed_device,candidate_k,max_length)
    can_list=[]
    for ele in candidate_list:
        can_list.append(ele.paragraph)
    evidence=rerank(req.question,can_list,rr_model,rr_tok,rr_dev,req.evidence_k,max_length)
    evidence_list=[]
    for ele in evidence:
        evidence_list.append(ele.paragraph.text)

    def event_stream():
        yield sse({"type": "evidence", "evidence": evidence_list})
        pieces = []
        for piece in stream_answer(req.question, evidence_list, gen_model, gen_tok):
            pieces.append(piece)
            yield sse({"type": "token", "text": piece})
        answer = "".join(pieces).strip()
        yield sse({"type": "done", "abstained": answer == "Unanswerable"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/ask/corpus",response_model=CorpusResponse)
def ask_corpus(req: CorpusRequest):
    ranked=rank_papers(req.question,*app.state.corpus,*app.state.embedder)
    paper_id=ranked[0][0]
    paper= load_paper("data/papers/" + paper_id + ".json")
    answer,evidence_list,abstained=run_pipeline(paper,req.question,req.evidence_k)

    return CorpusResponse(answer=answer,evidence=evidence_list,abstained=abstained,routed_paper_id=paper_id)



# 정적 데모 페이지 반환

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
