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

STATIC_DIR = Path(__file__).parent / "static"

candidate_k=20
max_length=512

@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.embedder=load_embedder(device="cpu")
    app.state.reranker=load_reranker(device="cpu")
    app.state.generator=load_generator()
    yield

app = FastAPI(lifespan=lifespan)

class AskRequest(BaseModel):
    paper_path: str
    question: str
    evidence_k: int = 5

class AskResponse(BaseModel):
    answer: str
    evidence: list[str]
    abstained: bool


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
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
    answer=generate_answer(req.question,evidence_list,gen_model,gen_tok)
    if answer=="Unanswerable":
        abstained=True
    else:
        abstained=False
    return AskResponse(answer=answer, evidence=evidence_list,abstained=abstained)


def sse(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


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


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
