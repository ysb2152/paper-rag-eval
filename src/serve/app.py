import json
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from src.retrieve.embed import load_embedder,search_embed
from src.retrieve.rerank import load_reranker,rerank
from src.generate.qwen import load_generator,generate_answer,stream_answer
from src.ingest.qasper import load_paper
import numpy as np
from src.index.corpus import rank_papers,VECTORS_PATH,META_PATH

STATIC_DIR = Path(__file__).parent / "static"

# /ask가 읽을 수 있는 논문은 이 디렉터리 안으로 제한한다(임의 로컬 경로 읽기 차단)
PAPERS_DIR = Path("data/papers").resolve()

candidate_k=20
max_length=512

# 서버 기동 때 임베더·리랭커(CPU)와 생성기(GPU)를 한 번 로딩해 상주시킨다

@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.embedder=load_embedder(device="cpu")
    app.state.reranker=load_reranker(device="cpu")
    app.state.generator=load_generator()
    # 코퍼스 인덱스는 /ask/corpus 첫 호출 때 지연 로딩한다. 그래야 인덱스가 없어도 /ask는 뜬다
    app.state.corpus=None
    yield

app = FastAPI(lifespan=lifespan)

# /ask 요청 바디 — 논문 경로·질문·근거 수

class AskRequest(BaseModel):
    paper_path: str
    question: str
    evidence_k: int = Field(5, ge=1, le=20)
# /ask 응답 — 답·근거 문단·보류 여부

class AskResponse(BaseModel):
    answer: str
    evidence: list[str]
    abstained: bool

class CorpusRequest(BaseModel):
    question: str
    evidence_k: int = Field(5, ge=1, le=20)

class CorpusResponse(BaseModel):
    answer : str
    evidence: list[str]
    abstained: bool
    routed_paper_id : str

# 모델이 대소문자·끝 마침표를 흔들어도 보류로 보도록 정규화해서 'Unanswerable'을 판정한다

def is_abstain(answer):
    return answer.strip().rstrip(".").lower() == "unanswerable"

# 요청 경로가 data/papers 안인지 확인하고 논문을 읽는다. 밖이면 400

def safe_paper(paper_path):
    resolved=Path(paper_path).resolve()
    if resolved!=PAPERS_DIR and PAPERS_DIR not in resolved.parents:
        raise HTTPException(status_code=400,detail="paper_path는 data/papers 아래의 논문이어야 합니다.")
    return load_paper(str(resolved))

# 코퍼스 인덱스를 지연 로딩한다. 파일이 없으면 생성 명령을 담아 503으로 안내한다

def load_corpus():
    if app.state.corpus is None:
        if not (Path(VECTORS_PATH).exists() and Path(META_PATH).exists()):
            raise HTTPException(status_code=503,
                                detail="코퍼스 인덱스가 없습니다. 'python -m src.index.corpus'로 먼저 생성하세요.")
        app.state.corpus=(np.load(VECTORS_PATH),json.load(open(META_PATH,encoding="utf-8")))
    return app.state.corpus

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
    abstained = is_abstain(answer)

    return (answer,evidence_list,abstained)

@app.post("/ask", response_model=AskResponse)

def ask(req: AskRequest):

    paper = safe_paper(req.paper_path)
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
    paper=safe_paper(req.paper_path)
    candidate_list=search_embed(paper.paragraphs,req.question,embed_model,embed_tok,embed_device,candidate_k,max_length)
    can_list=[]
    for ele in candidate_list:
        can_list.append(ele.paragraph)
    evidence=rerank(req.question,can_list,rr_model,rr_tok,rr_dev,req.evidence_k,max_length)
    evidence_list=[]
    for ele in evidence:
        evidence_list.append(ele.paragraph.text)

    # 클라이언트가 스트림 도중 끊기면 제너레이터가 닫히며 finally가 돌고, stop_event로 생성 스레드도 멈춘다
    stop_event=Event()

    def event_stream():
        try:
            yield sse({"type": "evidence", "evidence": evidence_list})
            pieces = []
            for piece in stream_answer(req.question, evidence_list, gen_model, gen_tok, stop_event=stop_event):
                pieces.append(piece)
                yield sse({"type": "token", "text": piece})
            answer = "".join(pieces).strip()
            yield sse({"type": "done", "abstained": is_abstain(answer)})
        finally:
            stop_event.set()

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/ask/corpus",response_model=CorpusResponse)
def ask_corpus(req: CorpusRequest):
    ranked=rank_papers(req.question,*load_corpus(),*app.state.embedder)
    paper_id=ranked[0][0]
    paper= load_paper("data/papers/" + paper_id + ".json")
    answer,evidence_list,abstained=run_pipeline(paper,req.question,req.evidence_k)

    return CorpusResponse(answer=answer,evidence=evidence_list,abstained=abstained,routed_paper_id=paper_id)



# 정적 데모 페이지 반환

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
