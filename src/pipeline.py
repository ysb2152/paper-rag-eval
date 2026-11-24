from src.retrieve.embed import search_embed
from src.retrieve.rerank import rerank
from src.generate.vllm_client import generate_with_usage
candidate_k=20
max_length=512

# 모델이 대소문자·끝 마침표를 흔들어도 보류로 보도록 정규화해서 'Unanswerable'을 판정한다

def is_abstain(answer):
    return answer.strip().rstrip(".").lower() == "unanswerable"


# 검색→리랭크→생성으로 답과 근거를 돌려준다. 'Unanswerable'이면 보류.
# 임베더·리랭커·엔드포인트를 인자로 받는 이유는 FastAPI 밖에서도 같은 경로를 써야 하기 때문이다.
# app.state 를 직접 뒤지면 서빙 프로세스 안에서만 쓸 수 있어서 에이전트의 MCP 도구가 못 쓴다.
# 재라우팅 판정에 쓸 리랭커 top-1 점수와 생성 토큰 사용량도 같이 돌려준다(P0-8).
# max_new_tokens 기본값 64 는 B-20·B-24 를 측정한 조건이라 /ask 쪽을 흔들지 않기 위해 유지한다.

def run_pipeline(paper,question,evidence_k,embedder,reranker,base_url,max_new_tokens=64):

    embed_model, embed_tok, embed_device = embedder
    rr_model, rr_tok, rr_dev = reranker
    candidate_list = search_embed(paper.paragraphs, question, embed_model, embed_tok, embed_device, candidate_k,
                                  max_length)
    can_list = []
    for ele in candidate_list:
        can_list.append(ele.paragraph)
    evidence = rerank(question, can_list, rr_model, rr_tok, rr_dev, evidence_k, max_length)
    if evidence:
        top_1_score=evidence[0].score
    else:
        top_1_score=None
        
    evidence_list = []
    for ele in evidence:
        evidence_list.append(ele.paragraph.text)
    answer_dict = generate_with_usage(question, evidence_list,base_url,max_new_tokens=max_new_tokens)
    abstained = is_abstain(answer_dict["answer"])

    return {"answer":answer_dict["answer"],
            "evidence":evidence_list,
            "abstained":abstained,
            "usage":answer_dict["usage"],
            "finish_reason":answer_dict["finish_reason"],
            "top_1_score":top_1_score}
