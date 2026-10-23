# 서로 다른 검색 방식의 순위를 RRF로 합친다. BM25 점수와 dense 코사인은 스케일이 달라
# 그대로 못 더하니, 원점수는 버리고 순위만 써서 문단마다 1/(k+순위)를 더한다. 한 방식이라도
# 상위에 올린 문단이 가점을 받는다. k는 최상위가 결과를 독식하지 않게 완화하는 상수(관례상 60).


# 여러 순위 리스트를 RRF로 합쳐 (문단id, 점수) 내림차순으로 돌려준다

def reciprocal_rank_fusion(rankings: list, k: int=60):
    scores,order={},{}
    for ranking in rankings:

        # RRF= 1/k+rank 이므로 rank starts at 1.

        for rank,paragraph_id in enumerate(ranking, start=1):

            if paragraph_id not in scores: scores[paragraph_id]=0.0
            scores[paragraph_id]+=1.0/(k+rank)

            # 처음 볼 경우만 넣기 아니면 현상유지

            order.setdefault(paragraph_id, len(order))

    return sorted(scores.items(), key=lambda item: (-item[1],order[item[0]]))
