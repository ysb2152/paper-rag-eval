"""서로 다른 검색 방식의 순위를 RRF로 합친다.

BM25 점수와 dense 코사인은 스케일이 달라 그대로 더할 수 없다. RRF(Reciprocal Rank
Fusion)는 원점수를 버리고 각 방식이 매긴 순위만 써서 문단마다 1/(k+순위)를 더한다.
어느 한 방식이라도 상위에 올린 문단이 가점을 받으므로, 어휘 매칭(BM25)과 의미 매칭
(dense)의 강점을 한 순위로 합칠 수 있다. k는 최상위 순위가 결과를 독식하지 않도록
완화하는 상수이며 관례상 60을 쓴다.
"""


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
