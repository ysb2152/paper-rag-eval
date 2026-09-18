"""서로 다른 검색 방식의 순위를 RRF로 합친다.

BM25 점수와 dense 코사인은 스케일이 달라 그대로 더할 수 없다. RRF(Reciprocal Rank
Fusion)는 원점수를 버리고 각 방식이 매긴 순위만 써서 문단마다 1/(k+순위)를 더한다.
어느 한 방식이라도 상위에 올린 문단이 가점을 받으므로, 어휘 매칭(BM25)과 의미 매칭
(dense)의 강점을 한 순위로 합칠 수 있다. k는 최상위 순위가 결과를 독식하지 않도록
완화하는 상수이며 관례상 60을 쓴다.
"""


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """여러 순위 목록(문단 ID의 정렬된 리스트)을 RRF 점수로 합쳐 내림차순 정렬한다.

    같은 점수면 먼저 등장한 문단을 앞에 두어 재실행 시 순서가 흔들리지 않게 한다.
    """
    scores: dict[str, float] = {}
    order: dict[str, int] = {}
    for ranking in rankings:
        for rank, paragraph_id in enumerate(ranking, start=1):
            scores[paragraph_id] = scores.get(paragraph_id, 0.0) + 1.0 / (k + rank)
            order.setdefault(paragraph_id, len(order))

    return sorted(scores.items(), key=lambda item: (-item[1], order[item[0]]))
