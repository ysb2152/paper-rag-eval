import json
import sys
import argparse
from pathlib import Path
from scripts.bm25_evaluation import compute_bm25_metrics
from src.eval.regression import find_regression_list


def read_helper(golden_path):

    if golden_path.exists():

        with open(golden_path,"r",encoding="utf-8") as f:

            golden=json.load(f)

            if not golden:
                return None

        return golden



def write_helper(golden_path,golden_dict):

    with open(golden_path,"w",encoding="utf-8") as f:

        json.dump(golden_dict,f,indent=2,ensure_ascii=False)


def report_regressions(regressions):

    if not regressions:
        print("회귀 없음, 통과")
        sys.exit(0)
    for row in regressions:
        print(f"{row['config']} {row['metric']} {row['baseline']} -> {row['current']} , {row['drop']}")
    sys.exit(1)

def run_to_compare(golden_path):

    golden=read_helper(golden_path)

    # 골든이 존재하지 않을 경우
    if golden == None:

        print(" go --update-baseline ")
        sys.exit(1)

    tuples=compute_bm25_metrics()

    metrics=tuples[0]

    # n_questions 변화할 경우
    if golden["n_questions"]!=tuples[1]:

        print('fixture 크기 바뀜')
        sys.exit(1)

    golden_bm25=golden["configs"]["bm25"]

    regressions=find_regression_list("bm25",golden_bm25,metrics,0.01)

    report_regressions(regressions)

def run_to_update(golden_path):

    tuples=compute_bm25_metrics()

    golden_shape={"n_questions":tuples[1],"configs":{"bm25":tuples[0]}}

    write_helper(golden_path,golden_shape)

    print(f"갱신됨  {golden_shape}")

def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--golden", type=Path, default=Path("baselines/retrieval.json"))
    args = parser.parse_args()
    if args.update_baseline:
        run_to_update(args.golden)
    else:
        run_to_compare(args.golden)


if __name__ == "__main__":
    main()
