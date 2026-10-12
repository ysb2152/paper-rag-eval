import json
import sys
import argparse
from pathlib import Path
from scripts.bm25_evaluation import compute_bm25_metrics
from src.eval.regression import find_regression_list
from scripts.pipeline_evaluation import compute_pipeline_metrics

tier_registry={"cpu":compute_bm25_metrics , "gpu":compute_pipeline_metrics}


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

def run_to_compare(golden_path,tier_list):

    golden=read_helper(golden_path)
    all_regressions=[]
    # 골든이 존재하지 않을 경우
    if golden == None:

        print(" go --update-baseline ")

        sys.exit(1)

    for tier in tier_list:

        configs,n=tier_registry[tier]()

        if golden["n_questions"] != n:
            print('fixture 크기 바뀜')
            sys.exit(1)

        for config_name in configs:

            if config_name not in golden["configs"]:

                print("go --update-baseline")

                sys.exit(1)

            all_regressions.extend(find_regression_list(config_name,golden["configs"][config_name],configs[config_name],0.01))

    report_regressions(all_regressions)

def run_to_update(golden_path,tier_list):

    golden = read_helper(golden_path)


    # 골든이 존재하지 않을 경우

    if golden == None:

        golden={"configs":{}}

    for tier in tier_list:

        configs, n = tier_registry[tier]()

        golden["n_questions"]=n

        golden["configs"].update(configs)

    write_helper(golden_path,golden)

    print(f"갱신됨 ")

def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--golden", type=Path, default=Path("baselines/retrieval.json"))
    parser.add_argument("--tier",type=str,choices=["cpu","gpu","all"],default="cpu")

    args = parser.parse_args()

    if args.tier=="all":
        tier_list=["cpu","gpu"]
    else:
        tier_list=[args.tier]

    if args.update_baseline:
        run_to_update(args.golden,tier_list)
    else:
        run_to_compare(args.golden,tier_list)


if __name__ == "__main__":
    main()
