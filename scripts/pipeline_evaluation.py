


with open("./baselines/regression_papers.txt","r",encoding="utf-8") as f:
    path_list=f.readlines()

paths = ["data/papers/"+path.strip()+".json" for path in path_list]

# 회귀셋을 dense·dense_rerank로 평가해 config별 지표와 문항 수를 돌려준다

def compute_pipeline_metrics():
    from scripts.pipeline_experiment import run
    report=run(paths)
    golden=dict()
    n_questions=report["evaluated_questions"]
    maps=["dense","dense_rerank"]
    for ele in maps:
        input_dict=dict()

        input_dict["hit@5"]=report["mean"][ele]["hit"]
        input_dict["recall"] = report["mean"][ele]["recall"]
        input_dict["mrr"]=report["mean"][ele]["rr"]

        golden[ele]=input_dict

    return (golden,n_questions)