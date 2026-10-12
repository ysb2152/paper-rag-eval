from pathlib import Path
from scripts.pipeline_experiment import run


paths = [path for path in Path("data/papers").glob("*.json")]

def compute_pipeline_metrics():
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