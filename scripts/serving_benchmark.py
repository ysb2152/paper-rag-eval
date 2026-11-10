import argparse
import json
import time
import statistics
import sys
from pathlib import Path
import concurrent.futures
from scripts.answer_experiment import oracle_evidence_paragraphs
from src.eval.answer import answer_f1_detail
from src.generate.vllm_client import generate_answer,MODEL_NAME
from src.ingest.qasper import load_paper


# oracle 근거로 7B가 답을 생성해 유형별·전체 평균 F1을 낸다

def run(paths: list[str], max_new_tokens: int = 64,base_url: str = "http://localhost:8000/v1") -> dict:

    rows = []
    for path in paths:
        paper = load_paper(path)
        for question in paper.questions:
            evidence = oracle_evidence_paragraphs(question)
            before_time=time.perf_counter()
            prediction = generate_answer(question.text, evidence,base_url, max_new_tokens)
            after_time=time.perf_counter()
            f1, matched_type = answer_f1_detail(prediction, question)
            rows.append({
                "paper_id": paper.id,
                "question_id": question.id,
                "question": question.text,
                "evidence": evidence,
                "prediction": prediction,
                "matched_type": matched_type,
                "has_evidence": bool(evidence),
                "f1": f1,
                "latency": after_time-before_time
            })
    latencies=[row["latency"] for row in rows]
    count = len(rows)
    types = sorted({row["matched_type"] for row in rows})
    by_type = {
        answer_type: {
            "count": sum(1 for row in rows if row["matched_type"] == answer_type),
            "mean_f1": sum(row["f1"] for row in rows if row["matched_type"] == answer_type)
            / sum(1 for row in rows if row["matched_type"] == answer_type),
        }
        for answer_type in types
    }
    latency = {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": statistics.quantiles(latencies, n=100)[94],
            "total_wall_time": sum(latencies)

        }

    return {
        "model": MODEL_NAME,
        "papers": len(paths),
        "questions": count,
        "max_new_tokens": max_new_tokens,
        "mean_f1": sum(row["f1"] for row in rows) / count if count else None,
        "by_type": by_type,
        "latency": latency,
        "rows": rows,

    }

def run_concurrent(paths,max_new_tokens,base_url,K):

    tuples=[]
    f1=[]
    map_dict=dict()
    for path in paths:
        paper = load_paper(path)
        for question in paper.questions:
            evidence = oracle_evidence_paragraphs(question)
            tuples.append((question,evidence))
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=K) as ex:
        for tuple in tuples:
            question,evidence=tuple
            future=ex.submit(generate_answer,question.text, evidence, base_url, max_new_tokens)
            map_dict[future]=question
        for future in concurrent.futures.as_completed(map_dict):
            prediction=future.result()
            question=map_dict[future]
            score,matched_type=(answer_f1_detail(prediction,question))
            f1.append(score)
    end=time.perf_counter()

    return {"wall_time":end-start, "throughput":len(f1)/(end-start), "speedup": 123/ (end-start), "mean_f1": sum(f1)/len(f1)}




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Qasper 단일 논문 JSON 파일들")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--out", type=Path, help="결과 JSON 저장 경로")
    parser.add_argument("--base_url", type=str,default="http://localhost:8000/v1")
    parser.add_argument("--concurrency_k",type=int,default=None)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.concurrency_k:
        report=run_concurrent(args.paths,args.max_new_tokens,args.base_url,args.concurrency_k)
    else:
        report = run(args.paths, args.max_new_tokens,args.base_url)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
