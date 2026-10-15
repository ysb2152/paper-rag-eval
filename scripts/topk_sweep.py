from pathlib import Path
from scripts.pipeline_experiment import run



paths = [path for path in Path("data/papers").glob("*.json")]

top_k=[3,5,10]

for k in top_k:

    report=run(paths,20,k,512)

    print(f"Top_k = {k}")

    print("| config | hit@k | recall | mrr |")

    print("|---|---|---|---|")

    for method in report["mean"]:

        print(f"| {method} | {report['mean'][method]['hit']:.3f} | {report['mean'][method]['recall']:.3f} | {report['mean'][method]['rr']:.3f} |")
