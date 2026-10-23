import json


from pathlib import Path


# 골든 JSON을 읽어 돌려준다. 없거나 비었으면 None

def read_helper(golden_path):

    if golden_path.exists():

        with open(golden_path,"r",encoding="utf-8") as f:

            golden=json.load(f)

            if not golden:
                return None

        return golden

golden=read_helper(Path("baselines/retrieval.json"))

print("| config | hit@5 | recall | mrr |")

print("|---|---|---|---|")

for ele in golden["configs"]:

    print(f"| {ele} | {golden['configs'][ele]['hit@5']:.3f} | {golden['configs'][ele]['recall']:.3f} | {golden['configs'][ele]['mrr']:.3f} |")

