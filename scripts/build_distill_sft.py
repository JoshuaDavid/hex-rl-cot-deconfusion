"""Build arm-B distillation SFT parquet: harvested no-think best-of-8
answers (own outputs, canonicalized) + a replay slice of armD2 train.

Weights: same token-importance scheme, computed against the MODEL'S OWN
perfect path (any 1.0-scoring answer supports the same counterfactual
deletion grading).

Output: data/armD2/distill_b/train.parquet (val reuses data/armD2/val.parquet)
Run: /venv/main/bin/python scripts/build_distill_sft.py
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from transformers import AutoTokenizer

from scripts.build_armD_witness import MODEL, cell_weights, tokenize_row

REPLAY_ROWS = 1000


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    for l in open("data/armD2/harvest_nothink.jsonl"):
        r = json.loads(l)
        path = r["answer_obj"]["path"]
        winner = r["answer_obj"]["winner"]
        rec = {"prompt_text": r["prompt"], "winner": winner,
               "gold_path": path,
               "cell_weights": cell_weights(r["gt"], winner, path)}
        t = tokenize_row(tok, rec)
        rows.append({
            "input_ids": t["input_ids"], "loss_mask": t["loss_mask"],
            "size": r["size"], "winner": winner, "path_len": r["plen"],
            "prompt_text": r["prompt"], "answer_text": t["answer_text"],
            "ground_truth": json.dumps(r["gt"]),
        })
    print(f"harvest rows: {len(rows)}")
    replay = pd.read_parquet("data/armD2/train.parquet").sample(
        n=REPLAY_ROWS, random_state=3)
    df = pd.concat([pd.DataFrame(rows), replay], ignore_index=True)
    df = df.sample(frac=1.0, random_state=4).reset_index(drop=True)
    os.makedirs("data/armD2/distill_b", exist_ok=True)
    df.to_parquet("data/armD2/distill_b/train.parquet")
    print(f"total train rows: {len(df)} -> data/armD2/distill_b/train.parquet")
    print("plen dist (harvest part):",
          pd.DataFrame(rows)["path_len"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
