"""R1 feasibility SFT data: teach the model to emit the tagged two-task format
and read 5x5 boards. Per board, one example per helper X in {A,B,D,E}:
  prompt   = r1_prompt(board, X, C)
  target   = <task-X>goldX</task-X>\n<task-C>goldC</task-C>
Full-completion loss (uniform mask). Reuses ArmDWitnessSFTDataset (input_ids +
float loss_mask).

Output: data/arme/r1_{train,val}.parquet
Run: /venv/main/bin/python scripts/build_arme_r1.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from transformers import AutoTokenizer

from hexenv.arme import board_from_gt, gold_answer_str, r1_prompt, SELECTABLE, EVALUATED
from scripts.build_armD_witness import MODEL, IM_END

VAL_BOARDS = 120


def build(tok, gts):
    rows = []
    for gt in gts:
        b = board_from_gt(gt)
        gc = gold_answer_str(EVALUATED, b)
        for x in SELECTABLE:
            gx = gold_answer_str(x, b)
            target = f"<task-{x}>{gx}</task-{x}>\n<task-{EVALUATED}>{gc}</task-{EVALUATED}>"
            prompt = r1_prompt(b, x, EVALUATED)
            pids = tok.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True,
                enable_thinking=False, tokenize=True)["input_ids"]
            comp = tok(target + IM_END, add_special_tokens=False)["input_ids"]
            ids = list(pids) + list(comp)
            if len(ids) > 2048:
                continue
            rows.append({"input_ids": ids,
                         "loss_mask": [0.0] * len(pids) + [1.0] * len(comp),
                         "size": gt["size"], "helper": x})
    return rows


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    train_gts = [json.loads(l) for l in open("data/arme/pool_train.jsonl")]
    test_gts = [json.loads(l) for l in open("data/arme/pool_test.jsonl")]
    train_rows = build(tok, train_gts)
    val_rows = build(tok, test_gts[:VAL_BOARDS])
    pd.DataFrame(train_rows).to_parquet("data/arme/r1_train.parquet")
    pd.DataFrame(val_rows).to_parquet("data/arme/r1_val.parquet")
    print(f"R1 SFT: train {len(train_rows)} (from {len(train_gts)} boards) "
          f"val {len(val_rows)}")


if __name__ == "__main__":
    main()
