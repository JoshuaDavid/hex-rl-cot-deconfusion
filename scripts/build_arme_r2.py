"""R2 SFT data: selection format, gold helper TEACHER-FORCED, gradient ONLY on
the evaluated (C) answer (armD-style).

Per board, one example with X ~ uniform {A,B,D,E}:
  prompt : select_prompt(board)
  ctx    : <selected-task>X</selected-task>\n<task-X>GOLD_X</task-X>\n<evaluated-task>   (loss 0)
  ans    : GOLD_C</evaluated-task><|im_end|>                                            (loss 1)
The selected token and helper are context only; the loss lands on C. This
measures whether a correctly-completed helper scaffolds the evaluated answer.

Output: data/arme/r2_{train,val}.parquet
Run: /venv/main/bin/python scripts/build_arme_r2.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from transformers import AutoTokenizer

from hexenv.arme import (board_from_gt, gold_answer_str, select_prompt,
                         split_ctx_ans, SELECTABLE, EVALUATED)
from scripts.build_armD_witness import MODEL, IM_END

VAL_BOARDS = 120


def build(tok, gts, rng, reps=1):
    rows, bad = [], 0
    for gt in gts:
        b = board_from_gt(gt)
        gc = gold_answer_str(EVALUATED, b)
        for _ in range(reps):
            x = rng.choice(SELECTABLE)
            gx = gold_answer_str(x, b)
            ctx = (f"<selected-task>{x}</selected-task>\n"
                   f"<task-{x}>{gx}</task-{x}>\n<evaluated-task>")
            ans = f"{gc}</evaluated-task>{IM_END}"
            pids = tok.apply_chat_template(
                [{"role": "user", "content": select_prompt(b)}],
                add_generation_prompt=True, enable_thinking=False,
                tokenize=True)["input_ids"]
            full_ids, astart = split_ctx_ans(tok, ctx, ans)
            ids = list(pids) + list(full_ids)
            if len(ids) > 2048:
                bad += 1
                continue
            mask = [0.0] * (len(pids) + astart) + [1.0] * (len(full_ids) - astart)
            assert len(mask) == len(ids)
            rows.append({"input_ids": ids, "loss_mask": mask,
                         "size": gt["size"], "helper": x})
    return rows, bad


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(4242)
    train_gts = [json.loads(l) for l in open("data/arme/pool_train.jsonl")]
    test_gts = [json.loads(l) for l in open("data/arme/pool_test.jsonl")]
    # 4 reps per train board so each helper is well represented (~uniform)
    train_rows, b1 = build(tok, train_gts, rng, reps=4)
    val_rows, b2 = build(tok, test_gts[:VAL_BOARDS], rng, reps=1)
    pd.DataFrame(train_rows).to_parquet("data/arme/r2_train.parquet")
    pd.DataFrame(val_rows).to_parquet("data/arme/r2_val.parquet")
    import collections
    bal = collections.Counter(r["helper"] for r in train_rows)
    print(f"R2 SFT: train {len(train_rows)} (dropped {b1}) val {len(val_rows)} "
          f"(dropped {b2}); helper balance {dict(bal)}")


if __name__ == "__main__":
    main()
