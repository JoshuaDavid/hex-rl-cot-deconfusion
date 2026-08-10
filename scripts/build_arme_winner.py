"""Winner-experiment SFT data (per user redirect 2026-08-10): evaluated task =
W (winner), useful helper = D (edge-connectivity -> winner is an intersection
check). Gradient flows through BOTH the helper AND the evaluated answer, and
SOLO-W examples are included so 'winner with no helper' is in-distribution (a
fair baseline for the differential). Select format so R4 follows directly.

Per board (SELECTABLE from ARME_HELPERS, e.g. A,C,D):
  - one select example per helper X: gradient on <task-X>..</task-X> AND
    <evaluated-task>W</evaluated-task>  (selection token X is teacher-forced ctx)
  - one solo example: solo_prompt(W), gradient on <task-W>W</task-W>

Output: data/arme/win_{train,val}.parquet
Run: ARME_EVAL=W ARME_HELPERS=A,C,D /venv/main/bin/python scripts/build_arme_winner.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from transformers import AutoTokenizer

from hexenv.arme import (board_from_gt, gold_answer_str, select_prompt, solo_prompt,
                         split_ctx_ans, SELECTABLE, EVALUATED)
from scripts.build_armD_witness import MODEL, IM_END

VAL_BOARDS = 120


def select_row(tok, b, x):
    gx = gold_answer_str(x, b)
    gw = gold_answer_str(EVALUATED, b)
    ctx0 = f"<selected-task>{x}</selected-task>\n"
    grad = (f"<task-{x}>{gx}</task-{x}>\n"
            f"<evaluated-task>{gw}</evaluated-task>{IM_END}")
    pids = tok.apply_chat_template(
        [{"role": "user", "content": select_prompt(b)}],
        add_generation_prompt=True, enable_thinking=False, tokenize=True)["input_ids"]
    full_ids, astart = split_ctx_ans(tok, ctx0, grad)
    ids = list(pids) + list(full_ids)
    mask = [0.0] * (len(pids) + astart) + [1.0] * (len(full_ids) - astart)
    return ids, mask


def solo_row(tok, b):
    gw = gold_answer_str(EVALUATED, b)
    comp = f"<task-{EVALUATED}>{gw}</task-{EVALUATED}>{IM_END}"
    pids = tok.apply_chat_template(
        [{"role": "user", "content": solo_prompt(b, EVALUATED)}],
        add_generation_prompt=True, enable_thinking=False, tokenize=True)["input_ids"]
    comp_ids = tok(comp, add_special_tokens=False)["input_ids"]
    return list(pids) + list(comp_ids), [0.0] * len(pids) + [1.0] * len(comp_ids)


def build(tok, gts):
    rows = []
    for gt in gts:
        b = board_from_gt(gt)
        for x in SELECTABLE:
            ids, mask = select_row(tok, b, x)
            if len(ids) <= 2048:
                rows.append({"input_ids": ids, "loss_mask": mask, "kind": f"sel_{x}"})
        ids, mask = solo_row(tok, b)
        if len(ids) <= 2048:
            rows.append({"input_ids": ids, "loss_mask": mask, "kind": "solo"})
    return rows


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    train_gts = [json.loads(l) for l in open("data/arme/pool_train.jsonl")]
    test_gts = [json.loads(l) for l in open("data/arme/pool_test.jsonl")]
    train_rows = build(tok, train_gts)
    val_rows = build(tok, test_gts[:VAL_BOARDS])
    pd.DataFrame(train_rows).to_parquet("data/arme/win_train.parquet")
    pd.DataFrame(val_rows).to_parquet("data/arme/win_val.parquet")
    import collections
    print(f"winner SFT: train {len(train_rows)} val {len(val_rows)}; "
          f"kinds {dict(collections.Counter(r['kind'] for r in train_rows))}")


if __name__ == "__main__":
    main()
