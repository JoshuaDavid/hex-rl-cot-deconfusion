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
    """Context ends with '<selected-task>' so X is generated as a standalone
    token (matches the R4 loop). Gradient on: the X token (X assigned uniformly
    over SELECTABLE -> installs an explorable ~uniform selection prior, like the
    marker 50/50 init), the helper content, and the winner. Structure tokens are
    mask 0. Char spans -> token mask via offset mapping (robust to merges)."""
    gx = gold_answer_str(x, b)
    gw = gold_answer_str(EVALUATED, b)
    base = tok.apply_chat_template(
        [{"role": "user", "content": select_prompt(b)}],
        add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ctx_ids = tok(base + "<selected-task>", add_special_tokens=False)["input_ids"]
    # build completion string and record char-spans to train
    comp = ""
    xs = len(comp); comp += x; xe = len(comp)                 # selection token
    comp += f"</selected-task>\n<task-{x}>"
    hs = len(comp); comp += gx; he = len(comp)                # helper content
    comp += f"</task-{x}>\n<evaluated-task>"
    ws = len(comp); comp += gw + "</evaluated-task>" + IM_END
    we = len(comp)                                            # winner + close
    train = [(xs, xe), (hs, he), (ws, we)]
    enc = tok(comp, add_special_tokens=False, return_offsets_mapping=True)
    cids, offs = enc["input_ids"], enc["offset_mapping"]
    cmask = [1.0 if any(o[0] < r[1] and o[1] > r[0] for r in train) else 0.0
             for o in offs]
    ids = list(ctx_ids) + list(cids)
    mask = [0.0] * len(ctx_ids) + cmask
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
