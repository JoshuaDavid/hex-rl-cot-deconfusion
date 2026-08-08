"""3-task interference study data: witness / occupancy / reading-order listing,
all derived from ONE shared constructive-board pool (sizes 5-7) so any
cross-task degradation is about the task mapping, not a board-distribution
shift. Test boards are disjoint from train boards for every task.

Tasks (all no-think, uniform answer-token loss weight):
  T_wit  (task=path)    : witness — winner + unique winning path
  T_cell (task=judge)   : "Which player, if any, has a stone on cell X?"
  T_list (task=listing) : "List ALL {Black|White|empty} cells in reading order"

Reuses the existing graders in hexenv/reward_verl.py (path/judge/listing).

Output: data/interference/{wit,cell,list}_{train,test}.parquet
Run: /venv/main/bin/python scripts/build_interference_data.py
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from transformers import AutoTokenizer

from hexenv.board import Board, BLACK, WHITE, EMPTY, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from hexenv.reward_verl import compute_score
from scripts.build_sft_certificates import Q, TAIL
from scripts.build_armD_witness import MODEL, IM_END
from scripts.build_armD_witness_v2 import fabricate_moves
from scripts.witness_constructive import gen_board

SIZES = [5, 6, 7]
N_TRAIN_BOARDS = 1600
N_TEST_BOARDS = 400
READING_ORDER_NOTE = (" Report them in reading order: row 1 left-to-right,"
                      " then row 2, and so on.")


def board_from(n, winner, wst, lst):
    b = Board(n)
    wcol = BLACK if winner == "Black" else WHITE
    lcol = WHITE if winner == "Black" else BLACK
    for x, y in wst:
        b.grid[y][x] = wcol
    for x, y in lst:
        b.grid[y][x] = lcol
    return b


def reading_order(b, kind):
    """kind in {'Black','White','empty'} -> list of cell names in reading order."""
    want = {"Black": BLACK, "White": WHITE, "empty": EMPTY}[kind]
    out = []
    for y in range(b.size):
        for x in range(b.size):
            if b.grid[y][x] == want:
                out.append(cell_name(x, y))
    return out


def make_pool(rng, n_boards, used):
    pool = []
    while len(pool) < n_boards:
        n = rng.choice(SIZES)
        winner = rng.choice(["Black", "White"])
        out = gen_board(rng, n, winner, p_forward=rng.uniform(0.2, 0.8),
                        extra_winner_frac=rng.uniform(0.1, 0.6))
        if out is None:
            continue
        wst, lst, path = out
        moves = fabricate_moves(rng, winner, wst, lst, path)
        key = (winner, frozenset((c, cell) for c, cell in moves))
        if key in used:
            continue
        used.add(key)
        b = board_from(n, winner, wst, lst)
        pool.append({"n": n, "winner": winner, "moves": moves,
                     "path": [cell_name(x, y) for x, y in path], "board": b})
    return pool


def wit_row(rec):
    b = rec["board"]
    gt = {"category": "wit_if", "task": "path", "size": b.size,
          "moves": rec["moves"], "path_winner": rec["winner"]}
    prompt = RULES.format(n=b.size, board=render_ascii(b)) + Q + TAIL
    answer = "Answer: " + json.dumps({"winner": rec["winner"], "path": rec["path"]})
    return prompt, answer, gt


def cell_row(rng, rec):
    b = rec["board"]
    # balance occupied vs empty queries ~50/50 so "Neither" isn't a lazy win
    occ = [(x, y) for y in range(b.size) for x in range(b.size)
           if b.grid[y][x] != EMPTY]
    emp = [(x, y) for y in range(b.size) for x in range(b.size)
           if b.grid[y][x] == EMPTY]
    bucket = occ if (rng.random() < 0.5 and occ) or not emp else emp
    x, y = rng.choice(bucket)
    cell = cell_name(x, y)
    v = b.grid[y][x]
    label = "Black" if v == BLACK else ("White" if v == WHITE else "Neither")
    q = (f"\nWhich player, if any, has a stone on cell {cell}?"
         "\nEnd your response with exactly one line of the form:"
         "\nAnswer: Black|White|Neither\n")
    gt = {"category": "cell_if", "task": "judge", "size": b.size,
          "moves": rec["moves"], "judge_label": label, "cell": cell}
    prompt = RULES.format(n=b.size, board=render_ascii(b)) + q
    return prompt, f"Answer: {label}", gt


def list_row(rng, rec):
    b = rec["board"]
    # only ask about a class that actually occurs (an empty target is
    # unrepresentable by the listing grader -> would score -1)
    kinds = [k for k in ["Black", "White", "empty"] if reading_order(b, k)]
    kind = rng.choice(kinds)
    target = reading_order(b, kind)
    noun = {"Black": "cells holding a Black stone",
            "White": "cells holding a White stone",
            "empty": "empty cells"}[kind]
    q = (f"\nList ALL {noun}.{READING_ORDER_NOTE}"
         "\nEnd your response with exactly one line of the form:"
         '\nAnswer: ["cell", "cell", ...] (a JSON array)\n')
    gt = {"category": "list_if", "task": "listing", "size": b.size,
          "moves": rec["moves"], "list_target": target, "list_kind": kind}
    prompt = RULES.format(n=b.size, board=render_ascii(b)) + q
    return prompt, "Answer: " + json.dumps(target), gt


def tokenize(tok, prompt, answer):
    pids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        enable_thinking=False, tokenize=True)["input_ids"]
    comp = tok(answer + IM_END, add_special_tokens=False)["input_ids"]
    return {"input_ids": list(pids) + list(comp),
            "loss_mask": [0.0] * len(pids) + [1.0] * len(comp)}


def build_split(tok, pool, rng, tag):
    rows = {"wit": [], "cell": [], "list": []}
    for rec in pool:
        for key, (prompt, answer, gt) in {
            "wit": wit_row(rec),
            "cell": cell_row(rng, rec),
            "list": list_row(rng, rec),
        }.items():
            # every gold answer must score 1.0 through the real grader
            assert compute_score("x", answer, json.dumps(gt))["score"] == 1.0, \
                (key, gt, answer)
            t = tokenize(tok, prompt, answer)
            rows[key].append({
                "input_ids": t["input_ids"], "loss_mask": t["loss_mask"],
                "size": rec["n"], "prompt_text": prompt,
                "answer_text": answer, "ground_truth": json.dumps(gt),
                "data_source": f"hex_{key}_if",
                "prompt": [{"role": "user", "content": prompt}],
                "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
                "ability": "hex", "extra_info": {"category": f"{key}_if",
                                                 "size": rec["n"]},
            })
    for key, rs in rows.items():
        pd.DataFrame(rs).to_parquet(f"data/interference/{key}_{tag}.parquet")
        print(f"{key}_{tag}: {len(rs)} rows")


def main():
    os.makedirs("data/interference", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(20260808)
    used = set()
    train_pool = make_pool(rng, N_TRAIN_BOARDS, used)
    test_pool = make_pool(rng, N_TEST_BOARDS, used)
    print(f"pools: train {len(train_pool)}, test {len(test_pool)} "
          f"(disjoint boards)")
    build_split(tok, train_pool, rng, "train")
    build_split(tok, test_pool, rng, "test")


if __name__ == "__main__":
    main()
