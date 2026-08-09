"""Occupancy training data on sizes 5-9 (for the decomposed verifier's
per-cell membership oracle). Balanced occupied/empty queries.

Output: data/occ/{train,test}.parquet
Run: /venv/main/bin/python scripts/build_occ.py
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from transformers import AutoTokenizer
from hexenv.board import Board, BLACK, WHITE, EMPTY, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from hexenv.reward_verl import compute_score
from scripts.build_armD_witness import MODEL, IM_END
from scripts.build_armD_witness_v2 import fabricate_moves
from scripts.witness_constructive import gen_board

OCC_Q = ("\nWhich player, if any, has a stone on cell {cell}?"
         "\nEnd your response with exactly one line of the form:"
         "\nAnswer: Black|White|Neither\n")


def occ_prompt(n, board, cell):
    return RULES.format(n=n, board=render_ascii(board)) + OCC_Q.format(cell=cell)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(7)
    rows = []
    while len(rows) < 4200:
        n = rng.choice([5, 6, 7, 8, 9])
        winner = rng.choice(["Black", "White"])
        g = gen_board(rng, n, winner, p_forward=rng.uniform(0.0, 0.5),
                      extra_winner_frac=rng.uniform(0.1, 0.6))
        if not g:
            continue
        wst, lst, path = g
        b = Board(n)
        for x, y in wst:
            b.grid[y][x] = BLACK if winner == "Black" else WHITE
        for x, y in lst:
            b.grid[y][x] = WHITE if winner == "Black" else BLACK
        occ = [(x, y) for y in range(n) for x in range(n) if b.grid[y][x] != EMPTY]
        emp = [(x, y) for y in range(n) for x in range(n) if b.grid[y][x] == EMPTY]
        bucket = occ if (rng.random() < 0.5 and occ) or not emp else emp
        x, y = rng.choice(bucket)
        cell = cell_name(x, y)
        v = b.grid[y][x]
        lab = "Black" if v == BLACK else ("White" if v == WHITE else "Neither")
        gt = {"category": "occ", "task": "judge", "size": n,
              "moves": fabricate_moves(random, winner, wst, lst, path),
              "judge_label": lab}
        ans = f"Answer: {lab}"
        assert compute_score("x", ans, json.dumps(gt))["score"] == 1.0
        pids = tok.apply_chat_template([{"role": "user", "content": occ_prompt(n, b, cell)}],
                                       add_generation_prompt=True, enable_thinking=False,
                                       tokenize=True)["input_ids"]
        comp = tok(ans + IM_END, add_special_tokens=False)["input_ids"]
        rows.append({"input_ids": list(pids) + list(comp),
                     "loss_mask": [0.0] * len(pids) + [1.0] * len(comp), "size": n})
    os.makedirs("data/occ", exist_ok=True)
    pd.DataFrame(rows[:3800]).to_parquet("data/occ/train.parquet")
    pd.DataFrame(rows[3800:]).to_parquet("data/occ/test.parquet")
    print(f"occ train {3800} test {len(rows) - 3800}")


if __name__ == "__main__":
    main()
