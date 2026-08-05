"""Emit per-category verl parquets into data/curriculum/, with the category
threaded into ground_truth (so the reward side channel logs it) and extra_info.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from hexenv.reward import board_from_gt
from hexenv.prompts import move_prompt, RULES
from hexenv.render import render_ascii
from hexenv.board import Board, BLACK, WHITE

OUT = "data/curriculum"
JUDGE_SUFFIX = (
    "\nHas either player ALREADY completed a winning connection on this board?"
    "\nEnd your response with exactly one line of the form:"
    "\nAnswer: Black|White|Neither\n"
)


def key_of(r):
    return (r["size"], tuple(map(tuple, r["moves"])), r.get("to_move", "J"))


def load_move(paths, val_keys):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        for l in open(p):
            r = json.loads(l)
            if r["winner"] != r["to_move"] or not (0 < len(r["winning_moves"]) < r["n_legal"]):
                continue
            if key_of(r) in val_keys:
                continue
            rows.append(r)
    return rows


def move_row(r, category):
    gt = {"category": category, "size": r["size"], "moves": r["moves"],
          "to_move": r["to_move"], "winning_moves": r["winning_moves"]}
    return {
        "data_source": "hex_solver",
        "prompt": [{"role": "user", "content": move_prompt(board_from_gt(gt))}],
        "ability": "hex",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
        "extra_info": {"category": category, "size": r["size"]},
    }


def judge_row(r, category):
    gt = {"category": category, "task": "judge", "size": r["size"],
          "moves": r["moves"], "judge_label": r["judge_label"]}
    b = Board(r["size"])
    for c, cell in r["moves"]:
        b.play(cell, BLACK if c == "B" else WHITE)
    prompt = RULES.format(n=r["size"], board=render_ascii(b)) + JUDGE_SUFFIX
    return {
        "data_source": "hex_solver",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "hex_judge",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
        "extra_info": {"category": category, "size": r["size"], "task": "judge"},
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    val_keys = set()
    for p in ["data/verl_hex/val_positions.jsonl", "data/verl_hex/val_edge_positions.jsonl"]:
        for l in open(p):
            val_keys.add(key_of(json.loads(l)))

    cats = {
        "edge_m1": load_move(["data/corpus_mate1_edge.jsonl"], val_keys),
        "gen_m1": load_move(["data/corpus_mate1_gen.jsonl", "data/corpus_mate1.jsonl"], val_keys),
        "mate2": load_move(["data/corpus_mate2.jsonl"], val_keys),
        "general": load_move(["data/corpus_5x5.jsonl", "data/corpus_5x5_v2.jsonl",
                              "data/corpus_6x6.jsonl", "data/corpus_6x6_v2.jsonl",
                              "data/corpus_7x7.jsonl", "data/corpus_7x7_v2.jsonl"], val_keys),
    }
    for cat, rows in cats.items():
        df = pd.DataFrame([move_row(r, cat) for r in rows])
        df.to_parquet(f"{OUT}/{cat}.parquet")
        print(f"{cat}: {len(df)} rows")

    judge = [json.loads(l) for l in open("data/corpus_judge.jsonl")]
    df = pd.DataFrame([judge_row(r, "judge") for r in judge])
    df.to_parquet(f"{OUT}/judge.parquet")
    print(f"judge: {len(df)} rows")


if __name__ == "__main__":
    main()
