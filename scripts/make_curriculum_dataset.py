"""Build staged curriculum datasets for arm B.

Stage mixes (fractions of train rows):
  B1: 0.35 edge-mate1, 0.35 general-mate1, 0.30 general
  B2: 0.20 edge-mate1, 0.20 general-mate1, 0.35 mate2, 0.25 general
  B3: 0.10 mate1(any), 0.20 mate2, 0.70 general
Val (shared, fixed): existing val_positions + 60 held-out edge-mate1 positions
(val_edge_positions.jsonl, disjoint from train by position key).

Usage: /venv/main/bin/python scripts/make_curriculum_dataset.py B1
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.reward import board_from_gt
from hexenv.prompts import move_prompt

MIXES = {
    "B1": {"judge": 0.15, "edge_m1": 0.30, "gen_m1": 0.30, "mate2": 0.0, "general": 0.25},
    "B2": {"judge": 0.05, "edge_m1": 0.20, "gen_m1": 0.15, "mate2": 0.35, "general": 0.25},
    "B3": {"judge": 0.0, "edge_m1": 0.05, "gen_m1": 0.05, "mate2": 0.20, "general": 0.70},
}
TRAIN_SIZE = 5000


def key_of(r):
    return (r["size"], tuple(map(tuple, r["moves"])), r.get("to_move", "J"))


def load(path):
    rows = []
    for l in open(path):
        r = json.loads(l)
        if r["winner"] == r["to_move"] and 0 < len(r["winning_moves"]) < r["n_legal"]:
            rows.append(r)
    return rows


JUDGE_SUFFIX = (
    "\nWhich player, if any, has ALREADY completed a winning connection on this board?"
    "\nEnd your response with exactly one line of the form:"
    "\nAnswer: Black|White|Neither\n"
)


def to_row(r):
    if "judge_label" in r:
        gt = {"task": "judge", "size": r["size"], "moves": r["moves"],
              "judge_label": r["judge_label"]}
        from hexenv.prompts import RULES
        from hexenv.render import render_ascii
        from hexenv.board import Board, BLACK, WHITE
        b = Board(r["size"])
        for c, cell in r["moves"]:
            b.play(cell, BLACK if c == "B" else WHITE)
        prompt = RULES.format(n=r["size"], board=render_ascii(b)) + JUDGE_SUFFIX
        return {
            "data_source": "hex_solver",
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "hex_judge",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
            "extra_info": {"size": r["size"], "task": "judge"},
        }
    gt = {"size": r["size"], "moves": r["moves"], "to_move": r["to_move"],
          "winning_moves": r["winning_moves"]}
    return {
        "data_source": "hex_solver",
        "prompt": [{"role": "user", "content": move_prompt(board_from_gt(gt))}],
        "ability": "hex",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
        "extra_info": {"size": r["size"], "n_stones": r["n_stones"],
                       "p_random_win": len(r["winning_moves"]) / r["n_legal"]},
    }


def main():
    stage = sys.argv[1]
    mix = MIXES[stage]
    rng = random.Random(99)

    edge_m1 = load("data/corpus_mate1_edge.jsonl")
    gen_m1 = load("data/corpus_mate1_gen.jsonl") + load("data/corpus_mate1.jsonl")
    mate2 = load("data/corpus_mate2.jsonl") if os.path.exists("data/corpus_mate2.jsonl") else []
    general = []
    for p in ["data/corpus_5x5.jsonl", "data/corpus_5x5_v2.jsonl",
              "data/corpus_6x6.jsonl", "data/corpus_6x6_v2.jsonl",
              "data/corpus_7x7.jsonl", "data/corpus_7x7_v2.jsonl"]:
        general += load(p)

    # held-out edge val slice: fixed 60 edge-mate1 positions, excluded from train
    rng_val = random.Random(4242)
    rng_val.shuffle(edge_m1)
    val_edge = edge_m1[:60]
    edge_m1 = edge_m1[60:]
    if not os.path.exists("data/verl_hex/val_edge_positions.jsonl"):
        with open("data/verl_hex/val_edge_positions.jsonl", "w") as f:
            for r in val_edge:
                f.write(json.dumps(r) + "\n")

    # exclude general-val positions from general pool
    val_keys = {key_of(json.loads(l)) for l in open("data/verl_hex/val_positions.jsonl")}
    val_keys |= {key_of(r) for r in val_edge}
    general = [r for r in general if key_of(r) not in val_keys]

    judge = [json.loads(l) for l in open("data/corpus_judge.jsonl")] \
        if os.path.exists("data/corpus_judge.jsonl") else []
    pools = {"judge": judge, "edge_m1": edge_m1, "gen_m1": gen_m1,
             "mate2": mate2, "general": general}
    rows, seen = [], set()
    for name, frac in mix.items():
        want = int(TRAIN_SIZE * frac)
        pool = pools[name][:]
        rng.shuffle(pool)
        got = 0
        for r in pool:
            if got >= want:
                break
            k = key_of(r)
            if k in val_keys or k in seen:
                continue
            seen.add(k)
            rows.append(to_row(r))
            got += 1
        print(f"{name}: {got}/{want} (pool {len(pool)})")

    rng.shuffle(rows)
    import pandas as pd
    out_dir = f"data/verl_hex_{stage}"
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(out_dir, "train.parquet"))
    # shared val = general val + edge val, in verl schema
    val_rows = []
    for p in ["data/verl_hex/val_positions.jsonl", "data/verl_hex/val_edge_positions.jsonl"]:
        for l in open(p):
            r = json.loads(l)
            if r["winner"] == r["to_move"]:
                val_rows.append(to_row(r))
    pd.DataFrame(val_rows).to_parquet(os.path.join(out_dir, "val.parquet"))
    print(f"train {len(rows)}, val {len(val_rows)} -> {out_dir}")


if __name__ == "__main__":
    main()
