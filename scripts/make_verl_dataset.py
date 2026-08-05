"""Convert solver-labeled corpora (build_corpus.py jsonl) into verl parquet.

Keeps only positions winning for the player to move (else zero group variance).
Splits train/val. Schema follows verl conventions (confirm against notes/verl_setup.md).
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.reward import board_from_gt
from hexenv.prompts import move_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpora", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    seen = set()
    for path in args.corpora:
        for line in open(path):
            r = json.loads(line)
            key = (r["size"], tuple(sorted(map(tuple, r["moves"]))), r["to_move"])
            if key in seen:
                continue
            seen.add(key)
            if r["winner"] != r["to_move"]:
                continue
            # exclude fully-determined positions (all moves win): zero group
            # variance under GRPO => no gradient, wasted rollouts
            if len(r["winning_moves"]) >= r["n_legal"]:
                continue
            gt = {
                "size": r["size"],
                "moves": r["moves"],
                "to_move": r["to_move"],
                "winning_moves": r["winning_moves"],
            }
            prompt_text = move_prompt(board_from_gt(gt))
            r["_corpus"] = {k: r[k] for k in ("size", "n_stones", "moves", "to_move",
                                              "winner", "winning_moves", "n_legal")}
            rows.append({
                "_corpus": r["_corpus"],
                "data_source": "hex_solver",
                "prompt": [{"role": "user", "content": prompt_text}],
                "ability": "hex",
                "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
                "extra_info": {"size": r["size"], "n_stones": r["n_stones"],
                               "p_random_win": len(r["winning_moves"]) / r["n_legal"]},
            })

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:n_val], rows[n_val:]

    import pandas as pd
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "val_positions.jsonl"), "w") as f:
        for row in val:
            f.write(json.dumps(row["_corpus"]) + "\n")
    for row in train + val:
        row.pop("_corpus")
    pd.DataFrame(train).to_parquet(os.path.join(args.out_dir, "train.parquet"))
    pd.DataFrame(val).to_parquet(os.path.join(args.out_dir, "val.parquet"))
    print(f"train {len(train)}, val {len(val)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
