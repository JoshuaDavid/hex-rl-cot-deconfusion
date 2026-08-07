"""Long-path-enriched witness eval set (phase-2 yardstick).

Stratified by gold path length: bins 14-17, 18-21, 22-25, 26+, target 125
boards each (500 total), sizes 7-9, winner-balanced within bins, disjoint
from armD2 train/val positions. Low p_forward walks make long paths common;
per-bin rejection fills the strata.

Output: data/armD2/test_longpath.parquet (eval format).
Run: /venv/main/bin/python scripts/build_armD_longpath_eval.py
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from hexenv.board import Board, BLACK, WHITE, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from scripts.build_sft_certificates import Q, TAIL
from scripts.build_armD_witness import grade
from scripts.build_armD_witness_v2 import fabricate_moves
from scripts.witness_constructive import gen_board

BINS = [(14, 17), (18, 21), (22, 25), (26, 99)]
PER_BIN = 125
SIZES = [7, 8, 9]
MAX_SECONDS = 600


def bin_of(plen):
    for i, (a, b) in enumerate(BINS):
        if a <= plen <= b:
            return i
    return None


def main():
    # positions already used in train/val (any split leakage is disqualifying)
    used = set()
    for f in ["train", "val", "test"]:
        df = pd.read_parquet(f"data/armD2/{f}.parquet")
        gts = (df["ground_truth"] if "ground_truth" in df.columns
               else df["reward_model"].map(lambda m: m["ground_truth"]))
        for gt in gts:
            g = json.loads(gt)
            used.add((g["path_winner"],
                      frozenset((c, cell) for c, cell in map(tuple, g["moves"]))))

    rng = random.Random(20260807)
    bins = [[] for _ in BINS]
    bin_winner = [{"Black": 0, "White": 0} for _ in BINS]
    t0 = time.time()
    attempts = 0
    while any(len(b) < PER_BIN for b in bins) and time.time() - t0 < MAX_SECONDS:
        attempts += 1
        n = rng.choice(SIZES)
        winner = rng.choice(["Black", "White"])
        out = gen_board(rng, n, winner,
                        p_forward=rng.uniform(0.0, 0.08),
                        extra_winner_frac=rng.uniform(0.0, 0.6))
        if out is None:
            continue
        wst, lst, path = out
        bi = bin_of(len(path))
        if bi is None or len(bins[bi]) >= PER_BIN:
            continue
        if bin_winner[bi][winner] >= PER_BIN // 2 + 5:
            continue
        moves = fabricate_moves(rng, winner, wst, lst, path)
        key = (winner, frozenset((c, cell) for c, cell in moves))
        if key in used:
            continue
        used.add(key)
        bin_winner[bi][winner] += 1

        b = Board(n)
        wcol = BLACK if winner == "Black" else WHITE
        lcol = WHITE if winner == "Black" else BLACK
        for x, y in wst:
            b.grid[y][x] = wcol
        for x, y in lst:
            b.grid[y][x] = lcol
        names = [cell_name(x, y) for x, y in path]
        gt = {"category": "witness_armD2_long", "task": "path", "size": n,
              "moves": moves, "path_winner": winner}
        assert grade(gt, winner, names) == 1.0
        bins[bi].append({
            "data_source": "hex_witness_armD2",
            "prompt": [{"role": "user",
                        "content": RULES.format(n=n, board=render_ascii(b)) + Q + TAIL}],
            "ability": "hex_path",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
            "extra_info": {"category": "witness_armD2_long", "size": n,
                           "task": "path", "path_len": len(names),
                           "gold_answer": "Answer: " + json.dumps(
                               {"winner": winner, "path": names})},
        })

    rows = [r for b in bins for r in b]
    for (a, z), b, wc in zip(BINS, bins, bin_winner):
        print(f"bin {a}-{z}: {len(b)} boards ({wc['Black']}B/{wc['White']}W)")
    print(f"total {len(rows)} in {attempts} attempts, {time.time()-t0:.0f}s")
    pd.DataFrame(rows).to_parquet("data/armD2/test_longpath.parquet")
    print("wrote data/armD2/test_longpath.parquet")


if __name__ == "__main__":
    main()
