"""Arm D v2 data: constructive witness boards, sizes 2x2-9x9.

Boards come from scripts/witness_constructive.py (planted unique induced
path + distractors). Per-board randomization:
  - 25% long-path bucket: p_forward ~ U(0.05, 0.2) (wandering walks)
    else p_forward ~ U(0.3, 0.9)
  - density: extra_winner_frac ~ U(0, 0.6)
Targets/weights identical to v1 (no-think, token-importance loss weights).

Outputs (data/armD2/): train.parquet / val.parquet (SFT format),
test.parquet (eval format), train_debug.jsonl (stratified by size).

Run: /venv/main/bin/python scripts/build_armD_witness_v2.py
"""

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from transformers import AutoTokenizer

from hexenv.board import Board, BLACK, WHITE, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from scripts.build_sft_certificates import Q, TAIL
from scripts.build_armD_witness import MODEL, cell_weights, grade, tokenize_row
from scripts.witness_constructive import gen_board

SIZES = [2, 3, 4, 5, 6, 7, 8, 9]
TRAIN_CAP = 1000
VAL_CAP = 50
TEST_CAP = 100
ATTEMPTS = 40_000
STALL = 8_000


def fabricate_moves(rng, winner, wstones, lstones, path):
    """Alternating move order (Black first) ending with the winner placing
    the final path cell. Order is irrelevant to grading; kept plausible."""
    last = path[-1]
    w_rest = [c for c in wstones if c != last]
    rng.shuffle(w_rest)
    w_seq = w_rest + [last]
    l_seq = list(lstones)
    rng.shuffle(l_seq)
    wtag = "B" if winner == "Black" else "W"
    ltag = "W" if winner == "Black" else "B"
    moves = []
    if winner == "Black":  # B W B W ... B
        for i, c in enumerate(w_seq):
            moves.append([wtag, cell_name(*c)])
            if i < len(l_seq):
                moves.append([ltag, cell_name(*l_seq[i])])
    else:  # B W B W ... W
        for i, c in enumerate(w_seq):
            moves.append([ltag, cell_name(*l_seq[i])])
            moves.append([wtag, cell_name(*c)])
    return moves


def build_pools():
    rng = random.Random(20260807)
    pools = {}
    target = TRAIN_CAP + VAL_CAP + TEST_CAP
    for n in SIZES:
        pool = {}
        winner_count = {"Black": 0, "White": 0}
        cap_w = int(target * 0.55)
        t0, last_new = time.time(), 0
        for attempt in range(ATTEMPTS):
            if len(pool) >= target or attempt - last_new > STALL:
                break
            winner = rng.choice(["Black", "White"])
            if winner_count[winner] >= cap_w:
                winner = "Black" if winner == "White" else "White"
                if winner_count[winner] >= cap_w:
                    break
            p_forward = (rng.uniform(0.05, 0.2) if rng.random() < 0.25
                         else rng.uniform(0.3, 0.9))
            out = gen_board(rng, n, winner, p_forward=p_forward,
                            extra_winner_frac=rng.uniform(0.0, 0.6))
            if out is None:
                continue
            wst, lst, path = out
            key = (winner, frozenset(wst), frozenset(lst))
            if key in pool:
                continue
            last_new = attempt
            winner_count[winner] += 1

            b = Board(n)
            wcol = BLACK if winner == "Black" else WHITE
            lcol = WHITE if winner == "Black" else BLACK
            for x, y in wst:
                b.grid[y][x] = wcol
            for x, y in lst:
                b.grid[y][x] = lcol
            names = [cell_name(x, y) for x, y in path]
            gt = {"category": "witness_armD2", "task": "path", "size": n,
                  "moves": fabricate_moves(rng, winner, wst, lst, path),
                  "path_winner": winner}
            assert grade(gt, winner, names) == 1.0, (gt, names)
            pool[key] = {
                "size": n, "winner": winner, "gold_path": names,
                "ground_truth": gt,
                "prompt_text": RULES.format(n=n, board=render_ascii(b)) + Q + TAIL,
                "cell_weights": cell_weights(gt, winner, names),
            }
        nB = sum(1 for r in pool.values() if r["winner"] == "Black")
        print(f"size {n}: pool {len(pool)} ({nB}B/{len(pool) - nB}W) "
              f"after {attempt + 1} attempts, {time.time() - t0:.0f}s", flush=True)
        pools[n] = pool
    return pools


def main():
    os.makedirs("data/armD2", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    pools = build_pools()
    rng = random.Random(11)
    train, val, test = [], [], []
    for n in SIZES:
        recs = list(pools[n].values())
        by_w = {"Black": [], "White": []}
        for r in recs:
            by_w[r["winner"]].append(r)
        for lst in by_w.values():
            rng.shuffle(lst)
        import itertools
        inter = [r for pair in itertools.zip_longest(by_w["Black"], by_w["White"])
                 for r in pair if r is not None]
        m = len(inter)
        n_test = min(TEST_CAP, max(2, m // 4))
        n_val = min(VAL_CAP, max(1, m // 10))
        test += inter[:n_test]
        val += inter[n_test:n_test + n_val]
        train += inter[n_test + n_val:n_test + n_val + TRAIN_CAP]
        print(f"size {n}: train {min(m - n_test - n_val, TRAIN_CAP)} "
              f"val {n_val} test {n_test}")

    for name, split in [("train", train), ("val", val)]:
        rows = []
        for r in split:
            t = tokenize_row(tok, r)
            rows.append({
                "input_ids": t["input_ids"], "loss_mask": t["loss_mask"],
                "size": r["size"], "winner": r["winner"],
                "path_len": len(r["gold_path"]),
                "prompt_text": r["prompt_text"], "answer_text": t["answer_text"],
                "ground_truth": json.dumps(r["ground_truth"]),
            })
        pd.DataFrame(rows).to_parquet(f"data/armD2/{name}.parquet")
        lens = sorted(len(x["input_ids"]) for x in rows)
        print(f"{name}: {len(rows)} rows, seq len med/max "
              f"{lens[len(lens)//2]}/{lens[-1]}")

    test_rows = []
    for r in test:
        test_rows.append({
            "data_source": "hex_witness_armD2",
            "prompt": [{"role": "user", "content": r["prompt_text"]}],
            "ability": "hex_path",
            "reward_model": {"style": "rule",
                             "ground_truth": json.dumps(r["ground_truth"])},
            "extra_info": {"category": "witness_armD2", "size": r["size"],
                           "task": "path", "path_len": len(r["gold_path"]),
                           "gold_answer": "Answer: " + json.dumps(
                               {"winner": r["winner"], "path": r["gold_path"]})},
        })
    pd.DataFrame(test_rows).to_parquet("data/armD2/test.parquet")
    print(f"test: {len(test_rows)} rows")

    with open("data/armD2/train_debug.jsonl", "w") as f:
        by_size = {}
        for r in train:
            by_size.setdefault(r["size"], []).append(r)
        for n in SIZES:
            for r in by_size.get(n, [])[:6]:
                t = tokenize_row(tok, r)
                f.write(json.dumps({
                    "size": r["size"], "winner": r["winner"],
                    "gold_path": r["gold_path"],
                    "cell_weights": r["cell_weights"],
                    "prompt_text": r["prompt_text"],
                    "answer_text": t["answer_text"],
                    "tokens_with_weights": list(zip(
                        t["completion_tokens"], t["completion_weights"])),
                }) + "\n")
    print("wrote data/armD2/train_debug.jsonl")


if __name__ == "__main__":
    main()
