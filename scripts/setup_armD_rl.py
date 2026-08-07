"""Arm C (RL-with-think) setup: merge bok adapter into a full HF model and
build the witness long-path RL dataset.

1. checkpoints/armD2_bok/hf_merged  <- Qwen3-1.7B + adapter_ep3 (bf16)
2. data/verl_witness_long/{train,val}.parquet — 4000/128 fresh constructive
   boards, plen 8-32 with ~50% mass at plen>=18, disjoint from every
   existing pool/eval set.

Run: /venv/verl/bin/python scripts/setup_armD_rl.py
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from hexenv.board import Board, BLACK, WHITE, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from scripts.build_sft_certificates import Q, TAIL
from scripts.build_armD_witness import grade
from scripts.build_armD_witness_v2 import fabricate_moves
from scripts.witness_constructive import gen_board
from scripts.harvest_pool import used_keys

BIN_TARGETS = [((8, 13), 800), ((14, 17), 800), ((18, 21), 800),
               ((22, 25), 900), ((26, 32), 828)]  # 4128 = 4000 train + 128 val


def merge():
    out = "checkpoints/armD2_bok/hf_merged"
    if os.path.exists(os.path.join(out, "config.json")):
        print("merged model exists, skipping")
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, "checkpoints/armD2_bok/adapter_ep3")
    model = model.merge_and_unload()
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B").save_pretrained(out)
    print(f"merged -> {out}")


def build_data():
    used = used_keys()
    for f in ["data/armD2/harvest_pool.jsonl"]:
        for l in open(f):
            g = json.loads(l)["gt"]
            used.add((g["path_winner"],
                      frozenset((c, cell) for c, cell in map(tuple, g["moves"]))))
    rng = random.Random(20260807 + 2)
    bins = [[] for _ in BIN_TARGETS]
    while any(len(b) < tgt for b, (_, tgt) in zip(bins, BIN_TARGETS)):
        n = rng.choice([7, 8, 9])
        winner = rng.choice(["Black", "White"])
        out = gen_board(rng, n, winner, p_forward=rng.uniform(0.0, 0.35),
                        extra_winner_frac=rng.uniform(0.0, 0.6))
        if out is None:
            continue
        wst, lst, path = out
        bi = next((i for i, ((a, z), _) in enumerate(BIN_TARGETS)
                   if a <= len(path) <= z), None)
        if bi is None or len(bins[bi]) >= BIN_TARGETS[bi][1]:
            continue
        moves = fabricate_moves(rng, winner, wst, lst, path)
        key = (winner, frozenset((c, cell) for c, cell in moves))
        if key in used:
            continue
        used.add(key)
        b = Board(n)
        wcol = BLACK if winner == "Black" else WHITE
        lcol = WHITE if winner == "Black" else BLACK
        for x, y in wst:
            b.grid[y][x] = wcol
        for x, y in lst:
            b.grid[y][x] = lcol
        names = [cell_name(x, y) for x, y in path]
        gt = {"category": "witness_rl", "task": "path", "size": n,
              "moves": moves, "path_winner": winner}
        assert grade(gt, winner, names) == 1.0
        bins[bi].append({
            "data_source": "hex_witness_rl",
            "prompt": [{"role": "user",
                        "content": RULES.format(n=n, board=render_ascii(b)) + Q + TAIL}],
            "ability": "hex_path",
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
            "extra_info": {"category": "witness_rl", "size": n, "task": "path",
                           "path_len": len(names)},
        })
    rows = [r for b in bins for r in b]
    rng.shuffle(rows)
    os.makedirs("data/verl_witness_long", exist_ok=True)
    pd.DataFrame(rows[128:]).to_parquet("data/verl_witness_long/train.parquet")
    pd.DataFrame(rows[:128]).to_parquet("data/verl_witness_long/val.parquet")
    print(f"train {len(rows)-128}, val 128; bins: "
          + ", ".join(f"{a}-{z}:{len(b)}" for ((a, z), _), b in zip(BIN_TARGETS, bins)))


if __name__ == "__main__":
    merge()
    build_data()
