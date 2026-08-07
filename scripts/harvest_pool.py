"""Shared harvest board pool + no-think best-of-k harvest (arm B), with the
same pool reused by arm T's with-think harvest.

Pool: fresh constructive boards, plen-stratified with deep bins oversampled
(8-13:400, 14-17:400, 18-21:400, 22-25:700, 26-32:700), sizes 7-9, disjoint
from armD2 train/val/test, test_longpath, and the pilot boards.

No-think harvest: ep8 adapter, k=8 temp 1.0; keep one grader-perfect answer
per board (the model's own, canonicalized to Answer: {json}).

Outputs:
  data/armD2/harvest_pool.jsonl        (boards: prompt, gt, plen, size)
  data/armD2/harvest_nothink.jsonl     (board + kept answer + stats)

Run: /venv/verl/bin/python scripts/harvest_pool.py [--pool-only]
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, BLACK, WHITE, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from scripts.build_sft_certificates import Q, TAIL
from scripts.build_armD_witness import grade
from scripts.build_armD_witness_v2 import fabricate_moves
from scripts.witness_constructive import gen_board

BIN_TARGETS = [((8, 13), 400), ((14, 17), 400), ((18, 21), 400),
               ((22, 25), 700), ((26, 32), 700)]
K = 8


def used_keys():
    import pandas as pd
    used = set()
    for f in ["train", "val", "test", "test_longpath"]:
        df = pd.read_parquet(f"data/armD2/{f}.parquet")
        gts = (df["ground_truth"] if "ground_truth" in df.columns
               else df["reward_model"].map(lambda m: m["ground_truth"]))
        for gt in gts:
            g = json.loads(gt)
            used.add((g["path_winner"],
                      frozenset((c, cell) for c, cell in map(tuple, g["moves"]))))
    from scripts.pilot_think_harvest import make_boards
    for b in make_boards():
        g = b["gt"]
        used.add((g["path_winner"],
                  frozenset((c, cell) for c, cell in map(tuple, g["moves"]))))
    return used


def build_pool():
    used = used_keys()
    rng = random.Random(20260807 + 1)
    bins = [[] for _ in BIN_TARGETS]
    t0 = time.time()
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
        gt = {"category": "witness_harvest", "task": "path", "size": n,
              "moves": moves, "path_winner": winner}
        assert grade(gt, winner, names) == 1.0
        bins[bi].append({
            "plen": len(path), "size": n, "gt": gt, "gold_path": names,
            "prompt": RULES.format(n=n, board=render_ascii(b)) + Q + TAIL,
        })
    pool = [r for b in bins for r in b]
    print(f"pool: {len(pool)} boards in {time.time()-t0:.0f}s; "
          + ", ".join(f"{a}-{z}:{len(b)}" for ((a, z), _), b in zip(BIN_TARGETS, bins)))
    with open("data/armD2/harvest_pool.jsonl", "w") as f:
        for r in pool:
            f.write(json.dumps(r) + "\n")
    return pool


def harvest_nothink(pool):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from hexenv.reward_verl import compute_score

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": r["prompt"]}],
        add_generation_prompt=True, enable_thinking=False, tokenize=False)
        for r in pool]
    llm = LLM(model="Qwen/Qwen3-1.7B", max_model_len=2048,
              gpu_memory_utilization=0.6, dtype="bfloat16",
              enable_lora=True, max_lora_rank=64)
    adapter = LoRARequest("ep8", 1, "checkpoints/armD2_sft_weighted/adapter_ep8")
    sp = SamplingParams(temperature=1.0, n=K, max_tokens=256)
    outs = llm.generate(prompts, sp, lora_request=adapter)

    kept, stats = [], {}
    for r, o in zip(pool, outs):
        n_perfect = 0
        first = None
        for s in o.outputs:
            d = compute_score("x", s.text, json.dumps(r["gt"]))
            if d["kind_win"]:
                n_perfect += 1
                if first is None:
                    m = json.loads(
                        __import__("re").search(r"\{.*\}", s.text, 16).group(0))
                    first = {"winner": m["winner"],
                             "path": [str(c).lower() for c in m["path"]]}
        b = next(i for i, ((a, z), _) in enumerate(BIN_TARGETS)
                 if a <= r["plen"] <= z)
        st = stats.setdefault(b, [0, 0])
        st[0] += 1
        if first:
            st[1] += 1
            kept.append({**{k: r[k] for k in ("plen", "size", "gt", "prompt")},
                         "answer_obj": first, "n_perfect": n_perfect})
    with open("data/armD2/harvest_nothink.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    for i, ((a, z), _) in enumerate(BIN_TARGETS):
        tot, hit = stats.get(i, [0, 0])
        print(f"bin {a}-{z}: {hit}/{tot} boards with a perfect sample")
    print(f"kept {len(kept)} rows -> data/armD2/harvest_nothink.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-only", action="store_true")
    args = ap.parse_args()
    if os.path.exists("data/armD2/harvest_pool.jsonl"):
        pool = [json.loads(l) for l in open("data/armD2/harvest_pool.jsonl")]
        print(f"loaded existing pool ({len(pool)})")
    else:
        pool = build_pool()
    if not args.pool_only:
        harvest_nothink(pool)
