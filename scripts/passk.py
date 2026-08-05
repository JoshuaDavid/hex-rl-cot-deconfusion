"""C1 pass@k envelope: sample k rollouts per position (forced-close scheme),
count positions where >=1 sampled move is win-preserving, for k in powers of 2.

Run per model (base + checkpoints) on THE SAME fixed position set.
Chunked sampling (n per call) to bound memory; accumulates counts.

/venv/vllm/bin/python scripts/passk.py --model Qwen/Qwen3-1.7B \
  --corpus data/verl_hex/val_positions.jsonl --k-max 1024 --limit 100 \
  --out results/passk/base_1p7b.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, BLACK, WHITE
from hexenv.prompts import move_prompt, extract_move
from hexenv.forced_close_gen import generate_forced_close


def board_from_record(rec):
    b = Board(rec["size"])
    for color, cell in rec["moves"]:
        b.play(cell, BLACK if color == "B" else WHITE)
    b.to_move = BLACK if rec["to_move"] == "B" else WHITE
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-max", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--think-budget", type=int, default=2160)
    ap.add_argument("--gpu-mem", type=float, default=0.8)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.corpus)]
    recs = [r for r in recs if r["winner"] == r["to_move"]
            and 0 < len(r["winning_moves"]) < r["n_legal"]]
    recs = recs[: args.limit]

    from vllm import LLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, max_model_len=4096,
              gpu_memory_utilization=args.gpu_mem)

    prompts = [move_prompt(board_from_record(r)) for r in recs]
    wins_sets = [set(r["winning_moves"]) for r in recs]
    legal_sets = [set(board_from_record(r).legal_moves()) for r in recs]

    # success_at[i][j] = 1 if sample j for position i was win-preserving
    successes = [[] for _ in recs]
    move_lists = [[] for _ in recs]
    n_done = 0
    seed = 0
    while n_done < args.k_max:
        n_this = min(args.chunk, args.k_max - n_done)
        outs = generate_forced_close(llm, tok, prompts, n=n_this,
                                     temperature=args.temperature,
                                     think_budget=args.think_budget, seed=seed)
        for i, row in enumerate(outs):
            for s in row:
                mv = extract_move(s["text"])
                ok = mv is not None and mv in legal_sets[i] and mv in wins_sets[i]
                successes[i].append(ok)
                move_lists[i].append(mv)
        n_done += n_this
        seed += 1
        # incremental save
        ks = [k for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] if k <= n_done]
        passk = {}
        for k in ks:
            cnt = sum(1 for s in successes if any(s[:k]))
            passk[k] = cnt / len(recs)
        result = {
            "model": args.model, "temperature": args.temperature,
            "n_positions": len(recs), "samples_done": n_done, "pass_at_k": passk,
            "move_support": [sorted({m for m in ml if m}) for ml in move_lists],
            "successes": ["".join("1" if x else "0" for x in s) for s in successes],
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"))
        print(f"samples {n_done}/{args.k_max}: pass@k {passk}", flush=True)


if __name__ == "__main__":
    main()
