"""Checkpoint eval with forced-close generation (matches training rollout scheme).

Win-preserving rate + legal rate + natural-close rate + think length on held-out
positions; stores full CoTs for C2/C3.

Usage: /venv/vllm/bin/python scripts/eval_checkpoint.py --model <hf-dir-or-name> \
    --corpus data/verl_hex/val_positions.jsonl --out results/checkpoints/<tag>.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, BLACK, WHITE
from hexenv.prompts import move_prompt, extract_move
from hexenv.forced_close_gen import generate_forced_close


def board_from_record(rec) -> Board:
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
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--think-budget", type=int, default=2160)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.8)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.corpus)]
    recs = [r for r in recs if r["winner"] == r["to_move"]]
    if args.limit:
        recs = recs[: args.limit]

    from vllm import LLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, max_model_len=4096,
              gpu_memory_utilization=args.gpu_mem)

    prompts = [move_prompt(board_from_record(r)) for r in recs]
    outs = generate_forced_close(llm, tok, prompts, n=args.k,
                                 temperature=args.temperature,
                                 think_budget=args.think_budget)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = n_legal = n_win = n_nat = 0
    think_tok = []
    per_size = {}
    with open(args.out, "w") as f:
        for r, row in zip(recs, outs):
            wins = set(r["winning_moves"])
            legal = set(board_from_record(r).legal_moves())
            moves = [extract_move(s["text"]) for s in row]
            for s, m in zip(row, moves):
                n += 1
                lg = m is not None and m in legal
                wn = lg and m in wins
                n_legal += lg
                n_win += wn
                n_nat += s["natural_close"]
                think_tok.append(s["think_tokens"])
                ps = per_size.setdefault(r["size"], [0, 0, 0])
                ps[0] += 1; ps[1] += lg; ps[2] += wn
            f.write(json.dumps({
                "position": {k: r[k] for k in ("size", "n_stones", "moves", "to_move",
                                               "winning_moves", "n_legal")},
                "moves": moves,
                "responses": [s["text"] for s in row],
                "natural_close": [s["natural_close"] for s in row],
                "p_random_win": len(wins) / len(legal),
            }) + "\n")

    think_tok.sort()
    summary = {
        "model": args.model, "k": args.k, "temperature": args.temperature,
        "n_samples": n, "legal_rate": n_legal / n, "win_rate": n_win / n,
        "natural_close_rate": n_nat / n,
        "think_tokens_p50": think_tok[len(think_tok) // 2],
        "per_size": {str(sz): {"n": v[0], "legal": round(v[1] / v[0], 3),
                               "win": round(v[2] / v[0], 3)}
                     for sz, v in sorted(per_size.items())},
    }
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
