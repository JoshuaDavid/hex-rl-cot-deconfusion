"""Phase 1 items 4+5: think/no_think move quality vs random baseline, and
GRPO group-variance check (k rollouts per position).

Consumes a solver-labeled corpus (build_corpus.py). Only uses positions where
the player to move is winning (so 'good move' = keeps the win).

Usage: python scripts/phase1_movequality.py --corpus data/corpus_5x5.jsonl \
    [--model Qwen/Qwen3-1.7B] [--no-think] [--k 8] [--n-positions 40]
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, BLACK, WHITE
from hexenv.prompts import move_prompt, extract_move

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "phase1")


def board_from_record(rec) -> Board:
    b = Board(rec["size"])
    for color, cell in rec["moves"]:
        b.play(cell, BLACK if color == "B" else WHITE)
    b.to_move = BLACK if rec["to_move"] == "B" else WHITE
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-positions", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.corpus)]
    winning = [r for r in recs if r["winner"] == r["to_move"] and r["n_stones"] >= 2]
    rng = random.Random(args.seed)
    rng.shuffle(winning)
    positions = winning[: args.n_positions]

    from hexenv.genbackend import Backend

    backend = Backend(args.model, enable_thinking=not args.no_think)
    prompts = [move_prompt(board_from_record(r)) for r in positions]
    outs = backend.generate(prompts, n=args.k, temperature=args.temperature,
                            max_tokens=args.max_tokens, seed=args.seed)

    os.makedirs(RESULTS, exist_ok=True)
    tag = f"movequality_{os.path.basename(args.corpus).split('.')[0]}_{args.model.split('/')[-1]}_{'nothink' if args.no_think else 'think'}"
    path = os.path.join(RESULTS, tag + ".jsonl")

    group_stats = []
    with open(path, "w") as f:
        for r, texts in zip(positions, outs):
            wins = set(r["winning_moves"])
            legal = set(board_from_record(r).legal_moves())
            rewards, moves = [], []
            for text in texts:
                mv = extract_move(text)
                if mv is None or mv not in legal:
                    rew = -1.0
                else:
                    rew = 1.0 if mv in wins else -1.0
                rewards.append(rew)
                moves.append(mv)
            rec = {
                "position": {k: r[k] for k in ("size", "n_stones", "moves", "to_move", "winning_moves", "n_legal")},
                "moves": moves, "rewards": rewards,
                "legal_flags": [m is not None and m in legal for m in moves],
                "p_random_win": len(wins) / len(legal),
                "responses": texts,
            }
            f.write(json.dumps(rec) + "\n")
            group_stats.append(rec)

    # summary
    import statistics
    all_r = [r for g in group_stats for r in g["rewards"]]
    frac_legal = sum(1 for g in group_stats for ok in g["legal_flags"] if ok) / len(all_r)
    frac_win = sum(1 for r in all_r if r > 0) / len(all_r)
    p_rand = statistics.mean(g["p_random_win"] for g in group_stats)
    nonzero_var = sum(1 for g in group_stats if len(set(g["rewards"])) > 1)
    print(f"wrote {path}")
    print(f"positions: {len(group_stats)}, k={args.k}")
    print(f"parse+legal rate: {frac_legal:.3f}")
    print(f"win-preserving move rate: {frac_win:.3f}  (random baseline: {p_rand:.3f})")
    print(f"groups with nonzero reward variance: {nonzero_var}/{len(group_stats)}")


if __name__ == "__main__":
    main()
