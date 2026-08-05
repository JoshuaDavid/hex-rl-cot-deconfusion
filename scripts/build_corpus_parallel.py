"""Parallel solver-labeled corpus builder (one mohex per worker process)."""

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, BLACK, WHITE
from hexenv.positions import random_position

_solver = None


def label_one(task):
    global _solver
    if _solver is None:
        from hexenv.solver import HexSolver
        _solver = HexSolver()
    size, n_stones, seed = task
    rng = random.Random(seed)
    b = random_position(size, n_stones, rng)
    t0 = time.time()
    try:
        winner, wins = _solver.exact_winning_moves(b)
    except Exception:
        from hexenv.solver import HexSolver
        _solver = HexSolver()
        winner, wins = _solver.exact_winning_moves(b)
    dt = time.time() - t0
    return {
        "size": size,
        "n_stones": n_stones,
        "moves": [["B" if c == BLACK else "W", cell] for c, cell in b.moves],
        "to_move": "B" if b.to_move == BLACK else "W",
        "winner": "B" if winner == BLACK else "W",
        "winning_moves": wins,
        "n_legal": len(b.legal_moves()),
        "solver_s": round(dt, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--n", type=int, default=150, help="positions per bucket")
    ap.add_argument("--stones", type=str, required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    buckets = [int(s) for s in args.stones.split(",")]
    tasks = []
    i = 0
    for ns in buckets:
        for _ in range(args.n):
            tasks.append((args.size, ns, args.seed * 1000003 + i))
            i += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    seen, n_written = set(), 0
    t_start = time.time()
    with Pool(args.workers) as pool, open(args.out, "w") as f:
        for rec in pool.imap_unordered(label_one, tasks, chunksize=1):
            key = (tuple(sorted(map(tuple, rec["moves"]))), rec["to_move"])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(rec) + "\n")
            n_written += 1
            if n_written % 50 == 0:
                f.flush()
                print(f"{n_written}/{len(tasks)} labeled, {time.time()-t_start:.0f}s", flush=True)
    print(f"wrote {n_written} to {args.out} in {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
