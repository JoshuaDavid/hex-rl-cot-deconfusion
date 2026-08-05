"""Generate solver-guided games for Phase 0 reading + latency benchmarking.

Both sides play a uniformly random *winning* move when winning, else a uniformly
random legal move. Logs per-move solver latency.
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, EMPTY, BLACK, WHITE
from hexenv.render import render_ascii
from hexenv.solver import HexSolver


def gen_game(solver, size, rng, opening=None):
    b = Board(size)
    if opening:
        b.play(opening)
    records = []
    while b.winner() == EMPTY:
        t0 = time.time()
        _, wins = solver.exact_winning_moves(b)
        dt = time.time() - t0
        mover = "B" if b.to_move == BLACK else "W"
        if wins:
            mv = rng.choice(wins)
            status = "winning"
        else:
            mv = rng.choice(b.legal_moves())
            status = "losing"
        records.append({
            "mover": mover, "status": status, "n_winning": len(wins),
            "n_legal": len(b.legal_moves()), "move": mv, "solver_s": round(dt, 3),
            "board_before": render_ascii(b),
        })
        b.play(mv)
    return records, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--n-games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    solver = HexSolver()
    out = args.out or f"results/phase0/games_{args.size}x{args.size}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    lat = []
    with open(out, "w") as f:
        for g in range(args.n_games):
            cells = Board(args.size).legal_moves()
            opening = rng.choice(cells)
            recs, final = gen_game(solver, args.size, rng, opening=opening)
            lat += [r["solver_s"] for r in recs]
            f.write(json.dumps({
                "game": g, "size": args.size, "opening": opening,
                "winner": {BLACK: "B", WHITE: "W"}[final.winner()],
                "n_moves": len(recs) + 1, "moves": recs,
                "final_board": render_ascii(final),
            }) + "\n")
            print(f"game {g}: opening {opening}, winner {'B' if final.winner()==BLACK else 'W'}, {len(recs)+1} moves, mean solver {sum(r['solver_s'] for r in recs)/len(recs):.3f}s max {max(r['solver_s'] for r in recs):.3f}s")

    lat.sort()
    print(f"\nsolver latency over {len(lat)} calls: mean {sum(lat)/len(lat):.3f}s "
          f"p50 {lat[len(lat)//2]:.3f}s p95 {lat[int(len(lat)*.95)]:.3f}s max {lat[-1]:.3f}s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
