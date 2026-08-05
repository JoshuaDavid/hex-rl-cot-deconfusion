"""Generate mate-in-1 positions: random deep positions filtered to those where
the mover has an immediate connection-completing move; exact-label the winning
set with the solver. Output schema = corpus + immediate_wins field."""

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


def gen_one(seed):
    global _solver
    from hexenv.solver import HexSolver
    if _solver is None:
        _solver = HexSolver()
    rng = random.Random(seed)
    for _ in range(400):
        size = rng.choice([5, 5, 6, 6, 7])
        lo = size * size // 3
        hi = size * size - 4
        ns = rng.randrange(lo, hi)
        try:
            b = random_position(size, ns, rng)
        except RuntimeError:
            continue
        mover = b.to_move
        iw = []
        for m in b.legal_moves():
            c = b.copy()
            c.play(m)
            if c.winner() == mover:
                iw.append(m)
        if not iw:
            continue
        winner, wins = _solver.exact_winning_moves(b)
        if winner != mover or not (0 < len(wins) < len(b.legal_moves())):
            continue
        return {
            "size": size, "n_stones": ns,
            "moves": [["B" if c == BLACK else "W", cell] for c, cell in b.moves],
            "to_move": "B" if mover == BLACK else "W",
            "winner": "B" if winner == BLACK else "W",
            "winning_moves": wins, "immediate_wins": iw,
            "n_legal": len(b.legal_moves()),
        }
    return None


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    out = "data/corpus_mate1_gen.jsonl"
    seen = set()
    n = 0
    t0 = time.time()
    with Pool(16) as pool, open(out, "w") as f:
        for rec in pool.imap_unordered(gen_one, range(target * 2)):
            if rec is None:
                continue
            key = (rec["size"], tuple(map(tuple, rec["moves"])), rec["to_move"])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(rec) + "\n")
            n += 1
            if n % 100 == 0:
                print(f"{n}/{target} ({time.time()-t0:.0f}s)", flush=True)
            if n >= target:
                break
    print(f"wrote {n} to {out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
