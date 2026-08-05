"""Generate mate-in-2 positions: winning for mover, NO immediate win, but there
exists a move M such that after M, every opponent reply leaves the mover with an
immediate win. Exact-labeled; schema = corpus + mate2_moves field.

Detection is board-logic-heavy but solver-light: candidate M must be in the
exact winning set (labeled first); the after-M-every-reply-leaves-mate-in-1
check is pure connectivity.
"""

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


def has_immediate_win(b, mover):
    for m in b.legal_moves():
        c = b.copy()
        c.play(m)
        if c.winner() == mover:
            return True
    return False


def immediate_wins(b, mover):
    out = []
    for m in b.legal_moves():
        c = b.copy()
        c.play(m)
        if c.winner() == mover:
            out.append(m)
    return out


def gen_one(seed):
    global _solver
    from hexenv.solver import HexSolver
    if _solver is None:
        _solver = HexSolver()
    rng = random.Random(2 * 10**7 + seed)
    for _ in range(400):
        size = rng.choice([5, 5, 6, 6, 7])
        ns = rng.randrange(size * size // 4, size * size - 6)
        try:
            b = random_position(size, ns, rng)
        except RuntimeError:
            continue
        mover = b.to_move
        if has_immediate_win(b, mover):
            continue
        winner, wins = _solver.exact_winning_moves(b)
        if winner != mover or not (0 < len(wins) < len(b.legal_moves())):
            continue
        mate2 = []
        for m in wins:
            c = b.copy()
            c.play(m)  # our winning move; now opponent to move
            ok = True
            for o in c.legal_moves():
                d = c.copy()
                d.play(o)
                if not has_immediate_win(d, mover):
                    ok = False
                    break
            if ok:
                mate2.append(m)
        if not mate2:
            continue
        return {
            "size": size, "n_stones": ns,
            "moves": [["B" if c_ == BLACK else "W", cell] for c_, cell in b.moves],
            "to_move": "B" if mover == BLACK else "W",
            "winner": "B" if winner == BLACK else "W",
            "winning_moves": wins, "mate2_moves": mate2,
            "n_legal": len(b.legal_moves()),
        }
    return None


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    out = "data/corpus_mate2.jsonl"
    seen, n = set(), 0
    t0 = time.time()
    with Pool(16) as pool, open(out, "w") as f:
        for rec in pool.imap_unordered(gen_one, range(target * 4)):
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
