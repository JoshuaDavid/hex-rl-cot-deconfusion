"""Shared 5x5 board pool for arm E (R1/R2/R3/R4 all draw from the same boards
so the instrumental differential is measured on a consistent, disjoint test
split). Each board -> gt dict {size, moves(all stones), winner}. gen_board
gives a definite winner (good: task B non-trivial) with varied fill.

Output: data/arme/pool_{train,test}.jsonl
Run: /venv/main/bin/python scripts/build_arme_pool.py
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, BLACK, WHITE, cell_name
from hexenv.arme import gold_payload

# Sparse-large regime (env-tunable): big grid + few stones so C (empties) is
# hard to enumerate alone but mechanical given A (all stones). See RESEARCH_LOG
# 2026-08-10 (C saturated at 5x5 -> escalate).
N = int(os.environ.get("ARME_N", "7"))
N_TRAIN = int(os.environ.get("ARME_NTRAIN", "5000"))
N_TEST = int(os.environ.get("ARME_NTEST", "600"))
PF = (0.55, 0.9)       # p_forward high -> short direct winning chain
EXTRA = (0.0, 0.15)    # few extra winner stones
LOSER = (0.05, 0.25)   # few loser stones


def make_board(rng):
    w = rng.choice(["Black", "White"])
    from scripts.witness_constructive import gen_board
    out = gen_board(rng, N, w, p_forward=rng.uniform(*PF),
                    extra_winner_frac=rng.uniform(*EXTRA),
                    loser_adjacent_frac=rng.uniform(*LOSER))
    if out is None:
        return None
    wst, lst, _ = out
    b = Board(N)
    wc = BLACK if w == "Black" else WHITE
    lc = WHITE if w == "Black" else BLACK
    for x, y in wst:
        b.grid[y][x] = wc
    for x, y in lst:
        b.grid[y][x] = lc
    return b, w


def board_key(b):
    return tuple(tuple(row) for row in b.grid)


def gt_of(b, winner):
    moves = []
    for y in range(N):
        for x in range(N):
            v = b.grid[y][x]
            if v == BLACK:
                moves.append(["B", cell_name(x, y)])
            elif v == WHITE:
                moves.append(["W", cell_name(x, y)])
    return {"size": N, "moves": moves, "winner": winner}


def main():
    os.makedirs("data/arme", exist_ok=True)
    rng = random.Random(20260810)
    seen = set()
    rows = []
    tries = 0
    target = N_TRAIN + N_TEST
    while len(rows) < target and tries < target * 50:
        tries += 1
        r = make_board(rng)
        if r is None:
            continue
        b, w = r
        k = board_key(b)
        if k in seen:
            continue
        seen.add(k)
        rows.append(gt_of(b, w))
    rng.shuffle(rows)
    test, train = rows[:N_TEST], rows[N_TEST:]
    for name, rs in [("train", train), ("test", test)]:
        with open(f"data/arme/pool_{name}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
    # quick stats: mean empties, winner balance
    import statistics
    from hexenv.arme import board_from_gt
    emp = [len(gold_payload("C", board_from_gt(r))) for r in test]
    wb = statistics.mean(1 for r in test if r["winner"] == "Black") if False else \
        sum(r["winner"] == "Black" for r in test) / len(test)
    print(f"pool: train {len(train)} test {len(test)} (tries {tries}); "
          f"test mean empties {statistics.mean(emp):.1f}; "
          f"test winner=Black frac {wb:.2f}")


if __name__ == "__main__":
    main()
