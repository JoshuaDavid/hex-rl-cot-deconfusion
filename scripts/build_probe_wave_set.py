"""Build the live-probe position set with within-winning-set structure labels.

For each selected position (>=2 winning moves, >=1 losing, sizes 5/6 + deep 7):
- noninferior_winners: benzene dfpn-solver-find-winning output (ICE-surviving
  winners — used deliberately as the domination-order instrument).
- robustness[m]: for winning move m, mean over opponent replies o of
  (fraction of our legal replies after o that keep the win). 2-ply win density.
- bridge_delta[m]: bridges(mover) after m minus before.
- center_dist[m]: normalized distance of m from board center.

Already run once (2026-08-05) -> data/probe_wave_positions.jsonl (64 positions).
"""

import json
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, BLACK, WHITE, parse_cell
from scripts.collect_activations import bridges

_solver = None


def board_from_record(rec):
    b = Board(rec["size"])
    for c, cell in rec["moves"]:
        b.play(cell, BLACK if c == "B" else WHITE)
    b.to_move = BLACK if rec["to_move"] == "B" else WHITE
    return b


def label_one(rec):
    global _solver
    from hexenv.solver import HexSolver
    if _solver is None:
        _solver = HexSolver()
    b = board_from_record(rec)
    mover = b.to_move
    t0 = time.time()

    ni = _solver.winning_moves(b)  # find-winning = non-inferior winners
    wins = rec["winning_moves"]

    robustness = {}
    bridge_delta = {}
    center = {}
    n = rec["size"]
    cx = cy = (n - 1) / 2
    base_bridges = bridges(b, mover)
    for m in wins:
        child = b.copy()
        child.play(m)
        bridge_delta[m] = bridges(child, mover) - base_bridges
        x, y = parse_cell(m)
        center[m] = round(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 / n, 3)
        densities = []
        for o in child.legal_moves():
            gc = child.copy()
            gc.play(o)
            _, our_wins = _solver.exact_winning_moves(gc)
            densities.append(len(our_wins) / max(1, len(gc.legal_moves())))
        robustness[m] = round(sum(densities) / max(1, len(densities)), 4)

    rec2 = dict(rec)
    rec2["noninferior_winners"] = ni
    rec2["robustness"] = robustness
    rec2["bridge_delta"] = bridge_delta
    rec2["center_dist"] = center
    rec2["label_s"] = round(time.time() - t0, 1)
    return rec2


def main():
    recs = [json.loads(l) for l in open("data/verl_hex/val_positions.jsonl")]
    ok = [r for r in recs if 2 <= len(r["winning_moves"]) < r["n_legal"]
          and (r["size"] <= 6 or r["n_stones"] >= 10)]
    rng = random.Random(31)
    rng.shuffle(ok)
    sel = ok[:64]
    print(f"{len(sel)} positions selected")
    out = "data/probe_wave_positions.jsonl"
    t0 = time.time()
    with Pool(12) as pool, open(out, "w") as f:
        for i, rec in enumerate(pool.imap_unordered(label_one, sel)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{i+1}/{len(sel)} ({rec['size']}x{rec['size']}, "
                  f"{len(rec['winning_moves'])} winners, {rec['label_s']}s)",
                  flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
