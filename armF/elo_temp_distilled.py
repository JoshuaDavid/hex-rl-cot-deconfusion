"""Joint Elo-vs-temperature ladder: original HexHex + rank-1024 distilled.

16 entities: {orig, dist} x 8 temps, full round-robin, paired random single-
move openings both colors (same protocol as elo_temp.py). Both models get
identical inference: 180-degree rotation-averaged logits, illegal-masked,
softmax(logits/t) sampling (argmax at t=0). Joint Bradley-Terry MLE anchored
at orig t=1.0 -> 1500, so distilled Elos land on the same scale as the
published original ladder.

Usage: /venv/main/bin/python armF/elo_temp_distilled.py --openings 8
"""
import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import fingerE_bottleneck as B  # noqa: E402
from elo_temp import fit_elo  # noqa: E402

from hexhex.logic.hexboard import Board  # noqa: E402
from hexhex.utils.utils import correct_position1d  # noqa: E402

DEV = "cuda"
CKPT = "checkpoints/armF_fingerE/bottleneck_anchored_ext.pt"


def load_distilled(cnn):
    boards = torch.load("armF/data/positions.pt", weights_only=False)["boards"]
    perm = torch.randperm(len(boards), generator=torch.Generator().manual_seed(0))
    Vs, mus, _ = B.pca_basis(cnn, boards[perm][:B.N_BASIS])
    student = B.BottleneckedCNN(cnn, Vs, mus).to(DEV)
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    student.load_state_dict(ck["state_dict"])
    student.eval()
    return student


@torch.no_grad()
def dist_logits(student, x):
    """Rotation-averaged distilled logits, same scheme as RotationWrapper."""
    x_flip = torch.flip(x, [2, 3])
    y_flip = torch.flip(student(x_flip), [1])
    return (student(x) + y_flip) / 2


def make_logit_fn(kind, cnn, student):
    if kind == "orig":
        return lambda x: W.policy_logits(cnn, x)
    return lambda x: dist_logits(student, x)


def pick_move(logit_fn, b, temp):
    x = b.board_tensor.unsqueeze(0).float().to(DEV)
    lg = logit_fn(x)[0]
    if temp <= 0.01:
        p1 = lg.argmax().item()
    else:
        p1 = torch.multinomial(torch.softmax(lg / temp, dim=0), 1).item()
    p1 = correct_position1d(p1, 11, b.player)
    mv = divmod(p1, 11)
    if mv not in b.legal_moves:
        mv = random.choice(sorted(b.legal_moves))
    return mv


@torch.no_grad()
def play(fn_first, t_first, fn_second, t_second, opening):
    b = Board(11, switch_allowed=False)
    b.set_stone(opening)
    fns = {0: (fn_first, t_first), 1: (fn_second, t_second)}
    while not b.winner and b.legal_moves:
        fn, t = fns[b.player]
        b.set_stone(pick_move(fn, b, t))
    return b.winner[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--openings", type=int, default=8,
                    help="paired openings per pair (2 games each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="armF/results/elo_temp_distilled.json")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cnn = W.load_model()
    for p in cnn.parameters():
        p.requires_grad_(False)
    student = load_distilled(cnn)
    print("distilled model loaded", flush=True)

    ents = [("orig", t) for t in args.temps] + [("dist", t) for t in args.temps]
    fns = [make_logit_fn(k, cnn, student) for k, _ in ents]
    n = len(ents)
    wins = [[0] * n for _ in range(n)]

    pairs = list(itertools.combinations(range(n), 2))
    t0 = time.time()
    for pi, (i, j) in enumerate(pairs):
        for _ in range(args.openings):
            op = divmod(random.randrange(121), 11)
            for a, bidx in ((i, j), (j, i)):
                wnr = play(fns[a], ents[a][1], fns[bidx], ents[bidx][1], op)
                winner = a if wnr == 0 else bidx
                loser = bidx if wnr == 0 else a
                wins[winner][loser] += 1
        print(f"[{pi+1}/{len(pairs)}] {ents[i][0]} t={ents[i][1]:.2f} vs "
              f"{ents[j][0]} t={ents[j][1]:.2f}: {wins[i][j]}-{wins[j][i]} "
              f"({time.time()-t0:.0f}s)", flush=True)

    anchor_idx = ents.index(("orig", 1.0))
    elos = fit_elo(ents, wins, anchor_idx)
    print("\nkind  temp   Elo (anchor: orig t=1.0 -> 1500)")
    for (k, t), e in sorted(zip(ents, elos), key=lambda z: -z[1]):
        print(f"{k:5s} {t:5.2f}  {e:7.1f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"entities": ents, "elos": elos, "wins": wins,
                   "openings_per_pair": args.openings, "seed": args.seed,
                   "ckpt": CKPT}, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
