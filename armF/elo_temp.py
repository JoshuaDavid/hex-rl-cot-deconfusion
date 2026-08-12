"""HexHex Elo vs sampling temperature.

Round-robin between HexHex-11 at fixed temperatures. Paired random openings
(same opening, both color assignments) cancel first-move advantage. Bradley-
Terry MLE, anchored: temp=1.0 -> Elo 1500.

Usage: /venv/main/bin/python armF/elo_temp.py --openings 20
"""
import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402

sys.path.insert(0, str(W.HEXHEX_ROOT))
from hexhex.logic.hexboard import Board  # noqa: E402
from hexhex.utils.utils import correct_position1d  # noqa: E402


def pick_move(model, b, temp):
    x = b.board_tensor.unsqueeze(0).cuda()
    lg = W.policy_logits(model, x)[0]
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
def play(model, temp_first, temp_second, opening):
    """Returns 0 if first player wins, 1 if second. opening = (x,y) forced
    first move."""
    b = Board(11, switch_allowed=False)
    b.set_stone(opening)
    temps = {0: temp_first, 1: temp_second}
    while not b.winner and b.legal_moves:
        b.set_stone(pick_move(model, b, temps[b.player]))
    return b.winner[0]


def fit_elo(temps, wins, anchor_idx, anchor_elo=1500.0):
    n = len(temps)
    r = torch.zeros(n, requires_grad=True)
    w = torch.tensor(wins, dtype=torch.float64)
    # virtual half-draw per pair keeps undefeated players' Elo finite
    w = w + 0.25 * (1 - torch.eye(n, dtype=torch.float64))
    opt = torch.optim.Adam([r], lr=0.1)
    for _ in range(4000):
        opt.zero_grad()
        d = r.unsqueeze(1) - r.unsqueeze(0)  # d[i,j] = r_i - r_j
        loss = -(w * torch.nn.functional.logsigmoid(d)).sum()
        loss.backward()
        opt.step()
    scale = 400.0 / torch.log(torch.tensor(10.0))
    elos = (r.detach() - r.detach()[anchor_idx]) * scale + anchor_elo
    return elos.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ap.add_argument("--openings", type=int, default=20,
                    help="paired openings per pair (2 games each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="armF/results/elo_temp.json")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = W.load_model()
    temps = args.temps
    n = len(temps)
    wins = [[0] * n for _ in range(n)]   # wins[i][j] = games i beat j
    games = [[0] * n for _ in range(n)]

    pairs = list(itertools.combinations(range(n), 2))
    for i, j in pairs:
        for _ in range(args.openings):
            op = divmod(random.randrange(121), 11)
            for a, bidx in ((i, j), (j, i)):
                wnr = play(model, temps[a], temps[bidx], op)
                winner = a if wnr == 0 else bidx
                loser = bidx if wnr == 0 else a
                wins[winner][loser] += 1
                games[a][bidx] += 1
                games[bidx][a] += 1
        wij, wji = wins[i][j], wins[j][i]
        print(f"t={temps[i]:.2f} vs t={temps[j]:.2f}: {wij}-{wji}", flush=True)

    anchor_idx = temps.index(1.0)
    elos = fit_elo(temps, wins, anchor_idx)
    print("\ntemp   Elo (anchor: t=1.0 -> 1500)")
    for t, e in sorted(zip(temps, elos), key=lambda z: -z[1]):
        print(f"{t:5.2f}  {e:7.1f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"temps": temps, "elos": elos, "wins": wins,
                   "openings_per_pair": args.openings, "seed": args.seed}, f,
                  indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
