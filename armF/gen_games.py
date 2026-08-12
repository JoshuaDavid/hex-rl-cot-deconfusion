"""Generate 11x11 games WITH move sequences for arm F r2 (moves format).

Per game stores: moves (T,2 absolute coords), per-ply canonical board tensors
(board AFTER each move, uint8), per-ply canonical cell index of the just-played
cell (in the frame of the board after the move: transpose iff next-to-move==1).

Usage: /venv/main/bin/python armF/gen_games.py --games 1800 --random-games 400
"""
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402

sys.path.insert(0, str(W.HEXHEX_ROOT))
from hexhex.logic.hexboard import Board  # noqa: E402
from hexhex.utils.utils import correct_position1d  # noqa: E402


def play_game(model, temperature, epsilon, random_only=False):
    b = Board(11, switch_allowed=False)
    moves, boards, cells = [], [], []
    ply = 0
    while not b.winner and b.legal_moves:
        if random_only or random.random() < epsilon:
            mv = random.choice(sorted(b.legal_moves))
        else:
            x = b.board_tensor.unsqueeze(0).cuda()
            lg = W.policy_logits(model, x)[0]
            t = temperature(ply)
            if t <= 0.01:
                p1 = lg.argmax().item()
            else:
                p1 = torch.multinomial(torch.softmax(lg / t, dim=0), 1).item()
            p1 = correct_position1d(p1, 11, b.player)
            mv = divmod(p1, 11)
            if mv not in b.legal_moves:
                mv = random.choice(sorted(b.legal_moves))
        b.set_stone(mv)
        moves.append(mv)
        boards.append(b.board_tensor.clone().to(torch.uint8))
        q = b.player  # to-move AFTER the stone was placed
        cx, cy = (mv[1], mv[0]) if q == 1 else mv
        cells.append(cx * 11 + cy)
        ply += 1
    return moves, boards, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1800)
    ap.add_argument("--random-games", type=int, default=400)
    ap.add_argument("--out", default="armF/data/games.pt")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = W.load_model()

    def temp(ply):
        return 1.5 if ply < 6 else (1.0 if ply < 20 else 0.5)

    games = []
    for g in range(args.games + args.random_games):
        rand_only = g >= args.games
        moves, boards, cells = play_game(model, temp, epsilon=0.05,
                                         random_only=rand_only)
        games.append({"moves": torch.tensor(moves, dtype=torch.int8),
                      "boards": torch.stack(boards),
                      "cells": torch.tensor(cells, dtype=torch.int16)})
        if (g + 1) % 200 == 0:
            print(f"game {g+1}/{args.games + args.random_games} "
                  f"(len {len(moves)})", flush=True)
    lens = torch.tensor([len(g["moves"]) for g in games])
    print(f"{len(games)} games, plies min/med/max "
          f"{lens.min()}/{lens.median()}/{lens.max()}, total {lens.sum()}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"games": games,
                "note": "boards = canonical uint8 AFTER each move; cells = "
                        "canonical idx of played cell in that frame"}, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
