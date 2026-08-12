"""Generate 11x11 positions for arm F: HexHex self-play (temperature + epsilon
noise) plus pure random games. Stores canonical (2,13,13) board tensors as uint8
(borders included), deduped. CNN acts are recomputed on the fly later.

Usage: /venv/main/bin/python armF/gen_positions.py --games 900 --random-games 300 \
           --out armF/data/positions.pt
"""
import argparse
import hashlib
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402

sys.path.insert(0, str(W.HEXHEX_ROOT))
from hexhex.logic.hexboard import Board  # noqa: E402
from hexhex.utils.utils import correct_position1d  # noqa: E402


def play_game(model, temperature, epsilon, random_only=False, collect_every=1):
    b = Board(11, switch_allowed=False)
    seen = []
    ply = 0
    while not b.winner and b.legal_moves:
        if ply % collect_every == 0:
            seen.append(b.board_tensor.clone().to(torch.uint8))
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
        ply += 1
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=900)
    ap.add_argument("--random-games", type=int, default=300)
    ap.add_argument("--out", default="armF/data/positions.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = W.load_model()
    dedup = {}

    def add(tensors):
        for t in tensors:
            h = hashlib.md5(t.numpy().tobytes()).hexdigest()
            if h not in dedup:
                dedup[h] = t

    def temp(ply):
        return 1.5 if ply < 6 else (1.0 if ply < 20 else 0.5)

    for g in range(args.games):
        add(play_game(model, temp, epsilon=0.05))
        if (g + 1) % 100 == 0:
            print(f"selfplay game {g+1}/{args.games}, unique positions {len(dedup)}", flush=True)
    for g in range(args.random_games):
        add(play_game(model, temp, epsilon=0.0, random_only=True, collect_every=2))
    print(f"total unique positions: {len(dedup)}")

    boards = torch.stack(list(dedup.values()))
    perm = torch.randperm(len(boards))
    boards = boards[perm]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"boards": boards, "note": "canonical uint8 (2,13,13), to-move perspective"}, out)
    print(f"wrote {out} shape {tuple(boards.shape)}")


if __name__ == "__main__":
    main()
