"""Finger T final eval: transformer vs distilled CNN vs original CNN.

Batched game harness (all games advance one ply per model call). Protocols:
  head2head: paired single-move openings (default all 121, both colors) at
             t=0 vs distilled and vs orig.
  ladder:    joint BT Elo, {orig, dist, tx} x temps, paired random 1-move
             openings, anchored orig t=1.0 -> 1500 (comparable to the
             published elo_temp_distilled ladder).
ALL logit paths here are rotation-averaged AND explicitly illegal-masked
(hexhex_wrap.policy_logits is NOT masked despite its docstring; the legacy
ladder relied on it raw).

Usage: /venv/main/bin/python armF/tx_eval.py --ckpt checkpoints/armF_txT18/best.pt
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
from tx_train import TxPolicy, boards_to_states  # noqa: E402
from elo_temp import fit_elo  # noqa: E402

sys.path.insert(0, str(W.HEXHEX_ROOT))
from hexhex.logic.hexboard import Board  # noqa: E402
from hexhex.utils.utils import correct_position1d  # noqa: E402

DEV = "cuda"


def load_tx(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    model = TxPolicy(**ck["cfg"]).to(DEV)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("step"), ck.get("val_kl")


def mask(lg, x):
    return lg - 1000.0 * x[:, :, 1:-1, 1:-1].sum(1).reshape(len(x), 121)


def make_fns(cnn, dist, tx):
    @torch.no_grad()
    def f_orig(x):
        return mask(W.policy_logits(cnn, x), x)

    @torch.no_grad()
    def f_dist(x):
        from elo_temp_distilled import dist_logits
        return mask(dist_logits(dist, x), x)

    @torch.no_grad()
    def f_tx(x):
        u8 = x.to(torch.uint8)
        lg = tx(boards_to_states(u8))
        lg = (lg + torch.flip(tx(boards_to_states(torch.flip(u8, [2, 3]))), [1])) / 2
        return mask(lg, x)

    return {"orig": f_orig, "dist": f_dist, "tx": f_tx}


@torch.no_grad()
def play_batched(fa, ta, fb, tb, openings, seed=0):
    """For each opening play 2 games (a first / b first). Returns a-wins list.
    All games advance in lockstep; each side's pending boards batched."""
    rng = random.Random(seed)
    games = []
    for op in openings:
        for a_is in (0, 1):
            b = Board(11, switch_allowed=False)
            b.set_stone(op)
            games.append({"b": b, "a_is": a_is, "done": False, "awin": None})
    while any(not g["done"] for g in games):
        for side in (0, 1):
            idx = [i for i, g in enumerate(games)
                   if not g["done"] and g["b"].player == (g["a_is"] ^ (0 if side == 0 else 1))]
            # side 0 -> player a, side 1 -> player b
            if not idx:
                continue
            fn, t = (fa, ta) if side == 0 else (fb, tb)
            X = torch.stack([games[i]["b"].board_tensor for i in idx]).float().to(DEV)
            lg = fn(X)
            if t <= 0.01:
                picks = lg.argmax(1).cpu()
            else:
                picks = torch.multinomial(torch.softmax(lg / t, 1), 1).squeeze(1).cpu()
            for j, i in enumerate(idx):
                g = games[i]
                p1 = correct_position1d(picks[j].item(), 11, g["b"].player)
                mv = divmod(p1, 11)
                if mv not in g["b"].legal_moves:
                    mv = rng.choice(sorted(g["b"].legal_moves))
                g["b"].set_stone(mv)
                if g["b"].winner or not g["b"].legal_moves:
                    g["done"] = True
                    g["awin"] = g["b"].winner == [g["a_is"]]
    return [g["awin"] for g in games]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/armF_txT18/best.pt")
    ap.add_argument("--ladder-temps", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--ladder-openings", type=int, default=10)
    ap.add_argument("--skip-ladder", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="armF/results/tx_eval.json")
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cnn = W.load_model()
    for p in cnn.parameters():
        p.requires_grad_(False)
    from elo_temp_distilled import load_distilled
    dist = load_distilled(cnn)
    tx, step, vkl = load_tx(args.ckpt)
    print(f"tx ckpt step {step} val_kl {vkl}", flush=True)
    fns = make_fns(cnn, dist, tx)
    res = {"ckpt": args.ckpt, "step": step, "val_kl": vkl}

    # head-to-head, all 121 single-move openings, both colors, t=0
    all_ops = [divmod(i, 11) for i in range(121)]
    t0 = time.time()
    for opp in ("dist", "orig"):
        wins = play_batched(fns["tx"], 0.0, fns[opp], 0.0, all_ops, seed=1)
        w = sum(wins)
        res[f"h2h_vs_{opp}"] = {"wins": w, "games": len(wins),
                                "frac": round(w / len(wins), 4)}
        print(f"tx vs {opp} t=0: {w}/{len(wins)} ({time.time()-t0:.0f}s)",
              flush=True)

    if not args.skip_ladder:
        ents = [(k, t) for k in ("orig", "dist", "tx") for t in args.ladder_temps]
        n = len(ents)
        wins = [[0] * n for _ in range(n)]
        rng = random.Random(args.seed)
        for i, j in itertools.combinations(range(n), 2):
            ops = [divmod(rng.randrange(121), 11)
                   for _ in range(args.ladder_openings)]
            aw = play_batched(fns[ents[i][0]], ents[i][1],
                              fns[ents[j][0]], ents[j][1], ops,
                              seed=100 + i * n + j)
            wins[i][j] += sum(aw)
            wins[j][i] += len(aw) - sum(aw)
            print(f"{ents[i]} vs {ents[j]}: {sum(aw)}-{len(aw)-sum(aw)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        anchor = ents.index(("orig", 1.0))
        elos = fit_elo(ents, wins, anchor)
        res["ladder"] = {"entities": ents, "elos": elos, "wins": wins,
                         "openings_per_pair": args.ladder_openings}
        print("\nkind  temp   Elo (anchor: orig t=1.0 -> 1500)")
        for (k, t), e in sorted(zip(ents, elos), key=lambda z: -z[1]):
            print(f"{k:5s} {t:5.2f}  {e:7.1f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
