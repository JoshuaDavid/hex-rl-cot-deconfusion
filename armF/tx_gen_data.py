"""Finger T phase 1: batched self-play position generation + teacher labels.

Plays many games in parallel (one batched CNN forward per ply-step) with a
mixed recipe (fingerE was data-bound at 225k; this targets ~2M unique):
  temp    45%  temp schedule 1.5/1.0/0.5 by ply, eps 0.05   (r1-style coverage)
  sharp   25%  temp 0.3 constant, eps 0.03                  (near-expert manifold)
  hybrid  20%  random first k in 2..8 plies, then temp 0.2  (P58 off-manifold lever)
  random  10%  fully random
Positions = canonical uint8 (2,13,13) boards BEFORE each move, deduped by hash.
Phase 2 precomputes rotation-averaged illegal-masked teacher logits (f16) for
every position so training never calls the CNN.

Usage: /venv/main/bin/python armF/tx_gen_data.py --target 2000000
"""
import argparse
import hashlib
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402

sys.path.insert(0, str(W.HEXHEX_ROOT))
from hexhex.logic.hexboard import Board  # noqa: E402
from hexhex.utils.utils import correct_position1d  # noqa: E402

DEV = "cuda"


@torch.no_grad()
def masked_teacher_logits(model, x):
    """Rotation-averaged AND explicitly illegal-masked. NB: W.policy_logits'
    docstring claims masking but RotationWrapperModel does none — the CNN
    puts high logits on occupied cells (verified 2026-08-21)."""
    lg = W.policy_logits(model, x)
    occ = x[:, :, 1:-1, 1:-1].sum(1).reshape(len(x), 121)
    return lg - 1000.0 * occ


def temp_schedule(ply):
    return 1.5 if ply < 6 else (1.0 if ply < 20 else 0.5)


def new_game(rng, mix="full"):
    if mix == "temp":
        return {"b": Board(11, switch_allowed=False), "ply": 0,
                "spec": {"kind": "temp", "eps": 0.05}}
    r = rng.random()
    if r < 0.45:
        spec = {"kind": "temp", "eps": 0.05}
    elif r < 0.70:
        spec = {"kind": "sharp", "eps": 0.03, "t": 0.3}
    elif r < 0.90:
        spec = {"kind": "hybrid", "eps": 0.0, "t": 0.2,
                "k_open": rng.randrange(2, 9)}
    else:
        spec = {"kind": "random"}
    return {"b": Board(11, switch_allowed=False), "ply": 0, "spec": spec}


def wants_random(g, rng):
    s = g["spec"]
    if s["kind"] == "random":
        return True
    if s["kind"] == "hybrid" and g["ply"] < s["k_open"]:
        return True
    return rng.random() < s.get("eps", 0.0)


def game_temp(g):
    s = g["spec"]
    if s["kind"] == "temp":
        return temp_schedule(g["ply"])
    return s["t"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000000)
    ap.add_argument("--par", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="armF/data/tx_positions.pt")
    ap.add_argument("--logits-out", default="armF/data/tx_teacher.pt")
    ap.add_argument("--mix", default="full", choices=["full", "temp"])
    args = ap.parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    model = W.load_model()
    seen = set()
    out_bytes = []
    active = [new_game(rng, args.mix) for _ in range(args.par)]
    t0 = time.time()
    n_moves = 0
    counts = {"temp": 0, "sharp": 0, "hybrid": 0, "random": 0}
    last_report = 0

    while len(out_bytes) < args.target:
        # record positions (board BEFORE the move about to be played)
        for g in active:
            bt = g["b"].board_tensor.to(torch.uint8)
            raw = bt.numpy().tobytes()
            h = hashlib.blake2b(raw, digest_size=12).digest()
            if h not in seen:
                seen.add(h)
                out_bytes.append(raw)
        # choose moves
        need_model = []
        for i, g in enumerate(active):
            g["_rand"] = wants_random(g, rng)
            if not g["_rand"]:
                need_model.append(i)
        if need_model:
            X = torch.stack([active[i]["b"].board_tensor for i in need_model]
                            ).float().to(DEV)
            lg = masked_teacher_logits(model, X)
            temps = torch.tensor([game_temp(active[i]) for i in need_model],
                                 device=DEV).unsqueeze(1)
            probs = torch.softmax(lg / temps.clamp_min(0.05), dim=1)
            samp = torch.multinomial(probs, 1).squeeze(1).cpu()
            amax = lg.argmax(1).cpu()
        for j, i in enumerate(need_model):
            g = active[i]
            p1 = (amax[j] if game_temp(g) <= 0.01 else samp[j]).item()
            p1 = correct_position1d(p1, 11, g["b"].player)
            mv = divmod(p1, 11)
            if mv not in g["b"].legal_moves:
                mv = rng.choice(sorted(g["b"].legal_moves))
            g["_mv"] = mv
        for g in active:
            if g["_rand"]:
                g["_mv"] = rng.choice(sorted(g["b"].legal_moves))
            g["b"].set_stone(g["_mv"])
            g["ply"] += 1
            n_moves += 1
        nxt = []
        for g in active:
            if g["b"].winner or not g["b"].legal_moves:
                counts[g["spec"]["kind"]] += 1
                nxt.append(new_game(rng, args.mix))
            else:
                nxt.append(g)
        active = nxt
        if len(out_bytes) - last_report >= 100000:
            last_report = len(out_bytes)
            el = time.time() - t0
            print(f"{len(out_bytes)} unique / {n_moves} moves, "
                  f"{el:.0f}s ({len(out_bytes)/el:.0f} pos/s) games {counts}",
                  flush=True)

    boards = torch.stack([
        torch.frombuffer(raw, dtype=torch.uint8).reshape(2, 13, 13)
        for raw in out_bytes[:args.target]])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"boards": boards, "counts": counts, "seed": args.seed,
                "note": "canonical uint8 boards BEFORE each move, deduped"},
               args.out)
    print(f"wrote {args.out}: {boards.shape} ({time.time()-t0:.0f}s)", flush=True)

    # phase 2: teacher logits (rotation-averaged, illegal-masked), f16
    n = len(boards)
    tl = torch.empty(n, 121, dtype=torch.float16)
    with torch.no_grad():
        for i in range(0, n, 2048):
            x = boards[i:i + 2048].float().to(DEV)
            tl[i:i + 2048] = masked_teacher_logits(model, x).half().cpu()
            if i % 204800 == 0:
                print(f"teacher {i}/{n} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"logits": tl}, args.logits_out)
    print(f"wrote {args.logits_out}: {tl.shape} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
