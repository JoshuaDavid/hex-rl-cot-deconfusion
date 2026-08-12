"""Arm F finger D: can ONE Qwen-style transformer MLP block emulate ONE conv
skip layer of the HexHex CNN?

Map: z2 (B,7744) -> D Linear(7744->2048) -> [Qwen3 MLP block: RMSNorm +
SwiGLU(2048->6144->2048) + residual] -> U Linear(2048->7744) -> z3.
No attention (single token; self-attention would be a no-op).
All params trained from random init, MSE in per-dim normalized space.

Controls (equal step budget):
  identity  per-dim scalar OLS z3 ~ a*z2+b (closed form, no training)
  lin2048   D,U only (rank-2048 linear bottleneck, no MLP)
  linfull   unconstrained Linear(7744->7744) (linear ceiling)
  mlp       the full thing

Functional eval: substitute emulator for skiplayers[2] in the CNN,
agreement (top1/top3/spearman) + paired-opening play vs pure CNN.

Usage: /venv/main/bin/python armF/fingerD_convmlp.py [--smoke] [--steps 8000]
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402

sys.path.insert(0, str(W.HEXHEX_ROOT))
from hexhex.logic.hexboard import Board  # noqa: E402
from hexhex.utils.utils import correct_position1d  # noqa: E402

DEV = "cuda"
D_Z = 64 * 11 * 11  # 7744
D_H = 2048
D_INTER = 6144
IN_IDX, OUT_IDX = 2, 3  # z2 -> z3; emulator replaces skiplayers[2]


# ---------------------------------------------------------------- data
@torch.no_grad()
def build_cache(cnn, boards_u8, batch=2048):
    """Returns z_in, z_out as (N, 7744) fp16 on DEV."""
    zi, zo = [], []
    m = W.inner(cnn)
    for i in range(0, len(boards_u8), batch):
        x = boards_u8[i:i + batch].float().to(DEV)
        h = m.conv(x)
        for sl in m.skiplayers[:IN_IDX - 1]:
            h = sl(h)
        z_in = m.skiplayers[IN_IDX - 1](h) if IN_IDX > 0 else h
        z_out = m.skiplayers[OUT_IDX - 1](z_in)
        zi.append(z_in.reshape(len(x), -1).half())
        zo.append(z_out.reshape(len(x), -1).half())
    return torch.cat(zi), torch.cat(zo)


# ---------------------------------------------------------------- models
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
        return (v * self.weight.float()).to(x.dtype)


class QwenMLPBlock(nn.Module):
    def __init__(self, d=D_H, inter=D_INTER):
        super().__init__()
        self.norm = RMSNorm(d)
        self.gate = nn.Linear(d, inter, bias=False)
        self.up = nn.Linear(d, inter, bias=False)
        self.down = nn.Linear(inter, d, bias=False)

    def forward(self, h):
        n = self.norm(h)
        return h + self.down(F.silu(self.gate(n)) * self.up(n))


def make_model(variant):
    if variant == "mlp":
        return nn.Sequential(nn.Linear(D_Z, D_H), QwenMLPBlock(),
                             nn.Linear(D_H, D_Z))
    if variant == "lin2048":
        return nn.Sequential(nn.Linear(D_Z, D_H), nn.Linear(D_H, D_Z))
    if variant == "linfull":
        return nn.Linear(D_Z, D_Z)
    raise ValueError(variant)


# ---------------------------------------------------------------- metrics
@torch.no_grad()
def val_r2(model, xv, yv, batch=4096):
    """Mean per-dim R2 on val, normalized space."""
    se = torch.zeros(D_Z, device=DEV)
    for i in range(0, len(xv), batch):
        pred = model(xv[i:i + batch].float())
        se += (pred - yv[i:i + batch].float()).pow(2).sum(0)
    var = yv.float().var(0, unbiased=False) * len(yv)
    return (1 - se / var).mean().item()


@torch.no_grad()
def identity_baseline(xt, yt, xv, yv):
    """Per-dim OLS y ~ a*x + b fit on train, R2 on val."""
    xtf, ytf = xt.float(), yt.float()
    mx, my = xtf.mean(0), ytf.mean(0)
    a = ((xtf - mx) * (ytf - my)).mean(0) / (xtf - mx).pow(2).mean(0).clamp_min(1e-8)
    b = my - a * mx
    pred = xv.float() * a + b
    se = (pred - yv.float()).pow(2).sum(0)
    var = yv.float().var(0, unbiased=False) * len(yv)
    return (1 - se / var).mean().item()


# ---------------------------------------------------------------- training
def train(model, xt, yt, xv, yv, steps, lr=1e-3, batch=1024, log_every=500):
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    warmup = 100

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        p = (s - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    n = len(xt)
    t0 = time.time()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        idx = torch.randint(0, n, (batch,), device=DEV)
        x, y = xt[idx].float(), yt[idx].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x)
        loss = F.mse_loss(pred.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (s + 1) % log_every == 0 or s + 1 == steps:
            r2 = val_r2(model, xv, yv)
            print(f"  step {s+1}/{steps} loss {loss.item():.4f} "
                  f"val R2 {r2:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return val_r2(model, xv, yv)


# ---------------------------------------------------------------- stitch
class Emulator:
    """Wraps a trained normalized-space model into a z2->z3-hat map."""

    def __init__(self, model, mu2, sd2, mu3, sd3):
        self.m, self.mu2, self.sd2, self.mu3, self.sd3 = model, mu2, sd2, mu3, sd3

    @torch.no_grad()
    def __call__(self, z2):  # (B,64,11,11) -> (B,64,11,11)
        x = (z2.reshape(len(z2), -1) - self.mu2) / self.sd2
        y = self.m(x.float()) * self.sd3 + self.mu3
        return y.reshape(-1, 64, 11, 11)


@torch.no_grad()
def emulated_logits(cnn, emu, boards_f):
    m = W.inner(cnn)
    x = boards_f.to(DEV)
    h = m.conv(x)
    for sl in m.skiplayers[:IN_IDX]:
        h = sl(h)
    zhat = emu(h)
    logits = W.stitched_logits(cnn, zhat, OUT_IDX)
    occ = x[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
    return logits - 1000.0 * occ


@torch.no_grad()
def pure_logits(cnn, boards_f):
    x = boards_f.to(DEV)
    logits = W.stitched_logits(cnn, W.dump_acts(cnn, x)[0], 0)
    occ = x[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
    return logits - 1000.0 * occ


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return (ra * rb).mean().item()


@torch.no_grad()
def agreement(cnn, emu, boards_u8, batch=256):
    top1 = top3 = tot = 0
    sps = []
    for i in range(0, len(boards_u8), batch):
        bf = boards_u8[i:i + batch].float()
        ref = pure_logits(cnn, bf)
        st = emulated_logits(cnn, emu, bf)
        occ = bf[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121).to(DEV)
        r1 = ref.argmax(1)
        top1 += (st.argmax(1) == r1).sum().item()
        top3 += (st.topk(3, dim=1).indices == r1[:, None]).any(1).sum().item()
        for j in range(len(bf)):
            legal = occ[j] < 0.5
            sps.append(spearman(st[j][legal], ref[j][legal]))
        tot += len(bf)
    return {"top1": top1 / tot, "top3": top3 / tot,
            "spearman": sum(sps) / len(sps)}


def make_emu_player(cnn, emu, illegal_ctr):
    def fn(b, moves):
        lg = emulated_logits(cnn, emu, b.board_tensor.unsqueeze(0).float())[0]
        p1 = correct_position1d(lg.argmax().item(), 11, b.player)
        mv = divmod(p1, 11)
        if mv not in b.legal_moves:
            illegal_ctr[0] += 1
            mv = random.choice(sorted(b.legal_moves))
        return mv
    return fn


def make_cnn_player(cnn):
    def fn(b, moves):
        lg = pure_logits(cnn, b.board_tensor.unsqueeze(0).float())[0]
        p1 = correct_position1d(lg.argmax().item(), 11, b.player)
        return divmod(p1, 11)
    return fn


def make_random_player():
    def fn(b, moves):
        return random.choice(sorted(b.legal_moves))
    return fn


def play_games(move_fn_a, move_fn_b, n_games, opening_plies=4):
    """Paired: same 4-ply random opening for games 2g, 2g+1 with sides swapped."""
    wins = 0
    for g in range(n_games):
        if g % 2 == 0:
            rng = random.Random(1000 + g)
            opening = None
            while opening is None:
                b = Board(11, switch_allowed=False)
                mvs = []
                for _ in range(opening_plies):
                    mv = rng.choice(sorted(b.legal_moves))
                    mvs.append(mv)
                    b.set_stone(mv)
                if not b.winner:
                    opening = mvs
        b = Board(11, switch_allowed=False)
        moves = []
        for mv in opening:
            b.set_stone(mv)
            moves.append(mv)
        a_is = g % 2
        while not b.winner:
            fn = move_fn_a if b.player == a_is else move_fn_b
            mv = fn(b, moves)
            b.set_stone(mv)
            moves.append(mv)
        if b.winner == [a_is]:
            wins += 1
    return wins


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--variants", default="linfull,lin2048,mlp")
    ap.add_argument("--n-val", type=int, default=4096)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="armF/results/fingerD.json")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.games = 200, 2

    torch.manual_seed(0)
    random.seed(0)
    cnn = W.load_model()
    boards = torch.load("armF/data/positions.pt", weights_only=False)["boards"]
    if args.smoke:
        boards = boards[:8000]
    perm = torch.randperm(len(boards), generator=torch.Generator().manual_seed(0))
    boards = boards[perm]
    print(f"positions: {len(boards)}", flush=True)

    z_in, z_out = build_cache(cnn, boards)
    print(f"cache: z{IN_IDX} {tuple(z_in.shape)} z{OUT_IDX} {tuple(z_out.shape)} "
          f"| z{IN_IDX} mean {z_in.float().mean():.3f} std {z_in.float().std():.3f} "
          f"| z{OUT_IDX} mean {z_out.float().mean():.3f} std {z_out.float().std():.3f}",
          flush=True)

    # sanity anchor: stitching TRUE z3 back must reproduce pure logits
    bf = boards[:64].float()
    zt = W.dump_acts(cnn, bf.to(DEV))[OUT_IDX]
    anchor = (W.stitched_logits(cnn, zt, OUT_IDX)
              - 1000.0 * bf.to(DEV)[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121))
    diff = (anchor - pure_logits(cnn, bf)).abs().max().item()
    print(f"sanity anchor (true-z{OUT_IDX} stitch vs pure) max|diff| = {diff:.2e}",
          flush=True)
    assert diff < 1e-3

    nv = args.n_val if not args.smoke else 1000
    xt, yt = z_in[:-nv], z_out[:-nv]
    xv, yv = z_in[-nv:], z_out[-nv:]
    mu2, sd2 = xt.float().mean(0), xt.float().std(0).clamp_min(1e-4)
    mu3, sd3 = yt.float().mean(0), yt.float().std(0).clamp_min(1e-4)
    xt = ((xt.float() - mu2) / sd2).half()
    xv = ((xv.float() - mu2) / sd2).half()
    yt = ((yt.float() - mu3) / sd3).half()
    yv = ((yv.float() - mu3) / sd3).half()

    results = {"n_train": len(xt), "n_val": nv, "steps": args.steps}
    results["identity"] = {"r2": identity_baseline(xt, yt, xv, yv)}
    print(f"identity baseline val R2 = {results['identity']['r2']:.4f}", flush=True)

    ckdir = Path("checkpoints/armF_fingerD")
    ckdir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants.split(","):
        model = make_model(variant)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"\n=== {variant} ({n_par/1e6:.1f}M params) ===", flush=True)
        r2 = train(model, xt, yt, xv, yv, args.steps)
        torch.save({"state_dict": model.state_dict(),
                    "mu2": mu2, "sd2": sd2, "mu3": mu3, "sd3": sd3},
                   ckdir / f"{variant}.pt")
        model.eval()
        emu = Emulator(model, mu2.to(DEV), sd2.to(DEV), mu3.to(DEV), sd3.to(DEV))
        val_boards = boards[-nv:][:1024]
        agr = agreement(cnn, emu, val_boards)
        random.seed(0)
        ictr = [0]
        pl = make_emu_player(cnn, emu, ictr)
        w_rand = play_games(pl, make_random_player(), max(2, args.games // 3))
        w_cnn = play_games(pl, make_cnn_player(cnn), args.games * 2)
        results[variant] = {
            "params_M": round(n_par / 1e6, 1), "r2": r2, "agreement": agr,
            "vs_random": f"{w_rand}/{max(2, args.games//3)}",
            "vs_cnn_openings": f"{w_cnn}/{args.games*2}",
            "illegal_argmax": ictr[0]}
        print(f"{variant}: R2 {r2:.4f} top1 {agr['top1']:.3f} top3 "
              f"{agr['top3']:.3f} sp {agr['spearman']:.3f} | vs random "
              f"{w_rand}/{max(2, args.games//3)} | vs CNN {w_cnn}/{args.games*2} "
              f"| illegal {ictr[0]}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
