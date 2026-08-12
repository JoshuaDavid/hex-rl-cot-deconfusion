"""Arm F finger D2: can a Qwen-style transformer MLP block imitate one layer
transition of the DISTILLED rank-1024 CNN (finger E artifact)?

The distilled net's native state is c_l = enc_l(z_l) in R^1024. Target map:
  c2 -> c3 = enc3( skiplayer2( dec2(c2) ) )        (1024 -> 1024)
same depth as finger D, but the Qwen block (2048 hidden) now EXPANDS instead
of compressing, and D.U can be full rank-1024 — finger D's bottleneck
confound is gone by construction.

Variants (equal budget): identity (per-dim OLS, closed form) / linfull
(Linear 1024->1024) / mlp (Lin 1024->2048 -> QwenMLPBlock -> Lin 2048->1024)
/ lin_mlp (linfull + parallel mlp path).

Stitch eval: substitute emulator for the c2->c3 transition inside the
distilled CNN; agreement + paired-opening play vs the UNMODIFIED distilled net.

Usage: /venv/main/bin/python armF/fingerD2_distilled.py [--smoke]
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
import fingerD_convmlp as FD  # noqa: E402
import fingerE_bottleneck as B  # noqa: E402

from hexhex.utils.utils import correct_position1d  # noqa: E402

DEV = "cuda"
D_C = 1024
D_H = 2048
CKPT = "checkpoints/armF_fingerE/bottleneck_anchored_ext.pt"


def load_distilled():
    cnn = W.load_model()
    dummyV = [torch.zeros(B.K, B.D_Z)] * B.N_LAYERS
    dummym = [torch.zeros(B.D_Z)] * B.N_LAYERS
    student = B.BottleneckedCNN(cnn, dummyV, dummym).to(DEV)
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    student.load_state_dict(ck["state_dict"])
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    return student


# ---------------------------------------------------------------- transition
@torch.no_grad()
def c2_from_boards(student, x):
    m = student.inner
    h = student.bns[0](m.conv(x))
    h = student.bns[1](m.skiplayers[0](h))
    z2t = m.skiplayers[1](h)
    return student.bns[2].enc(z2t.reshape(len(x), -1))


@torch.no_grad()
def c3_from_c2(student, c2):
    m = student.inner
    h2 = student.bns[2].dec(c2).reshape(-1, 64, 11, 11)
    z3t = m.skiplayers[2](h2)
    return student.bns[3].enc(z3t.reshape(len(c2), -1))


@torch.no_grad()
def logits_from_c3(student, c3, occ):
    m = student.inner
    h = student.bns[3].dec(c3).reshape(-1, 64, 11, 11)
    for l in range(3, 18):
        h = student.bns[l + 1](m.skiplayers[l](h))
    return m.policyconv(h).view(-1, 121) + m.bias - 1000.0 * occ


@torch.no_grad()
def build_cache(student, boards_u8, batch=2048):
    ci, co = [], []
    for i in range(0, len(boards_u8), batch):
        x = boards_u8[i:i + batch].float().to(DEV)
        c2 = c2_from_boards(student, x)
        ci.append(c2.half())
        co.append(c3_from_c2(student, c2).half())
    return torch.cat(ci), torch.cat(co)


# ---------------------------------------------------------------- models
class LinBypassMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D_C, D_C)
        self.down = nn.Linear(D_C, D_H)
        self.block = FD.QwenMLPBlock()
        self.up = nn.Linear(D_H, D_C, bias=False)

    def forward(self, x):
        return self.lin(x) + self.up(self.block(self.down(x)))


def make_model(variant):
    if variant == "mlp":
        return nn.Sequential(nn.Linear(D_C, D_H), FD.QwenMLPBlock(),
                             nn.Linear(D_H, D_C))
    if variant == "linfull":
        return nn.Linear(D_C, D_C)
    if variant == "lin_mlp":
        return LinBypassMLP()
    raise ValueError(variant)


# ---------------------------------------------------------------- metrics
@torch.no_grad()
def val_r2(model, xv, yv, batch=8192):
    se = torch.zeros(yv.shape[1], device=DEV)
    for i in range(0, len(xv), batch):
        pred = model(xv[i:i + batch].float())
        se += (pred - yv[i:i + batch].float()).pow(2).sum(0)
    var = yv.float().var(0, unbiased=False) * len(yv)
    return (1 - se / var).mean().item()


@torch.no_grad()
def identity_baseline(xt, yt, xv, yv):
    xtf, ytf = xt.float(), yt.float()
    mx, my = xtf.mean(0), ytf.mean(0)
    a = ((xtf - mx) * (ytf - my)).mean(0) / (xtf - mx).pow(2).mean(0).clamp_min(1e-8)
    b = my - a * mx
    se = (xv.float() * a + b - yv.float()).pow(2).sum(0)
    var = yv.float().var(0, unbiased=False) * len(yv)
    return (1 - se / var).mean().item()


def train(model, xt, yt, xv, yv, steps, lr=1e-3, batch=1024, log_every=1000):
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
            print(f"  step {s+1}/{steps} loss {loss.item():.4f} "
                  f"val R2 {val_r2(model, xv, yv):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return val_r2(model, xv, yv)


# ---------------------------------------------------------------- stitch
class Emulator:
    def __init__(self, model, mu2, sd2, mu3, sd3):
        self.m, self.mu2, self.sd2, self.mu3, self.sd3 = model, mu2, sd2, mu3, sd3

    @torch.no_grad()
    def __call__(self, c2):
        x = (c2 - self.mu2) / self.sd2
        return self.m(x.float()) * self.sd3 + self.mu3


@torch.no_grad()
def emulated_logits(student, emu, boards_f):
    x = boards_f.to(DEV)
    c2 = c2_from_boards(student, x)
    occ = x[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
    return logits_from_c3(student, emu(c2), occ)


@torch.no_grad()
def agreement(student, emu, boards_u8, batch=256):
    top1 = top3 = tot = 0
    sps = []
    for i in range(0, len(boards_u8), batch):
        bf = boards_u8[i:i + batch].float()
        ref = student(bf.to(DEV))
        st = emulated_logits(student, emu, bf)
        occ = bf[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121).to(DEV)
        r1 = ref.argmax(1)
        top1 += (st.argmax(1) == r1).sum().item()
        top3 += (st.topk(3, dim=1).indices == r1[:, None]).any(1).sum().item()
        for j in range(len(bf)):
            legal = occ[j] < 0.5
            sps.append(FD.spearman(st[j][legal], ref[j][legal]))
        tot += len(bf)
    return {"top1": top1 / tot, "top3": top3 / tot,
            "spearman": sum(sps) / len(sps)}


def make_emu_player(student, emu, illegal_ctr):
    def fn(b, moves):
        lg = emulated_logits(student, emu, b.board_tensor.unsqueeze(0).float())[0]
        p1 = correct_position1d(lg.argmax().item(), 11, b.player)
        mv = divmod(p1, 11)
        if mv not in b.legal_moves:
            illegal_ctr[0] += 1
            mv = random.choice(sorted(b.legal_moves))
        return mv
    return fn


def make_distilled_player(student):
    @torch.no_grad()
    def fn(b, moves):
        lg = student(b.board_tensor.unsqueeze(0).float().to(DEV))[0]
        p1 = correct_position1d(lg.argmax().item(), 11, b.player)
        return divmod(p1, 11)
    return fn


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--variants", default="linfull,mlp,lin_mlp")
    ap.add_argument("--n-val", type=int, default=4096)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="armF/results/fingerD2.json")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.games = 200, 2

    torch.manual_seed(0)
    random.seed(0)
    student = load_distilled()
    boards = torch.cat([
        torch.load("armF/data/positions.pt", weights_only=False)["boards"],
        torch.load("armF/data/positions2.pt", weights_only=False)["boards"]])
    if args.smoke:
        boards = boards[:8000]
    perm = torch.randperm(len(boards), generator=torch.Generator().manual_seed(0))
    boards = boards[perm]
    print(f"positions: {len(boards)}", flush=True)

    c_in, c_out = build_cache(student, boards)
    print(f"cache: c2 {tuple(c_in.shape)} c3 {tuple(c_out.shape)} | "
          f"c2 mean {c_in.float().mean():.3f} std {c_in.float().std():.3f} | "
          f"c3 mean {c_out.float().mean():.3f} std {c_out.float().std():.3f}",
          flush=True)

    # sanity anchor: stitching the TRUE c3 must reproduce the distilled logits
    bf = boards[:64].float()
    x = bf.to(DEV)
    occ = x[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
    anchor = logits_from_c3(student, c3_from_c2(student, c2_from_boards(student, x)), occ)
    diff = (anchor - student(x)).abs().max().item()
    print(f"sanity anchor (true-c3 stitch vs distilled) max|diff| = {diff:.2e}",
          flush=True)
    assert diff < 1e-3

    nv = args.n_val if not args.smoke else 1000
    xt, yt = c_in[:-nv], c_out[:-nv]
    xv, yv = c_in[-nv:], c_out[-nv:]
    mu2, sd2 = xt.float().mean(0), xt.float().std(0).clamp_min(1e-4)
    mu3, sd3 = yt.float().mean(0), yt.float().std(0).clamp_min(1e-4)
    xt = ((xt.float() - mu2) / sd2).half()
    xv = ((xv.float() - mu2) / sd2).half()
    yt = ((yt.float() - mu3) / sd3).half()
    yv = ((yv.float() - mu3) / sd3).half()

    results = {"n_train": len(xt), "n_val": nv, "steps": args.steps,
               "ckpt": CKPT}
    results["identity"] = {"r2": identity_baseline(xt, yt, xv, yv)}
    print(f"identity baseline val R2 = {results['identity']['r2']:.4f}", flush=True)

    ckdir = Path("checkpoints/armF_fingerD2")
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
        agr = agreement(student, emu, boards[-nv:][:1024])
        random.seed(0)
        ictr = [0]
        pl = make_emu_player(student, emu, ictr)
        w_rand = FD.play_games(pl, FD.make_random_player(), max(2, args.games // 3))
        w_dist = FD.play_games(pl, make_distilled_player(student), args.games * 2)
        results[variant] = {
            "params_M": round(n_par / 1e6, 1), "r2": r2, "agreement": agr,
            "vs_random": f"{w_rand}/{max(2, args.games//3)}",
            "vs_distilled_openings": f"{w_dist}/{args.games*2}",
            "illegal_argmax": ictr[0]}
        print(f"{variant}: R2 {r2:.4f} top1 {agr['top1']:.3f} top3 "
              f"{agr['top3']:.3f} sp {agr['spearman']:.3f} | vs random "
              f"{w_rand}/{max(2, args.games//3)} | vs distilled "
              f"{w_dist}/{args.games*2} | illegal {ictr[0]}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
