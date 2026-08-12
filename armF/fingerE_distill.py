"""Arm F finger E phase 2: distill HexHex into a rank-1024-state student.

Student: h0 = E_in(flat board 338) -> 18 blocks T_l(h) = Lin(h) + QwenMLP
delta (1024->3072->1024) -> policy. Two variants:
  anchored     19 decoders D_l(1024->7744) reconstruct true z_l (per-dim-norm
               MSE) + KL through the FROZEN CNN policy head on D_18(h_18).
  policy_only  trained Linear(1024->121) head, KL vs teacher only.

Teacher targets computed on the fly from the frozen CNN. Eval: decoded
per-layer R2 (anchored), top1/top3/spearman agreement, paired-opening play
vs pure CNN + vs random (finger D harness).

Usage: /venv/main/bin/python armF/fingerE_distill.py [--smoke] [--variants anchored,policy_only]
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

from hexhex.utils.utils import correct_position1d  # noqa: E402

DEV = "cuda"
D_Z = 7744
D_H = 1024
D_INTER = 3072
D_IN = 2 * 13 * 13  # 338
N_LAYERS = 19  # capture points; 18 transitions


def set_width(w):
    global D_H, D_INTER
    D_H, D_INTER = w, 3 * w


class MLPDelta(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = FD.RMSNorm(D_H)
        self.gate = nn.Linear(D_H, D_INTER, bias=False)
        self.up = nn.Linear(D_H, D_INTER, bias=False)
        self.down = nn.Linear(D_INTER, D_H, bias=False)

    def forward(self, h):
        n = self.norm(h)
        return self.down(F.silu(self.gate(n)) * self.up(n))


class Block(nn.Module):
    """Pre-norm residual: h + Lin(norm(h)) + MLPdelta(h). The free linear
    term lives in (I + Lin∘norm); unnormalized Lin-composition across 18
    blocks diverges (smoke run, 2026-08-12)."""

    def __init__(self):
        super().__init__()
        self.norm = FD.RMSNorm(D_H)
        self.lin = nn.Linear(D_H, D_H)
        self.mlp = MLPDelta()

    def forward(self, h):
        return h + self.lin(self.norm(h)) + self.mlp(h)


class Student(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.e_in = nn.Linear(D_IN, D_H)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS - 1))
        if variant == "anchored":
            self.dec = nn.ModuleList(nn.Linear(D_H, D_Z) for _ in range(N_LAYERS))
        else:
            self.head = nn.Linear(D_H, 121)

    def forward(self, xflat):
        h = self.e_in(xflat)
        hs = [h]
        for blk in self.blocks:
            h = blk(h)
            hs.append(h)
        return hs


# ---------------------------------------------------------------- teacher
@torch.no_grad()
def teacher_batch(cnn, boards_u8, idx, mu, sd, need_acts):
    bf = boards_u8[idx].float().to(DEV)
    logits = FD.pure_logits(cnn, bf)  # legal-masked
    zs = None
    if need_acts:
        acts = W.dump_acts(cnn, bf)
        zs = [((a.reshape(len(bf), -1) - mu[l]) / sd[l]) for l, a in enumerate(acts)]
    return bf.reshape(len(bf), -1), logits, zs


def student_logits(student, cnn, xflat, mu, sd):
    hs = student(xflat)
    if student.variant == "anchored":
        z18 = student.dec[18](hs[18]) * sd[18] + mu[18]
        m = W.inner(cnn)
        return m.policyconv(z18.reshape(-1, 64, 11, 11)).view(-1, 121) + m.bias
    return student.head(hs[18])


def masked(logits, xflat):
    occ = xflat.reshape(-1, 2, 13, 13)[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
    return logits - 1000.0 * occ


# ---------------------------------------------------------------- training
def train(student, cnn, boards_u8, val_idx0, mu, sd, steps, lr=1e-3,
          batch=512, log_every=500):
    n_all = len(boards_u8)

    @torch.no_grad()
    def quick_val_kl():
        idx = torch.arange(val_idx0, min(val_idx0 + 512, n_all))
        xflat, tlogits, _ = teacher_batch(cnn, boards_u8, idx, mu, sd, False)
        slog = masked(student_logits(student, cnn, xflat, mu, sd).float(), xflat)
        tp = F.softmax(tlogits, -1)
        return (tp * (F.log_softmax(tlogits, -1)
                      - F.log_softmax(slog, -1))).sum(-1).mean().item()

    student.to(DEV)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    warmup = 100
    anchored = student.variant == "anchored"

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        p = (s - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    n_train = val_idx0
    t0 = time.time()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        idx = torch.randint(0, n_train, (batch,))
        xflat, tlogits, zs = teacher_batch(cnn, boards_u8, idx, mu, sd, anchored)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = student(xflat)
            if anchored:
                act_loss = sum(
                    F.mse_loss(student.dec[l](hs[l]).float(), zs[l])
                    for l in range(N_LAYERS)) / N_LAYERS
                z18 = student.dec[18](hs[18]).float() * sd[18] + mu[18]
                m = W.inner(cnn)
                slog = m.policyconv(z18.reshape(-1, 64, 11, 11)).view(-1, 121) + m.bias
            else:
                act_loss = torch.zeros((), device=DEV)
                slog = student.head(hs[18])
        slog = masked(slog.float(), xflat)
        tp = F.softmax(tlogits, -1)
        kl = (tp * (F.log_softmax(tlogits, -1)
                    - F.log_softmax(slog, -1))).sum(-1).mean()
        loss = act_loss + kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if (s + 1) % log_every == 0 or s + 1 == steps:
            print(f"  step {s+1}/{steps} act {act_loss.item():.4f} "
                  f"kl {kl.item():.4f} val_kl {quick_val_kl():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- eval
@torch.no_grad()
def eval_val(student, cnn, boards_u8, mu, sd, batch=512):
    anchored = student.variant == "anchored"
    n = len(boards_u8)
    se = torch.zeros(N_LAYERS, D_Z, device=DEV) if anchored else None
    ssum = torch.zeros(N_LAYERS, D_Z, device=DEV) if anchored else None
    ssq = torch.zeros(N_LAYERS, D_Z, device=DEV) if anchored else None
    kls = []
    top1 = top3 = tot = 0
    sps = []
    for i in range(0, n, batch):
        idx = torch.arange(i, min(i + batch, n))
        xflat, tlogits, zs = teacher_batch(cnn, boards_u8, idx, mu, sd, anchored)
        hs = student(xflat)
        if anchored:
            for l in range(N_LAYERS):
                pred = student.dec[l](hs[l]).float()
                se[l] += (pred - zs[l]).pow(2).sum(0)
                ssum[l] += zs[l].sum(0)
                ssq[l] += zs[l].pow(2).sum(0)
        slog = masked(student_logits(student, cnn, xflat, mu, sd).float(), xflat)
        tp = F.softmax(tlogits, -1)
        kls.append((tp * (F.log_softmax(tlogits, -1)
                          - F.log_softmax(slog, -1))).sum(-1).mean().item())
        r1 = tlogits.argmax(1)
        top1 += (slog.argmax(1) == r1).sum().item()
        top3 += (slog.topk(3, dim=1).indices == r1[:, None]).any(1).sum().item()
        occ = xflat.reshape(-1, 2, 13, 13)[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
        for j in range(len(idx)):
            legal = occ[j] < 0.5
            sps.append(FD.spearman(slog[j][legal], tlogits[j][legal]))
        tot += len(idx)
    out = {"kl": sum(kls) / len(kls), "top1": top1 / tot, "top3": top3 / tot,
           "spearman": sum(sps) / len(sps)}
    if anchored:
        var = (ssq / n - (ssum / n).pow(2)).clamp_min(1e-8)
        r2 = (1 - se / (var * n)).mean(1).cpu().tolist()
        out["r2_per_layer"] = r2
        out["r2_mean"] = sum(r2) / len(r2)
    return out


def make_student_player(student, cnn, mu, sd, illegal_ctr):
    @torch.no_grad()
    def fn(b, moves):
        xflat = b.board_tensor.unsqueeze(0).float().to(DEV).reshape(1, -1)
        lg = masked(student_logits(student, cnn, xflat, mu, sd).float(), xflat)[0]
        p1 = correct_position1d(lg.argmax().item(), 11, b.player)
        mv = divmod(p1, 11)
        if mv not in b.legal_moves:
            illegal_ctr[0] += 1
            mv = random.choice(sorted(b.legal_moves))
        return mv
    return fn


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--variants", default="anchored,policy_only")
    ap.add_argument("--n-val", type=int, default=4096)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--out", default="armF/results/fingerE_distill.json")
    args = ap.parse_args()
    set_width(args.width)
    if args.smoke:
        args.steps, args.games = 200, 2

    torch.manual_seed(0)
    random.seed(0)
    cnn = W.load_model()
    boards = torch.load("armF/data/positions.pt", weights_only=False)["boards"]
    perm = torch.randperm(len(boards), generator=torch.Generator().manual_seed(0))
    boards = boards[perm]
    n_train = len(boards) - args.n_val
    print(f"positions: {len(boards)} train {n_train} val {args.n_val}", flush=True)

    # per-layer per-dim normalization stats from 10k train positions
    with torch.no_grad():
        ssum = torch.zeros(N_LAYERS, D_Z, device=DEV)
        ssq = torch.zeros(N_LAYERS, D_Z, device=DEV)
        for i in range(0, 10000, 500):
            acts = W.dump_acts(cnn, boards[i:i + 500].float().to(DEV))
            for l, a in enumerate(acts):
                flat = a.reshape(len(a), -1)
                ssum[l] += flat.sum(0)
                ssq[l] += flat.pow(2).sum(0)
        mu = ssum / 10000
        sd = (ssq / 10000 - mu.pow(2)).clamp_min(1e-8).sqrt().clamp_min(1e-4)

    results = {"n_train": n_train, "n_val": args.n_val, "steps": args.steps,
               "width": D_H}
    ckdir = Path("checkpoints/armF_fingerE")
    ckdir.mkdir(parents=True, exist_ok=True)
    val_boards = boards[n_train:]
    eval_boards = val_boards[:1024] if not args.smoke else val_boards[:128]
    for variant in args.variants.split(","):
        student = Student(variant)
        n_par = sum(p.numel() for p in student.parameters())
        print(f"\n=== {variant} ({n_par/1e6:.1f}M params) ===", flush=True)
        train(student, cnn, boards, n_train, mu, sd, args.steps)
        torch.save({"state_dict": student.state_dict(), "mu": mu.cpu(),
                    "sd": sd.cpu(), "variant": variant, "width": D_H},
                   ckdir / f"{variant}_w{D_H}.pt")
        student.eval()
        ev = eval_val(student, cnn, eval_boards, mu, sd)
        random.seed(0)
        ictr = [0]
        pl = make_student_player(student, cnn, mu, sd, ictr)
        w_rand = FD.play_games(pl, FD.make_random_player(), 20 if not args.smoke else 2)
        w_cnn = FD.play_games(pl, FD.make_cnn_player(cnn), args.games * 2)
        results[variant] = {"params_M": round(n_par / 1e6, 1), **ev,
                            "vs_random": f"{w_rand}/{20 if not args.smoke else 2}",
                            "vs_cnn_openings": f"{w_cnn}/{args.games*2}",
                            "illegal_argmax": ictr[0]}
        r2s = ev.get("r2_per_layer")
        if r2s:
            print("  decoded R2: " + " ".join(f"{r:.3f}" for r in r2s), flush=True)
        print(f"{variant}: kl {ev['kl']:.4f} top1 {ev['top1']:.3f} top3 "
              f"{ev['top3']:.3f} sp {ev['spearman']:.3f}"
              + (f" | R2 mean {ev['r2_mean']:.4f}" if r2s else "")
              + f" | vs random {w_rand} | vs CNN {w_cnn}/{args.games*2} "
              f"| illegal {ictr[0]}", flush=True)
        del student
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
