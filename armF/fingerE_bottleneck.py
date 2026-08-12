"""Arm F finger E phase 3: rank-1024 bottlenecked CNN, jointly fine-tuned.

Insert trainable Linear(7744->1024)->Linear(1024->7744) bottlenecks at every
capture point z_0..z_18 of a trainable copy of the HexHex CNN; warm-start
E/D from the per-layer PCA basis (init == the chained-PCA player), fine-tune
bottlenecks + conv weights against the frozen teacher.

Variants:
  kl_only   loss = KL(teacher || student policy logits)
  anchored  + mean_l per-dim-norm MSE(decoded z-hat_l, teacher z_l)

Usage: /venv/main/bin/python armF/fingerE_bottleneck.py [--smoke]
"""
import argparse
import copy
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
K = 1024
N_LAYERS = 19
N_BASIS = 10000


class Bottleneck(nn.Module):
    def __init__(self, V, mu):
        """V: (K, 7744) PCA rows, mu: (7744,). Init to project+reconstruct."""
        super().__init__()
        self.enc = nn.Linear(D_Z, K)
        self.dec = nn.Linear(K, D_Z)
        with torch.no_grad():
            self.enc.weight.copy_(V)
            self.enc.bias.copy_(-V @ mu)
            self.dec.weight.copy_(V.T)
            self.dec.bias.copy_(mu)

    def forward(self, z):  # (B,64,11,11) -> same, through rank-K state
        flat = z.reshape(len(z), -1)
        return self.dec(self.enc(flat)).reshape(-1, 64, 11, 11)


class BottleneckedCNN(nn.Module):
    def __init__(self, cnn, Vs, mus):
        super().__init__()
        self.inner = copy.deepcopy(W.inner(cnn))
        self.bns = nn.ModuleList(Bottleneck(Vs[l], mus[l])
                                 for l in range(N_LAYERS))

    def forward(self, x, capture=False):
        m = self.inner
        caps = []
        h = self.bns[0](m.conv(x))
        if capture:
            caps.append(h)
        for l, sl in enumerate(m.skiplayers):
            h = self.bns[l + 1](sl(h))
            if capture:
                caps.append(h)
        logits = m.policyconv(h).view(-1, 121) + m.bias
        occ = x[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
        logits = logits - 1000.0 * occ
        return (logits, caps) if capture else logits


@torch.no_grad()
def pca_basis(cnn, boards_u8):
    n = len(boards_u8)
    X = [torch.empty(n, D_Z) for _ in range(N_LAYERS)]
    for i in range(0, n, 250):
        bb = boards_u8[i:i + 250].float().to(DEV)
        acts = W.dump_acts(cnn, bb)
        for l, a in enumerate(acts):
            X[l][i:i + len(bb)] = a.reshape(len(bb), -1).cpu()
    Vs, mus, sds = [], [], []
    for l in range(N_LAYERS):
        x = X[l].to(DEV)
        mu = x.mean(0)
        sds.append(x.std(0).clamp_min(1e-4))
        xc = x - mu
        C = (xc.T @ xc) / n
        _, evec = torch.linalg.eigh(C)
        Vs.append(evec[:, -K:].flip(1).T.contiguous())
        mus.append(mu)
    return Vs, mus, torch.stack(sds)


def train(student, cnn, boards_u8, n_train, sd, steps, variant, lr=1e-4,
          batch=256, log_every=500):
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    warmup = 100
    anchored = variant == "anchored"

    def lr_at(s):
        if s < warmup:
            return lr * (s + 1) / warmup
        p = (s - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * p))

    @torch.no_grad()
    def quick_val_kl():
        bf = boards_u8[n_train:n_train + 512].float().to(DEV)
        tlogits = FD.pure_logits(cnn, bf)
        slog = student(bf)
        tp = F.softmax(tlogits, -1)
        return (tp * (F.log_softmax(tlogits, -1)
                      - F.log_softmax(slog, -1))).sum(-1).mean().item()

    t0 = time.time()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        idx = torch.randint(0, n_train, (batch,))
        bf = boards_u8[idx].float().to(DEV)
        with torch.no_grad():
            tlogits = FD.pure_logits(cnn, bf)
            tacts = W.dump_acts(cnn, bf) if anchored else None
        slog, caps = student(bf, capture=True)
        tp = F.softmax(tlogits, -1)
        kl = (tp * (F.log_softmax(tlogits, -1)
                    - F.log_softmax(slog, -1))).sum(-1).mean()
        if anchored:
            act_loss = sum(
                ((caps[l].reshape(len(bf), -1) - tacts[l].reshape(len(bf), -1))
                 / sd[l]).pow(2).mean() for l in range(N_LAYERS)) / N_LAYERS
        else:
            act_loss = torch.zeros((), device=DEV)
        loss = kl + act_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        if (s + 1) % log_every == 0 or s + 1 == steps:
            print(f"  step {s+1}/{steps} kl {kl.item():.4f} act "
                  f"{act_loss.item():.4f} val_kl {quick_val_kl():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def eval_student(student, cnn, boards_u8, sd, batch=256):
    n = len(boards_u8)
    kls, sps = [], []
    top1 = top3 = tot = 0
    se = torch.zeros(N_LAYERS, device=DEV)
    denom = torch.zeros(N_LAYERS, device=DEV)
    for i in range(0, n, batch):
        bf = boards_u8[i:i + batch].float().to(DEV)
        tlogits = FD.pure_logits(cnn, bf)
        slog, caps = student(bf, capture=True)
        tacts = W.dump_acts(cnn, bf)
        for l in range(N_LAYERS):
            d = (caps[l].reshape(len(bf), -1) - tacts[l].reshape(len(bf), -1)) / sd[l]
            se[l] += d.pow(2).sum()
            denom[l] += ((tacts[l].reshape(len(bf), -1)
                          - tacts[l].reshape(len(bf), -1).mean(0)) / sd[l]).pow(2).sum()
        tp = F.softmax(tlogits, -1)
        kls.append((tp * (F.log_softmax(tlogits, -1)
                          - F.log_softmax(slog, -1))).sum(-1).mean().item())
        r1 = tlogits.argmax(1)
        top1 += (slog.argmax(1) == r1).sum().item()
        top3 += (slog.topk(3, dim=1).indices == r1[:, None]).any(1).sum().item()
        occ = bf[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
        for j in range(len(bf)):
            legal = occ[j] < 0.5
            sps.append(FD.spearman(slog[j][legal], tlogits[j][legal]))
        tot += len(bf)
    r2 = (1 - se / denom).cpu().tolist()
    return {"kl": sum(kls) / len(kls), "top1": top1 / tot, "top3": top3 / tot,
            "spearman": sum(sps) / len(sps), "r2_per_layer": r2,
            "r2_mean": sum(r2) / len(r2)}


def make_bn_player(student, illegal_ctr):
    @torch.no_grad()
    def fn(b, moves):
        lg = student(b.board_tensor.unsqueeze(0).float().to(DEV))[0]
        p1 = correct_position1d(lg.argmax().item(), 11, b.player)
        mv = divmod(p1, 11)
        if mv not in b.legal_moves:
            illegal_ctr[0] += 1
            mv = random.choice(sorted(b.legal_moves))
        return mv
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--variants", default="kl_only,anchored")
    ap.add_argument("--n-val", type=int, default=4096)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="armF/results/fingerE_bottleneck.json")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.games = 200, 2

    torch.manual_seed(0)
    random.seed(0)
    cnn = W.load_model()
    for p in cnn.parameters():
        p.requires_grad_(False)
    boards = torch.load("armF/data/positions.pt", weights_only=False)["boards"]
    perm = torch.randperm(len(boards), generator=torch.Generator().manual_seed(0))
    boards = boards[perm]
    n_train = len(boards) - args.n_val
    print(f"positions: {len(boards)} train {n_train} val {args.n_val}", flush=True)

    Vs, mus, sd = pca_basis(cnn, boards[:N_BASIS])
    print("PCA basis built", flush=True)

    results = {"n_train": n_train, "steps": args.steps, "k": K}
    ckdir = Path("checkpoints/armF_fingerE")
    ckdir.mkdir(parents=True, exist_ok=True)
    eval_boards = boards[n_train:n_train + (1024 if not args.smoke else 128)]
    for variant in args.variants.split(","):
        student = BottleneckedCNN(cnn, Vs, mus).to(DEV)
        n_par = sum(p.numel() for p in student.parameters())
        print(f"\n=== {variant} ({n_par/1e6:.1f}M params) ===", flush=True)
        ev0 = eval_student(student, cnn, eval_boards[:256], sd)
        print(f"  init (chained-PCA): kl {ev0['kl']:.4f} top1 {ev0['top1']:.3f} "
              f"R2 mean {ev0['r2_mean']:.4f}", flush=True)
        train(student, cnn, boards, n_train, sd, args.steps, variant)
        torch.save({"state_dict": student.state_dict(), "variant": variant},
                   ckdir / f"bottleneck_{variant}.pt")
        student.eval()
        ev = eval_student(student, cnn, eval_boards, sd)
        random.seed(0)
        ictr = [0]
        pl = make_bn_player(student, ictr)
        w_rand = FD.play_games(pl, FD.make_random_player(), 20 if not args.smoke else 2)
        w_cnn = FD.play_games(pl, FD.make_cnn_player(cnn), args.games * 2)
        results[variant] = {"params_M": round(n_par / 1e6, 1), "init": ev0,
                            **ev,
                            "vs_random": f"{w_rand}/{20 if not args.smoke else 2}",
                            "vs_cnn_openings": f"{w_cnn}/{args.games*2}",
                            "illegal_argmax": ictr[0]}
        print(f"{variant}: kl {ev['kl']:.4f} top1 {ev['top1']:.3f} top3 "
              f"{ev['top3']:.3f} sp {ev['spearman']:.3f} | R2 mean "
              f"{ev['r2_mean']:.4f} | vs random {w_rand} | vs CNN "
              f"{w_cnn}/{args.games*2} | illegal {ictr[0]}", flush=True)
        del student
        torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
