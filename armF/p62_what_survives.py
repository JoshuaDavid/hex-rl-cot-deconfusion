"""P62: what does block 23 actually destroy? hs[23] vs hs[24] vs hs[28]
on the spliced polish model, at X-move tokens.

Probes: (a) occupancy, per-cell 3-class linear 2048->363 (canonical
frame: 0 empty / 1 own / 2 opp); (b) c18 containment R2, ridge
2048->1024 on normalized targets; (c) policy (from p61). Plus norm of
the block-23 write f(h) = hs24 - hs23 relative to ||hs23||.
"""
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_movesr4 as R4  # noqa: E402
import build_d04  # noqa: E402
import p59_policy_text as P59  # noqa: E402
from p61_depth_probe import gather  # noqa: E402

DEV = "cuda"


@torch.no_grad()
def c_targets(student, games, gis, mu, sd):
    """Normalized c18 targets at X plies (same convention as training)."""
    outs = []
    for gi in gis:
        bb = games[gi]["boards"][0::2].float().to(DEV)
        cs = R4.dump_c(student, bb)  # list of (B,1024)
        outs.append(((cs[18] - mu[18]) / sd[18]).cpu())
    return torch.cat(outs)


@torch.no_grad()
def occ_targets(games, gis):
    """Per-cell class in canonical frame: 0 empty, 1 ch0(own), 2 ch1."""
    outs = []
    for gi in gis:
        bb = games[gi]["boards"][0::2].float()
        own = bb[:, 0, 1:-1, 1:-1].reshape(-1, 121)
        opp = bb[:, 1, 1:-1, 1:-1].reshape(-1, 121)
        outs.append((own + 2 * opp).long())
    return torch.cat(outs)


def fit_occ(X, y, Xv, yv, steps=1500):
    X, y, Xv, yv = X.to(DEV), y.to(DEV), Xv.to(DEV), yv.to(DEV)
    probe = nn.Linear(2048, 363).to(DEV)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        idx = torch.randint(0, len(X), (2048,), device=DEV)
        lg = probe(X[idx]).reshape(-1, 121, 3)
        loss = nn.functional.cross_entropy(lg.reshape(-1, 3),
                                           y[idx].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        pred = probe(Xv).reshape(-1, 121, 3).argmax(-1)
        acc = (pred == yv).float().mean().item()
        occ_mask = yv > 0
        acc_occ = (pred[occ_mask] == yv[occ_mask]).float().mean().item()
    return acc, acc_occ


def ridge_r2(X, Y, Xv, Yv, lam=1e-2):
    X, Y, Xv, Yv = X.double(), Y.double(), Xv.double(), Yv.double()
    Xm, Ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - Xm, Y - Ym
    A = Xc.T @ Xc + lam * len(X) * torch.eye(X.shape[1], dtype=torch.double)
    Wt = torch.linalg.solve(A, Xc.T @ Yc)
    P = (Xv - Xm) @ Wt + Ym
    ss_res = ((Yv - P) ** 2).sum(0)
    ss_tot = ((Yv - Yv.mean(0)) ** 2).sum(0) + 1e-9
    return (1 - ss_res / ss_tot).mean().item()


def main():
    cnn = W.load_model()
    student = R4.load_student(cnn)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = P59.load_spliced_lm("checkpoints/armF_polish19b/final.pt")
    d = torch.load("armF/results/r4_cstats.pt", weights_only=False)
    mu, sd = d["mu"].to(DEV), d["sd"].to(DEV)

    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]
    rng = random.Random(7)
    val_gis = [gi for gi in range(len(games)) if gi % 15 == 0]
    tr_gis = rng.sample([gi for gi in range(len(games)) if gi % 15 != 0],
                        300)
    va_gis = rng.sample(val_gis, 40)

    ks = [23, 24, 28]
    Htr, _, _ = gather(model, tok, student, games, tr_gis, ks)
    Hva, _, _ = gather(model, tok, student, games, va_gis, ks)

    w = (Hva[24] - Hva[23]).norm(dim=-1) / Hva[23].norm(dim=-1)
    print(f"block-23 write ratio ||f(h)||/||h||: mean {w.mean():.3f} "
          f"p50 {w.median():.3f} p90 {w.quantile(0.9):.3f}", flush=True)

    ytr_o = occ_targets(games, tr_gis)
    yva_o = occ_targets(games, va_gis)
    ctr = c_targets(student, games, tr_gis, mu, sd)
    cva = c_targets(student, games, va_gis, mu, sd)

    res = {"write_ratio_mean": w.mean().item(),
           "write_ratio_p90": w.quantile(0.9).item()}
    for k in ks:
        acc, acc_occ = fit_occ(Htr[k], ytr_o, Hva[k], yva_o)
        r2 = ridge_r2(Htr[k], ctr, Hva[k], cva)
        res[k] = {"occ_acc": acc, "occ_acc_occupied": acc_occ,
                  "c18_r2": r2}
        print(f"hs[{k}]: occ acc {acc:.4f} (occupied-only {acc_occ:.4f}) "
              f"c18 ridge R2 {r2:.4f}", flush=True)

    Path("armF/results/p62_what_survives.json").write_text(
        json.dumps(res, indent=1))
    print("wrote armF/results/p62_what_survives.json")


if __name__ == "__main__":
    main()
