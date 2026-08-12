"""Frozen-Qwen affine probe baseline: ridge from Qwen hidden (layer 5+l, cell
tokens of copy 2) -> HexHex z_l, per l = 0..18. Closed form via accumulated
sufficient statistics. Control probe: affine from flattened board (338 dims).

Targets are per-layer-per-channel normalized (train stats). Reports val R^2.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import qwen_embed as Q  # noqa: E402

DEV = "cuda"


def cnn_targets(model, boards_f, mu=None, sd=None):
    """boards_f: (B,2,13,13) float cuda. Returns (19, B, 121, 64) normalized."""
    acts = W.dump_acts(model, boards_f)
    z = torch.stack([a.permute(0, 2, 3, 1).reshape(-1, 121, 64) for a in acts])
    if mu is not None:
        z = (z - mu[:, None, None, :]) / sd[:, None, None, :]
    return z


def patch_features(boards_f, r):
    """Shared-weight local patch per cell, matching conv geometry.
    boards_f: (B,2,13,13) cuda. Returns (B*121, 2*(2r+1)^2 + 1)."""
    import torch.nn.functional as F
    u = F.unfold(boards_f, kernel_size=2 * r + 1, padding=r - 1)  # (B, 2k^2, 121)
    x = u.permute(0, 2, 1).reshape(-1, u.shape[1])
    return torch.cat([x, torch.ones(len(x), 1, device=x.device)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default="armF/data/positions.pt")
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="armF/results/probe_frozen.json")
    ap.add_argument("--random-init", action="store_true")
    args = ap.parse_args()

    boards = torch.load(args.positions, weights_only=False)["boards"].float()
    assert len(boards) >= args.n_train + args.n_val, len(boards)
    train = boards[: args.n_train]
    val = boards[args.n_train: args.n_train + args.n_val]
    print(f"train {len(train)} val {len(val)}")

    cnn = W.load_model()
    tok, qwen = Q.load_qwen(random_init=args.random_init)
    H = qwen.config.hidden_size
    L = 19

    # pass 0: normalization stats from train CNN acts
    mus, sds = [], []
    with torch.no_grad():
        zs = []
        for i in range(0, min(len(train), 2000), 256):
            zs.append(cnn_targets(cnn, train[i:i+256].to(DEV)))
        zc = torch.cat(zs, dim=1)  # (19, n, 121, 64)
        mu = zc.mean(dim=(1, 2))
        sd = zc.std(dim=(1, 2)).clamp_min(1e-4)
    del zs, zc
    print("norm stats done")

    D = H + 1          # qwen features + bias
    RADII = [1, 3]
    DP = {r: 2 * (2 * r + 1) ** 2 + 1 for r in RADII}
    stats = {}
    for split, data in [("train", train), ("val", val)]:
        XtX = torch.zeros(L, D, D, device=DEV)
        XtY = torch.zeros(L, D, 64, device=DEV)
        PtP = {r: torch.zeros(DP[r], DP[r], device=DEV) for r in RADII}
        PtY = {r: torch.zeros(L, DP[r], 64, device=DEV) for r in RADII}
        Ysum = torch.zeros(L, 64, device=DEV)
        Ysq = torch.zeros(L, 64, device=DEV)
        n = 0
        t0 = time.time()
        for i in range(0, len(data), args.batch):
            bb = data[i:i+args.batch]
            z = cnn_targets(cnn, bb.to(DEV), mu, sd)  # (19,B,121,64)
            batch = Q.batch_prompts(tok, bb)
            hs = Q.hidden_at_cells(qwen, batch)  # list of (B,121,H)
            xp = {r: patch_features(bb.to(DEV), r) for r in RADII}
            for r in RADII:
                PtP[r] += xp[r].T @ xp[r]
            for l in range(L):
                x = hs[l].reshape(-1, H).float()
                x = torch.cat([x, torch.ones(len(x), 1, device=DEV)], 1)
                y = z[l].reshape(-1, 64)
                XtX[l] += x.T @ x
                XtY[l] += x.T @ y
                for r in RADII:
                    PtY[r][l] += xp[r].T @ y
                Ysum[l] += y.sum(0)
                Ysq[l] += (y * y).sum(0)
            n += len(bb) * 121
            if i // args.batch % 20 == 0:
                print(f"{split} {i}/{len(data)} ({time.time()-t0:.0f}s)", flush=True)
        stats[split] = dict(XtX=XtX, XtY=XtY, PtP=PtP, PtY=PtY, Ysum=Ysum, Ysq=Ysq, n=n)

    def r2_from(A, s, xtx, xty):
        # SSE = sum_c [ y'y - 2 a'x'y + a'x'x a ]
        sse = (s["Ysq"].sum(-1)
               - 2 * (A * xty).sum(dim=(-2, -1))
               + (A * (xtx @ A)).sum(dim=(-2, -1)))
        mean = s["Ysum"] / s["n"]
        sstot = (s["Ysq"] - s["n"] * mean * mean).sum(-1)
        return (1 - sse / sstot)

    results = {"lambda": {}, "qwen_r2": None}
    best = None
    for lam in [0.1, 1.0, 10.0, 100.0]:
        eyeD = torch.eye(D, device=DEV) * lam
        A = torch.linalg.solve(stats["train"]["XtX"] + eyeD, stats["train"]["XtY"])
        r2v = r2_from(A, stats["val"], stats["val"]["XtX"], stats["val"]["XtY"])
        results["lambda"][str(lam)] = [round(v, 4) for v in r2v.tolist()]
        print(f"lam {lam}: mean val R2 {r2v.mean().item():.4f}")
        if best is None or r2v.mean() > best[0]:
            best = (r2v.mean().item(), lam, A, r2v)
    results["qwen_r2"] = [round(v, 4) for v in best[3].tolist()]
    results["best_lambda"] = best[1]

    bs = {k: stats["val"][k] for k in ["Ysum", "Ysq"]}
    bs["n"] = stats["val"]["n"]
    for r in [1, 3]:
        dp = stats["train"]["PtP"][r].shape[0]
        eyeP = torch.eye(dp, device=DEV) * 0.1
        AP = torch.linalg.solve(stats["train"]["PtP"][r].unsqueeze(0) + eyeP,
                                stats["train"]["PtY"][r])
        r2p = r2_from(AP, bs, stats["val"]["PtP"][r].unsqueeze(0),
                      stats["val"]["PtY"][r])
        results[f"patch{r}_r2"] = [round(v, 4) for v in r2p.tolist()]

    print("\nlayer | qwen R2 | patch1 R2 | patch3 R2")
    for l in range(L):
        print(f"  z{l:2d} |  {results['qwen_r2'][l]:6.3f} | {results['patch1_r2'][l]:6.3f}"
              f" | {results['patch3_r2'][l]:6.3f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    torch.save({"A": best[2].cpu(), "mu": mu.cpu(), "sd": sd.cpu(),
                "lambda": best[1]}, out.with_suffix(".pt"))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
