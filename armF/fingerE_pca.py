"""Arm F finger E cheapest falsifier: chained frozen per-layer PCA compression.

At every capture point z_0..z_18, project the 7744-dim activation map onto its
top-k PCA basis and reconstruct, letting the real conv layers run in between.
Measures whether a frozen rank-k global state suffices for play, and how
per-layer compression error COMPOUNDS through the trunk.

Variants: chained k in {512, 1024, 2048}; single-layer z9-only k=1024
(compounding control: z9 is the variance trough, var@1024 = .872).

Metrics per variant: chained per-layer z-hat R2 (per-dim normalized, vs true
acts), top1/top3/spearman agreement, paired-opening play vs pure CNN and vs
random. Same harness conventions as finger D.

Usage: /venv/main/bin/python armF/fingerE_pca.py [--smoke]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import fingerD_convmlp as FD  # noqa: E402

from hexhex.utils.utils import correct_position1d  # noqa: E402

DEV = "cuda"
D_Z = 7744
K_MAX = 2048
N_BASIS = 10000


# ---------------------------------------------------------------- basis
@torch.no_grad()
def build_basis(cnn, boards_u8):
    """Top-K_MAX PCA basis per layer via covariance eigh.
    Returns mean (19, 7744), V (19, K_MAX, 7744) descending eig order."""
    n = len(boards_u8)
    X = [torch.empty(n, D_Z) for _ in range(19)]
    for i in range(0, n, 250):
        bb = boards_u8[i:i + 250].float().to(DEV)
        acts = W.dump_acts(cnn, bb)
        for l, a in enumerate(acts):
            X[l][i:i + len(bb)] = a.reshape(len(bb), -1).cpu()
    mean = torch.empty(19, D_Z, device=DEV)
    V = torch.empty(19, K_MAX, D_Z, device=DEV)
    for l in range(19):
        x = X[l].to(DEV)
        mu = x.mean(0)
        x = x - mu
        C = (x.T @ x) / n
        ev, evec = torch.linalg.eigh(C)  # ascending
        mean[l] = mu
        V[l] = evec[:, -K_MAX:].flip(1).T.contiguous()
        vf1024 = ev[-1024:].sum() / ev.clamp_min(0).sum()
        print(f"z{l:2d}: basis var@1024 {vf1024:.4f}", flush=True)
    return mean, V


# ---------------------------------------------------------------- forward
@torch.no_grad()
def compressed_logits(cnn, mean, V, k, boards_f, layers, capture=False):
    """Forward with PCA project+reconstruct at capture points in `layers`.
    Returns illegal-masked logits (B,121); if capture, also list of 19 z-hats."""
    m = W.inner(cnn)
    x = boards_f.to(DEV)

    def proj(h, l):
        flat = h.reshape(len(h), -1) - mean[l]
        rec = (flat @ V[l, :k].T) @ V[l, :k] + mean[l]
        return rec.reshape(-1, 64, 11, 11)

    caps = []
    h = m.conv(x)
    if 0 in layers:
        h = proj(h, 0)
    if capture:
        caps.append(h.clone())
    for l, sl in enumerate(m.skiplayers):
        h = sl(h)
        if l + 1 in layers:
            h = proj(h, l + 1)
        if capture:
            caps.append(h.clone())
    logits = m.policyconv(h).view(-1, 121) + m.bias
    occ = x[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
    logits = logits - 1000.0 * occ
    return (logits, caps) if capture else logits


@torch.no_grad()
def chained_r2(cnn, mean, V, k, layers, boards_u8, batch=256):
    """Per-layer mean per-dim R2 of chained z-hats vs true acts on val."""
    se = torch.zeros(19, D_Z, device=DEV)
    ssum = torch.zeros(19, D_Z, device=DEV)
    ssq = torch.zeros(19, D_Z, device=DEV)
    n = len(boards_u8)
    for i in range(0, n, batch):
        bf = boards_u8[i:i + batch].float()
        true = W.dump_acts(cnn, bf.to(DEV))
        _, caps = compressed_logits(cnn, mean, V, k, bf, layers, capture=True)
        for l in range(19):
            t = true[l].reshape(len(bf), -1)
            c = caps[l].reshape(len(bf), -1)
            se[l] += (c - t).pow(2).sum(0)
            ssum[l] += t.sum(0)
            ssq[l] += t.pow(2).sum(0)
    var = (ssq / n - (ssum / n).pow(2)).clamp_min(1e-8)
    r2 = 1 - se / (var * n)
    return r2.mean(1).cpu().tolist()


@torch.no_grad()
def agreement(cnn, mean, V, k, layers, boards_u8, batch=256):
    top1 = top3 = tot = 0
    sps = []
    for i in range(0, len(boards_u8), batch):
        bf = boards_u8[i:i + batch].float()
        ref = FD.pure_logits(cnn, bf)
        st = compressed_logits(cnn, mean, V, k, bf, layers)
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


def make_pca_player(cnn, mean, V, k, layers, illegal_ctr):
    def fn(b, moves):
        lg = compressed_logits(cnn, mean, V, k,
                               b.board_tensor.unsqueeze(0).float(), layers)[0]
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
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--n-val", type=int, default=4096)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="armF/results/fingerE_pca.json")
    args = ap.parse_args()
    n_games_cnn = args.games * 2 if not args.smoke else 4
    n_games_rand = 20 if not args.smoke else 2

    torch.manual_seed(0)
    random.seed(0)
    cnn = W.load_model()
    boards = torch.load("armF/data/positions.pt", weights_only=False)["boards"]
    perm = torch.randperm(len(boards), generator=torch.Generator().manual_seed(0))
    boards = boards[perm]
    train_boards, val_boards = boards[:-args.n_val], boards[-args.n_val:]
    print(f"positions: {len(boards)} (basis from first {N_BASIS} of train)",
          flush=True)

    t0 = time.time()
    mean, V = build_basis(cnn, train_boards[:N_BASIS])
    print(f"basis built in {time.time()-t0:.0f}s", flush=True)

    # sanity anchor: k=K_MAX chained should be close to pure; k=D_Z would be exact
    bf = val_boards[:64].float()
    ref = FD.pure_logits(cnn, bf)
    st = compressed_logits(cnn, mean, V, K_MAX, bf, layers=set(range(19)))
    print(f"sanity: chained k={K_MAX} top1-match on 64 val boards = "
          f"{(st.argmax(1) == ref.argmax(1)).float().mean():.3f}", flush=True)

    all_layers = set(range(19))
    variants = [
        ("chained_k512", 512, all_layers),
        ("chained_k1024", 1024, all_layers),
        ("chained_k2048", 2048, all_layers),
        ("single_z9_k1024", 1024, {9}),
    ]
    results = {"n_basis": N_BASIS, "n_val": args.n_val}
    agr_boards = val_boards[:1024] if not args.smoke else val_boards[:128]
    r2_boards = val_boards[:2048] if not args.smoke else val_boards[:128]
    for name, k, layers in variants:
        print(f"\n=== {name} ===", flush=True)
        r2s = chained_r2(cnn, mean, V, k, layers, r2_boards)
        print("  z-hat R2 per layer: " +
              " ".join(f"{r:.3f}" for r in r2s), flush=True)
        agr = agreement(cnn, mean, V, k, layers, agr_boards)
        random.seed(0)
        ictr = [0]
        pl = make_pca_player(cnn, mean, V, k, layers, ictr)
        w_rand = FD.play_games(pl, FD.make_random_player(), n_games_rand)
        w_cnn = FD.play_games(pl, FD.make_cnn_player(cnn), n_games_cnn)
        results[name] = {
            "k": k, "layers": sorted(layers), "r2_per_layer": r2s,
            "r2_mean": sum(r2s) / len(r2s), "r2_z18": r2s[18],
            "agreement": agr,
            "vs_random": f"{w_rand}/{n_games_rand}",
            "vs_cnn_openings": f"{w_cnn}/{n_games_cnn}",
            "illegal_argmax": ictr[0]}
        print(f"{name}: R2 mean {results[name]['r2_mean']:.4f} z18 "
              f"{r2s[18]:.4f} | top1 {agr['top1']:.3f} top3 {agr['top3']:.3f} "
              f"sp {agr['spearman']:.3f} | vs random {w_rand}/{n_games_rand} "
              f"| vs CNN {w_cnn}/{n_games_cnn} | illegal {ictr[0]}", flush=True)

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
