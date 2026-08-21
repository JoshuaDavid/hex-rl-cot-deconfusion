"""P75: baselines for tx-in-Qwen containment (guest = txT12 transformer).

1. Rank check of the guest residual stream: 13 capture points h_0..h_12
   (h_0 = emb+pos+registers, h_l = block-l output), each (B, 128, 1024).
   Per-token covariance spectrum (adapter-relevant) + full-map gram spectrum
   (position-coupled structure, comparable to armF/rank_check.py).
2. Frozen-probe ridge: Qwen hs[5+l] at copy-2 cell tokens + 7 appended
   register-slot tokens -> guest h_l (per-dim guest-normalized).
   Arms: pretrained Qwen, random-init Qwen (reservoir control), and a
   flatboard+position affine control (no-computation floor).

All prompts render to identical-length text (cells are 1 char), so cell/reg
token indices are computed once and reused.

Usage: /venv/main/bin/python armF/p75_baselines.py [--n-train 6000 --n-val 1000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import render11 as R  # noqa: E402
import qwen_embed as Q  # noqa: E402
from tx_train import TxPolicy, boards_to_states, N_CELL, N_REG  # noqa: E402

DEV = "cuda"
N_PTS = 13  # h_0 .. h_12
QWEN_LAYERS = list(range(5, 5 + N_PTS))  # hs[5..17]
REG_SUFFIX = "Register slots: a b c d e f g\n"


def load_guest(path="checkpoints/armF_txT12/best.pt"):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    m = TxPolicy(**ck["cfg"]).to(DEV)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def guest_acts(model, boards_u8):
    """(B,2,13,13) uint8 cuda -> (13, B, 128, 1024) fp32."""
    st = boards_to_states(boards_u8)
    x = model.emb(st) + model.pos.weight.unsqueeze(0)
    x = torch.cat([x, model.reg.unsqueeze(0).expand(len(x), -1, -1)], 1)
    hs = [x.float()]
    for blk in model.blocks:
        x = blk(x, model.rel)
        hs.append(x.float())
    return torch.stack(hs)


def render_with_regs(board):
    text, off1, off2 = R.render_two_copy(board)
    base = len(text)
    text = text + REG_SUFFIX
    # offsets of the 7 letters a..g in the suffix (search after the colon —
    # 'e'/'g' also occur in the word "Register")
    colon = REG_SUFFIX.index(":")
    reg_off = [base + REG_SUFFIX.index(ch, colon) for ch in "abcdefg"]
    for o in reg_off:
        assert text[o] in "abcdefg"
    return text, off2 + reg_off


def batch_prompts(tok, boards_u8):
    """Per-board token indices — BPE merges depend on cell symbols, so
    tokenization is board-dependent (same reason qwen_embed recomputes)."""
    texts, all_offs = [], []
    for b in boards_u8.cpu():
        text, offs = render_with_regs(b)
        texts.append(text)
        all_offs.append(offs)
    enc = tok(texts, return_offsets_mapping=True, padding=True,
              return_tensors="pt", add_special_tokens=False)
    B = len(texts)
    idxs = torch.zeros(B, N_CELL + N_REG, dtype=torch.long)
    for i in range(B):
        starts = {}
        for tj, (a, bnd) in enumerate(enc["offset_mapping"][i].tolist()):
            if a == bnd:
                continue
            for o in range(a, bnd):
                starts[o] = tj
        row = [starts[o] for o in all_offs[i]]
        assert len(set(row)) == len(row), "token collision"
        idxs[i] = torch.tensor(row)
    return enc["input_ids"], enc["attention_mask"], idxs


@torch.no_grad()
def qwen_feats(qwen, tok, boards_u8):
    """(B,2,13,13) uint8 -> (13, B, 128, H) fp32 at aligned tokens."""
    ids, am, idxs = batch_prompts(tok, boards_u8)
    out = qwen(input_ids=ids.to(DEV), attention_mask=am.to(DEV),
               output_hidden_states=True)
    idx = idxs.to(DEV)
    res = []
    for j in QWEN_LAYERS:
        h = out.hidden_states[j]
        g = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        res.append(g.float())
    return torch.stack(res)


def control_feats(boards_u8):
    """flat board (338) + position one-hot (128) + 1 -> (B,128,467)."""
    B = len(boards_u8)
    flat = boards_u8.reshape(B, -1).float()  # 338
    f = flat.unsqueeze(1).expand(B, N_CELL + N_REG, 338)
    eye = torch.eye(N_CELL + N_REG, device=boards_u8.device
                    ).unsqueeze(0).expand(B, -1, -1)
    ones = torch.ones(B, N_CELL + N_REG, 1, device=boards_u8.device)
    return torch.cat([f, eye, ones], -1)


def run_probe(qwen, tok, guest, boards, mu, sd, n_train, batch=32):
    D_G = 1024
    H = qwen.config.hidden_size if qwen is not None else None
    D = (H + 1) if qwen is not None else 467
    stats = {}
    for split, lo, hi in [("train", 0, n_train), ("val", n_train, len(boards))]:
        XtX = torch.zeros(N_PTS, D, D, device=DEV)
        XtY = torch.zeros(N_PTS, D, D_G, device=DEV)
        Ysum = torch.zeros(N_PTS, D_G, device=DEV)
        Ysq = torch.zeros(N_PTS, D_G, device=DEV)
        n = 0
        t0 = time.time()
        for i in range(lo, hi, batch):
            bb = boards[i:i + batch].to(DEV)
            z = (guest_acts(guest, bb) - mu[:, None, None]) / sd[:, None, None]
            if qwen is not None:
                xs = qwen_feats(qwen, tok, bb)
                xs = torch.cat([xs, torch.ones(*xs.shape[:3], 1, device=DEV)], -1)
            else:
                cf = control_feats(bb)
            for l in range(N_PTS):
                x = (xs[l] if qwen is not None else cf).reshape(-1, D)
                y = z[l].reshape(-1, D_G)
                XtX[l] += x.T @ x
                XtY[l] += x.T @ y
                Ysum[l] += y.sum(0)
                Ysq[l] += (y * y).sum(0)
            n += len(bb) * (N_CELL + N_REG)
            if (i - lo) // batch % 40 == 0:
                print(f"  {split} {i-lo}/{hi-lo} ({time.time()-t0:.0f}s)",
                      flush=True)
        stats[split] = dict(XtX=XtX, XtY=XtY, Ysum=Ysum, Ysq=Ysq, n=n)

    def r2_of(A, s):
        sse = (s["Ysq"].sum(-1) - 2 * (A * s["XtY"]).sum(dim=(-2, -1))
               + (A * (s["XtX"] @ A)).sum(dim=(-2, -1)))
        mean = s["Ysum"] / s["n"]
        sstot = (s["Ysq"] - s["n"] * mean * mean).sum(-1)
        return 1 - sse / sstot

    best = None
    for lam in [0.1, 1.0, 10.0, 100.0]:
        A = torch.linalg.solve(stats["train"]["XtX"]
                               + torch.eye(D, device=DEV) * lam,
                               stats["train"]["XtY"])
        r2 = r2_of(A, stats["val"])
        if best is None or r2.mean() > best[1].mean():
            best = (lam, r2, A)
    return best


@torch.no_grad()
def rank_check(guest, boards, n=10000, n_map=4000, batch=500):
    tok_cov = torch.zeros(N_PTS, 1024, 1024, device=DEV)
    tok_sum = torch.zeros(N_PTS, 1024, device=DEV)
    maps = torch.empty(N_PTS, n_map, 128 * 1024, dtype=torch.float16)
    cnt = 0
    for i in range(0, n, batch):
        bb = boards[i:i + batch].to(DEV)
        hs = guest_acts(guest, bb)  # (13,B,128,1024)
        for l in range(N_PTS):
            x = hs[l].reshape(-1, 1024)
            tok_cov[l] += x.T @ x
            tok_sum[l] += x.sum(0)
        if i < n_map:
            take = min(batch, n_map - i)
            maps[:, i:i + take] = hs[:, :take].reshape(N_PTS, take, -1).half()
        cnt += len(bb) * 128
    out = {}
    for l in range(N_PTS):
        cov = tok_cov[l] / cnt - torch.outer(tok_sum[l] / cnt, tok_sum[l] / cnt)
        ev = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0).cpu()
        cum = ev.cumsum(0) / ev.sum()
        m = maps[l].float().to(DEV)
        m = m - m.mean(0)
        g = (m @ m.T)
        evm = torch.linalg.eigvalsh(g).clamp_min(0).flip(0).cpu()
        cumm = evm.cumsum(0) / evm.sum()
        out[l] = {
            "tok_rank90": int((cum < .90).sum()) + 1,
            "tok_pr": round((ev.sum() ** 2 / (ev ** 2).sum()).item(), 1),
            "map_rank90": int((cumm < .90).sum()) + 1,
            "map_pr": round((evm.sum() ** 2 / (evm ** 2).sum()).item(), 1),
            "map_var_top2048": round(cumm[2047].item(), 4),
        }
        print(f"h{l:2d}: tok_rank90 {out[l]['tok_rank90']:4d} tok_PR "
              f"{out[l]['tok_pr']:7.1f} | map_rank90 {out[l]['map_rank90']:4d} "
              f"map_PR {out[l]['map_pr']:7.1f} var@2048 "
              f"{out[l]['map_var_top2048']:.4f}", flush=True)
        del m, g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--positions", default="armF/data/tx_positions.pt")
    ap.add_argument("--out", default="armF/results/p75_baselines.json")
    ap.add_argument("--skip-rank", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(0)

    all_boards = torch.load(args.positions, weights_only=False)["boards"]
    perm = torch.randperm(len(all_boards),
                          generator=torch.Generator().manual_seed(0))
    boards = all_boards[perm[:max(args.n_train + args.n_val, 10000)]]
    guest = load_guest()
    print("guest loaded", flush=True)

    results = {}
    if not args.skip_rank:
        print("== rank check ==", flush=True)
        results["rank"] = rank_check(guest, boards)

    # normalization stats over 2000 train boards
    with torch.no_grad():
        zs = []
        for i in range(0, 2000, 500):
            zs.append(guest_acts(guest, boards[i:i + 500].to(DEV)))
        zc = torch.cat(zs, 1)
        mu = zc.mean(dim=(1, 2))
        sd = zc.std(dim=(1, 2)).clamp_min(1e-4)
        del zs, zc
    print("norm stats done", flush=True)

    pb = boards[: args.n_train + args.n_val]
    tok, qwen = Q.load_qwen()

    print("== control probe (flatboard+pos) ==", flush=True)
    lam, r2c, _ = run_probe(None, None, guest, pb, mu, sd, args.n_train)
    results["control_r2"] = [round(v, 4) for v in r2c.tolist()]
    print("control:", " ".join(f"{v:.3f}" for v in r2c.tolist()), flush=True)

    print("== pretrained Qwen probe ==", flush=True)
    lam, r2p, A = run_probe(qwen, tok, guest, pb, mu, sd, args.n_train)
    results["pretrained_r2"] = [round(v, 4) for v in r2p.tolist()]
    results["pretrained_lambda"] = lam
    torch.save({"A": A.cpu(), "mu": mu.cpu(), "sd": sd.cpu(), "lambda": lam},
               "armF/results/p75_probe_pretrained.pt")
    del qwen
    torch.cuda.empty_cache()

    print("== random-init Qwen probe ==", flush=True)
    tok, qwen_r = Q.load_qwen(random_init=True)
    lam, r2r, Ar = run_probe(qwen_r, tok, guest, pb, mu, sd, args.n_train)
    results["randinit_r2"] = [round(v, 4) for v in r2r.tolist()]
    torch.save({"A": Ar.cpu(), "mu": mu.cpu(), "sd": sd.cpu(), "lambda": lam},
               "armF/results/p75_probe_randinit.pt")

    print("\nlayer | pretrained | randinit | control")
    for l in range(N_PTS):
        print(f"  h{l:2d} |   {r2p[l]:6.3f}   |  {r2r[l]:6.3f}  | {r2c[l]:6.3f}")
    print(f"mean  |   {r2p.mean():6.3f}   |  {r2r.mean():6.3f}  "
          f"| {r2c.mean():6.3f}")

    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
