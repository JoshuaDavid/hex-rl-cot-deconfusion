"""Can a single MLP+skip, outside Qwen, repro the CNN c0->c1 transition?
(Joshua 2026-08-13). Cells: linear(c0) ref, skipMLP(c0) at hidden 4096 and
16384, skipMLP(board) control (board = full information by construction; if
board succeeds where c0 fails, the rank-1024 c0 projection is the
bottleneck, not MLP capacity). All games (~150k X-ply samples), 60 epochs.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_movesr4 as R4  # noqa: E402

DEV = "cuda"
K = 1024


class SkipMLP(nn.Module):
    def __init__(self, din, hidden):
        super().__init__()
        self.lin = nn.Linear(din, K)
        self.mlp = nn.Sequential(nn.Linear(din, hidden), nn.GELU(),
                                 nn.Linear(hidden, K))

    def forward(self, x):
        return self.lin(x) + self.mlp(x)


@torch.no_grad()
def build(student, games, mu, sd):
    C0, C1, B = [], [], []
    for i in range(0, len(games), 32):
        bb = torch.cat([g["boards"][0::2].float()
                        for g in games[i:i + 32]]).to(DEV)
        cs = list(R4.dump_c(student, bb))
        C0.append(((cs[0] - mu[0]) / sd[0]).cpu())
        C1.append(((cs[1] - mu[1]) / sd[1]).cpu())
        B.append(bb.reshape(len(bb), -1).cpu())
    return torch.cat(C0), torch.cat(C1), torch.cat(B)


def r2(pred, tgt, sst):
    return (1 - ((pred - tgt) ** 2).sum() / sst).item()


def fit(model, Xtr, Ytr, Xte, Yte, sst, epochs=60, bs=8192, lr=1e-3, tag=""):
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    n = len(Xtr)
    best = -1e9
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            sel = perm[i:i + bs]
            loss = ((model(Xtr[sel]) - Ytr[sel]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = r2(model(Xte), Yte, sst)
        best = max(best, v)
        if ep % 10 == 9:
            print(f"  [{tag}] ep{ep + 1} val R2 {v:.4f} (best {best:.4f})",
                  flush=True)
    return best


def main():
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]
    te_gi = [gi for gi in range(len(games)) if gi % 15 == 0][:60]
    tr_gi = [gi for gi in range(len(games)) if gi % 15 != 0]
    cnn = W.load_model()
    student = R4.load_student(cnn)
    mu, sd = R4.c_stats(student, games)
    C0tr, C1tr, Btr = build(student, [games[i] for i in tr_gi], mu, sd)
    C0te, C1te, Bte = build(student, [games[i] for i in te_gi], mu, sd)
    print(f"{len(C0tr)} train samples, {len(C0te)} test", flush=True)

    C0tr, C1tr, Btr = C0tr.to(DEV), C1tr.to(DEV), Btr.to(DEV)
    C0te, C1te, Bte = C0te.to(DEV), C1te.to(DEV), Bte.to(DEV)
    sst = ((C1te - C1te.mean(0)) ** 2).sum()

    X = torch.cat([C0tr, torch.ones(len(C0tr), 1, device=DEV)], 1)
    G = X.T @ X + 1.0 * torch.eye(X.shape[1], device=DEV)
    Wr = torch.linalg.solve(G, X.T @ C1tr)
    Xe = torch.cat([C0te, torch.ones(len(C0te), 1, device=DEV)], 1)
    res = {"linear_c0": r2(Xe @ Wr, C1te, sst)}
    print(f"linear(c0) {res['linear_c0']:.4f}", flush=True)

    res["skip4096_c0"] = fit(SkipMLP(K, 4096), C0tr, C1tr, C0te, C1te, sst,
                             tag="skip4096_c0")
    res["skip16384_c0"] = fit(SkipMLP(K, 16384), C0tr, C1tr, C0te, C1te, sst,
                              tag="skip16384_c0")
    res["skip4096_board"] = fit(SkipMLP(Btr.shape[1], 4096), Btr, C1tr, Bte,
                                C1te, sst, tag="skip4096_board")
    print(json.dumps(res, indent=1))
    Path("armF/results/mlp_skip_c0c1.json").write_text(json.dumps(
        {"n_train": len(C0tr), "res": res}, indent=1))


if __name__ == "__main__":
    main()
