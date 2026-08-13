"""Is the c1 wall readout-side or backbone-side? (Joshua test (c), 2026-08-13)

Frozen c1-trained backbone (armF_movesc1). At X tokens, fit heads -> c1:
  linear(hs6), MLP(hs6), MLP(hs5), and references linear(c0true), MLP(c0true)
where c0true is the normalized native c0. MLP(c0true) is the D2-style
ceiling (the CNN transition itself, MLP-learnable?); MLP(hs6)-linear(hs6)
is the discriminator: big gap = readout-side wall, no gap = backbone-side.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_containment as T  # noqa: E402
import train_movesr4 as R4  # noqa: E402
import train_movesr4t as R4T  # noqa: E402

DEV = "cuda"
K = 1024
CKPT = "checkpoints/armF_movesc1/final.pt"
N_TRAIN_G = 1500


@torch.no_grad()
def collect(backbone, games, recs, sel, layers=(5, 6), batch=8):
    Hs = {k: [] for k in layers}
    for i in range(0, len(sel), batch):
        chunk = sel[i:i + batch]
        lens = [len(recs[r]["ids"]) for r in chunk]
        Tlen = max(lens)
        ids = torch.full((len(chunk), Tlen), 151643, dtype=torch.long)
        for j, r in enumerate(chunk):
            ids[j, :lens[j]] = recs[r]["ids"].long()
        am = (torch.arange(Tlen)[None] < torch.tensor(lens)[:, None]).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = backbone(input_ids=ids.to(DEV), attention_mask=am.to(DEV),
                           output_hidden_states=True, use_cache=False)
        for j, r in enumerate(chunk):
            mt = recs[r]["mt"].long().to(DEV)
            for k in layers:
                Hs[k].append(out.hidden_states[k][j, mt].float().cpu())
    return {k: torch.cat(v) for k, v in Hs.items()}


@torch.no_grad()
def c_targets(student, games, mu, sd):
    out = {0: [], 1: []}
    for i in range(0, len(games), 32):
        bb = torch.cat([g["boards"][0::2].float()
                        for g in games[i:i + 32]]).to(DEV)
        cs = list(R4.dump_c(student, bb))
        for l in (0, 1):
            out[l].append(((cs[l] - mu[l]) / sd[l]).cpu())
    return {l: torch.cat(v) for l, v in out.items()}


def r2(pred, tgt, sst):
    return (1 - ((pred - tgt) ** 2).sum() / sst).item()


def fit_linear(Xtr, Ytr, Xte, Yte, sst, lam=1.0):
    X = torch.cat([Xtr, torch.ones(len(Xtr), 1, device=DEV)], 1)
    G = X.T @ X + lam * torch.eye(X.shape[1], device=DEV)
    Wr = torch.linalg.solve(G, X.T @ Ytr)
    Xe = torch.cat([Xte, torch.ones(len(Xte), 1, device=DEV)], 1)
    return r2(Xe @ Wr, Yte, sst)


def fit_mlp(Xtr, Ytr, Xte, Yte, sst, hidden=4096, epochs=30, bs=4096,
            lr=1e-3, tag=""):
    din = Xtr.shape[1]
    mlp = nn.Sequential(nn.Linear(din, hidden), nn.GELU(),
                        nn.Linear(hidden, K)).to(DEV)
    opt = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=0.0)
    n = len(Xtr)
    best = -1e9
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            sel = perm[i:i + bs]
            loss = ((mlp(Xtr[sel]) - Ytr[sel]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = r2(mlp(Xte), Yte, sst)
        best = max(best, v)
        if ep % 5 == 4:
            print(f"  [{tag}] ep{ep + 1} val R2 {v:.4f} (best {best:.4f})",
                  flush=True)
    return best


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]
    tr_gi = [gi for gi in range(len(games)) if gi % 15 != 0][:N_TRAIN_G]
    te_gi = [gi for gi in range(len(games)) if gi % 15 == 0][:60]
    cnn = W.load_model()
    student = R4.load_student(cnn)
    mu, sd = R4.c_stats(student, games)

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    backbone = T.load_backbone()
    missing, _ = backbone.load_state_dict(
        {k: v.float() for k, v in ck["backbone"].items()}, strict=False)
    assert not [m for m in missing if "rotary" not in m]
    backbone.eval()

    tr_g = [games[i] for i in tr_gi]
    te_g = [games[i] for i in te_gi]
    recs_tr = R4T.build_seqs_t(tok, tr_g)
    recs_te = R4T.build_seqs_t(tok, te_g)
    print(f"collecting hs for {len(tr_g)} train / {len(te_g)} test games")
    Htr = collect(backbone, tr_g, recs_tr, list(range(len(recs_tr))))
    Hte = collect(backbone, te_g, recs_te, list(range(len(recs_te))))
    Ctr = c_targets(student, tr_g, mu, sd)
    Cte = c_targets(student, te_g, mu, sd)
    assert len(Htr[5]) == len(Ctr[0]), (len(Htr[5]), len(Ctr[0]))
    print(f"{len(Htr[5])} train tokens, {len(Hte[5])} test")

    Y1tr, Y1te = Ctr[1].to(DEV), Cte[1].to(DEV)
    sst = ((Y1te - Y1te.mean(0)) ** 2).sum()
    res = {}
    res["linear_hs6"] = fit_linear(Htr[6].to(DEV), Y1tr, Hte[6].to(DEV),
                                   Y1te, sst)
    res["linear_c0true"] = fit_linear(Ctr[0].to(DEV), Y1tr, Cte[0].to(DEV),
                                      Y1te, sst)
    print(f"linear(hs6)->c1 {res['linear_hs6']:.4f} | "
          f"linear(c0true)->c1 {res['linear_c0true']:.4f}", flush=True)
    res["mlp_hs6"] = fit_mlp(Htr[6].to(DEV), Y1tr, Hte[6].to(DEV), Y1te,
                             sst, tag="mlp_hs6")
    res["mlp_hs5"] = fit_mlp(Htr[5].to(DEV), Y1tr, Hte[5].to(DEV), Y1te,
                             sst, tag="mlp_hs5")
    res["mlp_c0true"] = fit_mlp(Ctr[0].to(DEV), Y1tr, Cte[0].to(DEV), Y1te,
                                sst, tag="mlp_c0true")
    print(json.dumps(res, indent=1))
    Path("armF/results/head_capacity_c1.json").write_text(json.dumps(
        {"ckpt": CKPT, "n_train_tokens": len(Htr[5]), "res": res}, indent=1))


if __name__ == "__main__":
    main()
