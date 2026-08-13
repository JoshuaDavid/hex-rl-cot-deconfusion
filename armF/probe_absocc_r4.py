"""P33: absolute-frame occupancy ridge probe on the r4 checkpoint.

At every move (color) token, probe hs[layer] -> per-cell 3-way occupancy
{empty, X, O} in the FIXED absolute frame (no canonical flip). Per layer,
per parity, and per prefix-length bin. Discriminates frame-gating
bottleneck (probes near-perfect) vs relay error accumulation (probes
degrade with ply).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_movesr4 as R4  # noqa: E402
import eval_stitch_r4 as E4  # noqa: E402
import train_movesonly_z0 as Z  # noqa: E402

DEV = "cuda"
NH = 24  # hidden_states entries (emb + 23 blocks)


@torch.no_grad()
def collect(backbone, tok, games, recs, sel, batch=4):
    """Returns hs (N, NH, 2048), occ targets (N, 121) in {0,1,2}, ply (N,)."""
    H, Y, P = [], [], []
    for i in range(0, len(sel), batch):
        chunk = sel[i:i + batch]
        lens = [len(recs[r]["ids"]) for r in chunk]
        Tlen = max(lens)
        ids = torch.full((len(chunk), Tlen), 151643, dtype=torch.long)
        for j, r in enumerate(chunk):
            ids[j, :lens[j]] = recs[r]["ids"].long()
        am = (torch.arange(Tlen)[None]
              < torch.tensor(lens)[:, None]).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = backbone(input_ids=ids.to(DEV), attention_mask=am.to(DEV),
                           output_hidden_states=True, use_cache=False)
        for j, r in enumerate(chunk):
            mt = recs[r]["mt"].long()
            H.append(torch.stack(
                [out.hidden_states[k][j, mt.to(DEV)].float().cpu()
                 for k in range(NH)], 1))
            moves = games[recs[r]["gi"]]["moves"]
            occ = torch.zeros(121, dtype=torch.long)
            for t, mv in enumerate(moves.tolist()):
                occ[mv[0] * 11 + mv[1]] = 1 + t % 2  # 1=X, 2=O
                Y.append(occ.clone())
                P.append(t)
    return torch.cat(H), torch.stack(Y), torch.tensor(P)


def fit_eval(Htr, Ytr, Hte, Yte, Pte, lam=1e1):
    """Ridge to one-hot (121x3); returns overall/occupied acc + breakdowns."""
    X = torch.cat([Htr, torch.ones(len(Htr), 1, device=DEV)], 1)
    Yoh = torch.zeros(len(Ytr), 121, 3, device=DEV)
    Yoh.scatter_(2, Ytr.to(DEV)[:, :, None], 1.0)
    G = X.T @ X + lam * torch.eye(X.shape[1], device=DEV)
    Wr = torch.linalg.solve(G, X.T @ Yoh.reshape(len(X), -1))
    Xe = torch.cat([Hte, torch.ones(len(Hte), 1, device=DEV)], 1)
    pred = (Xe @ Wr).reshape(len(Xe), 121, 3).argmax(-1)
    Yte = Yte.to(DEV)
    correct = pred == Yte
    occ_mask = Yte > 0
    out = {"acc": correct.float().mean().item(),
           "acc_occupied": correct[occ_mask].float().mean().item(),
           "acc_empty": correct[~occ_mask].float().mean().item()}
    for par in (0, 1):
        m = (Pte % 2 == par).to(DEV)
        out[f"acc_par{par}"] = correct[m].float().mean().item()
        out[f"acc_occ_par{par}"] = correct[m][occ_mask[m]].float().mean().item()
    bins = [(0, 20), (20, 40), (40, 60), (60, 200)]
    for lo, hi in bins:
        m = ((Pte >= lo) & (Pte < hi)).to(DEV)
        if m.sum() > 0:
            out[f"acc_occ_ply{lo}-{hi}"] = \
                correct[m][occ_mask[m]].float().mean().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/armF_movesr4/best.pt")
    ap.add_argument("--n-games", type=int, default=90)
    ap.add_argument("--out", default="armF/results/probe_absocc_r4.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    tok.padding_side = "right"
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]
    val_g = [games[gi] for gi in range(len(games))
             if gi % 15 == 0][:args.n_games]
    recs = Z.build_seqs(tok, val_g, "numbered")
    backbone, _, _, _, step = E4.load_trained(args.ckpt)
    print(f"ckpt step {step}, {len(val_g)} val games")

    n_fit_g = args.n_games * 2 // 3
    fit_sel = [i for i in range(len(recs)) if recs[i]["gi"] < n_fit_g]
    te_sel = [i for i in range(len(recs)) if recs[i]["gi"] >= n_fit_g]
    Htr, Ytr, _ = collect(backbone, tok, val_g, recs, fit_sel)
    Hte, Yte, Pte = collect(backbone, tok, val_g, recs, te_sel)
    print(f"fit {len(Ytr)} prefixes, test {len(Yte)}")

    res = {}
    for k in range(NH):
        res[k] = fit_eval(Htr[:, k].to(DEV), Ytr, Hte[:, k].to(DEV),
                          Yte, Pte)
        print(f"hs{k:2d} acc {res[k]['acc']:.3f} occ {res[k]['acc_occupied']:.3f}"
              f" | par0 {res[k]['acc_occ_par0']:.3f} par1 "
              f"{res[k]['acc_occ_par1']:.3f}", flush=True)
    best = max(res, key=lambda k: res[k]["acc"])
    print(f"\nbest layer hs{best}: {json.dumps(res[best], indent=1)}")
    Path(args.out).write_text(json.dumps(
        {"step": step, "best_layer": best, "per_layer": res}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
