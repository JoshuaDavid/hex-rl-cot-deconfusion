"""Joshua's hs5 test on the r4x (frame-consistent) checkpoint.

At X move (color) tokens: per layer, (a) ridge probe hs -> per-cell 3-way
occupancy (absolute frame; a fixed frame transform is absorbed by the linear
map), (b) ridge probe hs -> normalized c0 (1024-dim), per-dim R2. Tests the
prediction that occupancy probes are near-perfect by ~hs5.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_movesr4 as R4  # noqa: E402
import train_movesr4x as R4X  # noqa: E402
import eval_stitch_r4 as E4  # noqa: E402

DEV = "cuda"
NH = 24


@torch.no_grad()
def collect(backbone, games, recs, sel, batch=2):
    H, Y, P = [], [], []
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
            mt = recs[r]["mt"].long()
            H.append(torch.stack(
                [out.hidden_states[k][j, mt.to(DEV)].float().cpu()
                 for k in range(NH)], 1))
            moves = games[recs[r]["gi"]]["moves"]
            occ = torch.zeros(121, dtype=torch.long)
            for t, mv in enumerate(moves.tolist()):
                occ[mv[0] * 11 + mv[1]] = 1 + t % 2
                if t % 2 == 0:  # supervised (X) tokens only
                    Y.append(occ.clone())
                    P.append(t)
    return torch.cat(H), torch.stack(Y), torch.tensor(P)


def ridge(Xtr, Ytr, Xte, lam=1e1):
    X = torch.cat([Xtr, torch.ones(len(Xtr), 1, device=DEV)], 1)
    G = X.T @ X + lam * torch.eye(X.shape[1], device=DEV)
    Wr = torch.linalg.solve(G, X.T @ Ytr)
    Xe = torch.cat([Xte, torch.ones(len(Xte), 1, device=DEV)], 1)
    return Xe @ Wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/armF_movesr4x/best.pt")
    ap.add_argument("--n-games", type=int, default=90)
    ap.add_argument("--out", default="armF/results/probe_absocc_r4x.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]
    val_g = [games[gi] for gi in range(len(games))
             if gi % 15 == 0][:args.n_games]
    recs = R4X.build_seqs_x(tok, val_g)
    backbone, _, _, _, step = E4.load_trained(args.ckpt)
    print(f"ckpt step {step}, {len(val_g)} val games")

    cnn = W.load_model()
    student = R4.load_student(cnn)
    mu, sd = R4.c_stats(student, games)

    n_fit_g = args.n_games * 2 // 3
    fit_sel = [i for i in range(len(recs)) if recs[i]["gi"] < n_fit_g]
    te_sel = [i for i in range(len(recs)) if recs[i]["gi"] >= n_fit_g]
    Htr, Ytr, _ = collect(backbone, val_g, recs, fit_sel)
    Hte, Yte, Pte = collect(backbone, val_g, recs, te_sel)
    print(f"fit {len(Ytr)} X-tokens, test {len(Yte)}")

    def c0_targets(sel_games):
        bb = torch.cat([g["boards"][0::2].float() for g in sel_games]).to(DEV)
        c0 = next(iter(R4.dump_c(student, bb)))
        return (c0 - mu[0]) / sd[0]

    Ctr = c0_targets(val_g[:n_fit_g])
    Cte = c0_targets(val_g[n_fit_g:])
    sst = ((Cte - Cte.mean(0)) ** 2).sum()

    Yoh_tr = torch.zeros(len(Ytr), 121, 3, device=DEV)
    Yoh_tr.scatter_(2, Ytr.to(DEV)[:, :, None], 1.0)
    Yte_d = Yte.to(DEV)
    occ_mask = Yte_d > 0

    res = {}
    for k in range(NH):
        pred = ridge(Htr[:, k].to(DEV), Yoh_tr.reshape(len(Ytr), -1),
                     Hte[:, k].to(DEV)).reshape(len(Yte), 121, 3).argmax(-1)
        correct = pred == Yte_d
        chat = ridge(Htr[:, k].to(DEV), Ctr, Hte[:, k].to(DEV))
        res[k] = {
            "acc": correct.float().mean().item(),
            "acc_occupied": correct[occ_mask].float().mean().item(),
            "acc_empty": correct[~occ_mask].float().mean().item(),
            "c0_r2": (1 - ((chat - Cte) ** 2).sum() / sst).item(),
        }
        for lo, hi in [(0, 20), (20, 40), (40, 200)]:
            m = ((Pte >= lo) & (Pte < hi)).to(DEV)
            if m.sum() > 0:
                res[k][f"acc_occ_ply{lo}-{hi}"] = \
                    correct[m][occ_mask[m]].float().mean().item()
        print(f"hs{k:2d} occ-acc {res[k]['acc_occupied']:.3f} "
              f"(all {res[k]['acc']:.3f}) | c0 R2 {res[k]['c0_r2']:.3f}",
              flush=True)
    Path(args.out).write_text(json.dumps(
        {"step": step, "ckpt": args.ckpt, "per_layer": res}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
