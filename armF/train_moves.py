"""Arm F r2: moves-format joint fine-tune with THREE dense supervision streams.

Sequence = preamble + move list (cut at random ply) + single board render.
 1. columns: at each move token t, the CNN's 19x64 activation column at the
    just-played cell, board AFTER move t (adapters C_l: 2048->64).
 2. pca: at each move token t, top-128 whitened PCA coords of the FULL 7744-dim
    map after move t (adapters P_l: 2048->128).
 3. render: full per-cell map at the 121 render cell tokens, as r1
    (adapters A_l warm-started from the frozen probe).
Loss = mean of the three per-layer-mean MSEs. ~1000 supervised scalars/token.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_containment as T  # noqa: E402

DEV = "cuda"
L = 19


class SmallAdapters(nn.Module):
    def __init__(self, out_dim, H=2048):
        super().__init__()
        self.maps = nn.ModuleList([nn.Linear(H, out_dim) for _ in range(L)])


def load_data():
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    toks = torch.load("armF/data/tokens_moves.pt", weights_only=False)
    return games, toks


@torch.no_grad()
def batch_targets(cnn, games, toks, sel, mu, sd, pca):
    """Returns (col (M,19,64), coords (M,19,128), render (B,19,121,64),
    seq_of (M,), last_of (B,)) where M = total prefix plies in the batch."""
    boards, cells, seq_of, last_of = [], [], [], []
    for j, i in enumerate(sel.tolist()):
        g = games[int(toks["game_id"][i])]
        cut = int(toks["cuts"][i])
        boards.append(g["boards"][:cut].float())
        cells.append(g["cells"][:cut].long())
        seq_of.append(torch.full((cut,), j))
        last_of.append(sum(len(b) for b in boards) - 1)
    bb = torch.cat(boards).to(DEV)
    cell = torch.cat(cells).to(DEV)
    seq_of = torch.cat(seq_of).to(DEV)
    last_of = torch.tensor(last_of, device=DEV)
    M = len(bb)
    acts = W.dump_acts(cnn, bb)  # 19 x (M,64,11,11)
    col = torch.empty(M, L, 64, device=DEV)
    coords = torch.empty(M, L, pca["m"], device=DEV)
    render = torch.empty(len(sel), L, 121, 64, device=DEV)
    for l, a in enumerate(acts):
        flat_c = a.reshape(M, 64, 121)
        col[:, l] = (flat_c[torch.arange(M, device=DEV), :, cell]
                     - mu[l]) / sd[l]
        fm = a.reshape(M, -1)  # channel-major, matches pca_basis
        coords[:, l] = ((fm - pca["mean"][l]) @ pca["V"][l].T) / pca["eig"][l].sqrt()
        z = a.permute(0, 2, 3, 1).reshape(M, 121, 64)
        render[:, l] = (z[last_of] - mu[l]) / sd[l]
    return col, coords, render, seq_of, last_of


def gather_tok(h, idx):
    return torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))


def forward_batch(backbone, adA, adC, adP, cnn, games, toks, sel, mu, sd, pca,
                  want_loss=True):
    """Returns dict of per-layer squared-error sums + counts per stream, and the
    scalar loss (mean of per-layer-mean MSEs over 3 streams)."""
    ids = toks["input_ids"][sel].long().to(DEV)
    Tm_len = int(toks["lens"][sel].max())
    ids = ids[:, :Tm_len]
    am = (torch.arange(Tm_len, device=DEV)[None]
          < toks["lens"][sel][:, None].to(DEV)).long()
    mt = toks["move_tok"][sel].long().to(DEV)
    mmask = mt >= 0
    ncut = int(mmask.sum(1).max())
    mt, mmask = mt[:, :ncut], mmask[:, :ncut]
    ct = toks["cell_idx"][sel].long().to(DEV)
    col, coords, render, seq_of, _ = batch_targets(cnn, games, toks, sel, mu, sd, pca)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = backbone(input_ids=ids, attention_mask=am,
                       output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    loss_terms = {"col": [], "pca": [], "render": []}
    B = len(sel)
    flat_mask = mmask.reshape(-1)
    # map (seq, ply) -> row in col/coords: rows are ordered seq-major
    for l in range(L):
        h = hs[5 + l].float()
        hm = gather_tok(h, mt.clamp(min=0)).reshape(B * ncut, -1)[flat_mask]
        pc = adC.maps[l](hm)
        pp = adP.maps[l](hm)
        hc = gather_tok(h, ct)
        pr = adA.maps[l](hc)
        loss_terms["col"].append(((pc - col[:, l]) ** 2).mean())
        loss_terms["pca"].append(((pp - coords[:, l]) ** 2).mean())
        loss_terms["render"].append(((pr - render[:, l]) ** 2).mean())
    means = {k: torch.stack(v).mean() for k, v in loss_terms.items()}
    loss = (means["col"] + means["pca"] + means["render"]) / 3
    stats = {k: [t.detach() for t in v] for k, v in loss_terms.items()}
    return loss, means, stats, (col, coords, render)


@torch.no_grad()
def evaluate(backbone, adA, adC, adP, cnn, games, toks, idx, mu, sd, pca, batch=8):
    backbone.eval()
    keys = ("col", "pca", "render")
    sse = {k: torch.zeros(L, device=DEV) for k in keys}
    ssum = {k: torch.zeros(L, device=DEV) for k in keys}
    ssq = {k: torch.zeros(L, device=DEV) for k in keys}
    n = {k: 0 for k in keys}
    for i in range(0, len(idx), batch):
        sel = idx[i:i+batch]
        _, _, stats, tgts = forward_batch(backbone, adA, adC, adP, cnn, games,
                                          toks, sel, mu, sd, pca)
        for k, tg in zip(keys, tgts):
            nk = tg[:, 0].numel()
            n[k] += nk
            for l in range(L):
                sse[k][l] += stats[k][l] * nk
                ssum[k][l] += tg[:, l].sum()
                ssq[k][l] += (tg[:, l] ** 2).sum()
    r2 = {}
    for k in keys:
        mean = ssum[k] / n[k]
        sstot = ssq[k] - n[k] * mean * mean
        r2[k] = (1 - sse[k] / sstot).cpu()
    backbone.train()
    return r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--run-name", default="armF_moves_r2")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_dir = Path(f"checkpoints/{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    games, toks = load_data()
    pca = torch.load("armF/results/pca_basis.pt", weights_only=False)
    for k in ("V", "eig", "mean"):
        pca[k] = pca[k].to(DEV)
    N = len(toks["lens"])
    val_games = torch.tensor([g % 15 == 0 for g in toks["game_id"].tolist()])
    val_idx = torch.nonzero(val_games).squeeze(1)[:args.n_val]
    train_idx = torch.nonzero(~val_games).squeeze(1)
    print(f"N {N} train {len(train_idx)} val {len(val_idx)}")

    cnn = W.load_model()
    backbone = T.load_backbone()
    adA = T.Adapters().to(DEV)
    mu, sd = adA.warm_start("armF/results/probe_frozen.pt")
    mu, sd = mu.to(DEV), sd.to(DEV)
    adC = SmallAdapters(64).to(DEV)
    adP = SmallAdapters(pca["m"]).to(DEV)
    backbone.train()

    qwen_params = [p for p in backbone.parameters() if p.requires_grad]
    ad_params = (list(adA.parameters()) + list(adC.parameters())
                 + list(adP.parameters()))
    opt = torch.optim.AdamW([
        {"params": qwen_params, "lr": args.lr},
        {"params": ad_params, "lr": args.adapter_lr},
    ], weight_decay=0.0)

    def lr_scale(step):
        if step < args.warmup:
            return step / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    if not args.smoke:
        import wandb
        wandb.init(project="hex-rl-cot-deconfusion", name=args.run_name,
                   config=vars(args))

    def log_eval(step):
        r2 = evaluate(backbone, adA, adC, adP, cnn, games, toks, val_idx,
                      mu, sd, pca)
        msg = " ".join(f"{k} {r2[k].mean().item():.4f}" for k in r2)
        print(f"step {step} val R2 mean: {msg}", flush=True)
        if not args.smoke:
            wandb.log({f"val/r2_{k}_mean": r2[k].mean().item() for k in r2}
                      | {f"val/r2_{k}_z{l}": r2[k][l].item()
                         for k in r2 for l in (0, 9, 18)}, step=max(step, 1))
        return r2

    log_eval(0)
    if args.smoke:
        sel = train_idx[:args.batch]
        loss, means, _, _ = forward_batch(backbone, adA, adC, adP, cnn, games,
                                          toks, sel, mu, sd, pca)
        loss.backward()
        print("smoke loss:", {k: round(v.item(), 3) for k, v in means.items()})
        print(f"smoke ok, mem {torch.cuda.max_memory_allocated()/2**30:.1f}GB")
        return

    g = torch.Generator().manual_seed(0)
    step, t0 = 0, time.time()
    best = -1e9
    while step < args.steps:
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        for i in range(0, len(perm) - args.batch, args.batch):
            sel = perm[i:i+args.batch]
            loss, means, _, _ = forward_batch(backbone, adA, adC, adP, cnn,
                                              games, toks, sel, mu, sd, pca)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(qwen_params + ad_params, 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 25 == 0:
                wandb.log({"train/loss": loss.item(), "train/grad_norm": gn.item(),
                           "train/lr": sched.get_last_lr()[0],
                           **{f"train/mse_{k}": v.item() for k, v in means.items()}},
                          step=step)
                print(f"step {step} loss {loss.item():.4f} "
                      f"col {means['col'].item():.3f} pca {means['pca'].item():.3f} "
                      f"render {means['render'].item():.3f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
            if step % args.eval_every == 0:
                r2 = log_eval(step)
                score = sum(r2[k].mean().item() for k in r2)
                if score > best:
                    best = score
                    torch.save({"backbone": {k: v.bfloat16() for k, v in
                                             backbone.state_dict().items()},
                                "adA": adA.state_dict(), "adC": adC.state_dict(),
                                "adP": adP.state_dict(), "step": step,
                                "r2": {k: v.tolist() for k, v in r2.items()},
                                "args": vars(args)}, out_dir / "best.pt")
            if step >= args.steps:
                break
    r2 = log_eval(step)
    (out_dir / "final_r2.json").write_text(json.dumps(
        {k: v.tolist() for k, v in r2.items()} | {"step": step}))
    wandb.finish()


if __name__ == "__main__":
    main()
