"""Depth sweep: render-free z0 full-map target read from EVERY hidden depth.

Same setup as train_movesonly_z0.py but 23 adapters, one per hs[1..23], each a
Linear(2048 -> 7744) predicting the normalized z0 map after each move, at that
move's last token. Loss = mean over depths. Reports per-depth val R2.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_containment as T  # noqa: E402
import train_movesonly_z0 as Z  # noqa: E402

DEV = "cuda"
DEPTHS = list(range(1, 24))  # hs[0] = embeddings (frozen, skip)


def forward_batch(backbone, ads, cnn, games, recs, sel, mu0, sd0):
    tgt = Z.batch_targets(cnn, games, recs, sel, mu0, sd0)  # (M,7744)
    lens = [len(recs[i]["ids"]) for i in sel]
    Tlen = max(lens)
    ids = torch.full((len(sel), Tlen), 151643, dtype=torch.long)
    for j, i in enumerate(sel):
        ids[j, :lens[j]] = recs[i]["ids"].long()
    ids = ids.to(DEV)
    am = (torch.arange(Tlen, device=DEV)[None]
          < torch.tensor(lens, device=DEV)[:, None]).long()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = backbone(input_ids=ids, attention_mask=am,
                       output_hidden_states=True, use_cache=False)
    sses = []
    for d, ad in zip(DEPTHS, ads):
        h = out.hidden_states[d].float()
        hm = torch.cat([h[j, recs[i]["mt"].long().to(DEV)]
                        for j, i in enumerate(sel)])
        sses.append(((ad(hm) - tgt) ** 2).sum())
    loss = torch.stack(sses).mean() / tgt.numel()
    return loss, torch.stack(sses).detach(), tgt


@torch.no_grad()
def evaluate(backbone, ads, cnn, games, recs, idx, mu0, sd0, batch=8):
    backbone.eval()
    sse = torch.zeros(len(DEPTHS), device=DEV)
    ssum = ssq = 0.0
    n = 0
    for i in range(0, len(idx), batch):
        sel = idx[i:i+batch]
        _, s, tgt = forward_batch(backbone, ads, cnn, games, recs, sel, mu0, sd0)
        sse += s
        ssum += tgt.sum().item()
        ssq += (tgt ** 2).sum().item()
        n += tgt.numel()
    backbone.train()
    mean = ssum / n
    sstot = ssq - n * mean * mean
    return (1 - sse.cpu() / sstot).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--n-val", type=int, default=60)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    recs = Z.build_seqs(tok, games)
    val = [i for i in range(len(recs)) if recs[i]["gi"] % 15 == 0][:args.n_val]
    train = [i for i in range(len(recs)) if recs[i]["gi"] % 15 != 0]

    cnn = W.load_model()
    backbone = T.load_backbone()
    d = torch.load("armF/results/probe_frozen.pt", weights_only=False)
    mu0, sd0 = d["mu"][0].to(DEV), d["sd"][0].to(DEV)
    ads = nn.ModuleList([nn.Linear(2048, 7744) for _ in DEPTHS]).to(DEV)
    backbone.train()

    qwen_params = [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": qwen_params, "lr": args.lr},
        {"params": ads.parameters(), "lr": args.adapter_lr},
    ], weight_decay=0.0)

    def lr_scale(step):
        if step < args.warmup:
            return step / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    def show(step, r2):
        best = max(range(len(r2)), key=lambda i: r2[i])
        print(f"step {step} peak hs[{DEPTHS[best]}] R2 {r2[best]:.4f} | "
              + " ".join(f"{DEPTHS[i]}:{r2[i]:.2f}"
                         for i in range(0, len(r2), 3)), flush=True)

    r2 = evaluate(backbone, ads, cnn, games, recs, val, mu0, sd0)
    show(0, r2)
    hist = [(0, r2)]

    rng = random.Random(0)
    step, t0 = 0, time.time()
    while step < args.steps:
        sel = rng.sample(train, args.batch)
        loss, _, _ = forward_batch(backbone, ads, cnn, games, recs, sel,
                                   mu0, sd0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qwen_params + list(ads.parameters()), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step % args.eval_every == 0:
            r2 = evaluate(backbone, ads, cnn, games, recs, val, mu0, sd0)
            show(step, r2)
            hist.append((step, r2))

    Path("armF/results/movesonly_sweep.json").write_text(json.dumps(
        {"depths": DEPTHS, "hist": hist, "args": vars(args)}))
    final = hist[-1][1]
    print("FINAL per-depth R2: "
          + " ".join(f"hs[{d}]:{v:.3f}" for d, v in zip(DEPTHS, final)))


if __name__ == "__main__":
    main()
