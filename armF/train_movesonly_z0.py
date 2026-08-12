"""Mini falsifier for r3: render-free, z0 only, full-map adapter.

Input = PREAMBLE_M + full move list (no render, no cuts). At each move token t
a single Linear(2048 -> 7744) must reconstruct the ENTIRE per-channel-normalized
z0 map (channel-major flatten) of the board after move t. Joint fine-tune of the
backbone (hidden_states[5] readout, depth-aligned). 1000 steps, ~10 min.
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
import render_moves as RM  # noqa: E402

DEV = "cuda"


def build_seqs(tok, games, fmt="plain"):
    recs = []
    for gi, g in enumerate(games):
        parts = [RM.PREAMBLE_M]
        pos = len(RM.PREAMBLE_M)
        spans = []
        for t, mv in enumerate(g["moves"].tolist()):
            if fmt == "numbered":
                s = f"\n{t + 1}. {RM.move_str(mv)} {'X' if t % 2 == 0 else 'O'}"
            else:
                s = " " + RM.move_str(mv)
            parts.append(s)
            spans.append((pos + 1, pos + len(s)))
            pos += len(s)
        text = "".join(parts)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        mt = RM.move_token_indices(enc["offset_mapping"], spans)
        recs.append({"ids": torch.tensor(enc["input_ids"], dtype=torch.int32),
                     "mt": torch.tensor(mt, dtype=torch.int16), "gi": gi,
                     "text": text})
    return recs


@torch.no_grad()
def batch_targets(cnn, games, recs, sel, mu0, sd0):
    boards = [games[recs[i]["gi"]]["boards"].float() for i in sel]
    bb = torch.cat(boards).to(DEV)
    a = W.dump_acts(cnn, bb)[0]  # (M,64,11,11)
    z = (a - mu0[None, :, None, None]) / sd0[None, :, None, None]
    return z.reshape(len(bb), -1)  # channel-major 7744


def forward_batch(backbone, ad, cnn, games, recs, sel, mu0, sd0):
    tgt = batch_targets(cnn, games, recs, sel, mu0, sd0)
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
    h = out.hidden_states[5].float()
    hm = torch.cat([h[j, recs[i]["mt"].long().to(DEV)]
                    for j, i in enumerate(sel)])  # (M,2048), game-major
    pred = ad(hm)
    sse = ((pred - tgt) ** 2).sum()
    return sse / tgt.numel(), sse.detach(), tgt


@torch.no_grad()
def evaluate(backbone, ad, cnn, games, recs, idx, mu0, sd0, batch=8):
    backbone.eval()
    sse = ssum = ssq = 0.0
    n = 0
    for i in range(0, len(idx), batch):
        sel = idx[i:i+batch]
        _, s, tgt = forward_batch(backbone, ad, cnn, games, recs, sel, mu0, sd0)
        sse += s.item()
        ssum += tgt.sum().item()
        ssq += (tgt ** 2).sum().item()
        n += tgt.numel()
    backbone.train()
    mean = ssum / n
    return 1 - sse / (ssq - n * mean * mean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--fmt", default="plain", choices=["plain", "numbered"])
    ap.add_argument("--out", default="armF/results/movesonly_z0.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    recs = build_seqs(tok, games, args.fmt)
    print(f"{len(recs)} seqs, maxlen {max(len(r['ids']) for r in recs)}")
    for r in random.Random(0).sample(recs, 3):
        print("--- sample ---")
        print(r["text"][:400])

    val = [i for i in range(len(recs)) if recs[i]["gi"] % 15 == 0][:args.n_val]
    train = [i for i in range(len(recs)) if recs[i]["gi"] % 15 != 0]
    cnn = W.load_model()
    backbone = T.load_backbone()
    d = torch.load("armF/results/probe_frozen.pt", weights_only=False)
    mu0, sd0 = d["mu"][0].to(DEV), d["sd"][0].to(DEV)
    ad = nn.Linear(2048, 7744).to(DEV)
    backbone.train()

    qwen_params = [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": qwen_params, "lr": args.lr},
        {"params": ad.parameters(), "lr": args.adapter_lr},
    ], weight_decay=0.0)

    def lr_scale(step):
        if step < args.warmup:
            return step / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    r2 = evaluate(backbone, ad, cnn, games, recs, val, mu0, sd0)
    print(f"step 0 val R2 {r2:.4f}", flush=True)
    hist = [(0, r2)]

    rng = random.Random(0)
    step, t0 = 0, time.time()
    while step < args.steps:
        sel = rng.sample(train, args.batch)
        loss, _, _ = forward_batch(backbone, ad, cnn, games, recs, sel, mu0, sd0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(qwen_params + list(ad.parameters()), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step % args.eval_every == 0:
            r2 = evaluate(backbone, ad, cnn, games, recs, val, mu0, sd0)
            print(f"step {step} val R2 {r2:.4f}", flush=True)
            hist.append((step, r2))

    Path(args.out).write_text(json.dumps({"hist": hist, "args": vars(args)}))
    print(f"FINAL z0 render-free R2 {hist[-1][1]:.4f}")


if __name__ == "__main__":
    main()
