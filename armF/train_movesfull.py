"""Arm F r3: render-free full-stack containment.

Input = preamble + numbered/colored move list ONLY ("\n1. g1 X\n2. d4 O ...").
At each move token (the color token), 19 adapters Linear(2048 -> 7744) predict
the FULL per-channel-normalized CNN map z_l after that move, depth-aligned
z_l <-> hs[5+l]. No cuts: every move token supervises its own prefix
(~73 prefixes/seq, ~147k supervised scalars per move token, ~301M adapter
params). Joint FT as r1/r2 (backbone lr 1e-5, adapters 1e-3).
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
L = 19


@torch.no_grad()
def batch_targets(cnn, games, recs, sel, mu, sd):
    """(M, 19, 7744) normalized channel-major maps for all prefix plies."""
    boards = [games[recs[i]["gi"]]["boards"].float() for i in sel]
    bb = torch.cat(boards).to(DEV)
    acts = W.dump_acts(cnn, bb)
    tgt = torch.empty(len(bb), L, 7744, device=DEV)
    for l, a in enumerate(acts):
        z = (a - mu[l][None, :, None, None]) / sd[l][None, :, None, None]
        tgt[:, l] = z.reshape(len(bb), -1)
    return tgt


def forward_batch(backbone, ads, cnn, games, recs, sel, mu, sd):
    tgt = batch_targets(cnn, games, recs, sel, mu, sd)
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
    for l in range(L):
        h = out.hidden_states[5 + l].float()
        hm = torch.cat([h[j, recs[i]["mt"].long().to(DEV)]
                        for j, i in enumerate(sel)])
        sses.append(((ads[l](hm) - tgt[:, l]) ** 2).sum())
    sses = torch.stack(sses)
    loss = sses.mean() / tgt[:, 0].numel()
    return loss, sses.detach(), tgt


@torch.no_grad()
def evaluate(backbone, ads, cnn, games, recs, idx, mu, sd, batch=4):
    backbone.eval()
    sse = torch.zeros(L, device=DEV)
    ssum = torch.zeros(L, device=DEV)
    ssq = torch.zeros(L, device=DEV)
    n = 0
    for i in range(0, len(idx), batch):
        sel = idx[i:i+batch]
        _, s, tgt = forward_batch(backbone, ads, cnn, games, recs, sel, mu, sd)
        sse += s
        ssum += tgt.sum(dim=(0, 2))
        ssq += (tgt ** 2).sum(dim=(0, 2))
        n += tgt[:, 0].numel()
    backbone.train()
    mean = ssum / n
    return (1 - sse / (ssq - n * mean * mean)).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--run-name", default="armF_movesfull_r3")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_dir = Path(f"checkpoints/{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    recs = Z.build_seqs(tok, games, "numbered")
    print(f"{len(recs)} seqs, maxlen {max(len(r['ids']) for r in recs)}")
    for r in random.Random(0).sample(recs, 3):
        print("--- sample ---")
        print(r["text"][:220].replace("\n", "\\n"))

    val = [i for i in range(len(recs)) if recs[i]["gi"] % 15 == 0][:args.n_val]
    train = [i for i in range(len(recs)) if recs[i]["gi"] % 15 != 0]
    cnn = W.load_model()
    backbone = T.load_backbone()
    d = torch.load("armF/results/probe_frozen.pt", weights_only=False)
    mu, sd = d["mu"].to(DEV), d["sd"].to(DEV)
    ads = nn.ModuleList([nn.Linear(2048, 7744) for _ in range(L)]).to(DEV)
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

    if not args.smoke:
        import wandb
        wandb.init(project="hex-rl-cot-deconfusion", name=args.run_name,
                   config=vars(args))

    def log_eval(step):
        r2 = evaluate(backbone, ads, cnn, games, recs, val, mu, sd)
        print(f"step {step} val R2 mean {r2.mean().item():.4f} | "
              + " ".join(f"z{l}:{r2[l].item():.2f}" for l in range(0, L, 3))
              + f" z18:{r2[18].item():.2f}", flush=True)
        if not args.smoke:
            wandb.log({"val/r2_mean": r2.mean().item()}
                      | {f"val/r2_z{l}": r2[l].item() for l in range(L)},
                      step=max(step, 1))
        return r2

    log_eval(0)
    if args.smoke:
        sel = train[:args.batch]
        loss, _, _ = forward_batch(backbone, ads, cnn, games, recs, sel, mu, sd)
        loss.backward()
        opt.step()  # materialize Adam states for the true memory peak
        print(f"smoke loss {loss.item():.3f}, "
              f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}GB")
        return

    rng = random.Random(0)
    step, t0 = 0, time.time()
    best = -1e9
    while step < args.steps:
        sel = rng.sample(train, args.batch)
        loss, _, _ = forward_batch(backbone, ads, cnn, games, recs, sel, mu, sd)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            qwen_params + list(ads.parameters()), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 25 == 0:
            wandb.log({"train/loss": loss.item(), "train/grad_norm": gn.item(),
                       "train/lr": sched.get_last_lr()[0]}, step=step)
            print(f"step {step} loss {loss.item():.4f} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step % args.eval_every == 0:
            r2 = log_eval(step)
            score = r2.mean().item()
            if score > best:
                best = score
                torch.save({"backbone": {k: v.bfloat16() for k, v in
                                         backbone.state_dict().items()},
                            "ads": {k: v.bfloat16() for k, v in
                                    ads.state_dict().items()},
                            "step": step, "r2": r2.tolist(),
                            "args": vars(args)}, out_dir / "best.pt")
    r2 = log_eval(step)
    (out_dir / "final_r2.json").write_text(json.dumps(
        {"r2": r2.tolist(), "step": step}))
    wandb.finish()


if __name__ == "__main__":
    main()
