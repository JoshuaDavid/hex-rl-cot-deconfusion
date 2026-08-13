"""Arm F c0-only speed run (Joshua 2026-08-13): how fast CAN the board map go?

Single-token-cell format (r4t), loss on c0 ONLY (adapter at hs5), adapter LR
10x (1e-2). Tests whether the 19-layer dense loss and adapter LR were
throttling c0; c0 is exactly affine in occupancy so R2 -> 1 is achievable
in principle.
"""
import argparse
import json
import random
import sys
import time
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
HS = 5  # readout layer for c0


@torch.no_grad()
def batch_targets(student, games, recs, sel, mu0, sd0):
    boards = [games[recs[i]["gi"]]["boards"][0::2].float() for i in sel]
    bb = torch.cat(boards).to(DEV)
    c0 = next(iter(R4.dump_c(student, bb)))
    return (c0 - mu0) / sd0


def forward_batch(backbone, ad, student, games, recs, sel, mu0, sd0):
    tgt = batch_targets(student, games, recs, sel, mu0, sd0)
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
    h = out.hidden_states[HS].float()
    hm = torch.cat([h[j, recs[i]["mt"].long().to(DEV)]
                    for j, i in enumerate(sel)])
    sse = ((ad(hm) - tgt) ** 2).sum()
    return sse / tgt.numel(), sse.detach(), tgt


@torch.no_grad()
def evaluate(backbone, ad, student, games, recs, idx, mu0, sd0, batch=4):
    backbone.eval()
    sse, ssum, ssq, n = 0.0, 0.0, 0.0, 0
    for i in range(0, len(idx), batch):
        _, s, tgt = forward_batch(backbone, ad, student, games, recs,
                                  idx[i:i+batch], mu0, sd0)
        sse += s
        ssum += tgt.sum()
        ssq += (tgt ** 2).sum()
        n += tgt.numel()
    backbone.train()
    mean = ssum / n
    return (1 - sse / (ssq - n * mean * mean)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter-lr", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--run-name", default="armF_movesc0")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_dir = Path(f"checkpoints/{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]
    recs = R4T.build_seqs_t(tok, games)
    print(f"{len(recs)} seqs, maxlen {max(len(r['ids']) for r in recs)}")

    val = [i for i in range(len(recs)) if recs[i]["gi"] % 15 == 0][:args.n_val]
    train = [i for i in range(len(recs)) if recs[i]["gi"] % 15 != 0]
    cnn = W.load_model()
    student = R4.load_student(cnn)
    mu, sd = R4.c_stats(student, games)
    mu0, sd0 = mu[0], sd[0]
    backbone = T.load_backbone()
    ad = nn.Linear(2048, K).to(DEV)
    backbone.train()

    qwen_params = [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": ad.parameters(), "lr": args.adapter_lr},
         {"params": qwen_params, "lr": args.lr}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, step / args.warmup))

    if not args.smoke:
        import wandb
        wandb.init(project="hex-rl-cot-deconfusion", name=args.run_name,
                   config=vars(args))

    hist = []

    def log_eval(step):
        r2 = evaluate(backbone, ad, student, games, recs, val, mu0, sd0)
        hist.append({"step": step, "c0_r2": r2})
        print(f"step {step} val c0 R2 {r2:.4f}", flush=True)
        if not args.smoke:
            wandb.log({"val/r2_c0": r2}, step=max(step, 1))
        return r2

    log_eval(0)
    if args.smoke:
        loss, _, _ = forward_batch(backbone, ad, student, games, recs,
                                   train[:args.batch], mu0, sd0)
        loss.backward()
        opt.step()
        print(f"smoke loss {loss.item():.3f}, "
              f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}GB")
        return

    rng = random.Random(0)
    step, t0 = 0, time.time()
    while step < args.steps:
        sel = rng.sample(train, args.batch)
        loss, _, _ = forward_batch(backbone, ad, student, games, recs, sel,
                                   mu0, sd0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(
            qwen_params + list(ad.parameters()), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 25 == 0:
            wandb.log({"train/loss": loss.item(), "train/grad_norm": gn.item(),
                       "train/lr": sched.get_last_lr()[0]}, step=step)
            print(f"step {step} loss {loss.item():.4f} "
                  f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step % args.eval_every == 0:
            log_eval(step)
    torch.save({"backbone": {k: v.bfloat16() for k, v in
                             backbone.state_dict().items()},
                "ad": {k: v.bfloat16() for k, v in ad.state_dict().items()},
                "step": step, "args": vars(args)}, out_dir / "final.pt")
    (out_dir / "final_c0.json").write_text(json.dumps({"hist": hist}))
    wandb.finish()


if __name__ == "__main__":
    main()
