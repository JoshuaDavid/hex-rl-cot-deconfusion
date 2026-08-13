"""Arm F r4x: frame-consistent supervision (Joshua's proposal 2026-08-13).

Same task as r4 (contain the distilled net's native c_l in R^1024) but the
canonical frame no longer alternates across supervised positions: format =
preamble + "\nPlayer: X" + numbered move list, loss ONLY at X move (color)
tokens, whose targets all share the O-to-move frame. WSD schedule (warmup
-> constant; no anneal) so extension is free.
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
import train_movesr4 as R4  # noqa: E402

DEV = "cuda"
L = 19
K = 1024
HDR = "\nPlayer: X"


def build_seqs_x(tok, games):
    """Numbered format + Player header; move-token spans for X moves only."""
    recs = []
    for gi, g in enumerate(games):
        parts = [RM.PREAMBLE_M, HDR]
        pos = len(RM.PREAMBLE_M) + len(HDR)
        spans = []
        for t, mv in enumerate(g["moves"].tolist()):
            s = f"\n{t + 1}. {RM.move_str(mv)} {'X' if t % 2 == 0 else 'O'}"
            parts.append(s)
            if t % 2 == 0:
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
def batch_targets(student, games, recs, sel, mu, sd):
    """(M, 19, 1024) normalized native states, X plies (even t) only."""
    boards = [games[recs[i]["gi"]]["boards"][0::2].float() for i in sel]
    bb = torch.cat(boards).to(DEV)
    tgt = torch.empty(len(bb), L, K, device=DEV)
    for l, c in enumerate(R4.dump_c(student, bb)):
        tgt[:, l] = (c - mu[l]) / sd[l]
    return tgt


def forward_batch(backbone, ads, student, games, recs, sel, mu, sd):
    tgt = batch_targets(student, games, recs, sel, mu, sd)
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
def evaluate(backbone, ads, student, games, recs, idx, mu, sd, batch=4):
    backbone.eval()
    sse = torch.zeros(L, device=DEV)
    ssum = torch.zeros(L, device=DEV)
    ssq = torch.zeros(L, device=DEV)
    n = 0
    for i in range(0, len(idx), batch):
        sel = idx[i:i+batch]
        _, s, tgt = forward_batch(backbone, ads, student, games, recs, sel,
                                  mu, sd)
        sse += s
        ssum += tgt.sum(dim=(0, 2))
        ssq += (tgt ** 2).sum(dim=(0, 2))
        n += tgt[:, 0].numel()
    backbone.train()
    mean = ssum / n
    return (1 - sse / (ssq - n * mean * mean)).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--run-name", default="armF_movesr4x")
    ap.add_argument("--init-ckpt", default=None,
                    help="warm restart (weights only)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_dir = Path(f"checkpoints/{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]
    recs = build_seqs_x(tok, games)
    print(f"{len(recs)} seqs, maxlen {max(len(r['ids']) for r in recs)}, "
          f"X-tokens/seq ~{sum(len(r['mt']) for r in recs)/len(recs):.1f}")
    for r in random.Random(0).sample(recs, 3):
        print("--- sample ---")
        print(r["text"][:260].replace("\n", "\\n"))

    val = [i for i in range(len(recs)) if recs[i]["gi"] % 15 == 0][:args.n_val]
    train = [i for i in range(len(recs)) if recs[i]["gi"] % 15 != 0]
    cnn = W.load_model()
    student = R4.load_student(cnn)
    mu, sd = R4.c_stats(student, games)
    backbone = T.load_backbone()
    ads = nn.ModuleList([nn.Linear(2048, K) for _ in range(L)]).to(DEV)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location="cpu",
                        weights_only=False)
        missing, _ = backbone.load_state_dict(
            {k: v.float() for k, v in ck["backbone"].items()}, strict=False)
        assert not [m for m in missing if "rotary" not in m], missing
        ads.load_state_dict({k: v.float() for k, v in ck["ads"].items()})
        print(f"warm start from {args.init_ckpt} (step {ck.get('step')})")
    backbone.train()

    qwen_params = [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": ads.parameters(), "lr": args.adapter_lr},
         {"params": qwen_params, "lr": args.lr}], weight_decay=0.0)

    def lr_scale(step):  # WSD without the D: warmup then constant
        return min(1.0, step / args.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    if not args.smoke:
        import wandb
        wandb.init(project="hex-rl-cot-deconfusion", name=args.run_name,
                   config=vars(args))

    def log_eval(step):
        r2 = evaluate(backbone, ads, student, games, recs, val, mu, sd)
        print(f"step {step} val R2 mean {r2.mean().item():.4f} | "
              + " ".join(f"c{l}:{r2[l].item():.2f}" for l in range(0, L, 3))
              + f" c18:{r2[18].item():.2f}", flush=True)
        if not args.smoke:
            wandb.log({"val/r2_mean": r2.mean().item()}
                      | {f"val/r2_c{l}": r2[l].item() for l in range(L)},
                      step=max(step, 1))
        return r2

    if args.smoke:
        r = recs[0]
        g = games[0]
        n_x = (len(g["moves"]) + 1) // 2
        assert len(r["mt"]) == n_x, (len(r["mt"]), n_x)
        assert len(g["boards"][0::2]) == n_x
        print("X-only span/target alignment: OK")

    log_eval(0)
    if args.smoke:
        sel = train[:args.batch]
        loss, _, _ = forward_batch(backbone, ads, student, games, recs, sel,
                                   mu, sd)
        loss.backward()
        opt.step()
        print(f"smoke loss {loss.item():.3f}, "
              f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}GB")
        return

    rng = random.Random(0)
    step, t0 = 0, time.time()
    best = -1e9
    while step < args.steps:
        sel = rng.sample(train, args.batch)
        loss, _, _ = forward_batch(backbone, ads, student, games, recs, sel,
                                   mu, sd)
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
