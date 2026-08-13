"""Arm F single-layer containment runs (Joshua 2026-08-13).

Single-token-cell format (r4t), loss on ONE c-layer (adapter at hs[5+layer]),
adapter LR 10x (1e-2). --layer selects the target; --init-ckpt warm-restarts
the backbone (and the adapter if the saved layer matches; otherwise the saved
adapter is kept FROZEN as an aux drift diagnostic, e.g. watch c0 while
training c1).
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


@torch.no_grad()
def batch_targets(student, games, recs, sel, mu, sd, layers):
    boards = [games[recs[i]["gi"]]["boards"][0::2].float() for i in sel]
    bb = torch.cat(boards).to(DEV)
    cs = list(R4.dump_c(student, bb))
    return {l: (cs[l] - mu[l]) / sd[l] for l in layers}


def gather_h(out, recs, sel, hs_idx):
    h = out.hidden_states[hs_idx].float()
    return torch.cat([h[j, recs[i]["mt"].long().to(DEV)]
                      for j, i in enumerate(sel)])


def forward_batch(backbone, ad, layer, student, games, recs, sel, mu, sd,
                  aux=None, hs_idx=None):
    if hs_idx is None:
        hs_idx = 5 + layer
    layers = [layer] + ([aux[0]] if aux else [])
    tgt = batch_targets(student, games, recs, sel, mu, sd, layers)
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
    hm = gather_h(out, recs, sel, hs_idx)
    sse = ((ad(hm) - tgt[layer]) ** 2).sum()
    aux_sse = None
    if aux:
        with torch.no_grad():
            ha = gather_h(out, recs, sel, 5 + aux[0])
            aux_sse = ((aux[1](ha) - tgt[aux[0]]) ** 2).sum()
    return sse / tgt[layer].numel(), sse.detach(), tgt, aux_sse


@torch.no_grad()
def ridge_init(backbone, ad, layer, hs_idx, student, games, recs, idx, mu,
               sd, lam=10.0, batch=8):
    backbone.eval()
    Hs, Ys = [], []
    for i in range(0, len(idx), batch):
        sel = idx[i:i + batch]
        tgt = batch_targets(student, games, recs, sel, mu, sd, [layer])
        lens = [len(recs[j]["ids"]) for j in sel]
        Tlen = max(lens)
        ids = torch.full((len(sel), Tlen), 151643, dtype=torch.long)
        for jj, j in enumerate(sel):
            ids[jj, :lens[jj]] = recs[j]["ids"].long()
        am = (torch.arange(Tlen, device=DEV)[None]
              < torch.tensor(lens, device=DEV)[:, None]).long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = backbone(input_ids=ids.to(DEV), attention_mask=am,
                           output_hidden_states=True, use_cache=False)
        Hs.append(gather_h(out, recs, sel, hs_idx).cpu())
        Ys.append(tgt[layer].cpu())
    X = torch.cat(Hs).to(DEV)
    Y = torch.cat(Ys).to(DEV)
    X1 = torch.cat([X, torch.ones(len(X), 1, device=DEV)], 1)
    G = X1.T @ X1 + lam * torch.eye(X1.shape[1], device=DEV)
    Wb = torch.linalg.solve(G, X1.T @ Y)
    ad.weight.copy_(Wb[:-1].T)
    ad.bias.copy_(Wb[-1])
    backbone.train()
    print(f"ridge-init adapter from {len(X)} tokens")


@torch.no_grad()
def evaluate(backbone, ad, layer, student, games, recs, idx, mu, sd,
             aux=None, batch=4, hs_idx=None):
    backbone.eval()
    acc = {l: [0.0, 0.0, 0.0] for l in ([layer] + ([aux[0]] if aux else []))}
    n = 0
    for i in range(0, len(idx), batch):
        _, s, tgt, aux_sse = forward_batch(backbone, ad, layer, student,
                                           games, recs, idx[i:i+batch],
                                           mu, sd, aux, hs_idx)
        acc[layer][0] += s
        if aux:
            acc[aux[0]][0] += aux_sse
        for l in acc:
            acc[l][1] += tgt[l].sum()
            acc[l][2] += (tgt[l] ** 2).sum()
        n += tgt[layer].numel()
    backbone.train()
    out = {}
    for l, (sse, ssum, ssq) in acc.items():
        mean = ssum / n
        out[l] = (1 - sse / (ssq - n * mean * mean)).item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter-lr", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--readout-hs", type=int, default=None,
                    help="override hidden-state index (default 5+layer)")
    ap.add_argument("--freeze-below", type=int, default=None,
                    help="freeze transformer blocks with index < N")
    ap.add_argument("--ridge-init", type=int, default=0,
                    help="closed-form init adapter from N train seqs")
    ap.add_argument("--init-ckpt", default=None)
    ap.add_argument("--fmt", default="r4t", choices=["r4t", "d04"])
    ap.add_argument("--run-name", default="armF_movesc0")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_dir = Path(f"checkpoints/{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]
    if args.fmt == "d04":
        import build_d04
        recs = build_d04.build_seqs_d(tok, games)
    else:
        recs = R4T.build_seqs_t(tok, games)
    print(f"{len(recs)} seqs, maxlen {max(len(r['ids']) for r in recs)}")

    val = [i for i in range(len(recs)) if recs[i]["gi"] % 15 == 0][:args.n_val]
    train = [i for i in range(len(recs)) if recs[i]["gi"] % 15 != 0]
    cnn = W.load_model()
    student = R4.load_student(cnn)
    mu, sd = R4.c_stats(student, games)
    backbone = T.load_backbone()
    ad = nn.Linear(2048, K).to(DEV)
    aux = None
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location="cpu",
                        weights_only=False)
        missing, _ = backbone.load_state_dict(
            {k: v.float() for k, v in ck["backbone"].items()}, strict=False)
        assert not [m for m in missing if "rotary" not in m], missing
        prev_layer = ck["args"].get("layer", 0)
        prev_ad = nn.Linear(2048, K).to(DEV)
        prev_ad.load_state_dict({k: v.float() for k, v in ck["ad"].items()})
        if prev_layer == args.layer:
            ad = prev_ad
        else:
            prev_ad.requires_grad_(False)
            aux = (prev_layer, prev_ad)
        print(f"warm start from {args.init_ckpt} (step {ck.get('step')}, "
              f"saved layer {prev_layer}, training layer {args.layer})")
    backbone.train()
    if args.freeze_below is not None:
        for i in range(args.freeze_below):
            backbone.layers[i].requires_grad_(False)
        print(f"froze blocks 0..{args.freeze_below - 1}")

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
    L = args.layer
    HS = args.readout_hs if args.readout_hs is not None else 5 + L
    if args.ridge_init:
        ridge_init(backbone, ad, L, HS, student, games, recs,
                   train[:args.ridge_init], mu, sd)

    def log_eval(step):
        r2 = evaluate(backbone, ad, L, student, games, recs, val, mu, sd, aux,
                      hs_idx=HS)
        hist.append({"step": step} | {f"c{l}_r2": v for l, v in r2.items()})
        print(f"step {step} " + " ".join(
            f"val c{l} R2 {v:.4f}" for l, v in sorted(r2.items())), flush=True)
        if not args.smoke:
            wandb.log({f"val/r2_c{l}": v for l, v in r2.items()},
                      step=max(step, 1))
        return r2

    log_eval(0)
    if args.smoke:
        loss, _, _, _ = forward_batch(backbone, ad, L, student, games, recs,
                                      train[:args.batch], mu, sd, aux, HS)
        loss.backward()
        opt.step()
        print(f"smoke loss {loss.item():.3f}, "
              f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}GB")
        return

    rng = random.Random(0)
    step, t0 = 0, time.time()
    while step < args.steps:
        sel = rng.sample(train, args.batch)
        loss, _, _, _ = forward_batch(backbone, ad, L, student, games, recs,
                                      sel, mu, sd, hs_idx=HS)
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
