"""Joint containment of the last 3 c-layers (Joshua 2026-08-13).

c16/c17/c18 read at hs[21..23], trained JOINTLY (mean of 3 normalized MSEs,
3 ridge-init adapters, blocks 20-22 trainable) on the frozen chain-c15 bottom.
Init = exact state at the start of chain stage c16: blocks 0..19 from
armF_chain_c18/final.pt (frozen ever since), blocks 20..22 pretrained (they
had zero gradient through stages c1..c15). Both arms share the same frozen
bottom, so this isolates within-tail shared supervision from bottom-reshaping.
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
import train_movesc0 as C0  # noqa: E402
import build_d04  # noqa: E402

DEV = "cuda"
K = 1024
LAYERS = [16, 17, 18]  # overwritten from --layers in main()


def pad_batch(recs, sel):
    lens = [len(recs[i]["ids"]) for i in sel]
    Tlen = max(lens)
    ids = torch.full((len(sel), Tlen), 151643, dtype=torch.long)
    for j, i in enumerate(sel):
        ids[j, :lens[j]] = recs[i]["ids"].long()
    am = (torch.arange(Tlen, device=DEV)[None]
          < torch.tensor(lens, device=DEV)[:, None]).long()
    return ids.to(DEV), am


def forward_joint(backbone, ads, student, games, recs, sel, mu, sd):
    tgt = C0.batch_targets(student, games, recs, sel, mu, sd, LAYERS)
    ids, am = pad_batch(recs, sel)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = backbone(input_ids=ids, attention_mask=am,
                       output_hidden_states=True, use_cache=False)
    loss = 0.0
    sses = {}
    for l in LAYERS:
        hm = C0.gather_h(out, recs, sel, 5 + l)
        sse = ((ads[l](hm) - tgt[l]) ** 2).sum()
        sses[l] = sse.detach()
        loss = loss + sse / tgt[l].numel()
    return loss / len(LAYERS), sses, tgt


@torch.no_grad()
def ridge_init_joint(backbone, ads, student, games, recs, idx, mu, sd,
                     lam=10.0, batch=8):
    backbone.eval()
    Hs = {l: [] for l in LAYERS}
    Ys = {l: [] for l in LAYERS}
    for i in range(0, len(idx), batch):
        sel = idx[i:i + batch]
        tgt = C0.batch_targets(student, games, recs, sel, mu, sd, LAYERS)
        ids, am = pad_batch(recs, sel)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = backbone(input_ids=ids, attention_mask=am,
                           output_hidden_states=True, use_cache=False)
        for l in LAYERS:
            Hs[l].append(C0.gather_h(out, recs, sel, 5 + l).cpu())
            Ys[l].append(tgt[l].cpu())
    for l in LAYERS:
        X = torch.cat(Hs[l]).to(DEV)
        Y = torch.cat(Ys[l]).to(DEV)
        X1 = torch.cat([X, torch.ones(len(X), 1, device=DEV)], 1)
        G = X1.T @ X1 + lam * torch.eye(X1.shape[1], device=DEV)
        Wb = torch.linalg.solve(G, X1.T @ Y)
        ads[l].weight.copy_(Wb[:-1].T)
        ads[l].bias.copy_(Wb[-1])
    backbone.train()
    print(f"ridge-init 3 adapters from "
          f"{sum(len(h) for h in Hs[LAYERS[0]])} tokens ({len(idx)} seqs)")


@torch.no_grad()
def evaluate(backbone, ads, student, games, recs, idx, mu, sd, batch=4):
    backbone.eval()
    acc = {l: [0.0, 0.0, 0.0] for l in LAYERS}
    n = 0
    for i in range(0, len(idx), batch):
        _, sses, tgt = forward_joint(backbone, ads, student, games, recs,
                                     idx[i:i + batch], mu, sd)
        for l in LAYERS:
            acc[l][0] += sses[l]
            acc[l][1] += tgt[l].sum()
            acc[l][2] += (tgt[l] ** 2).sum()
        n += tgt[LAYERS[0]].numel()
    backbone.train()
    out = {}
    for l, (sse, ssum, ssq) in acc.items():
        mean = ssum / n
        out[l] = (1 - sse / (ssq - n * mean * mean)).item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=9500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--adapter-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--ridge-init", type=int, default=1200)
    ap.add_argument("--layers", default="16,17,18")
    ap.add_argument("--freeze-below", type=int, default=20)
    ap.add_argument("--truncate-blocks", type=int, default=23,
                    help="drop blocks above the deepest readout (pure speed)")
    ap.add_argument("--bottom-ckpt",
                    default="checkpoints/armF_chain_c18/final.pt",
                    help="'none' = pure pretrained init")
    ap.add_argument("--early-stop-window", type=int, default=4)
    ap.add_argument("--early-stop-delta", type=float, default=0.01)
    ap.add_argument("--run-name", default="armF_jointtail")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    global LAYERS
    LAYERS = [int(x) for x in args.layers.split(",")]
    assert args.truncate_blocks >= 5 + max(LAYERS) - 1 + 1
    out_dir = Path(f"checkpoints/{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]
    recs = build_d04.build_seqs_d(tok, games, numbers=False)
    print(f"{len(recs)} seqs, maxlen {max(len(r['ids']) for r in recs)}")

    val = [i for i in range(len(recs)) if recs[i]["gi"] % 15 == 0][:args.n_val]
    train = [i for i in range(len(recs)) if recs[i]["gi"] % 15 != 0]
    cnn = W.load_model()
    student = R4.load_student(cnn)
    mu, sd = R4.c_stats(student, games)
    backbone = T.load_backbone(train_blocks=args.truncate_blocks)
    FB = args.freeze_below
    if args.bottom_ckpt != "none":
        ck = torch.load(args.bottom_ckpt, map_location="cpu",
                        weights_only=False)
        trainable = range(FB, args.truncate_blocks)
        keep = {k: v.float() for k, v in ck["backbone"].items()
                if not any(k.startswith(f"layers.{i}.") for i in trainable)}
        missing, _ = backbone.load_state_dict(keep, strict=False)
        bad = [m for m in missing if "rotary" not in m
               and not any(m.startswith(f"layers.{i}.") for i in trainable)]
        assert not bad, bad
        print(f"bottom 0..{FB - 1} from {args.bottom_ckpt}; "
              f"blocks {FB}.. pretrained")
    else:
        print("pure pretrained init")
    backbone.train()
    for i in range(FB):
        backbone.layers[i].requires_grad_(False)
    print(f"froze blocks 0..{FB - 1}" if FB else "nothing frozen")

    ads = {l: nn.Linear(2048, K).to(DEV) for l in LAYERS}
    ad_params = [p for a in ads.values() for p in a.parameters()]
    qwen_params = [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": ad_params, "lr": args.adapter_lr},
         {"params": qwen_params, "lr": args.lr}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, step / args.warmup))

    if not args.smoke:
        import wandb
        wandb.init(project="hex-rl-cot-deconfusion", name=args.run_name,
                   config=vars(args))

    hist = []
    if args.ridge_init:
        ridge_init_joint(backbone, ads, student, games, recs,
                         train[:args.ridge_init], mu, sd)

    def log_eval(step):
        r2 = evaluate(backbone, ads, student, games, recs, val, mu, sd)
        mean = sum(r2.values()) / len(r2)
        hist.append({"step": step, "mean_r2": mean}
                    | {f"c{l}_r2": v for l, v in r2.items()})
        print(f"step {step} " + " ".join(
            f"val c{l} R2 {v:.4f}" for l, v in sorted(r2.items()))
            + f" mean {mean:.4f}", flush=True)
        if not args.smoke:
            wandb.log({f"val/r2_c{l}": v for l, v in r2.items()}
                      | {"val/r2_mean": mean}, step=max(step, 1))
        return r2

    log_eval(0)
    if args.smoke:
        loss, _, _ = forward_joint(backbone, ads, student, games, recs,
                                   train[:args.batch], mu, sd)
        loss.backward()
        opt.step()
        print(f"smoke loss {loss.item():.3f}, "
              f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}GB")
        return

    rng = random.Random(0)
    step, t0 = 0, time.time()
    while step < args.steps:
        sel = rng.sample(train, args.batch)
        loss, _, _ = forward_joint(backbone, ads, student, games, recs,
                                   sel, mu, sd)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(qwen_params + ad_params, 1.0)
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
            W_ = args.early_stop_window
            if W_ and len(hist) > W_:
                gain = hist[-1]["mean_r2"] - hist[-1 - W_]["mean_r2"]
                if gain < args.early_stop_delta:
                    print(f"early stop at {step}: window gain {gain:.4f}",
                          flush=True)
                    break
    torch.save({"backbone": {k: v.bfloat16() for k, v in
                             backbone.state_dict().items()},
                "ads": {l: {k: v.bfloat16() for k, v in
                            a.state_dict().items()} for l, a in ads.items()},
                "step": step, "args": vars(args)}, out_dir / "final.pt")
    (out_dir / "final.json").write_text(json.dumps({"hist": hist}))
    wandb.finish()


if __name__ == "__main__":
    main()
