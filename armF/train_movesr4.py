"""Arm F r4: render-free containment of the DISTILLED rank-1024 CNN.

Teacher = finger E artifact (BottleneckedCNN, bottleneck_anchored_ext.pt,
1867 Elo @ t=0). Targets = the teacher's NATIVE state c_l = enc_l(z_l) in
R^1024 at all 19 capture points, per-dim normalized; adapters are 19x
Linear(2048 -> 1024) (~40M total vs r3's 301M). Input/readout identical to
r3: numbered move list, hs[5+l] at move (color) tokens, every move token
supervises its own prefix. Data = games.pt + games2.pt (4400 games).
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
import fingerE_bottleneck as B  # noqa: E402

DEV = "cuda"
L = 19
K = 1024
CKPT_DISTILLED = "checkpoints/armF_fingerE/bottleneck_anchored_ext.pt"


def load_student(cnn):
    """Distilled teacher; PCA-init bases are dummy (overwritten by ckpt)."""
    Vs = [torch.zeros(K, B.D_Z) for _ in range(L)]
    mus = [torch.zeros(B.D_Z) for _ in range(L)]
    student = B.BottleneckedCNN(cnn, Vs, mus).to(DEV)
    ck = torch.load(CKPT_DISTILLED, map_location=DEV, weights_only=False)
    student.load_state_dict(ck["state_dict"])
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    return student


@torch.no_grad()
def dump_c(student, x):
    """Native rank-1024 states at all 19 capture points. x: (B,2,13,13)."""
    m = student.inner
    cs = []
    z = m.conv(x)
    for l in range(L):
        c = student.bns[l].enc(z.reshape(len(z), -1))
        cs.append(c)
        if l < L - 1:
            h = student.bns[l].dec(c).reshape(-1, 64, 11, 11)
            z = m.skiplayers[l](h)
    return cs


@torch.no_grad()
def c_stats(student, games, n_boards=20000, batch=256):
    path = Path("armF/results/r4_cstats.pt")
    if path.exists():
        d = torch.load(path, weights_only=False)
        return d["mu"].to(DEV), d["sd"].to(DEV)
    boards = torch.cat([g["boards"].float() for g in games])
    idx = torch.randperm(len(boards),
                         generator=torch.Generator().manual_seed(0))[:n_boards]
    boards = boards[idx]
    s = torch.zeros(L, K, device=DEV)
    ss = torch.zeros(L, K, device=DEV)
    n = 0
    for i in range(0, len(boards), batch):
        bb = boards[i:i + batch].to(DEV)
        for l, c in enumerate(dump_c(student, bb)):
            s[l] += c.sum(0)
            ss[l] += (c * c).sum(0)
        n += len(bb)
    mu = s / n
    sd = (ss / n - mu * mu).clamp_min(1e-8).sqrt().clamp_min(1e-4)
    torch.save({"mu": mu.cpu(), "sd": sd.cpu()}, path)
    return mu, sd


@torch.no_grad()
def batch_targets(student, games, recs, sel, mu, sd):
    """(M, 19, 1024) normalized native states for all prefix plies."""
    boards = [games[recs[i]["gi"]]["boards"].float() for i in sel]
    bb = torch.cat(boards).to(DEV)
    tgt = torch.empty(len(bb), L, K, device=DEV)
    for l, c in enumerate(dump_c(student, bb)):
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
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=60)
    ap.add_argument("--run-name", default="armF_movesr4")
    ap.add_argument("--games2", default="armF/data/games2.pt")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="reservoir control: adapters only")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_dir = Path(f"checkpoints/{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    if args.games2:
        games = games + torch.load(args.games2, weights_only=False)["games"]
    recs = Z.build_seqs(tok, games, "numbered")
    print(f"{len(recs)} seqs, maxlen {max(len(r['ids']) for r in recs)}")
    for r in random.Random(0).sample(recs, 3):
        print("--- sample ---")
        print(r["text"][:220].replace("\n", "\\n"))

    val = [i for i in range(len(recs)) if recs[i]["gi"] % 15 == 0][:args.n_val]
    train = [i for i in range(len(recs)) if recs[i]["gi"] % 15 != 0]
    cnn = W.load_model()
    student = load_student(cnn)
    mu, sd = c_stats(student, games)
    print(f"c stats: mu norm {mu.norm(dim=1).mean():.2f}, "
          f"sd med {sd.median():.3f}")
    backbone = T.load_backbone()
    ads = nn.ModuleList([nn.Linear(2048, K) for _ in range(L)]).to(DEV)
    if args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad_(False)
    backbone.train()

    qwen_params = [p for p in backbone.parameters() if p.requires_grad]
    groups = [{"params": ads.parameters(), "lr": args.adapter_lr}]
    if qwen_params:
        groups.append({"params": qwen_params, "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)

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
        bb = games[0]["boards"][:4].float().to(DEV)
        _, caps = student(bb, capture=True)
        cs = dump_c(student, bb)
        for l in range(L):
            rec = student.bns[l].dec(cs[l]).reshape(-1, 64, 11, 11)
            assert torch.allclose(rec, caps[l], atol=1e-4), l
        print("dump_c == capture forward: OK")

    log_eval(0)
    if args.smoke:
        sel = train[:args.batch]
        loss, _, _ = forward_batch(backbone, ads, student, games, recs, sel,
                                   mu, sd)
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
