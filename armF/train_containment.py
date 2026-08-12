"""Arm F joint fine-tune: Qwen3-1.7B (blocks 0..23, embeddings frozen) + 19
affine adapters, loss = mean_l MSE(A_l h_{5+l}, z_l_normalized). No LM loss.

z_l <-> hidden_states[5+l] (HF convention: hidden_states[j] = residual stream
entering block j, i.e. output of block j-1; deepest read = output of block 22).
Backbone truncated to 24 blocks so all read points stay pre-final-norm.

Adapters warm-started from the frozen-probe ridge solution; step-0 val R^2 must
therefore reproduce armF/results/probe_frozen.json (pipeline check).
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

DEV = "cuda"
L = 19
QLAYERS = list(range(5, 24))


def cnn_targets(cnn, boards_f, mu, sd):
    acts = W.dump_acts(cnn, boards_f)
    z = torch.stack([a.permute(0, 2, 3, 1).reshape(-1, 121, 64) for a in acts])
    return (z - mu[:, None, None, :]) / sd[:, None, None, :]


def load_backbone(train_blocks=23, random_init=False):
    """23 blocks (0..22): hidden_states entries [emb, out0..out21, norm(out22)].
    norm replaced by Identity so entry 23 (deepest read, z18) is raw out22 —
    matching the full-model probe convention hidden_states[5+l] = out_{4+l}."""
    from transformers import AutoConfig, AutoModel
    if random_init:
        torch.manual_seed(0)
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")
        m = AutoModel.from_config(cfg, torch_dtype=torch.float32)
    else:
        m = AutoModel.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.float32)
    m.layers = nn.ModuleList(list(m.layers)[:train_blocks])
    m.config.num_hidden_layers = train_blocks
    m.norm = nn.Identity()
    m.embed_tokens.weight.requires_grad_(False)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.to(DEV)
    return m


class Adapters(nn.Module):
    def __init__(self, H=2048, C=64):
        super().__init__()
        self.maps = nn.ModuleList([nn.Linear(H, C) for _ in range(L)])

    def warm_start(self, probe_pt):
        d = torch.load(probe_pt, weights_only=False)
        A = d["A"]  # (19, 2049, 64)
        for l in range(L):
            self.maps[l].weight.data.copy_(A[l, :-1].T)
            self.maps[l].bias.data.copy_(A[l, -1])
        return d["mu"], d["sd"]


def forward_hiddens(backbone, ids, am):
    out = backbone(input_ids=ids, attention_mask=am, output_hidden_states=True,
                   use_cache=False)
    return out.hidden_states  # tuple


def gather_cells(h, cell):
    return torch.gather(h, 1, cell.unsqueeze(-1).expand(-1, -1, h.shape[-1]))


@torch.no_grad()
def evaluate(backbone, adapters, cnn, tokens, boards, idx, mu, sd, batch=32):
    backbone.eval()
    sse = torch.zeros(L, device=DEV)
    ssum = torch.zeros(L, 64, device=DEV)
    ssq = torch.zeros(L, 64, device=DEV)
    n = 0
    for i in range(0, len(idx), batch):
        sel = idx[i:i+batch]
        ids = tokens["input_ids"][sel].long().to(DEV)
        Tm = int(tokens["lens"][sel].max())
        ids = ids[:, :Tm]
        am = (torch.arange(Tm, device=DEV)[None] < tokens["lens"][sel][:, None].to(DEV)).long()
        cell = tokens["cell_idx"][sel].long().to(DEV)
        z = cnn_targets(cnn, boards[sel].float().to(DEV), mu, sd)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = forward_hiddens(backbone, ids, am)
        for l in range(L):
            h = gather_cells(hs[5 + l].float(), cell)
            pred = adapters.maps[l](h)
            y = z[l]
            sse[l] += ((pred - y) ** 2).sum()
            ssum[l] += y.sum(dim=(0, 1))
            ssq[l] += (y * y).sum(dim=(0, 1))
        n += len(sel) * 121
    mean = ssum / n
    sstot = (ssq - n * mean * mean).sum(-1)
    r2 = 1 - sse / sstot
    backbone.train()
    return r2.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--run-name", default="armF_containment_r1")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--probe", default=None,
                    help="warm-start probe .pt (default matches init type)")
    ap.add_argument("--freeze-qwen", action="store_true", help="adapters only (sanity)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir or f"checkpoints/{args.run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    boards = torch.load("armF/data/positions.pt", weights_only=False)["boards"]
    tokens = torch.load("armF/data/tokens.pt", weights_only=False)
    N = len(boards)
    val_idx = torch.arange(6000, 7000)
    train_idx = torch.cat([torch.arange(0, 6000), torch.arange(7000, N)])
    print(f"N {N} train {len(train_idx)} val {len(val_idx)}")

    cnn = W.load_model()
    backbone = load_backbone(random_init=args.random_init)
    adapters = Adapters().to(DEV)
    probe = args.probe or ("armF/results/probe_frozen_randinit.pt" if args.random_init
                           else "armF/results/probe_frozen.pt")
    print(f"warm-start probe: {probe}")
    mu, sd = adapters.warm_start(probe)
    mu, sd = mu.to(DEV), sd.to(DEV)
    if args.freeze_qwen:
        for p in backbone.parameters():
            p.requires_grad_(False)
    backbone.train()

    qwen_params = [p for p in backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": qwen_params, "lr": args.lr},
        {"params": adapters.parameters(), "lr": args.adapter_lr},
    ], weight_decay=0.0)

    def lr_scale(step):
        if step < args.warmup:
            return step / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    import wandb
    wandb.init(project="hex-rl-cot-deconfusion", name=args.run_name,
               config=vars(args))

    r2 = evaluate(backbone, adapters, cnn, tokens, boards, val_idx, mu, sd)
    print("step 0 val R2 (should ~match frozen probe):",
          [round(v, 3) for v in r2.tolist()], flush=True)
    wandb.log({"val/r2_mean": r2.mean().item(),
               **{f"val/r2_z{l}": r2[l].item() for l in range(L)}}, step=0)

    g = torch.Generator().manual_seed(0)
    step, t0 = 0, time.time()
    best = -1
    while step < args.steps:
        perm = train_idx[torch.randperm(len(train_idx), generator=g)]
        for i in range(0, len(perm) - args.batch, args.batch):
            sel = perm[i:i+args.batch]
            ids = tokens["input_ids"][sel].long().to(DEV)
            Tm = int(tokens["lens"][sel].max())
            ids = ids[:, :Tm]
            am = (torch.arange(Tm, device=DEV)[None] < tokens["lens"][sel][:, None].to(DEV)).long()
            cell = tokens["cell_idx"][sel].long().to(DEV)
            with torch.no_grad():
                z = cnn_targets(cnn, boards[sel].float().to(DEV), mu, sd)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hs = forward_hiddens(backbone, ids, am)
            loss = 0
            layer_mse = []
            for l in range(L):
                h = gather_cells(hs[5 + l].float(), cell)
                pred = adapters.maps[l](h)
                m = ((pred - z[l]) ** 2).mean()
                layer_mse.append(m.detach())
                loss = loss + m
            loss = loss / L
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                list(qwen_params) + list(adapters.parameters()), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 25 == 0:
                wandb.log({"train/loss": loss.item(), "train/grad_norm": gn.item(),
                           "train/lr": sched.get_last_lr()[0],
                           **{f"train/mse_z{l}": layer_mse[l].item() for l in (0, 9, 18)}},
                          step=step)
                print(f"step {step} loss {loss.item():.4f} gn {gn.item():.2f} "
                      f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
            if step % args.eval_every == 0:
                r2 = evaluate(backbone, adapters, cnn, tokens, boards, val_idx, mu, sd)
                wandb.log({"val/r2_mean": r2.mean().item(),
                           **{f"val/r2_z{l}": r2[l].item() for l in range(L)}}, step=step)
                print(f"step {step} val R2 mean {r2.mean().item():.4f} "
                      f"z0 {r2[0]:.3f} z9 {r2[9]:.3f} z18 {r2[18]:.3f}", flush=True)
                if r2.mean().item() > best:
                    best = r2.mean().item()
                    torch.save({"backbone": {k: v.bfloat16() for k, v in backbone.state_dict().items()},
                                "adapters": adapters.state_dict(),
                                "step": step, "r2": r2, "args": vars(args)},
                               out_dir / "best.pt")
            if step % args.save_every == 0:
                torch.save({"backbone": {k: v.bfloat16() for k, v in backbone.state_dict().items()},
                            "adapters": adapters.state_dict(),
                            "step": step, "args": vars(args)},
                           out_dir / "last.pt")
            if step >= args.steps:
                break
    r2 = evaluate(backbone, adapters, cnn, tokens, boards, val_idx, mu, sd)
    print("FINAL val R2:", [round(v, 4) for v in r2.tolist()])
    (out_dir / "final_r2.json").write_text(json.dumps(
        {"r2": r2.tolist(), "step": step}))
    wandb.finish()


if __name__ == "__main__":
    main()
