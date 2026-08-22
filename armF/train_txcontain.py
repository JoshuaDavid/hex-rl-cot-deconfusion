"""P76: joint containment of txT12 (transformer guest) in Qwen3-1.7B.

Mirror of train_containment.py (r1 recipe: fp32 + bf16 autocast, truncated
backbone, frozen embeddings, cosine, ridge warm-start, batch 16, lr 1e-5 /
adapter 1e-3, 9k steps) with:
  guest    txT12 h_l (l=0..12), 128 tokens (121 cells + 7 registers) x 1024,
           read at copy-2 cell tokens + appended register-slot tokens,
           h_l <-> hidden_states[5+l] (backbone truncated to 17 blocks).
  guards   stream-norm penalty (P62: normalized-MSE silently collapses hs
           norms; pin per-layer RMS at aligned tokens to init value) and
           spike-skip with abort (P52-era: shallow containment spikes).

Usage: /venv/main/bin/python armF/train_txcontain.py --steps 9000 --tag p76
"""
import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from p75_baselines import (batch_prompts, guest_acts, load_guest,  # noqa: E402
                           render_with_regs)
from tx_train import boards_to_states  # noqa: E402
import qwen_embed as Q  # noqa: E402

DEV = "cuda"
L = 13
QLAYERS = list(range(5, 5 + L))  # hs[5..17]
D_G = 1024


def load_backbone(train_blocks=17, random_init=False):
    """hidden_states: [emb, out0..out15, norm(out16)]; norm -> Identity so
    entry 17 (deepest read) is raw out16."""
    from transformers import AutoConfig, AutoModel
    if random_init:
        torch.manual_seed(0)
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-1.7B")
        m = AutoModel.from_config(cfg, torch_dtype=torch.float32)
    else:
        m = AutoModel.from_pretrained("Qwen/Qwen3-1.7B",
                                      torch_dtype=torch.float32)
    m.layers = nn.ModuleList(list(m.layers)[:train_blocks])
    m.config.num_hidden_layers = train_blocks
    m.norm = nn.Identity()
    m.embed_tokens.weight.requires_grad_(False)
    m.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    m.to(DEV)
    return m


class Adapters(nn.Module):
    def __init__(self, H=2048):
        super().__init__()
        self.maps = nn.ModuleList([nn.Linear(H, D_G) for _ in range(L)])

    def warm_start(self, probe_pt):
        d = torch.load(probe_pt, weights_only=False)
        A = d["A"]  # (13, 2049, 1024)
        for l in range(L):
            self.maps[l].weight.data.copy_(A[l, :-1].T)
            self.maps[l].bias.data.copy_(A[l, -1])
        return d["mu"].to(DEV), d["sd"].to(DEV)


def bf16_sd(module):
    return {k: v.to(torch.bfloat16) for k, v in module.state_dict().items()}


def gather_tok(h, idx):
    return torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))


EMIT_TAIL = "Next move:"


def hiddens_at(backbone, tok, boards_u8, tail=""):
    """Returns (hidden_states, aligned_idx, last_idx). With tail, prompts end
    at the emission site ("...Next move:") and last_idx points at it."""
    if not tail:
        ids, am, idxs = batch_prompts(tok, boards_u8)
    else:
        texts, all_offs = [], []
        for b in boards_u8.cpu():
            text, offs = render_with_regs(b)
            texts.append(text + tail)
            all_offs.append(offs)
        enc = tok(texts, return_offsets_mapping=True, padding=True,
                  return_tensors="pt", add_special_tokens=False)
        idxs = torch.zeros(len(texts), 128, dtype=torch.long)
        for i in range(len(texts)):
            starts = {}
            for tj, (a, bnd) in enumerate(enc["offset_mapping"][i].tolist()):
                if a == bnd:
                    continue
                for o in range(a, bnd):
                    starts[o] = tj
            idxs[i] = torch.tensor([starts[o] for o in all_offs[i]])
        ids, am = enc["input_ids"], enc["attention_mask"]
    out = backbone(input_ids=ids.to(DEV), attention_mask=am.to(DEV),
                   output_hidden_states=True, use_cache=False)
    last_idx = (am.sum(1) - 1).to(DEV)
    return out.hidden_states, idxs.to(DEV), last_idx


@torch.no_grad()
def guest_final_logits(guest, boards_u8):
    st = boards_to_states(boards_u8)
    return guest(st) - 1000.0 * (st > 0).float()


@torch.no_grad()
def evaluate(backbone, adapters, tok, guest, boards, mu, sd, batch=32,
             emit=None, tail="", cellhead=None):
    backbone.eval()
    sse = torch.zeros(L, device=DEV)
    ssum = torch.zeros(L, D_G, device=DEV)
    ssq = torch.zeros(L, D_G, device=DEV)
    n = 0
    emit_top1 = emit_n = 0
    for i in range(0, len(boards), batch):
        bb = boards[i:i + batch].to(DEV)
        z = (guest_acts(guest, bb) - mu[:, None, None]) / sd[:, None, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs, idx, last = hiddens_at(backbone, tok, bb, tail)
        for l in range(L):
            pred = adapters.maps[l](gather_tok(hs[5 + l].float(), idx))
            y = z[l]
            sse[l] += ((pred - y) ** 2).sum()
            ssum[l] += y.sum(dim=(0, 1))
            ssq[l] += (y * y).sum(dim=(0, 1))
        if emit is not None:
            hlast = hs[5 + L - 1][torch.arange(len(bb), device=DEV), last]
            ref = guest_final_logits(guest, bb).argmax(1)
            emit_top1 += (emit(hlast.float()).argmax(1) == ref).sum().item()
            emit_n += len(bb)
        if cellhead is not None:
            hc = gather_tok(hs[5 + L - 1].float(), idx)[:, :121]
            pred = cellhead(hc).squeeze(-1)
            pred = pred - 1000.0 * (boards_to_states(bb) > 0).float()
            ref = guest_final_logits(guest, bb).argmax(1)
            emit_top1 += (pred.argmax(1) == ref).sum().item()
            emit_n += len(bb)
        n += len(bb) * 128
    mean = ssum / n
    sstot = (ssq - n * mean * mean).sum(-1)
    backbone.train()
    r2 = (1 - sse / sstot).cpu()
    if emit is not None or cellhead is not None:
        return r2, emit_top1 / emit_n
    return r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--adapter-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--norm-wt", type=float, default=0.05)
    ap.add_argument("--skip-mult", type=float, default=10.0)
    ap.add_argument("--max-consec-skips", type=int, default=200)
    ap.add_argument("--n-boards", type=int, default=120000)
    ap.add_argument("--n-val", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--data", default="armF/data/tx_positions.pt")
    ap.add_argument("--emit-wt", type=float, default=0.0,
                    help=">0 enables emission-site head (KL vs guest final "
                         "logits at the last 'Next move:' token)")
    ap.add_argument("--emit-lr", type=float, default=1e-3)
    ap.add_argument("--cellhead-wt", type=float, default=0.0,
                    help=">0: per-cell scalar logit head at each cell token "
                         "(fixed-slot), assembled KL vs guest masked softmax")
    ap.add_argument("--resume-ckpt", default=None,
                    help="warm-restart backbone+adapters (weights only)")
    ap.add_argument("--probe", default="armF/results/p75_probe_pretrained.pt")
    ap.add_argument("--random-init", action="store_true")
    ap.add_argument("--tag", default="p76")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.eval_every, args.n_boards = 30, 10, 2000
        args.n_val = 128
    torch.manual_seed(0)

    all_boards = torch.load(args.data, weights_only=False)["boards"]
    perm = torch.randperm(len(all_boards),
                          generator=torch.Generator().manual_seed(1))
    train_b = all_boards[perm[: args.n_boards]]
    val_b = all_boards[perm[args.n_boards: args.n_boards + args.n_val]]
    print(f"train {len(train_b)} val {len(val_b)}", flush=True)

    guest = load_guest()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Q.QWEN)
    backbone = load_backbone(random_init=args.random_init)
    adapters = Adapters().to(DEV)
    mu, sd = adapters.warm_start(args.probe)
    print("adapters warm-started from probe", flush=True)
    tail = EMIT_TAIL if args.emit_wt > 0 else ""
    emit = torch.nn.Linear(2048, 121).to(DEV) if args.emit_wt > 0 else None
    cellhead = (torch.nn.Linear(2048, 1).to(DEV)
                if args.cellhead_wt > 0 else None)
    if args.resume_ckpt:
        ck = torch.load(args.resume_ckpt, map_location=DEV, weights_only=False)
        backbone.load_state_dict({k: v.float() for k, v in ck["backbone"].items()})
        adapters.load_state_dict(ck["adapters"])
        mu, sd = ck["mu"].to(DEV), ck["sd"].to(DEV)
        if emit is not None and "emit" in ck:
            emit.load_state_dict(ck["emit"])
        print(f"warm-restarted from {args.resume_ckpt} "
              f"(step {ck.get('step')})", flush=True)

    # norm-guard reference: per-layer RMS at aligned tokens, at init
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hs, idx, _last = hiddens_at(backbone, tok, val_b[:128].to(DEV), tail)
        rms_base = torch.stack([
            gather_tok(hs[5 + l].float(), idx).pow(2).mean(-1).sqrt().mean()
            for l in range(L)])
    print("rms_base:", " ".join(f"{v:.1f}" for v in rms_base.tolist()),
          flush=True)

    use_wandb = not args.no_wandb and not args.smoke
    if use_wandb:
        import wandb
        wandb.init(project="hex-rl-cot-deconfusion", name=f"armF_{args.tag}",
                   config=vars(args))

    qwen_params = [p for p in backbone.parameters() if p.requires_grad]
    groups = [
        {"params": qwen_params, "lr": args.lr},
        {"params": adapters.parameters(), "lr": args.adapter_lr},
    ]
    if emit is not None:
        groups.append({"params": emit.parameters(), "lr": args.emit_lr})
    if cellhead is not None:
        groups.append({"params": cellhead.parameters(), "lr": args.emit_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)

    def lr_scale(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    outdir = Path(f"checkpoints/armF_{args.tag}")
    outdir.mkdir(parents=True, exist_ok=True)
    hist = collections.deque(maxlen=200)
    consec_skips = 0
    best = -1e9
    r2_log = []
    t0 = time.time()
    step = 0
    backbone.train()
    while step < args.steps:
        sel = torch.randint(0, len(train_b), (args.batch,))
        bb = train_b[sel].to(DEV)
        with torch.no_grad():
            z = (guest_acts(guest, bb) - mu[:, None, None]) / sd[:, None, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs, idx, last = hiddens_at(backbone, tok, bb, tail)
        loss = 0
        rms_pen = 0
        for l in range(L):
            h = gather_tok(hs[5 + l].float(), idx)
            loss = loss + ((adapters.maps[l](h) - z[l]) ** 2).mean()
            rms = h.pow(2).mean(-1).sqrt().mean()
            rms_pen = rms_pen + (rms / rms_base[l] - 1.0) ** 2
        loss = loss / L
        rms_pen = rms_pen / L
        emit_kl = torch.zeros((), device=DEV)
        if emit is not None:
            hlast = hs[5 + L - 1][torch.arange(len(bb), device=DEV), last]
            with torch.no_grad():
                tgt = guest_final_logits(guest, bb)
            tp = torch.softmax(tgt, -1)
            emit_kl = (tp * (torch.log_softmax(tgt, -1)
                             - torch.log_softmax(emit(hlast.float()), -1))
                       ).sum(-1).mean()
        cell_kl = torch.zeros((), device=DEV)
        if cellhead is not None:
            hc = gather_tok(hs[5 + L - 1].float(), idx)[:, :121]  # cells only
            pred = cellhead(hc).squeeze(-1)  # (B,121)
            occ = (boards_to_states(bb) > 0).float()
            pred = pred - 1000.0 * occ
            with torch.no_grad():
                tgt = guest_final_logits(guest, bb)
            tp = torch.softmax(tgt, -1)
            cell_kl = (tp * (torch.log_softmax(tgt, -1)
                             - torch.log_softmax(pred, -1))).sum(-1).mean()
        total = (loss + args.norm_wt * rms_pen + args.emit_wt * emit_kl
                 + args.cellhead_wt * cell_kl)

        med = sorted(hist)[len(hist) // 2] if len(hist) >= 50 else None
        if med is not None and loss.item() > args.skip_mult * med:
            consec_skips += 1
            if consec_skips > args.max_consec_skips:
                print(f"ABORT: {consec_skips} consecutive skips at step {step}",
                      flush=True)
                break
            opt.zero_grad(set_to_none=True)
            continue
        consec_skips = 0
        hist.append(loss.item())

        opt.zero_grad(set_to_none=True)
        total.backward()
        train_params = (list(qwen_params) + list(adapters.parameters())
                        + (list(emit.parameters()) if emit is not None else [])
                        + (list(cellhead.parameters())
                           if cellhead is not None else []))
        gn = torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        sched.step()
        step += 1

        if step % 50 == 0 and use_wandb:
            import wandb
            wandb.log({"loss": loss.item(), "rms_pen": rms_pen.item(),
                       "emit_kl": emit_kl.item(), "grad_norm": gn.item(),
                       "lr": sched.get_last_lr()[0]}, step=step)
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(backbone, adapters, tok, guest, val_b, mu, sd,
                          emit=emit, tail=tail, cellhead=cellhead)
            has_head = emit is not None or cellhead is not None
            r2, emit_t1 = ev if has_head else (ev, float("nan"))
            r2_log.append({"step": step, "r2": [round(v, 4) for v in r2.tolist()],
                           "emit_top1": None if emit is None
                           else round(emit_t1, 4)})
            sps = step / (time.time() - t0)
            print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                  f"rms_pen {rms_pen.item():.4f} emit_kl {emit_kl.item():.4f} "
                  f"cell_kl {cell_kl.item():.4f} "
                  f"head_top1 {emit_t1:.3f} mean R2 {r2.mean():.4f} "
                  f"[{' '.join(f'{v:.3f}' for v in r2.tolist())}] "
                  f"({sps:.2f} it/s)", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"val_r2_mean": r2.mean().item(),
                           "emit_top1": emit_t1,
                           **{f"val_r2_h{l}": r2[l].item() for l in range(L)}},
                          step=step)
            score = r2.mean().item() + (emit_t1 if has_head else 0)
            extra = {**({"emit": emit.state_dict()} if emit is not None else {}),
                     **({"cellhead": cellhead.state_dict()}
                        if cellhead is not None else {})}
            if score > best:
                best = score
                torch.save({"backbone": bf16_sd(backbone),
                            "adapters": adapters.state_dict(), **extra,
                            "mu": mu.cpu(), "sd": sd.cpu(), "step": step,
                            "r2": r2.tolist()}, outdir / "best.pt")
        if step % args.save_every == 0 or step == args.steps:
            extra = {**({"emit": emit.state_dict()} if emit is not None else {}),
                     **({"cellhead": cellhead.state_dict()}
                        if cellhead is not None else {})}
            torch.save({"backbone": bf16_sd(backbone),
                        "adapters": adapters.state_dict(), **extra,
                        "mu": mu.cpu(), "sd": sd.cpu(), "step": step},
                       outdir / "last.pt")
            Path(f"armF/results/{args.tag}_r2log.json").write_text(
                json.dumps(r2_log, indent=1))

    print(f"done {step} steps in {(time.time()-t0)/3600:.2f}h; "
          f"best mean R2 {best:.4f}", flush=True)
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
