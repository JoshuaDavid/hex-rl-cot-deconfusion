"""P78: teach the tx-contained Qwen to SAY the guest's move.

Full Qwen3-1.7B; blocks 0-16 optionally replaced by P76 containment bottom.
Frozen: embeddings, blocks 0-16, lm_head (tied). Trainable: blocks 17-27 +
final norm (P70 scope=top). All-token CE on
  render_with_regs(board) + "Next move: <RowCol>"   (Row A-K, Col 01-11,
canonical frame; label = guest masked argmax). Eval: greedy 4-token emission
on held-out boards -> parse -> top1 vs guest argmax + legality.

Usage:
  /venv/main/bin/python armF/train_p78ft.py --bottom contained --tag p78_cont
  /venv/main/bin/python armF/train_p78ft.py --bottom original  --tag p78_orig
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from p75_baselines import render_with_regs, load_guest  # noqa: E402
from tx_train import boards_to_states  # noqa: E402
import qwen_embed as Q  # noqa: E402

DEV = "cuda"
# NB: prompt must NOT end with a trailing space — BPE merges the space into
# the answer token (' F'), which desyncs prompt-length offsets and makes
# generation off-distribution. Answer therefore carries the leading space.
PROMPT_TAIL = "Next move:"


def cell_str(idx):
    return f"{chr(65 + idx // 11)}{idx % 11 + 1:02d}"


def parse_cell(s):
    s = s.strip()
    if len(s) < 3 or not ("A" <= s[0] <= "K") or not s[1:3].isdigit():
        return None
    col = int(s[1:3])
    if not 1 <= col <= 11:
        return None
    return (ord(s[0]) - 65) * 11 + (col - 1)


def load_model(bottom, ckpt="checkpoints/armF_p76/best.pt"):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(Q.QWEN,
                                                torch_dtype=torch.bfloat16)
    if bottom == "contained":
        bk = torch.load(ckpt, map_location="cpu", weights_only=False)["backbone"]
        sd = model.model.state_dict()
        loaded = 0
        for k, v in bk.items():
            if k.startswith("layers.") and int(k.split(".")[1]) < 17:
                sd[k].copy_(v)
                loaded += 1
        print(f"contained bottom: {loaded} tensors loaded", flush=True)
    model.model.embed_tokens.weight.requires_grad_(False)
    for i, blk in enumerate(model.model.layers):
        blk.requires_grad_(i >= 17)
    model.model.norm.requires_grad_(True)
    model.lm_head.weight.requires_grad_(False)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(DEV)
    return model


@torch.no_grad()
def guest_labels(guest, boards_u8):
    st = boards_to_states(boards_u8)
    lg = guest(st) - 1000.0 * (st > 0).float()
    return lg.argmax(1)


def make_batch(tok, boards_u8, labels):
    """Loss on ANSWER tokens only. Safe here (unlike P60's masked-loss trap):
    the prompt is always teacher-provided at eval, never free-generated."""
    prompts = [render_with_regs(b)[0] + PROMPT_TAIL for b in boards_u8.cpu()]
    fulls = [p + " " + cell_str(l.item()) for p, l in zip(prompts, labels.cpu())]
    p_lens = [len(tok(p, add_special_tokens=False)["input_ids"])
              for p in prompts]
    enc = tok(fulls, return_tensors="pt", padding=True,
              add_special_tokens=False)
    ids = enc["input_ids"]
    lab = torch.full_like(ids, -100)
    for i in range(len(ids)):
        n = int(enc["attention_mask"][i].sum())
        lab[i, p_lens[i]:n] = ids[i, p_lens[i]:n]
    return ids, enc["attention_mask"], lab, torch.tensor(p_lens)


@torch.no_grad()
def gen_eval(model, tok, guest, boards_u8, batch=32):
    model.eval()
    top1 = legal = parsed = tot = 0
    refs = guest_labels(guest, boards_u8.to(DEV))
    occ = (boards_to_states(boards_u8.to(DEV)) > 0)
    for i in range(0, len(boards_u8), batch):
        bb = boards_u8[i:i + batch]
        prompts = [render_with_regs(b)[0] + PROMPT_TAIL for b in bb.cpu()]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left", add_special_tokens=False)
        out = model.generate(input_ids=enc["input_ids"].to(DEV),
                             attention_mask=enc["attention_mask"].to(DEV),
                             max_new_tokens=4, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        for j in range(len(bb)):
            txt = tok.decode(out[j, enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            c = parse_cell(txt)
            k = i + j
            if c is not None:
                parsed += 1
                if not occ[k, c]:
                    legal += 1
                if c == refs[k].item():
                    top1 += 1
            tot += 1
    model.train()
    return top1 / tot, legal / tot, parsed / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bottom", choices=["contained", "original"],
                    default="contained")
    ap.add_argument("--ckpt", default="checkpoints/armF_p76/best.pt")
    ap.add_argument("--aux-wt", type=float, default=0.0,
                    help=">0: auxiliary KL at hs[--aux-layer] @ last prompt "
                         "token vs guest final logits (P79R placement signal)")
    ap.add_argument("--aux-layer", type=int, default=22)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--tag", default="p78")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.eval_every, args.n_eval = 30, 15, 32
    torch.manual_seed(0)

    boards = torch.load("armF/data/tx_positions.pt",
                        weights_only=False)["boards"]
    perm = torch.randperm(len(boards), generator=torch.Generator().manual_seed(2))
    train_b = boards[perm[:200000]]
    val_b = boards[perm[200000:200000 + args.n_eval]]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Q.QWEN)
    guest = load_guest()
    model = load_model(args.bottom, args.ckpt)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"bottom={args.bottom}, trainable {n_tr/1e6:.0f}M", flush=True)

    use_wandb = not args.no_wandb and not args.smoke
    if use_wandb:
        import wandb
        wandb.init(project="hex-rl-cot-deconfusion", name=f"armF_{args.tag}",
                   config=vars(args))

    aux = None
    if args.aux_wt > 0:
        aux = torch.nn.Linear(2048, 121).to(DEV)
    groups = [{"params": [p for p in model.parameters() if p.requires_grad],
               "lr": args.lr}]
    if aux is not None:
        groups.append({"params": aux.parameters(), "lr": 1e-3})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    t0 = time.time()
    hist = []
    model.train()
    for s in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        sel = torch.randint(0, len(train_b), (args.batch,))
        bb = train_b[sel]
        labels = guest_labels(guest, bb.to(DEV))
        ids, am, lab, p_lens = make_batch(tok, bb, labels)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids.to(DEV), attention_mask=am.to(DEV),
                        labels=lab.to(DEV),
                        output_hidden_states=aux is not None)
        loss = out.loss
        aux_kl = torch.zeros((), device=DEV)
        if aux is not None:
            hlast = out.hidden_states[args.aux_layer][
                torch.arange(len(bb), device=DEV), (p_lens - 1).to(DEV)]
            st = boards_to_states(bb.to(DEV))
            with torch.no_grad():
                g = guest(st) - 1000.0 * (st > 0).float()
            tp = torch.softmax(g, -1)
            aux_kl = (tp * (torch.log_softmax(g, -1)
                            - torch.log_softmax(aux(hlast.float()), -1))
                      ).sum(-1).mean()
            loss = loss + args.aux_wt * aux_kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad]
            + (list(aux.parameters()) if aux is not None else []), 1.0)
        opt.step()
        if (s + 1) % 50 == 0 and use_wandb:
            import wandb
            wandb.log({"loss": out.loss.item(), "aux_kl": aux_kl.item()},
                      step=s + 1)
        if (s + 1) % args.eval_every == 0 or s + 1 == args.steps:
            top1, legalr, parsedr = gen_eval(model, tok, guest, val_b)
            aux_t1 = float("nan")
            if aux is not None:
                model.eval()
                hits = 0
                with torch.no_grad():
                    refs = guest_labels(guest, val_b.to(DEV))
                    for i in range(0, len(val_b), 32):
                        vb = val_b[i:i + 32]
                        vids, vam, _vl, vpl = make_batch(
                            tok, vb, refs[i:i + 32])
                        vo = model(input_ids=vids.to(DEV),
                                   attention_mask=vam.to(DEV),
                                   output_hidden_states=True)
                        h = vo.hidden_states[args.aux_layer][
                            torch.arange(len(vb), device=DEV),
                            (vpl - 1).to(DEV)]
                        hits += (aux(h.float()).argmax(1)
                                 == refs[i:i + 32]).sum().item()
                aux_t1 = hits / len(val_b)
                model.train()
            hist.append({"step": s + 1, "top1": round(top1, 4),
                         "legal": round(legalr, 4),
                         "parsed": round(parsedr, 4),
                         "aux_top1": None if aux is None
                         else round(aux_t1, 4)})
            print(f"step {s+1}/{args.steps} loss {out.loss.item():.4f} "
                  f"aux_kl {aux_kl.item():.4f} aux_top1 {aux_t1:.3f} "
                  f"gen top1 {top1:.3f} legal {legalr:.3f} parsed "
                  f"{parsedr:.3f} ({(s+1)/(time.time()-t0):.2f} it/s)",
                  flush=True)
            if use_wandb:
                import wandb
                wandb.log({"gen_top1": top1, "gen_legal": legalr,
                           "aux_top1": aux_t1}, step=s + 1)

    outdir = Path("checkpoints/armF_p78")
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save({"top": {k: v.to(torch.bfloat16)
                        for k, v in model.state_dict().items()
                        if any(f"layers.{i}." in k for i in range(17, 28))
                        or "model.norm" in k},
                "bottom": args.bottom, "hist": hist},
               outdir / f"{args.tag}.pt")
    Path(f"armF/results/{args.tag}_hist.json").write_text(
        json.dumps(hist, indent=1))
    print(f"done in {(time.time()-t0)/3600:.2f}h", flush=True)
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
