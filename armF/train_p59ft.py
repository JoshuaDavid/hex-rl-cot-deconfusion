"""P60 (P59 rung 3): LM-loss FT on next-move text — does contained state
speed verbalization?

Arms: --init {polish,base} x --scope {full,top}. polish = splice polished
blocks 0..22 into full CausalLM; top = train only blocks 23..27 + norm.
Loss on move-CELL tokens only (color token excluded). Greedy-gen top1 vs
distilled argmax at --gen-steps.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_movesr4 as R4  # noqa: E402
import build_d04  # noqa: E402
import p59_policy_text as P59  # noqa: E402

DEV = "cuda"


def tokenize_games(tok, games, all_tokens=False):
    pre = tok(build_d04.RM.PREAMBLE_M + build_d04.R4X.HDR,
              add_special_tokens=False)["input_ids"]
    out = []
    for g in games:
        ids = list(pre)
        mask = [0] * len(pre)
        for t, mv in enumerate([tuple(m) for m in g["moves"].tolist()]):
            line = (f"\n{build_d04.cell_str(mv)} "
                    f"{'X' if t % 2 == 0 else 'O'}")
            lids = tok(line, add_special_tokens=False)["input_ids"]
            ids += lids
            if all_tokens:
                mask += [1] * len(lids)
            else:
                mask += [1] * (len(lids) - 1) + [0]
        out.append((torch.tensor(ids), torch.tensor(mask)))
    return out


def batch_nll(model, recs, idxs, pad_id):
    seqs = [recs[i] for i in idxs]
    L = max(len(s[0]) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    msk = torch.zeros((len(seqs), L), dtype=torch.bool)
    for j, (i_, m_) in enumerate(seqs):
        ids[j, :len(i_)] = i_
        msk[j, :len(m_)] = m_.bool()
    ids, msk = ids.to(DEV), msk.to(DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids).logits
    lg = logits[:, :-1].float()
    tgt = ids[:, 1:]
    m = msk[:, 1:]
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                         reduction="none").reshape(tgt.shape)
    return (ce * m).sum() / m.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["polish", "base"], required=True)
    ap.add_argument("--scope", choices=["full", "top", "head"], required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--gen-steps", default="500,1000,2000")
    ap.add_argument("--ckpt", default="checkpoints/armF_polish19b/final.pt")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="rescale hs going into block 23 (0 = off)")
    ap.add_argument("--loss-all-tokens", action="store_true")
    args = ap.parse_args()
    tag = f"{args.init}_{args.scope}"
    if args.alpha:
        tag += f"_a{args.alpha:g}"
    if args.loss_all_tokens:
        tag += "_all"

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    cnn = W.load_model()
    student = R4.load_student(cnn)

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", torch_dtype=torch.float32).to(DEV)
    if args.init == "polish":
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for k, v in ck["backbone"].items():
                if k.startswith("layers."):
                    obj = model.model
                    parts = k.split(".")
                    for p in parts[:-1]:
                        obj = (getattr(obj, p) if not p.isdigit()
                               else obj[int(p)])
                    getattr(obj, parts[-1]).copy_(v.float().to(DEV))
        del ck
    model.model.embed_tokens.weight.requires_grad_(False)
    model.lm_head.weight.requires_grad_(False)
    if args.scope == "top":
        for i, blk in enumerate(model.model.layers):
            if i < 23:
                for p in blk.parameters():
                    p.requires_grad_(False)
    elif args.scope == "head":
        # untie: fresh independent lm_head initialized from embeddings;
        # ONLY it trains — linear probe through the real generation path
        import torch.nn as nn
        model.lm_head.weight = nn.Parameter(
            model.model.embed_tokens.weight.detach().clone())
        for p in model.model.parameters():
            p.requires_grad_(False)
        model.lm_head.weight.requires_grad_(True)
    if args.scope != "head":
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.alpha:
        a = args.alpha
        model.model.layers[23].register_forward_pre_hook(
            lambda m, ar: (ar[0] * a,) + ar[1:])
    model.train()

    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]
    recs = tokenize_games(tok, games, all_tokens=args.loss_all_tokens)
    val_idx = [i for i in range(len(recs)) if i % 15 == 0][:40]
    tr_idx = [i for i in range(len(recs)) if i % 15 != 0]
    pad_id = tok.eos_token_id
    gen_at = {int(s) for s in args.gen_steps.split(",")}

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[{tag}] trainable params "
          f"{sum(p.numel() for p in params)/1e6:.0f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup = 50
    rng = random.Random(0)
    hist = {"val_nll": [], "gen": {}}

    for step in range(1, args.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = args.lr * min(1.0, step / warmup)
        idxs = rng.sample(tr_idx, args.bs)
        loss = batch_nll(model, recs, idxs, pad_id)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 200 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                v = sum(batch_nll(model, recs, val_idx[i:i + 8], pad_id
                                  ).item() for i in range(0, 40, 8)) / 5
            model.train()
            hist["val_nll"].append([step, v])
            print(f"[{tag}] step {step} train {loss.item():.4f} "
                  f"val_nll {v:.4f}", flush=True)
        if step in gen_at:
            model.eval()
            g = P59.gen_eval(model, tok, student, games, n_prefixes=100)
            model.train()
            hist["gen"][step] = g
            print(f"[{tag}] step {step} gen legal {g['legal_rate']:.3f} "
                  f"top1|legal {g['top1_given_legal']:.3f}", flush=True)

    out = Path(f"armF/results/p60_{tag}.json")
    out.write_text(json.dumps({"args": vars(args), "hist": hist}, indent=1))
    print(f"[{tag}] wrote {out}", flush=True)
    ckdir = Path("checkpoints/armF_p60")
    ckdir.mkdir(parents=True, exist_ok=True)
    sd = {k: v.bfloat16() for k, v in model.named_parameters()
          if v.requires_grad}
    torch.save({"trainable": sd, "args": vars(args)}, ckdir / f"{tag}.pt")
    print(f"[{tag}] saved {ckdir / f'{tag}.pt'}", flush=True)


if __name__ == "__main__":
    main()
