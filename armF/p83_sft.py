"""P83 SFT: teach the P82 artifact to self-canonicalize (conversational hex).

Model: p82 bottom (2-format parser + containment + cellhead) + p82_ft top
(mixed hand-wire + emission). All blocks trainable (embeddings + tied head
frozen). Loss: CE on the TARGET region (stone-list CoT + canonical block +
" <move>") + aux KL at the generated ':' (guest logits, fresh head).

Eval (generation-based, greedy ~520 tokens): per-family re-render CELL
accuracy vs true board, end-to-end move top1/legal, + canonical-passthrough
regression (P82-style prompt -> move).

Usage: /venv/main/bin/python armF/p83_sft.py --steps 4000 --tag p83
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from p75_baselines import load_guest  # noqa: E402
from p83_data import target_text  # noqa: E402
from train_p78ft import (load_model, guest_labels, parse_cell,  # noqa: E402
                         gen_eval)
from train_txcontain import guest_final_logits  # noqa: E402
from tx_train import boards_to_states  # noqa: E402
import qwen_embed as Q  # noqa: E402

DEV = "cuda"
SEP = "\n\n"
MARK = "The same board again:\n\n"
SYM2CH = {"X": 0, "O": 1}


def parse_render(txt, true_board):
    """Cell accuracy of the SECOND copy in generated text vs true board."""
    i = txt.find(MARK)
    if i < 0:
        return None
    lines = txt[i + len(MARK):].split("\n")[:11]
    if len(lines) < 11:
        return None
    ok = tot = 0
    for x in range(11):
        syms = lines[x].strip().split()
        if len(syms) != 11:
            return None
        for y in range(11):
            true = ("X" if true_board[0, x + 1, y + 1] > 0.5 else
                    "O" if true_board[1, x + 1, y + 1] > 0.5 else ".")
            ok += int(syms[y] == true)
            tot += 1
    return ok / tot


def make_batch(tok, recs):
    prompts = [r["inp"] + SEP for r in recs]
    fulls = [p + target_text(r["board"], r["label"])
             for p, r in zip(prompts, recs)]
    p_lens = [len(tok(p, add_special_tokens=False)["input_ids"])
              for p in prompts]
    enc = tok(fulls, return_tensors="pt", padding=True,
              add_special_tokens=False)
    ids = enc["input_ids"]
    lab = torch.full_like(ids, -100)
    colon_idx = torch.zeros(len(ids), dtype=torch.long)
    for i in range(len(ids)):
        n = int(enc["attention_mask"][i].sum())
        lab[i, p_lens[i]:n] = ids[i, p_lens[i]:n]
        # ':' of "Next move:" = 5 tokens from the end (" F", "0", "4" = 3,
        # preceded by ":"); find robustly via char offset
        colon_char = len(fulls[i]) - len(" F04") - 1
        assert fulls[i][colon_char] == ":", fulls[i][-20:]
        colon_idx[i] = -1  # fill via offsets below
    enc2 = tok(fulls, return_offsets_mapping=True, padding=True,
               add_special_tokens=False)
    for i in range(len(ids)):
        colon_char = len(fulls[i]) - len(" F04") - 1
        for tj, (a, b) in enumerate(enc2["offset_mapping"][i]):
            if a <= colon_char < b:
                colon_idx[i] = tj
                break
        assert colon_idx[i] >= 0
    return ids, enc["attention_mask"], lab, colon_idx


@torch.no_grad()
def full_eval(model, tok, guest, val_recs, batch=16, max_new=1000):
    model.eval()
    fam_acc, fam_n = {}, {}
    move_top1 = move_legal = move_tot = 0
    for i in range(0, len(val_recs), batch):
        rr = val_recs[i:i + batch]
        prompts = [r["inp"] + SEP for r in rr]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left", add_special_tokens=False)
        out = model.generate(input_ids=enc["input_ids"].to(DEV),
                             attention_mask=enc["attention_mask"].to(DEV),
                             max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        for j, r in enumerate(rr):
            txt = tok.decode(out[j, enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            acc = parse_render(txt, r["board"])
            fam_acc[r["fam"]] = fam_acc.get(r["fam"], 0.0) + (acc or 0.0)
            fam_n[r["fam"]] = fam_n.get(r["fam"], 0) + 1
            m = re.search(r"Next move: ([A-K]\d\d)", txt)
            move_tot += 1
            if m:
                c = parse_cell(" " + m.group(1))
                if c is not None:
                    st = boards_to_states(
                        r["board"].unsqueeze(0).to(DEV))[0]
                    if st[c] == 0:
                        move_legal += 1
                    ref = guest_labels(guest,
                                       r["board"].unsqueeze(0).to(DEV))[0]
                    if c == ref.item():
                        move_top1 += 1
    model.train()
    return ({f: fam_acc[f] / fam_n[f] for f in fam_acc},
            move_top1 / move_tot, move_legal / move_tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--aux-wt", type=float, default=1.0)
    ap.add_argument("--aux-layer", type=int, default=22)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-eval", type=int, default=96)
    ap.add_argument("--bottom", default="checkpoints/armF_p82/bottom.pt")
    ap.add_argument("--top", default="checkpoints/armF_p78/p82_ft.pt")
    ap.add_argument("--tag", default="p83")
    ap.add_argument("--freeze-mid", action="store_true",
                    help="freeze blocks 5-16 (containment core)")
    ap.add_argument("--stone-wt", type=float, default=1.0,
                    help="CE upweight for ' X'/' O' target tokens")
    ap.add_argument("--preserve-every", type=int, default=0,
                    help="every Nth step: canonical-FT preservation batch")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.eval_every, args.n_eval = 30, 15, 8
    torch.manual_seed(0)

    recs = torch.load("armF/data/p83_sft.pt", weights_only=False)["records"]
    train_recs, val_recs = recs[:-512], recs[-512:]
    val_eval = val_recs[: args.n_eval]
    print(f"train {len(train_recs)} val {len(val_recs)}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Q.QWEN)
    guest = load_guest()
    model = load_model("contained", args.bottom)
    ft = torch.load(args.top, map_location=DEV, weights_only=False)
    sd = model.state_dict()
    for k, v in ft["top"].items():
        sd[k].copy_(v)
    # blocks trainable (embeddings + tied head stay frozen)
    for i, blk in enumerate(model.model.layers):
        blk.requires_grad_(not (args.freeze_mid and 5 <= i <= 16))
    model.model.norm.requires_grad_(True)
    ID_X = tok(" X", add_special_tokens=False)["input_ids"]
    ID_O = tok(" O", add_special_tokens=False)["input_ids"]
    assert len(ID_X) == 1 and len(ID_O) == 1
    ID_X, ID_O = ID_X[0], ID_O[0]
    aux = torch.nn.Linear(2048, 121).to(DEV)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable {n_tr/1e6:.0f}M", flush=True)

    use_wandb = not args.no_wandb and not args.smoke
    if use_wandb:
        import wandb
        wandb.init(project="hex-rl-cot-deconfusion", name=f"armF_{args.tag}",
                   config=vars(args))

    opt = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad],
          "lr": args.lr},
         {"params": aux.parameters(), "lr": 1e-3}], weight_decay=0.0)

    # canonical passthrough regression reference boards
    all_boards = torch.load("armF/data/tx_positions.pt",
                            weights_only=False)["boards"]
    pperm = torch.randperm(len(all_boards),
                           generator=torch.Generator().manual_seed(6))
    pass_b = all_boards[pperm[200000:200128]]

    hist = []
    t0 = time.time()
    model.train()
    for s in range(args.steps):
        lr = args.lr * min(1.0, (s + 1) / args.warmup) * 0.5 * (
            1 + math.cos(math.pi * s / args.steps))
        opt.param_groups[0]["lr"] = lr
        if args.preserve_every and (s + 1) % args.preserve_every == 0:
            # canonical-FT preservation batch (P82 objective)
            from train_p78ft import make_batch as ft_make_batch
            psel = torch.randint(0, len(train_recs), (args.batch,))
            pb = torch.stack([train_recs[i]["board"] for i in psel])
            plab = guest_labels(guest, pb.to(DEV))
            pfmt = "A" if (s // args.preserve_every) % 2 == 0 else "B"
            pids, pam, pl, ppl = ft_make_batch(tok, pb, plab, pfmt)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pout = model(input_ids=pids.to(DEV),
                             attention_mask=pam.to(DEV), labels=pl.to(DEV))
            loss = pout.loss
            aux_kl = torch.zeros((), device=DEV)
            out = pout
        else:
            sel = torch.randint(0, len(train_recs), (args.batch,))
            rr = [train_recs[i] for i in sel]
            ids, am, lab, colon = make_batch(tok, rr)
            lab_d = lab.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=ids.to(DEV), attention_mask=am.to(DEV),
                            output_hidden_states=True)
            # weighted CE (stone tokens upweighted — empty-board attractor)
            logits = out.logits[:, :-1]
            tgt = lab_d[:, 1:]
            wts = torch.ones_like(tgt, dtype=torch.float)
            wts[(tgt == ID_X) | (tgt == ID_O)] = args.stone_wt
            ce_rows = []
            for i0 in range(0, len(tgt), 2):
                ce = F.cross_entropy(
                    logits[i0:i0 + 2].reshape(-1, logits.shape[-1]).float(),
                    tgt[i0:i0 + 2].reshape(-1), ignore_index=-100,
                    reduction="none")
                ce_rows.append(ce)
            ce = torch.cat(ce_rows)
            wf = wts.reshape(-1) * (tgt.reshape(-1) != -100)
            ce_loss = (ce * wf).sum() / wf.sum()
            bts = torch.stack([r["board"] for r in rr]).to(DEV)
            h = out.hidden_states[args.aux_layer][
                torch.arange(len(rr), device=DEV), colon.to(DEV)]
            with torch.no_grad():
                g = guest_final_logits(guest, bts)
            tp = torch.softmax(g, -1)
            aux_kl = (tp * (torch.log_softmax(g, -1)
                            - torch.log_softmax(aux(h.float()), -1))
                      ).sum(-1).mean()
            loss = ce_loss + args.aux_wt * aux_kl
            out.loss = ce_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad]
            + list(aux.parameters()), 1.0)
        opt.step()
        if (s + 1) % 50 == 0 and use_wandb:
            import wandb
            wandb.log({"loss": out.loss.item(), "aux_kl": aux_kl.item()},
                      step=s + 1)
        if (s + 1) % args.eval_every == 0 or s + 1 == args.steps:
            fam, mt1, mlg = full_eval(model, tok, guest, val_eval)
            pt1, plg, _pp = gen_eval(model, tok, guest, pass_b)
            model.train()
            rec = {"step": s + 1, "fam_cellacc": {k: round(v, 4)
                                                  for k, v in fam.items()},
                   "move_top1": round(mt1, 4), "move_legal": round(mlg, 4),
                   "passthrough_top1": round(pt1, 4)}
            hist.append(rec)
            print(f"step {s+1}/{args.steps} loss {out.loss.item():.4f} "
                  f"aux {aux_kl.item():.3f} | cellacc "
                  + " ".join(f"{k}:{v:.3f}" for k, v in sorted(fam.items()))
                  + f" | move top1 {mt1:.3f} legal {mlg:.3f} | passthru "
                  f"{pt1:.3f} ({(s+1)/(time.time()-t0):.2f} it/s)",
                  flush=True)
            if use_wandb:
                import wandb
                wandb.log({"move_top1": mt1, "move_legal": mlg,
                           "passthrough_top1": pt1,
                           **{f"cellacc_{k}": v for k, v in fam.items()}},
                          step=s + 1)
            Path(f"armF/results/{args.tag}_hist.json").write_text(
                json.dumps(hist, indent=1))
            outdir = Path("checkpoints/armF_p83")
            outdir.mkdir(parents=True, exist_ok=True)
            torch.save({"model": {k: v.to(torch.bfloat16)
                                  for k, v in model.state_dict().items()
                                  if "layers." in k or "model.norm" in k},
                        "step": s + 1, "hist": hist},
                       outdir / f"{args.tag}_last.pt")
    print(f"done in {(time.time()-t0)/3600:.2f}h", flush=True)
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
