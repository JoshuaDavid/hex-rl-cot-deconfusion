"""P82: 2-format input robustness by retraining ONLY blocks 0-4.

Format A = the P75-81 render (X/O/. + technical preamble). Format B = same
token-layout skeleton, new surface: glyphs w/b/_ and conversational text.
Blocks 0-4 (the h_0<->hs[5] input parser) trained on 50/50 mixed batches
against the FROZEN adapter_0 interface (guest-normalized h_0 MSE at 121
cell + 7 register tokens). Everything above stays frozen: blocks 5-16,
adapters, cellhead (P80), hand-wired layer 17 + FT top (p81b).

Eval per format: h-layer R2 (13), cellhead top1, gen top1.

Usage:
  /venv/main/bin/python armF/p82_format.py --baseline   # zero-shot B
  /venv/main/bin/python armF/p82_format.py --train
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import render11 as R  # noqa: E402
from p75_baselines import (REG_SUFFIX, load_guest, guest_acts,  # noqa: E402
                           render_with_regs)
from train_txcontain import (load_backbone, Adapters, gather_tok,  # noqa: E402
                             bf16_sd, guest_final_logits, EMIT_TAIL)
from train_p78ft import (load_model, guest_labels, cell_str,  # noqa: E402
                         parse_cell, PROMPT_TAIL)
from tx_train import boards_to_states  # noqa: E402
import qwen_embed as Q  # noqa: E402

DEV = "cuda"

PRE_B = ("We are playing hex on an 11 by 11 board. Stones of the player to "
         "move are w (they want a chain from the top edge to the bottom "
         "edge); opponent stones are b (left edge to right edge); empty "
         "points are _. A point's lower-left and upper-right diagonal "
         "neighbours touch it.\n\n")
MID_B = "\nOnce more, the same position:\n\n"
POST_B = "\nThink it over.\n"
GLYPH_B = {"X": "w", "O": "b", ".": "_"}


def render_b(canonical, want_off1=False):
    """Same skeleton as render11.render (indent + ' g' per cell) with
    format-B surface. Returns (text, offsets2[, offsets1])."""
    def body(offset0):
        lines, offs = [], []
        pos = offset0
        for x in range(11):
            line = " " * x
            pos += x
            for y in range(11):
                if canonical[0, x + 1, y + 1] > 0.5:
                    c = GLYPH_B["X"]
                elif canonical[1, x + 1, y + 1] > 0.5:
                    c = GLYPH_B["O"]
                else:
                    c = GLYPH_B["."]
                line += " " + c
                pos += 1
                offs.append(pos)
                pos += 1
            lines.append(line)
            pos += 1
        return "\n".join(lines), offs

    b1, off1 = body(len(PRE_B))
    text1 = PRE_B + b1
    b2, off2 = body(len(text1) + len(MID_B))
    text = text1 + MID_B + b2 + POST_B
    for o in off2 + off1:
        assert text[o] in "wb_", (o, text[o])
    if want_off1:
        return text, off2, off1
    return text, off2


def prompt_of(board, fmt, tail=True, want_off1=False):
    if fmt == "A":
        if want_off1:
            text0, off1, off2 = R.render_two_copy(board)
        text, offs = render_with_regs(board)
    else:
        if want_off1:
            _t, off2, off1 = render_b(board, want_off1=True)
        text, off2b = render_b(board)
        base = len(text)
        text = text + REG_SUFFIX
        colon = REG_SUFFIX.index(":")
        offs = off2b + [base + REG_SUFFIX.index(ch, colon)
                        for ch in "abcdefg"]
    if tail:
        text = text + EMIT_TAIL
    if want_off1:
        return text, offs, off1
    return text, offs


def batch_fmt(tok, boards_u8, fmt, want_off1=False):
    texts, all_offs, all_off1 = [], [], []
    for b in boards_u8.cpu():
        if want_off1:
            t, o, o1 = prompt_of(b, fmt, want_off1=True)
            all_off1.append(o1)
        else:
            t, o = prompt_of(b, fmt)
        texts.append(t)
        all_offs.append(o)
    enc = tok(texts, return_offsets_mapping=True, padding=True,
              return_tensors="pt", add_special_tokens=False)
    idxs = torch.zeros(len(texts), 128, dtype=torch.long)
    idx1 = torch.zeros(len(texts), 121, dtype=torch.long)
    for i in range(len(texts)):
        starts = {}
        for tj, (a, bnd) in enumerate(enc["offset_mapping"][i].tolist()):
            for o in range(a, bnd):
                starts[o] = tj
        row = [starts[o] for o in all_offs[i]]
        assert len(set(row)) == len(row)
        idxs[i] = torch.tensor(row)
        if want_off1:
            idx1[i] = torch.tensor([starts[o] for o in all_off1[i]])
    if want_off1:
        return enc["input_ids"], enc["attention_mask"], idxs, idx1
    return enc["input_ids"], enc["attention_mask"], idxs


# ---------------------------------------------------------------- eval
@torch.no_grad()
def eval_stack(backbone17, adapters, cellhead, tok, guest, boards, mu, sd,
               fmt, batch=32):
    L = 13
    sse = torch.zeros(L, device=DEV)
    ssum = torch.zeros(L, 1024, device=DEV)
    ssq = torch.zeros(L, 1024, device=DEV)
    n = 0
    top1 = tot = 0
    for i in range(0, len(boards), batch):
        bb = boards[i:i + batch].to(DEV)
        z = (guest_acts(guest, bb) - mu[:, None, None]) / sd[:, None, None]
        ids, am, idxs = batch_fmt(tok, bb, fmt)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = backbone17(input_ids=ids.to(DEV),
                             attention_mask=am.to(DEV),
                             output_hidden_states=True, use_cache=False)
        idx = idxs.to(DEV)
        for l in range(L):
            pred = adapters.maps[l](gather_tok(out.hidden_states[5 + l].float(),
                                               idx))
            y = z[l]
            sse[l] += ((pred - y) ** 2).sum()
            ssum[l] += y.sum(dim=(0, 1))
            ssq[l] += (y * y).sum(dim=(0, 1))
        hc = gather_tok(out.hidden_states[17].float(), idx)[:, :121]
        pl = cellhead(hc).squeeze(-1) - 1000.0 * (boards_to_states(bb) > 0).float()
        ref = guest_final_logits(guest, bb).argmax(1)
        top1 += (pl.argmax(1) == ref).sum().item()
        tot += len(bb)
        n += len(bb) * 128
    mean = ssum / n
    sstot = (ssq - n * mean * mean).sum(-1)
    r2 = (1 - sse / sstot).cpu()
    return r2, top1 / tot


@torch.no_grad()
def gen_eval_fmt(full_model, tok, guest, boards, fmt, batch=32):
    top1 = legal = tot = 0
    refs = guest_labels(guest, boards.to(DEV))
    occ = (boards_to_states(boards.to(DEV)) > 0)
    for i in range(0, len(boards), batch):
        bb = boards[i:i + batch]
        prompts = [prompt_of(b, fmt)[0] for b in bb.cpu()]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left", add_special_tokens=False)
        out = full_model.generate(input_ids=enc["input_ids"].to(DEV),
                                  attention_mask=enc["attention_mask"].to(DEV),
                                  max_new_tokens=4, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        for j in range(len(bb)):
            txt = tok.decode(out[j, enc["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            c = parse_cell(txt)
            k = i + j
            if c is not None:
                if not occ[k, c]:
                    legal += 1
                if c == refs[k].item():
                    top1 += 1
            tot += 1
    return top1 / tot, legal / tot


def build_eval_models(p80_ckpt, p81_top, blocks04_sd=None):
    ck = torch.load(p80_ckpt, map_location=DEV, weights_only=False)
    backbone17 = load_backbone()
    backbone17.load_state_dict({k: v.float() for k, v in ck["backbone"].items()})
    adapters = Adapters().to(DEV)
    adapters.load_state_dict(ck["adapters"])
    cellhead = torch.nn.Linear(2048, 1).to(DEV)
    cellhead.load_state_dict(ck["cellhead"])
    full = load_model("contained", p80_ckpt)
    ft = torch.load(p81_top, map_location=DEV, weights_only=False)
    sd = full.state_dict()
    for k, v in ft["top"].items():
        sd[k].copy_(v)
    if blocks04_sd is not None:
        b17 = backbone17.state_dict()
        fsd = full.state_dict()
        for k, v in blocks04_sd.items():
            if k.startswith("layers.") and int(k.split(".")[1]) < 5:
                b17[k].copy_(v)
                fsd["model." + k].copy_(v)
    backbone17.eval()
    full.eval()
    mu, sd_ = ck["mu"].to(DEV), ck["sd"].to(DEV)
    return backbone17, adapters, cellhead, full, mu, sd_


def report(tag, backbone17, adapters, cellhead, full, tok, guest, val_b,
           mu, sd, results):
    for fmt in ("A", "B"):
        r2, ct1 = eval_stack(backbone17, adapters, cellhead, tok, guest,
                             val_b, mu, sd, fmt)
        gt1, gl = gen_eval_fmt(full, tok, guest, val_b[:256], fmt)
        results[f"{tag}_{fmt}"] = {
            "r2_mean": round(r2.mean().item(), 4),
            "r2_h0": round(r2[0].item(), 4),
            "cellhead_top1": round(ct1, 4),
            "gen_top1": round(gt1, 4), "gen_legal": round(gl, 4)}
        print(f"[{tag}] fmt {fmt}: R2 mean {r2.mean():.4f} h0 {r2[0]:.4f} "
              f"| cellhead {ct1:.3f} | gen {gt1:.3f} legal {gl:.3f}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--distill-wt", type=float, default=1.0)
    ap.add_argument("--n-val", type=int, default=512)
    ap.add_argument("--p80", default="checkpoints/armF_p80/best.pt")
    ap.add_argument("--p81", default="checkpoints/armF_p78/p81b_ft.pt")
    ap.add_argument("--out", default="armF/results/p82_format.json")
    args = ap.parse_args()
    torch.manual_seed(0)

    boards = torch.load("armF/data/tx_positions.pt",
                        weights_only=False)["boards"]
    perm = torch.randperm(len(boards),
                          generator=torch.Generator().manual_seed(6))
    train_b = boards[perm[:200000]]
    val_b = boards[perm[200000:200000 + args.n_val]]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Q.QWEN)
    guest = load_guest()
    results = {}
    outp = Path(args.out)
    if outp.exists():
        results = json.loads(outp.read_text())

    if args.baseline:
        b17, ad, chd, full, mu, sd = build_eval_models(args.p80, args.p81)
        report("zeroshot", b17, ad, chd, full, tok, guest, val_b, mu, sd,
               results)
        outp.write_text(json.dumps(results, indent=1))
        del b17, full
        torch.cuda.empty_cache()

    if args.train:
        ck = torch.load(args.p80, map_location=DEV, weights_only=False)
        sub = {k: v.float() for k, v in ck["backbone"].items()
               if k == "embed_tokens.weight"
               or (k.startswith("layers.") and int(k.split(".")[1]) < 5)}
        parser = load_backbone(train_blocks=5)
        parser.load_state_dict(sub, strict=False)
        parser.train()
        teacher = load_backbone(train_blocks=5)
        teacher.load_state_dict(sub, strict=False)
        teacher.eval()
        teacher.requires_grad_(False)
        adapter0 = torch.nn.Linear(2048, 1024).to(DEV)
        a0 = {k.split("maps.0.")[1]: v for k, v in ck["adapters"].items()
              if k.startswith("maps.0.")}
        adapter0.load_state_dict(a0)
        adapter0.requires_grad_(False)
        mu0 = ck["mu"][0].to(DEV)
        sd0 = ck["sd"][0].to(DEV)
        opt = torch.optim.AdamW(
            [p for p in parser.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.0)
        K_SFX = 10  # shared suffix tokens (register slots + "Next move:")
        t0 = time.time()
        for s in range(args.steps):
            lr = args.lr * (min(1.0, (s + 1) / 100)
                            * (0.5 * (1 + torch.cos(torch.tensor(
                                3.14159 * s / args.steps)).item())))
            for g in opt.param_groups:
                g["lr"] = lr
            sel = torch.randint(0, len(train_b), (args.batch,))
            bb = train_b[sel].to(DEV)
            fmt = "A" if s % 2 == 0 else "B"
            with torch.no_grad():
                h0 = (guest_acts(guest, bb)[0] - mu0) / sd0
            ids, am, idxs, idx1 = batch_fmt(tok, bb, fmt, want_off1=True)
            ids_d, am_d = ids.to(DEV), am.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = parser(input_ids=ids_d, attention_mask=am_d,
                             output_hidden_states=True, use_cache=False)
            hs5 = out.hidden_states[5].float()
            with torch.no_grad():
                pred = adapter0(gather_tok(hs5, idxs.to(DEV)))
                anchor = ((pred - h0) ** 2).mean()  # monitor only
            if fmt == "A":
                # full-sequence preservation (v1 collapse: non-cell states
                # incl. sinks are load-bearing for the frozen upper stack)
                with torch.no_grad(), torch.autocast("cuda",
                                                     dtype=torch.bfloat16):
                    ths5 = teacher(input_ids=ids_d, attention_mask=am_d,
                                   output_hidden_states=True, use_cache=False
                                   ).hidden_states[5].float()
                m = am_d.unsqueeze(-1).float()
                distill = (((hs5 - ths5) ** 2) * m).sum() / m.sum() / 2048
            else:
                # B v3: FULL-STATE teacher distillation at all structurally
                # corresponding tokens — copy-1 cells (the stack reads them:
                # v2 hole #1), copy-2 cells+regs in raw hs space (adapter
                # nullspace was free: v2 hole #2), shared suffix, sink tok 0
                ids_a, am_a, idxs_a, idx1_a = batch_fmt(tok, bb, "A",
                                                        want_off1=True)
                with torch.no_grad(), torch.autocast("cuda",
                                                     dtype=torch.bfloat16):
                    ths5 = teacher(input_ids=ids_a.to(DEV),
                                   attention_mask=am_a.to(DEV),
                                   output_hidden_states=True, use_cache=False
                                   ).hidden_states[5].float()
                d2 = gather_tok(hs5, idxs.to(DEV)) \
                    - gather_tok(ths5, idxs_a.to(DEV))
                d1 = gather_tok(hs5, idx1.to(DEV)) \
                    - gather_tok(ths5, idx1_a.to(DEV))
                nb = am_d.sum(1)
                na = am_a.sum(1).to(DEV)
                db = torch.stack([hs5[i, nb[i] - K_SFX:nb[i]]
                                  for i in range(len(bb))])
                da = torch.stack([ths5[i, na[i] - K_SFX:na[i]]
                                  for i in range(len(bb))])
                distill = ((d2 ** 2).mean() + (d1 ** 2).mean()
                           + ((db - da) ** 2).mean()
                           + ((hs5[:, 0] - ths5[:, 0]) ** 2).mean()) / 4
            loss = args.distill_wt * distill
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in parser.parameters() if p.requires_grad], 1.0)
            opt.step()
            if (s + 1) % 250 == 0:
                print(f"step {s+1}/{args.steps} [{fmt}] h0-mse "
                      f"{anchor.item():.4f} distill {distill.item():.4f} "
                      f"({(s+1)/(time.time()-t0):.2f} it/s)", flush=True)
        sd04 = {k: v for k, v in bf16_sd(parser).items()
                if k.startswith("layers.")}
        Path("checkpoints/armF_p82").mkdir(parents=True, exist_ok=True)
        torch.save({"blocks04": sd04}, "checkpoints/armF_p82/parser.pt")
        del parser
        torch.cuda.empty_cache()
        b17, ad, chd, full, mu, sd = build_eval_models(
            args.p80, args.p81,
            {k: v.float() for k, v in sd04.items()})
        report("trained", b17, ad, chd, full, tok, guest, val_b, mu, sd,
               results)
        outp.write_text(json.dumps(results, indent=1))

    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
