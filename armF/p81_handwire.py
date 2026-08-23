"""P81: hand-wire the selection head SGD won't build (block 17, kv head 0).

Construction (all fits are ridge on LN17(hs[17]) from the P80 bottom):
  k rows (kv head 0): only low-freq RoPE dims 63/127 nonzero (pair rotates
    ~4e-4 rad over the prompt — RoPE-safe). Targets: dim63 = 3*standardized
    cellhead logit at EMPTY cell tokens, -6 at occupied cells AND all
    non-cell tokens (repulsion doubles as legality); dim127 = 1 (angle ref).
  q rows (q head 0): only dim63, ridge target 1.0 at every token (sign is
    what matters; sparse-vector RMSNorm supplies gain ~sqrt(128)).
  v rows (kv head 0): dims 0..120 = ridge to one-hot cell id at cell
    tokens, 0 elsewhere.
  o_proj (q head 0 cols): fixed random-orthogonal 121-dim residual
    subspace, write norm ~3. q head 1 (shares kv0) zeroed at q and o.
Checks at init (no training): k-fit corr, attended-argmax-cell vs guest
argmax / vs cellhead argmax, and softmax concentration.

Saves surgical tensors to checkpoints/armF_p81/handwire.pt for FT via
train_p78ft --top-init.

Usage: /venv/main/bin/python armF/p81_handwire.py [--check-only]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_p78ft import load_model, PROMPT_TAIL  # noqa: E402
from p75_baselines import render_with_regs, load_guest  # noqa: E402
from tx_train import boards_to_states  # noqa: E402
import qwen_embed as Q  # noqa: E402

DEV = "cuda"
HD = 128  # head dim
LOW = 63  # lowest-frequency rotary dim (pairs with LOW+64)


def capture(model, tok, boards_u8, layer=17, batch=32, fmt="A"):
    """hs[layer] (B,T,2048) + aligned cell idx (B,121) + last idx (B,)."""
    hs_all, idx_all, last_all, occ_all = [], [], [], []
    for i in range(0, len(boards_u8), batch):
        bb = boards_u8[i:i + batch]
        texts, offs = [], []
        for b in bb.cpu():
            if fmt == "A":
                t, o = render_with_regs(b)
                t = t + PROMPT_TAIL
            else:
                from p82_format import prompt_of
                t, o = prompt_of(b, fmt)
            texts.append(t)
            offs.append(o[:121])
        enc = tok(texts, return_offsets_mapping=True, padding=True,
                  return_tensors="pt", add_special_tokens=False)
        idxs = torch.zeros(len(bb), 121, dtype=torch.long)
        for j in range(len(bb)):
            starts = {}
            for tj, (a, bnd) in enumerate(enc["offset_mapping"][j].tolist()):
                for o in range(a, bnd):
                    starts[o] = tj
            idxs[j] = torch.tensor([starts[o] for o in offs[j]])
        with torch.no_grad():
            out = model.model(input_ids=enc["input_ids"].to(DEV),
                              attention_mask=enc["attention_mask"].to(DEV),
                              output_hidden_states=True)
        hs_all.append(out.hidden_states[layer].float().cpu())
        idx_all.append(idxs)
        last_all.append(enc["attention_mask"].sum(1) - 1)
        occ_all.append((boards_to_states(bb.to(DEV)) > 0).cpu())
    T = max(h.shape[1] for h in hs_all)
    hs = torch.zeros(len(boards_u8), T, 2048)
    for i, h in enumerate(hs_all):
        hs[i * 32:i * 32 + len(h), :h.shape[1]] = h
    return hs, torch.cat(idx_all), torch.cat(last_all), torch.cat(occ_all)


def ridge(X, y, lam=1.0):
    """NO intercept — Qwen3 q/k/v projections have no bias, so any constant
    component must be carried by real feature directions (LN output has a
    stable token-independent mean component to fit through)."""
    X = X.to(DEV)
    y = y.to(DEV)
    A = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device=DEV),
                           X.T @ y)
    return A  # (2048, dy): weight rows = A.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/armF_p80/best.pt")
    ap.add_argument("--n-fit", type=int, default=768)
    ap.add_argument("--n-val", type=int, default=256)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--mix-fmt", action="store_true",
                    help="fit + check on 50/50 format A/B (p82)")
    ap.add_argument("--out", default="checkpoints/armF_p81/handwire.pt")
    args = ap.parse_args()
    torch.manual_seed(0)

    boards = torch.load("armF/data/tx_positions.pt",
                        weights_only=False)["boards"]
    perm = torch.randperm(len(boards),
                          generator=torch.Generator().manual_seed(4))
    fit_b = boards[perm[:args.n_fit]]
    val_b = boards[perm[args.n_fit:args.n_fit + args.n_val]]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Q.QWEN)
    guest = load_guest()
    model = load_model("contained", args.ckpt)
    model.eval()
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    cellhead = torch.nn.Linear(2048, 1).to(DEV)
    cellhead.load_state_dict(ck["cellhead"])

    attn = model.model.layers[17].self_attn
    ln17 = model.model.layers[17].input_layernorm

    print("capturing fit set...", flush=True)
    gather = lambda x, i: torch.gather(  # noqa: E731
        x, 1, i.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    fmts = ["A", "B"] if args.mix_fmt else ["A"]
    Xrows, tkrows, tvrows = [], [], []
    for fi, fmt in enumerate(fmts):
        fb = fit_b[fi::len(fmts)]
        hs, cidx, last, occ = capture(model, tok, fb, fmt=fmt)
        B, T, _ = hs.shape
        with torch.no_grad():
            lnx = ln17(hs.to(DEV).to(torch.bfloat16)).float().cpu()
            cell_states = gather(hs, cidx)
            logits = cellhead(cell_states.to(DEV)).squeeze(-1).cpu()
        mu_l, sd_l = logits[~occ].mean(), logits[~occ].std()
        tk = torch.full((B, T), -6.0)
        tv = torch.zeros(B, T, 121)
        for b in range(B):
            for c in range(121):
                t = cidx[b, c].item()
                tv[b, t, c] = 1.0
                if not occ[b, c]:
                    tk[b, t] = (2.0 * (logits[b, c] - mu_l)
                                / sd_l).clamp(-8, 8)
        Xrows.append(lnx.reshape(-1, 2048))
        tkrows.append(tk.reshape(-1))
        tvrows.append(tv.reshape(-1, 121))
    Xf = torch.cat(Xrows)
    tkf = torch.cat(tkrows)
    tvf = torch.cat(tvrows)
    Ak = ridge(Xf, torch.stack([tkf, torch.ones(len(tkf))], 1))
    Aq = ridge(Xf, torch.ones(len(tkf), 1))
    Av = ridge(Xf, tvf)
    pred_k = (Xf.to(DEV) @ Ak[:, 0]).cpu()
    pred_c = (Xf.to(DEV) @ Ak[:, 1]).cpu()
    pred_q = (Xf.to(DEV) @ Aq[:, 0]).cpu()
    corr = torch.corrcoef(torch.stack([pred_k, tkf]))[0, 1]
    print(f"k-fit corr {corr:.3f} | const-fit k127 mean {pred_c.mean():.3f} "
          f"std {pred_c.std():.3f} | q mean {pred_q.mean():.3f} "
          f"std {pred_q.std():.3f}", flush=True)

    # ---- surgery (weights are bf16) ----
    with torch.no_grad():
        kw = attn.k_proj.weight  # (8*128, 2048)
        kw[0 * HD:(0 + 1) * HD] = 0
        kw[LOW] = Ak[:, 0].to(kw.dtype)
        kw[LOW + 64] = Ak[:, 1].to(kw.dtype)
        qw = attn.q_proj.weight  # (16*128, 2048)
        qw[0:2 * HD] = 0  # q heads 0 and 1
        qw[LOW] = Aq[:, 0].to(qw.dtype)
        vw = attn.v_proj.weight
        vw[0 * HD:(0 + 1) * HD] = 0
        vw[0:121] = Av.T.to(vw.dtype)
        ow = attn.o_proj.weight  # (2048, 16*128)
        code = torch.linalg.qr(torch.randn(2048, 121))[0] * 25.0
        ow[:, 0:2 * HD] = 0
        ow[:, 0:121] = code.to(ow.dtype)

    # ---- init checks on val (per format) ----
    model.config._attn_implementation = "eager"
    res = {"k_fit_corr": corr.item()}
    for fmt in fmts:
        hsv, cidxv, lastv, occv = capture(model, tok, val_b, fmt=fmt)
        with torch.no_grad():
            lnv = ln17(hsv.to(DEV).to(torch.bfloat16)).float()
            k63 = lnv @ Ak[:, 0]
            k127 = lnv @ Ak[:, 1]
            q63 = lnv @ Aq[:, 0]
            krms = ((k63 ** 2 + k127 ** 2) / HD + 1e-6).sqrt()
            k63h = k63 / krms * attn.k_norm.weight[LOW].float()
            qsign = torch.sign(torch.gather(q63, 1, lastv.to(DEV)[:, None]))
            qmag = HD ** 0.5 * attn.q_norm.weight[LOW].float()
            scores = (qsign * qmag) * k63h / HD ** 0.5
            st = boards_to_states(val_b.to(DEV))
            g = guest(st) - 1000.0 * (st > 0).float()
            gref = g.argmax(1).cpu()
            cref = cellhead(gather(hsv, cidxv).to(DEV)).squeeze(-1)
            cref = (cref - 1000.0 * occv.to(DEV).float()).argmax(1).cpu()
            att_cell, conc = [], []
            for b in range(len(val_b)):
                sc = scores[b, :int(lastv[b]) + 1]
                p = torch.softmax(sc, 0)
                t = p.argmax().item()
                hits = (cidxv[b] == t).nonzero()
                att_cell.append(hits[0, 0].item() if len(hits) else -1)
                conc.append(p.max().item())
            att_cell = torch.tensor(att_cell)
            vs_guest = (att_cell == gref).float().mean().item()
            vs_cell = (att_cell == cref).float().mean().item()
        print(f"init checks [{fmt}]: attended vs guest {vs_guest:.3f} | "
              f"vs cellhead {vs_cell:.3f} | conc "
              f"{sum(conc)/len(conc):.3f}", flush=True)
        res[f"att_vs_guest_{fmt}"] = vs_guest
        res[f"att_vs_cellhead_{fmt}"] = vs_cell
        res[f"conc_{fmt}"] = sum(conc) / len(conc)
    Path("armF/results/p81_init.json").write_text(json.dumps(res, indent=1))

    if not args.check_only:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        sd = model.model.layers[17].state_dict()
        torch.save({"layer17": {k: v.clone() for k, v in sd.items()},
                    "checks": res}, args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
