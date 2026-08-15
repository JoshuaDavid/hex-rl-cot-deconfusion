"""P59 rungs 1-2: is the policy recoverable from Qwen acts as TEXT / linearly?

--gen: zero-shot greedy continuation of d04n prefixes at O-to-move points;
       measure format-legal rate + top1/rank vs distilled policy.
--probe: direct linear 2048->121 CE probe from hs[5+k] at X-move tokens to
       distilled argmax (canonical frame), k in {0,6,12,18}; val top1.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_movesr4 as R4  # noqa: E402
import eval_stitch_polish as SP  # noqa: E402
import build_d04  # noqa: E402

DEV = "cuda"


def load_spliced_lm(ckpt_path):
    """Full pretrained CausalLM with polished blocks 0..22 spliced in
    (blocks 23..27, norm, lm_head untouched) — eval_language.py recipe."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16).cuda().eval()
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with torch.no_grad():
        for k, v in ck["backbone"].items():
            if k.startswith("layers."):
                obj = model.model
                parts = k.split(".")
                for p in parts[:-1]:
                    obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
                getattr(obj, parts[-1]).copy_(v.bfloat16().cuda())
    return model


@torch.no_grad()
def gen_eval(backbone, tok, student, games, n_prefixes=200, seed=7):
    from hexhex.utils.utils import correct_position1d
    rng = random.Random(seed)
    val_gis = [gi for gi in range(len(games)) if gi % 15 == 0]
    prefixes = []
    for gi in val_gis:
        T = len(games[gi]["moves"])
        for t in range(1, T, 2):  # odd move-count: O to move
            prefixes.append((gi, t))
    prefixes = rng.sample(prefixes, n_prefixes)
    pat = re.compile(r"^\n([A-K])(\d{2}) O")
    n_fmt = n_legal = n_top1 = 0
    ranks = []
    for gi, t in prefixes:
        g = games[gi]
        moves = [tuple(m) for m in g["moves"][:t].tolist()]
        txt = SP.d04n_text(moves)
        enc = tok(txt, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = backbone.generate(ids, max_new_tokens=6, do_sample=False,
                                    pad_token_id=tok.eos_token_id)
        cont = tok.decode(out[0, ids.shape[1]:])
        m = pat.match(cont)
        # canonical-frame reference for O to move (player 1)
        bb = g["boards"][t - 1].unsqueeze(0).float().to(DEV)
        ref = student(bb)[0]
        occ = bb[0, :, 1:-1, 1:-1].sum(0).reshape(121)
        refm = ref - 1000.0 * occ
        order = torch.argsort(refm, descending=True)
        if not m:
            continue
        n_fmt += 1
        mv = (ord(m.group(1)) - ord("A"), int(m.group(2)) - 1)
        if not (0 <= mv[1] < 11) or mv in moves:
            continue
        n_legal += 1
        # absolute -> canonical (player 1 to move => transpose)
        p_can = correct_position1d(mv[0] * 11 + mv[1], 11, 1)
        r = (order == p_can).nonzero().item()
        ranks.append(r)
        n_top1 += r == 0
    n = len(prefixes)
    t_r = torch.tensor(ranks, dtype=torch.float)
    res = {"n": n, "fmt_rate": n_fmt / n, "legal_rate": n_legal / n,
           "top1_given_legal": n_top1 / max(n_legal, 1),
           "mean_rank_given_legal": t_r.mean().item() if len(t_r) else None,
           "median_rank": t_r.median().item() if len(t_r) else None}
    print(f"gen: n {n} fmt {res['fmt_rate']:.3f} legal {res['legal_rate']:.3f}"
          f" top1|legal {res['top1_given_legal']:.3f} "
          f"mean rank {res['mean_rank_given_legal']} "
          f"median {res['median_rank']}", flush=True)
    return res


@torch.no_grad()
def gather(backbone, tok, student, games, gis, ks):
    """hs[5+k] at X-move tokens + distilled argmax targets + occ."""
    recs = build_d04.build_seqs_d(tok, [games[gi] for gi in gis],
                                  numbers=False)
    H = {k: [] for k in ks}
    ys, occs = [], []
    for rec, gi in zip(recs, gis):
        bb = games[gi]["boards"][0::2].float().to(DEV)
        ref = student(bb)
        occ = bb[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
        ys.append((ref - 1000.0 * occ).argmax(-1).cpu())
        occs.append(occ.cpu())
        ids = rec["ids"].long()[None].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = backbone(input_ids=ids, output_hidden_states=True,
                         use_cache=False)
        mt = rec["mt"].long().to(DEV)
        for k in ks:
            H[k].append(o.hidden_states[5 + k].float()[0, mt].cpu())
    return ({k: torch.cat(v) for k, v in H.items()}, torch.cat(ys),
            torch.cat(occs))


def probe_eval(backbone, tok, student, games, ks, n_train=600, n_val=40,
               steps=2000, seed=7):
    rng = random.Random(seed)
    val_gis = [gi for gi in range(len(games)) if gi % 15 == 0]
    tr_gis = rng.sample([gi for gi in range(len(games)) if gi % 15 != 0],
                        n_train)
    va_gis = rng.sample(val_gis, n_val)
    Htr, ytr, _ = gather(backbone, tok, student, games, tr_gis, ks)
    Hva, yva, occva = gather(backbone, tok, student, games, va_gis, ks)
    res = {}
    for k in ks:
        X, y = Htr[k].to(DEV), ytr.to(DEV)
        Xv, yv = Hva[k].to(DEV), yva.to(DEV)
        probe = nn.Linear(2048, 121).to(DEV)
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3,
                                weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for s in range(steps):
            idx = torch.randint(0, len(X), (4096,), device=DEV)
            loss = nn.functional.cross_entropy(probe(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            lg = probe(Xv) - 1000.0 * occva.to(DEV)
            top1 = (lg.argmax(-1) == yv).float().mean().item()
            top3 = (lg.topk(3, -1).indices == yv[:, None]).any(-1)
            top3 = top3.float().mean().item()
        res[k] = {"top1": top1, "top3": top3,
                  "n_train": len(X), "n_val": len(Xv)}
        print(f"probe k={k}: val top1 {top1:.3f} top3 {top3:.3f} "
              f"(train {len(X)} pos)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", action="store_true")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    cnn = W.load_model()
    student = R4.load_student(cnn)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    ckpt = "checkpoints/armF_polish19b/final.pt"
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]

    res = {}
    if args.gen:
        lm = load_spliced_lm(ckpt)
        res["gen"] = gen_eval(lm, tok, student, games)
        del lm
        torch.cuda.empty_cache()
    if args.probe:
        backbone, ads, mu, sd, step = SP.load_trained(ckpt)
        res["probe"] = probe_eval(backbone, tok, student, games,
                                  [0, 6, 12, 18])
    out = Path("armF/results/p59_policy_text.json")
    prev = json.loads(out.read_text()) if out.exists() else {}
    prev.update(res)
    out.write_text(json.dumps(prev, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
