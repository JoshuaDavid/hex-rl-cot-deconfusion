"""P61 diagnostic: where does the contained policy die between hs[23]
(block 22 out, .478 linearly recoverable) and the lm_head input?

Linear 2048->121 CE probes at hs[23..28] of the SPLICED FULL polish model
(original blocks 23..27 + final norm on top of polished 0..22), at X-move
tokens, targets = distilled argmax. hs[28] = post-norm = lm_head input.
"""
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_movesr4 as R4  # noqa: E402
import build_d04  # noqa: E402
import p59_policy_text as P59  # noqa: E402

DEV = "cuda"


@torch.no_grad()
def gather(model, tok, student, games, gis, ks):
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
            o = model(input_ids=ids, output_hidden_states=True,
                      use_cache=False)
        mt = rec["mt"].long().to(DEV)
        for k in ks:
            H[k].append(o.hidden_states[k].float()[0, mt].cpu())
    return ({k: torch.cat(v) for k, v in H.items()}, torch.cat(ys),
            torch.cat(occs))


def fit_probe(X, y, Xv, yv, occv, steps=2000):
    X, y = X.to(DEV), y.to(DEV)
    Xv, yv, occv = Xv.to(DEV), yv.to(DEV), occv.to(DEV)
    probe = nn.Linear(2048, 121).to(DEV)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        idx = torch.randint(0, len(X), (4096,), device=DEV)
        loss = nn.functional.cross_entropy(probe(X[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        lg = probe(Xv) - 1000.0 * occv
        top1 = (lg.argmax(-1) == yv).float().mean().item()
        top3 = (lg.topk(3, -1).indices == yv[:, None]).any(-1)
    return top1, top3.float().mean().item()


def main():
    cnn = W.load_model()
    student = R4.load_student(cnn)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = P59.load_spliced_lm("checkpoints/armF_polish19b/final.pt")

    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]
    rng = random.Random(7)
    val_gis = [gi for gi in range(len(games)) if gi % 15 == 0]
    tr_gis = rng.sample([gi for gi in range(len(games)) if gi % 15 != 0],
                        600)
    va_gis = rng.sample(val_gis, 40)

    ks = [23, 24, 25, 26, 27, 28]
    Htr, ytr, _ = gather(model, tok, student, games, tr_gis, ks)
    Hva, yva, occva = gather(model, tok, student, games, va_gis, ks)
    res = {}
    for k in ks:
        t1, t3 = fit_probe(Htr[k], ytr, Hva[k], yva, occva)
        name = "post-norm (lm_head input)" if k == 28 else f"block {k-1} out"
        res[k] = {"top1": t1, "top3": t3}
        print(f"hs[{k}] ({name}): top1 {t1:.3f} top3 {t3:.3f}", flush=True)

    Path("armF/results/p61_depth_probe.json").write_text(
        json.dumps(res, indent=1))
    print("wrote armF/results/p61_depth_probe.json")


if __name__ == "__main__":
    main()
