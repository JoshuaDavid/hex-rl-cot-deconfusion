"""P65: does the payload survive to the newline slot (where the cell
letter is emitted)?

Linear 2048->121 CE probes at NEWLINE tokens (position mt+1, the '\n'
right after each ' X' color token) on the spliced polish model, at
hs[23] (raw) and hs[28] under alpha=400 (the P64 generation path).
Targets = same distilled argmax as the color-token probes (.478/.394
ceilings). Color-token hs[23] re-run as sanity anchor.
"""
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_movesr4 as R4  # noqa: E402
import build_d04  # noqa: E402
import p59_policy_text as P59  # noqa: E402
from p61_depth_probe import fit_probe  # noqa: E402

DEV = "cuda"
ALPHA = 400.0


@torch.no_grad()
def gather_nl(model, tok, student, games, gis, ks):
    """hs at newline (mt+1) AND color (mt) tokens; drop plies whose move
    line is the last in the sequence (no following newline)."""
    recs = build_d04.build_seqs_d(tok, [games[gi] for gi in gis],
                                  numbers=False)
    H = {(k, w): [] for k in ks for w in ("nl", "col")}
    ys, occs = [], []
    for rec, gi in zip(recs, gis):
        bb = games[gi]["boards"][0::2].float().to(DEV)
        ref = student(bb)
        occ = bb[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
        y = (ref - 1000.0 * occ).argmax(-1).cpu()
        ids = rec["ids"].long()[None].to(DEV)
        mt = rec["mt"].long()
        keep = (mt + 1) < ids.shape[1]
        mt = mt[keep].to(DEV)
        ys.append(y[keep.cpu()])
        occs.append(occ.cpu()[keep.cpu()])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(input_ids=ids, output_hidden_states=True,
                      use_cache=False)
        for k in ks:
            hs = o.hidden_states[k].float()[0]
            H[(k, "nl")].append(hs[mt + 1].cpu())
            H[(k, "col")].append(hs[mt].cpu())
    return ({kk: torch.cat(v) for kk, v in H.items()}, torch.cat(ys),
            torch.cat(occs))


def main():
    cnn = W.load_model()
    student = R4.load_student(cnn)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = P59.load_spliced_lm("checkpoints/armF_polish19b/final.pt")
    model.model.layers[23].register_forward_pre_hook(
        lambda m, ar: (ar[0] * ALPHA,) + ar[1:])

    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]
    rng = random.Random(7)
    val_gis = [gi for gi in range(len(games)) if gi % 15 == 0]
    tr_gis = rng.sample([gi for gi in range(len(games)) if gi % 15 != 0],
                        600)
    va_gis = rng.sample(val_gis, 40)

    ks = [23, 28]
    Htr, ytr, _ = gather_nl(model, tok, student, games, tr_gis, ks)
    Hva, yva, occva = gather_nl(model, tok, student, games, va_gis, ks)

    res = {}
    for kk in [(23, "col"), (23, "nl"), (28, "nl")]:
        t1, t3 = fit_probe(Htr[kk], ytr, Hva[kk], yva, occva)
        name = f"hs[{kk[0]}] @ {kk[1]}"
        res[f"{kk[0]}_{kk[1]}"] = {"top1": t1, "top3": t3}
        print(f"{name}: top1 {t1:.3f} top3 {t3:.3f}", flush=True)

    Path("armF/results/p65_newline_probe.json").write_text(
        json.dumps(res, indent=1))
    print("wrote armF/results/p65_newline_probe.json")


if __name__ == "__main__":
    main()
