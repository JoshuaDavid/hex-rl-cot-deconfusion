"""P68 rescore: generations of the stitched-label head (polish_head_a400_all
_slab) vs the STITCHED readout reference (the training target: hs[23]@' X'
-> adapter18 -> distilled head argmax) AND the distilled-oracle argmax, on
the same 100 prefixes as gen_eval/p66 (seed 7).

P68a: top1_vs_stitched >= .35. P68b: vs_stitched > vs_distilled on the
same generations (faithfulness: the head should track the model's own
readout, errors included). Stitched ref is causal (payload at the ' X'
token of the last prefix move), so full-game forward + index = prefix ref.
"""
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
import build_d04  # noqa: E402
import eval_stitch_r4 as ER4  # noqa: E402
import eval_stitch_polish as SP  # noqa: E402
import p59_policy_text as P59  # noqa: E402

DEV = "cuda"
ALPHA = 400.0
CKPT = "checkpoints/armF_p60/polish_head_a400_all_slab.pt"


@torch.no_grad()
def main():
    from hexhex.utils.utils import correct_position1d
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    cnn = W.load_model()
    student = R4.load_student(cnn)
    model = P59.load_spliced_lm("checkpoints/armF_polish19b/final.pt")
    model.lm_head.weight = nn.Parameter(
        model.model.embed_tokens.weight.detach().clone())
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.lm_head.weight.copy_(ck["trainable"]["lm_head.weight"]
                               .bfloat16().to(DEV))
    model.model.layers[23].register_forward_pre_hook(
        lambda m, ar: (ar[0] * ALPHA,) + ar[1:])
    sbb, ads, mu, sd, _ = SP.load_trained("checkpoints/armF_polish19b/final.pt")

    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]
    rng = random.Random(7)
    val_gis = [gi for gi in range(len(games)) if gi % 15 == 0]
    prefixes = []
    for gi in val_gis:
        T = len(games[gi]["moves"])
        for t in range(1, T, 2):
            prefixes.append((gi, t))
    prefixes = rng.sample(prefixes, 100)

    # per-game stitched + distilled argmaxes at X plies (index j = t//2)
    need = sorted({gi for gi, _ in prefixes})
    srecs = build_d04.build_seqs_d(tok, [games[gi] for gi in need],
                                   numbers=False)
    refs = {}
    for rec, gi in zip(srecs, need):
        g = games[gi]
        bb = g["boards"][0::2].float().to(DEV)
        occ = bb[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
        ids_ = rec["ids"].long()[None].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = sbb(input_ids=ids_, output_hidden_states=True,
                    use_cache=False)
        h = o.hidden_states[23].float()[0, rec["mt"].long()]
        lg_s = ER4.chat_to_logits(student, ads, h, 18, mu, sd)
        am_s = (lg_s - 1000.0 * occ).argmax(-1).cpu()
        am_d = (student(bb) - 1000.0 * occ).argmax(-1).cpu()
        refs[gi] = (am_s, am_d)

    pat = re.compile(r"^\n([A-K])(\d{2})")
    n_legal = 0
    hit_s = hit_d = agree_sd = 0
    for gi, t in prefixes:
        g = games[gi]
        moves = [tuple(m) for m in g["moves"][:t].tolist()]
        txt = SP.d04n_text(moves)
        ids = tok(txt, return_tensors="pt",
                  add_special_tokens=False)["input_ids"].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(ids, max_new_tokens=6, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        cont = tok.decode(out[0, ids.shape[1]:])
        j = (t - 1) // 2
        am_s, am_d = refs[gi][0][j].item(), refs[gi][1][j].item()
        agree_sd += am_s == am_d
        m = pat.match(cont)
        if not m:
            continue
        mv = (ord(m.group(1)) - ord("A"), int(m.group(2)) - 1)
        if not (0 <= mv[1] < 11) or mv in moves:
            continue
        n_legal += 1
        p_can = correct_position1d(mv[0] * 11 + mv[1], 11, 1)
        hit_s += p_can == am_s
        hit_d += p_can == am_d

    res = {"n": 100, "legal": n_legal,
           "top1_vs_stitched": hit_s / max(n_legal, 1),
           "top1_vs_distilled": hit_d / max(n_legal, 1),
           "stitched_distilled_agree": agree_sd / 100}
    print(json.dumps(res, indent=1), flush=True)
    Path("armF/results/p68_rescore.json").write_text(json.dumps(res,
                                                                indent=1))


if __name__ == "__main__":
    main()
