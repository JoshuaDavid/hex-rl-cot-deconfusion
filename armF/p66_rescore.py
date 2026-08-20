"""P66: rescore P64 generations vs BOTH references (distilled argmax and
orig-CNN argmax) on the same 100 prefixes as gen_eval (seed 7).

Corpus = orig-CNN temp play, so a perfect text-statistics LM greedy-decodes
~orig argmax; orig-vs-distilled argmax agreement is only ~.21 -> the old
gen top1 (scored vs distilled) had a ~.21 text-stats ceiling. Triangulate:
gen-vs-orig high & gen-vs-distilled ~ .21*that => head fits corpus stats;
gen-vs-distilled >> chance given orig => payload leakage.
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
import eval_stitch_polish as SP  # noqa: E402
import p59_policy_text as P59  # noqa: E402

DEV = "cuda"
ALPHA = 400.0


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
    ck = torch.load("checkpoints/armF_p60/polish_head_a400_all.pt",
                    map_location="cpu", weights_only=False)
    model.lm_head.weight.copy_(ck["trainable"]["lm_head.weight"]
                               .bfloat16().to(DEV))
    model.model.layers[23].register_forward_pre_hook(
        lambda m, ar: (ar[0] * ALPHA,) + ar[1:])

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

    pat = re.compile(r"^\n([A-K])(\d{2})")
    n_legal = 0
    hit_d = hit_o = agree_do = 0
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
        bb = g["boards"][t - 1].unsqueeze(0).float().to(DEV)
        occ = bb[0, :, 1:-1, 1:-1].sum(0).reshape(121)
        am_d = (student(bb)[0] - 1000.0 * occ).argmax().item()
        am_o = (W.policy_logits(cnn, bb)[0] - 1000.0 * occ).argmax().item()
        agree_do += am_d == am_o
        m = pat.match(cont)
        if not m:
            continue
        mv = (ord(m.group(1)) - ord("A"), int(m.group(2)) - 1)
        if not (0 <= mv[1] < 11) or mv in moves:
            continue
        n_legal += 1
        p_can = correct_position1d(mv[0] * 11 + mv[1], 11, 1)
        hit_d += p_can == am_d
        hit_o += p_can == am_o

    res = {"n": 100, "legal": n_legal,
           "top1_vs_distilled": hit_d / max(n_legal, 1),
           "top1_vs_orig": hit_o / max(n_legal, 1),
           "orig_distilled_agree": agree_do / 100}
    print(json.dumps(res, indent=1), flush=True)
    Path("armF/results/p66_rescore.json").write_text(json.dumps(res,
                                                                indent=1))


if __name__ == "__main__":
    main()
