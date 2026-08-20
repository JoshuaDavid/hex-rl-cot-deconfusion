"""P73: occupancy-constrained decoding for the text-path player. Instead of
greedy-generate-then-fallback, score every LEGAL cell's 3 tokens
(letter,d1,d2 after forced \\n) and play the argmax. Same protocol/opponent
as P72 (plays second vs distilled argmax, seeds 2000+g, 4-ply openings).

Side stats per position: greedy emission class (legal/occ_own/occ_other/
malformed) and whether the constrained pick equals the MASKED distilled
argmax — discriminates payload-intact-under-copy-spike vs destroyed
off-manifold.
"""
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
import eval_stitch_moves as EM  # noqa: E402
import eval_stitch_polish as SP  # noqa: E402
import eval_textplay as TP  # noqa: E402

DEV = "cuda"


def masked_am(student, b):
    from hexhex.utils.utils import correct_position1d
    lg = student(b.board_tensor.unsqueeze(0).float().to(DEV))[0]
    best, bmv = None, None
    for i in range(121):
        mv = divmod(correct_position1d(i, 11, b.player), 11)
        if mv in b.legal_moves and (best is None or lg[i] > best):
            best, bmv = lg[i], mv
    return bmv


def make_constrained_player(model, tok, student, st):
    nl = tok("\n", add_special_tokens=False)["input_ids"][0]

    @torch.no_grad()
    def fn(b, moves):
        st["n"] += 1
        txt = SP.d04n_text(list(moves))
        ctx = tok(txt, return_tensors="pt",
                  add_special_tokens=False)["input_ids"][0].tolist()
        legal = sorted(b.legal_moves)
        cand = []
        for mv in legal:
            cids = tok("\n" + build_d04.cell_str(mv),
                       add_special_tokens=False)["input_ids"]
            assert len(cids) == 4 and cids[0] == nl
            cand.append(ctx + cids)
        L = len(cand[0])
        scores = []
        for i in range(0, len(cand), 48):
            ids = torch.tensor(cand[i:i + 48], device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids).logits
            lp = F.log_softmax(logits[:, L - 4:L - 1].float(), -1)
            tgt = ids[:, L - 3:L]
            scores += lp.gather(-1, tgt[:, :, None])[:, :, 0].sum(-1).tolist()
        pick = legal[max(range(len(legal)), key=lambda i: scores[i])]

        am = masked_am(student, b)
        st["pick_eq_am"] += pick == am
        ids = torch.tensor([ctx], device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(ids, max_new_tokens=6, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        m = TP.PAT.match(tok.decode(out[0, ids.shape[1]:]))
        if not m:
            st["g_malformed"] += 1
        else:
            g = (ord(m.group(1)) - ord("A"), int(m.group(2)) - 1)
            if g in b.legal_moves:
                st["g_legal"] += 1
                st["g_legal_eq_pick"] += g == pick
            elif g in moves and moves.index(g) % 2 == 1:
                st["g_occ_own"] += 1
            else:
                st["g_occ_other"] += 1
        return pick
    return fn


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    cnn = W.load_model()
    student = R4.load_student(cnn)
    model = TP.load_textpath_model()
    import fingerE_bottleneck as B

    random.seed(0)
    st = {k: 0 for k in ("n", "pick_eq_am", "g_legal", "g_legal_eq_pick",
                         "g_malformed", "g_occ_own", "g_occ_other")}
    cp = make_constrained_player(model, tok, student, st)
    w_rand = SP.play_games_p2(cp, EM.make_random_player(), 20)
    print(f"vs random: {w_rand}/20  {st}", flush=True)
    w4 = SP.play_games_p2(cp, B.make_bn_player(student, [0]), 40,
                          opening_plies=4)
    print(f"vs distilled 4-ply: {w4}/40  {st}", flush=True)

    res = {"vs_random": f"{w_rand}/20", "vs_dist_4ply": f"{w4}/40", **st,
           "pick_eq_masked_am": st["pick_eq_am"] / max(st["n"], 1),
           "greedy_occ_rate": (st["g_occ_own"] + st["g_occ_other"])
           / max(st["n"], 1)}
    print(json.dumps(res, indent=1), flush=True)
    Path("armF/results/p73_constrained.json").write_text(
        json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
