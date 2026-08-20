"""P72 anatomy 2: WHICH own move gets re-emitted? Replay self-play games
(text player second vs distilled 4-ply, same protocol as eval_textplay) and
for every occupied-own emission record the copied move's ply offset from the
current ply. Distribution answers: own LAST move (offset 2)? induction off an
earlier X? uniform copy?"""
import collections
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_movesr4 as R4  # noqa: E402
import fingerE_bottleneck as B  # noqa: E402
import eval_stitch_polish as SP  # noqa: E402
import eval_textplay as TP  # noqa: E402

DEV = "cuda"


@torch.no_grad()
def main():
    from transformers import AutoTokenizer
    from hexhex.logic.hexboard import Board
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    cnn = W.load_model()
    student = R4.load_student(cnn)
    model = TP.load_textpath_model()
    opp = B.make_bn_player(student, [0])

    offs = collections.Counter()      # t - t0 (plies since copied own move)
    idx_frac = []                     # t0 / t (where in the game it came from)
    n_occ_own = n_occ_other = n_legal = 0
    rng = random.Random(0)
    for g in range(12):
        grng = random.Random(3000 + g)
        b = Board(11)
        moves = []
        for _ in range(4):
            mv = grng.choice(sorted(b.legal_moves))
            b.set_stone(mv)
            moves.append(mv)
        if b.winner:
            continue
        if len(moves) % 2 == 0:  # text player moves second: X to move now
            mv = opp(b, moves)
            b.set_stone(mv)
            moves.append(mv)
        while not b.winner:
            t = len(moves)
            txt = SP.d04n_text(list(moves))
            ids = tok(txt, return_tensors="pt",
                      add_special_tokens=False)["input_ids"].to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.generate(ids, max_new_tokens=6, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            cont = tok.decode(out[0, ids.shape[1]:])
            m = TP.PAT.match(cont)
            mv = None
            if m:
                mv = (ord(m.group(1)) - ord("A"), int(m.group(2)) - 1)
            if mv is not None and mv in b.legal_moves:
                n_legal += 1
                play = mv
            else:
                if mv is not None and mv in moves:
                    t0 = moves.index(mv)
                    if t0 % 2 == 1:  # own (O) earlier move
                        n_occ_own += 1
                        offs[t - t0] += 1
                        idx_frac.append(t0 / t)
                    else:
                        n_occ_other += 1
                play = rng.choice(sorted(b.legal_moves))
            b.set_stone(play)
            moves.append(play)
            if b.winner:
                break
            mv = opp(b, moves)
            b.set_stone(mv)
            moves.append(mv)

    tot = n_occ_own + n_occ_other + n_legal
    res = {"n_emissions": tot, "legal": n_legal, "occ_own": n_occ_own,
           "occ_other": n_occ_other,
           "offset_hist": dict(sorted(offs.items())),
           "frac_offset2": offs[2] / max(n_occ_own, 1),
           "frac_offset_le6": sum(v for k, v in offs.items() if k <= 6)
           / max(n_occ_own, 1),
           "mean_idx_frac": sum(idx_frac) / max(len(idx_frac), 1)}
    print(json.dumps(res, indent=1), flush=True)
    Path("armF/results/p72_whichmove.json").write_text(
        json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
