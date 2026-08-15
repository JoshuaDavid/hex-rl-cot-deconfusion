"""P58: deconfound opening-shift vs argmax-play-shift for deep stitch readout.

Spectate stitch ref-ranks (k=18, k=0) at player-1 decisions on:
  A) orig@temp both sides — exact training-game generator (temp schedule
     1.5/1.0/0.5, eps 0.05), fresh seed. Expect val-level agreement.
  B) orig, temp schedule for first 6 plies then argmax, both sides —
     policy-shaped openings, argmax mid-game.
"""
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_movesr4 as R4  # noqa: E402
import eval_stitch_r4 as SR4  # noqa: E402
import eval_stitch_polish as SP  # noqa: E402
from probe_stitch_gap import ref_rank, summarize  # noqa: E402

DEV = "cuda"


def orig_move(model, b, ply, temp_fn, eps, gen):
    from hexhex.utils.utils import correct_position1d
    if eps and random.random() < eps:
        return random.choice(sorted(b.legal_moves))
    x = b.board_tensor.unsqueeze(0).float().to(DEV)
    lg = W.policy_logits(model, x)[0]
    t = temp_fn(ply)
    if t <= 0.01:
        p1 = lg.argmax().item()
    else:
        p1 = torch.multinomial(torch.softmax(lg / t, dim=0), 1,
                               generator=gen).item()
    p1 = correct_position1d(p1, 11, b.player)
    mv = divmod(p1, 11)
    if mv not in b.legal_moves:
        mv = random.choice(sorted(b.legal_moves))
    return mv


@torch.no_grad()
def spectate(orig, student, backbone, ads, tok, mu, sd, ks, n_games,
             temp_fn, eps, seed0):
    from hexhex.logic.hexboard import Board
    ranks = {k: [] for k in ks}
    for g in range(n_games):
        gen = torch.Generator(device=DEV).manual_seed(seed0 + g)
        b = Board(11, switch_allowed=False)
        moves = []
        while not b.winner:
            if b.player == 1:
                bb = b.board_tensor.unsqueeze(0).float().to(DEV)
                ref = student(bb)[0]
                occ = bb[0, :, 1:-1, 1:-1].sum(0).reshape(121)
                refm = ref - 1000.0 * occ
                hs = SP.hidden_last(backbone, tok, list(moves))
                for k in ks:
                    st = SR4.chat_to_logits(student, ads, hs[5 + k][None],
                                            k, mu, sd)[0] - 1000.0 * occ
                    r = ref_rank(st, refm, occ)
                    if r is not None:
                        ranks[k].append(r)
            mv = orig_move(orig, b, len(moves), temp_fn, eps, gen)
            b.set_stone(mv)
            moves.append(mv)
    return ranks


def main():
    cnn = W.load_model()
    student = R4.load_student(cnn)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    backbone, ads, mu, sd, step = SP.load_trained(
        "checkpoints/armF_polish19b/final.pt")

    def temp_train(ply):
        return 1.5 if ply < 6 else (1.0 if ply < 20 else 0.5)

    def temp_open_only(ply):
        return 1.5 if ply < 6 else 0.0

    res = {}
    random.seed(0)
    ra = spectate(cnn, student, backbone, ads, tok, mu, sd, [0, 18], 40,
                  temp_train, 0.05, 4000)
    res["A_temp_k18"] = summarize("A (temp both sides) k=18", ra[18])
    res["A_temp_k0"] = summarize("A (temp both sides) k=0", ra[0])

    random.seed(0)
    rb = spectate(cnn, student, backbone, ads, tok, mu, sd, [0, 18], 40,
                  temp_open_only, 0.0, 5000)
    res["B_argmax_k18"] = summarize("B (temp open, argmax mid) k=18", rb[18])
    res["B_argmax_k0"] = summarize("B (temp open, argmax mid) k=0", rb[0])

    Path("armF/results/p58_spectate.json").write_text(
        json.dumps(res, indent=1))
    print("wrote armF/results/p58_spectate.json")


if __name__ == "__main__":
    main()
