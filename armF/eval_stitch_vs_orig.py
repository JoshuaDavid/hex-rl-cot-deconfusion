"""P57: stitch vs ORIGINAL HexHex CNN at temperature (no random openings).

Opponent temp gives game variety while openings stay on the training-game
manifold (gen_games temp schedule: 1.5/1.0/0.5, but no epsilon-random).
Stitch plays second (d04n protocol). Ref-ranks recorded at every stitch
decision. Baseline: distilled t=0 in the stitch's seat, same seeds.
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


def make_orig_temp_player(model, seed):
    from hexhex.utils.utils import correct_position1d
    gen = torch.Generator(device=DEV).manual_seed(seed)

    def fn(b, moves):
        x = b.board_tensor.unsqueeze(0).float().to(DEV)
        lg = W.policy_logits(model, x)[0]
        ply = len(moves)
        t = 1.5 if ply < 6 else (1.0 if ply < 20 else 0.5)
        p1 = torch.multinomial(torch.softmax(lg / t, dim=0), 1,
                               generator=gen).item()
        p1 = correct_position1d(p1, 11, b.player)
        mv = divmod(p1, 11)
        if mv not in b.legal_moves:
            mv = random.choice(sorted(b.legal_moves))
        return mv
    return fn


@torch.no_grad()
def play_vs_orig(orig, student, n_games, stitch=None, k=None,
                 backbone=None, ads=None, tok=None, mu=None, sd=None):
    """Player 0 = orig@temp, player 1 = stitch cut k (or distilled t=0
    baseline when stitch is None). Returns (ranks, wins)."""
    from hexhex.logic.hexboard import Board
    from hexhex.utils.utils import correct_position1d
    ranks, wins = [], 0
    for g in range(n_games):
        opp = make_orig_temp_player(orig, 3000 + g)
        b = Board(11, switch_allowed=False)
        moves = []
        while not b.winner:
            if b.player == 1:
                bb = b.board_tensor.unsqueeze(0).float().to(DEV)
                ref = student(bb)[0]
                occ = bb[0, :, 1:-1, 1:-1].sum(0).reshape(121)
                refm = ref - 1000.0 * occ
                if stitch is None:
                    lg = refm
                else:
                    hs = SP.hidden_last(backbone, tok, list(moves))
                    lg = SR4.chat_to_logits(student, ads, hs[5 + k][None],
                                            k, mu, sd)[0] - 1000.0 * occ
                    r = ref_rank(lg, refm, occ)
                    if r is not None:
                        ranks.append(r)
                p1 = correct_position1d(lg.argmax().item(), 11, b.player)
                mv = divmod(p1, 11)
                if mv not in b.legal_moves:
                    mv = random.choice(sorted(b.legal_moves))
            else:
                mv = opp(b, moves)
            b.set_stone(mv)
            moves.append(mv)
        wins += b.winner == [1]
    return ranks, wins


def main():
    cnn = W.load_model()
    student = R4.load_student(cnn)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    backbone, ads, mu, sd, step = SP.load_trained(
        "checkpoints/armF_polish19b/final.pt")

    n = 40
    res = {}
    random.seed(0)
    _, wb = play_vs_orig(cnn, student, n)
    print(f"baseline distilled t=0 vs orig@temp: {wb}/{n}", flush=True)
    res["baseline_distilled"] = {"wins": wb, "n": n}

    for k in [18, 0]:
        random.seed(0)
        rk, wk = play_vs_orig(cnn, student, n, stitch=True, k=k,
                              backbone=backbone, ads=ads, tok=tok,
                              mu=mu, sd=sd)
        print(f"cut {k} vs orig@temp: {wk}/{n}", flush=True)
        res[f"cut{k}"] = summarize(f"cut{k} ranks", rk) | {"wins": wk}

    Path("armF/results/stitch_vs_orig.json").write_text(
        json.dumps(res, indent=1))
    print("wrote armF/results/stitch_vs_orig.json")


if __name__ == "__main__":
    main()
