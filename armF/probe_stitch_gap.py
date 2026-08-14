"""P55: dissect the cut18 agreement-play dissociation.

A) Replay the cut18 stitch-vs-distilled 4-ply games (same seeds as
   eval_stitch_polish) and measure agreement + ref-rank of the stitch's
   argmax ON THE ACTUAL DECISION POSITIONS (own-play distribution).
B) On val-game positions, ref-rank blunder tails at k=0 vs k=18.
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
import fingerE_bottleneck as B  # noqa: E402
import build_d04  # noqa: E402

DEV = "cuda"


def ref_rank(st, ref, occ):
    """Rank of st's argmax in ref's legal-move order (0 = agree)."""
    legal = occ < 0.5
    if not legal.any():
        return None
    pick = st[legal].argmax()
    order = torch.argsort(ref[legal], descending=True)
    return (order == pick).nonzero().item()


@torch.no_grad()
def replay_cut(backbone, ads, tok, student, k, mu, sd, n_games,
               opening_plies=4):
    from hexhex.logic.hexboard import Board
    from hexhex.utils.utils import correct_position1d
    ranks, wins = [], 0
    for g in range(n_games):
        opening = None
        rng = random.Random(2000 + g)
        while opening is None:
            b = Board(11, switch_allowed=False)
            mvs = []
            for _ in range(opening_plies):
                mv = rng.choice(sorted(b.legal_moves))
                mvs.append(mv)
                b.set_stone(mv)
            if not b.winner:
                opening = mvs
        b = Board(11, switch_allowed=False)
        moves = []
        for mv in opening:
            b.set_stone(mv)
            moves.append(mv)
        while not b.winner:
            if b.player == 1:  # stitch decision: measure then move
                bb = b.board_tensor.unsqueeze(0).float().to(DEV)
                ref = student(bb)[0]
                occ = bb[0, :, 1:-1, 1:-1].sum(0).reshape(121)
                hs = SP.hidden_last(backbone, tok, list(moves))
                st = SR4.chat_to_logits(student, ads, hs[5 + k][None],
                                        k, mu, sd)[0]
                st = st - 1000.0 * occ
                refm = ref - 1000.0 * occ
                ranks.append(ref_rank(st, refm, occ))
                p1 = correct_position1d(st.argmax().item(), 11, b.player)
                mv = divmod(p1, 11)
                if mv not in b.legal_moves:
                    mv = random.choice(sorted(b.legal_moves))
            else:
                mv = B.make_bn_player(student, [0])(b, moves)
            b.set_stone(mv)
            moves.append(mv)
        wins += b.winner == [1]
    return ranks, wins


@torch.no_grad()
def val_ranks(backbone, ads, tok, student, games, mu, sd, ks, n_games,
              seed=7):
    rng = random.Random(seed)
    val_gis = [gi for gi in range(len(games)) if gi % 15 == 0]
    gis = rng.sample(val_gis, n_games)
    recs = build_d04.build_seqs_d(tok, [games[gi] for gi in gis],
                                  numbers=False)
    out = {k: [] for k in ks}
    for rec, gi in zip(recs, gis):
        bb = games[gi]["boards"][0::2].float().to(DEV)
        ref = student(bb)
        occ = bb[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
        ids = rec["ids"].long()[None].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = backbone(input_ids=ids, output_hidden_states=True,
                         use_cache=False)
        mt = rec["mt"].long().to(DEV)
        for k in ks:
            h = o.hidden_states[5 + k].float()[0, mt]
            st = (SR4.chat_to_logits(student, ads, h, k, mu, sd)
                  - 1000.0 * occ)
            refm = ref - 1000.0 * occ
            for j in range(len(bb)):
                r = ref_rank(st[j], refm[j], occ[j])
                if r is not None:
                    out[k].append(r)
    return out


def summarize(name, ranks):
    n = len(ranks)
    t = torch.tensor(ranks, dtype=torch.float)
    s = {"n": n, "top1": (t == 0).float().mean().item(),
         "rank_gt5": (t > 5).float().mean().item(),
         "rank_gt10": (t > 10).float().mean().item(),
         "mean_rank": t.mean().item(), "max_rank": t.max().item()}
    print(f"{name}: n {n} top1 {s['top1']:.3f} P(rank>5) {s['rank_gt5']:.3f} "
          f"P(rank>10) {s['rank_gt10']:.3f} mean {s['mean_rank']:.2f} "
          f"max {int(s['max_rank'])}", flush=True)
    return s


def main():
    cnn = W.load_model()
    student = R4.load_student(cnn)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    backbone, ads, mu, sd, step = SP.load_trained(
        "checkpoints/armF_polish19b/final.pt")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]

    res = {}
    vr = val_ranks(backbone, ads, tok, student, games, mu, sd, [0, 18], 40)
    res["val_k0"] = summarize("val k=0", vr[0])
    res["val_k18"] = summarize("val k=18", vr[18])

    random.seed(0)
    ranks18, w18 = replay_cut(backbone, ads, tok, student, 18, mu, sd, 40)
    print(f"replay cut18 wins {w18}/40 (eval had 3/40)")
    res["ownplay_k18"] = summarize("own-play k=18", ranks18) | {"wins": w18}

    random.seed(0)
    ranks0, w0 = replay_cut(backbone, ads, tok, student, 0, mu, sd, 40)
    print(f"replay cut0 wins {w0}/40 (eval had 26/40)")
    res["ownplay_k0"] = summarize("own-play k=0", ranks0) | {"wins": w0}

    Path("armF/results/probe_stitch_gap.json").write_text(
        json.dumps(res, indent=1))
    print("wrote armF/results/probe_stitch_gap.json")


if __name__ == "__main__":
    main()
