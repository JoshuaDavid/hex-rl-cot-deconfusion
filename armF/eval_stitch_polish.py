"""Stitch eval of the polished full-19 joint backbone (armF_polish19b).

d04n format. Supervision exists ONLY at X-move color tokens (boards[0::2]),
so agreement uses even-ply positions and the stitched player always plays
SECOND (every decision follows an X move) — protocol difference vs
eval_stitch_r4 (both sides); noted in results. Decode path (dec_k + remaining
skiplayers + policy head) and exactness anchor reused from eval_stitch_r4.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_containment as T  # noqa: E402
import eval_stitch as ES  # noqa: E402
import eval_stitch_moves as EM  # noqa: E402
import fingerE_bottleneck as B  # noqa: E402
import train_movesr4 as R4  # noqa: E402
import eval_stitch_r4 as SR4  # noqa: E402
import build_d04  # noqa: E402

DEV = "cuda"
L = 19
K = 1024


def load_trained(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = T.load_backbone(random_init=True)
    sd_ = {k: v.float() for k, v in ck["backbone"].items()}
    missing, _ = backbone.load_state_dict(sd_, strict=False)
    assert not [m for m in missing if "rotary" not in m], missing
    ads = {l: nn.Linear(2048, K).to(DEV) for l in range(L)}
    for l in range(L):
        ads[l].load_state_dict(
            {k: v.float() for k, v in ck["ads"][l].items()})
    backbone.to(DEV).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    for a in ads.values():
        for p in a.parameters():
            p.requires_grad_(False)
    d = torch.load("armF/results/r4_cstats.pt", weights_only=False)
    return backbone, ads, d["mu"].to(DEV), d["sd"].to(DEV), ck.get("step")


def d04n_text(moves):
    parts = [build_d04.RM.PREAMBLE_M, build_d04.R4X.HDR]
    for t, mv in enumerate(moves):
        parts.append(f"\n{build_d04.cell_str(mv)} "
                     f"{'X' if t % 2 == 0 else 'O'}")
    return "".join(parts)


@torch.no_grad()
def agreement_eval(backbone, ads, tok, student, games, mu, sd, n_games,
                   seed=7):
    """All X plies of n_games val games, all 19 cuts."""
    rng = random.Random(seed)
    val_gis = [gi for gi in range(len(games)) if gi % 15 == 0]
    gis = rng.sample(val_gis, n_games)
    recs = build_d04.build_seqs_d(tok, [games[gi] for gi in gis],
                                  numbers=False)
    res = {k: {"top1": 0, "top3": 0, "sp": []} for k in range(L)}
    n = 0
    for rec, gi in zip(recs, gis):
        bb = games[gi]["boards"][0::2].float().to(DEV)
        ref = student(bb)
        occ = bb[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
        ids = rec["ids"].long()[None].to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = backbone(input_ids=ids, output_hidden_states=True,
                           use_cache=False)
        mt = rec["mt"].long().to(DEV)
        assert len(mt) == len(bb)
        for k in range(L):
            h = out.hidden_states[5 + k].float()[0, mt]
            st = (SR4.chat_to_logits(student, ads, h, k, mu, sd)
                  - 1000.0 * occ)
            res[k]["top1"] += (st.argmax(-1) == ref.argmax(-1)).sum().item()
            top3 = st.topk(3, dim=-1).indices
            res[k]["top3"] += (top3 == ref.argmax(-1, keepdim=True)
                               ).any(-1).sum().item()
            for j in range(len(bb)):
                legal = occ[j] < 0.5
                res[k]["sp"].append(ES.spearman(st[j][legal], ref[j][legal]))
        n += len(bb)
    return {k: {"top1": r["top1"] / n, "top3": r["top3"] / n,
                "spearman": sum(r["sp"]) / len(r["sp"])}
            for k, r in res.items()}, n


@torch.no_grad()
def hidden_last(backbone, tok, moves):
    enc = tok(d04n_text(moves), return_tensors="pt",
              add_special_tokens=False)
    ids = enc["input_ids"].to(DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = backbone(input_ids=ids, output_hidden_states=True,
                       use_cache=False)
    return [hs.float()[0, -1] for hs in out.hidden_states]


def make_stitched_player(backbone, ads, tok, student, k, mu, sd, illegal_ctr):
    from hexhex.utils.utils import correct_position1d

    def fn(b, moves):
        assert len(moves) % 2 == 1  # plays second: last move is X's
        hs = hidden_last(backbone, tok, list(moves))
        lg = SR4.chat_to_logits(student, ads, hs[5 + k][None], k, mu, sd)[0]
        occ = b.board_tensor.to(torch.uint8).float()[
            :, 1:-1, 1:-1].sum(0).reshape(121).to(DEV)
        lg = lg - 1000.0 * occ
        p1 = correct_position1d(lg.argmax().item(), 11, b.player)
        mv = divmod(p1, 11)
        if mv not in b.legal_moves:
            illegal_ctr[0] += 1
            mv = random.choice(sorted(b.legal_moves))
        return mv
    return fn


def play_games_p2(move_fn, opp_fn, n_games, opening_plies=4):
    """move_fn always plays player 1 (second); fresh opening per game."""
    from hexhex.logic.hexboard import Board
    wins = 0
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
            fn = move_fn if b.player == 1 else opp_fn
            mv = fn(b, moves)
            b.set_stone(mv)
            moves.append(mv)
        if b.winner == [1]:
            wins += 1
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/armF_polish19b/final.pt")
    ap.add_argument("--n-agree-games", type=int, default=40)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--cuts-play", default="0,9,18")
    ap.add_argument("--out", default="armF/results/stitch_eval_polish.json")
    args = ap.parse_args()

    cnn = W.load_model()
    student = R4.load_student(cnn)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    backbone, ads, mu, sd, step = load_trained(args.ckpt)
    print(f"loaded ckpt step {step}")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]

    SR4.check_exact(student, games)

    agree, n_pos = agreement_eval(backbone, ads, tok, student, games, mu, sd,
                                  args.n_agree_games)
    print(f"agreement over {n_pos} X-ply positions")
    print("cut | top1 | top3 | spearman")
    for k in range(L):
        print(f" {k:2d} | {agree[k]['top1']:.3f} | {agree[k]['top3']:.3f} | "
              f"{agree[k]['spearman']:.3f}", flush=True)

    random.seed(0)
    play = {}
    for k in [int(c) for c in args.cuts_play.split(",")]:
        ictr = [0]
        sp = make_stitched_player(backbone, ads, tok, student, k, mu, sd,
                                  ictr)
        w_rand = play_games_p2(sp, EM.make_random_player(), args.games)
        w4 = play_games_p2(sp, B.make_bn_player(student, [0]),
                           args.games * 2, opening_plies=4)
        w1 = play_games_p2(sp, B.make_bn_player(student, [0]),
                           args.games * 2, opening_plies=1)
        play[k] = {"vs_random": f"{w_rand}/{args.games}",
                   "vs_dist_4ply": f"{w4}/{args.games*2}",
                   "vs_dist_1ply": f"{w1}/{args.games*2}",
                   "illegal_argmax": ictr[0]}
        print(f"cut {k}: vs random {w_rand}/{args.games}, vs distilled "
              f"4-ply {w4}/{args.games*2}, 1-ply {w1}/{args.games*2}, "
              f"illegal argmax {ictr[0]}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"step": step, "protocol": "player2-only, X-ply agreement",
         "agreement": agree, "play": play}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
