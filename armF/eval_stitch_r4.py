"""Arm F r4 stitching eval: Qwen reads the numbered move list, adapter k at
the LAST move token reconstructs the distilled net's native state c_k
(R^1024); dec_k + remaining bottlenecked skiplayers + policy head finish.

Reference player is the DISTILLED net (bottleneck_anchored_ext, 1867 Elo
@ t=0), not the original CNN. Play at cuts with BOTH 4-ply paired openings
(comparability with r1-r3) and 1-ply openings (honest protocol per the
finger-E Elo caveat).
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
import eval_stitch_full as EF  # noqa: E402
import fingerE_bottleneck as B  # noqa: E402
import train_movesr4 as R4  # noqa: E402

DEV = "cuda"
L = 19
K = 1024


def load_trained(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = T.load_backbone(random_init=True)  # arch only; overwritten
    sd_ = {k: v.float() for k, v in ck["backbone"].items()}
    missing, _ = backbone.load_state_dict(sd_, strict=False)
    assert not [m for m in missing if "rotary" not in m], missing
    ads = nn.ModuleList([nn.Linear(2048, K) for _ in range(L)]).to(DEV)
    ads.load_state_dict({k: v.float() for k, v in ck["ads"].items()})
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    for p in ads.parameters():
        p.requires_grad_(False)
    d = torch.load("armF/results/r4_cstats.pt", weights_only=False)
    return backbone, ads, d["mu"].to(DEV), d["sd"].to(DEV), ck.get("step")


@torch.no_grad()
def stitched_c_logits(student, c, k):
    """c: (B,1024) native state at capture k. Returns (B,121) raw logits
    (occ mask NOT applied)."""
    m = student.inner
    h = student.bns[k].dec(c).reshape(-1, 64, 11, 11)
    for l in range(k, L - 1):
        h = student.bns[l + 1](m.skiplayers[l](h))
    return m.policyconv(h).view(-1, 121) + m.bias


@torch.no_grad()
def check_exact(student, games):
    """Stitching from TRUE c_k must reproduce the distilled net's logits."""
    bb = games[0]["boards"][2:8].float().to(DEV)
    ref = student(bb)
    occ = bb[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
    cs = R4.dump_c(student, bb)
    for k in (0, 9, 18):
        st = stitched_c_logits(student, cs[k], k) - 1000.0 * occ
        assert torch.allclose(st, ref, atol=1e-3), \
            (k, (st - ref).abs().max().item())
    print("exactness anchor: true-c stitch == distilled forward at k=0,9,18")


def chat_to_logits(student, ads, h, k, mu, sd):
    """h: (B,2048) last-move-token states -> (B,121) raw logits."""
    c = ads[k](h) * sd[k] + mu[k]
    return stitched_c_logits(student, c, k)


@torch.no_grad()
def agreement_eval(backbone, ads, tok, student, games, mu, sd, n, batch=16,
                   seed=7):
    rng = random.Random(seed)
    val_gis = [gi for gi in range(len(games)) if gi % 15 == 0]
    picks = [(gi, rng.randint(2, len(games[gi]["moves"])))
             for gi in rng.choices(val_gis, k=n)]
    res = {k: {"top1": 0, "top3": 0, "sp": []} for k in range(L)}
    for i in range(0, n, batch):
        chunk = picks[i:i+batch]
        moves_b = [games[gi]["moves"][:cut].tolist() for gi, cut in chunk]
        bb = torch.stack([games[gi]["boards"][cut - 1]
                          for gi, cut in chunk]).float().to(DEV)
        ref = student(bb)
        occ = bb[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
        hs = EF.hidden_at_last_move(backbone, tok, moves_b)
        for k in range(L):
            st = (chat_to_logits(student, ads, hs[5 + k], k, mu, sd)
                  - 1000.0 * occ)
            res[k]["top1"] += (st.argmax(-1) == ref.argmax(-1)).sum().item()
            top3 = st.topk(3, dim=-1).indices
            res[k]["top3"] += (top3 == ref.argmax(-1, keepdim=True)
                               ).any(-1).sum().item()
            for j in range(len(bb)):
                legal = occ[j] < 0.5
                res[k]["sp"].append(ES.spearman(st[j][legal], ref[j][legal]))
    return {k: {"top1": r["top1"] / n, "top3": r["top3"] / n,
                "spearman": sum(r["sp"]) / len(r["sp"])}
            for k, r in res.items()}


def make_stitched_player(backbone, ads, tok, student, k, mu, sd, illegal_ctr):
    from hexhex.utils.utils import correct_position1d

    def fn(b, moves):
        hs = EF.hidden_at_last_move(backbone, tok, [list(moves)])
        lg = chat_to_logits(student, ads, hs[5 + k], k, mu, sd)[0]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/armF_movesr4/best.pt")
    ap.add_argument("--n-agree", type=int, default=304)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--cuts-play", default="0,9,18")
    ap.add_argument("--out", default="armF/results/stitch_eval_r4.json")
    args = ap.parse_args()

    cnn = W.load_model()
    student = R4.load_student(cnn)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    tok.padding_side = "right"
    backbone, ads, mu, sd, step = load_trained(args.ckpt)
    print(f"loaded ckpt step {step}")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    games += torch.load("armF/data/games2.pt", weights_only=False)["games"]

    check_exact(student, games)

    agree = agreement_eval(backbone, ads, tok, student, games, mu, sd,
                           args.n_agree)
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
        w_rand = EM.play_games(sp, EM.make_random_player(), args.games)
        w4 = EM.play_games(sp, B.make_bn_player(student, [0]),
                           args.games * 2, opening_plies=4)
        w1 = EM.play_games(sp, B.make_bn_player(student, [0]),
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
    out.write_text(json.dumps({"step": step, "agreement": agree,
                               "play": play}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
