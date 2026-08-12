"""r3 render-free stitching eval: Qwen reads preamble + numbered move list
ONLY; adapter k at the LAST move token reconstructs the full z_k map -> CNN
skiplayers[k:] -> policy head.

Agreement on val games at random cut plies; play with live prompts (numbered
format). Reuses play harness conventions from eval_stitch_moves.
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
import render_moves as RM  # noqa: E402
import eval_stitch as ES  # noqa: E402
import eval_stitch_moves as EM  # noqa: E402

DEV = "cuda"
L = 19


def load_trained(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = T.load_backbone(random_init=True)  # arch only; overwritten
    sd_ = {k: v.float() for k, v in ck["backbone"].items()}
    missing, _ = backbone.load_state_dict(sd_, strict=False)
    assert not [m for m in missing if "rotary" not in m], missing
    ads = nn.ModuleList([nn.Linear(2048, 7744) for _ in range(L)]).to(DEV)
    ads.load_state_dict({k: v.float() for k, v in ck["ads"].items()})
    d = torch.load("armF/results/probe_frozen.pt", weights_only=False)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    for p in ads.parameters():
        p.requires_grad_(False)
    return backbone, ads, d["mu"].to(DEV), d["sd"].to(DEV), ck.get("step")


def numbered_text(moves):
    parts = [RM.PREAMBLE_M]
    for t, mv in enumerate(moves):
        parts.append(f"\n{t + 1}. {RM.move_str(mv)} {'X' if t % 2 == 0 else 'O'}")
    return "".join(parts)


def zhat_to_logits(cnn, ads, h, k, mu, sd):
    """h: (B,2048) last-move-token states. Returns (B,121) logits."""
    zn = ads[k](h).reshape(-1, 64, 11, 11)
    z = zn * sd[k][None, :, None, None] + mu[k][None, :, None, None]
    return W.stitched_logits(cnn, z.contiguous(), k)


@torch.no_grad()
def hidden_at_last_move(backbone, tok, moves_batch):
    """Tokenize numbered texts, forward once, return hs (25,B,2048) at each
    text's last token (== last move's color token)."""
    texts = [numbered_text(m) for m in moves_batch]
    enc = tok(texts, return_tensors="pt", padding=True,
              add_special_tokens=False)
    ids = enc["input_ids"].to(DEV)
    am = enc["attention_mask"].to(DEV)
    last = am.sum(1) - 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = backbone(input_ids=ids, attention_mask=am,
                       output_hidden_states=True, use_cache=False)
    idx = last[:, None, None].expand(-1, 1, 2048)
    return [torch.gather(hs.float(), 1, idx).squeeze(1)
            for hs in out.hidden_states]


@torch.no_grad()
def agreement_eval(backbone, ads, tok, cnn, games, mu, sd, n, batch=16,
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
                          for gi, cut in chunk]).float()
        ref = ES.pure_cnn_logits(cnn, bb)
        occ = bb[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121).to(DEV)
        hs = hidden_at_last_move(backbone, tok, moves_b)
        for k in range(L):
            st = zhat_to_logits(cnn, ads, hs[5 + k], k, mu, sd) - 1000.0 * occ
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


def make_stitched_player(backbone, ads, tok, cnn, k, mu, sd, illegal_ctr):
    from hexhex.utils.utils import correct_position1d

    def fn(b, moves):
        hs = hidden_at_last_move(backbone, tok, [list(moves)])
        lg = zhat_to_logits(cnn, ads, hs[5 + k], k, mu, sd)[0]
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
    ap.add_argument("--ckpt", default="checkpoints/armF_movesfull_r3/best.pt")
    ap.add_argument("--n-agree", type=int, default=304)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--cuts-play", default="0,9,18")
    ap.add_argument("--out", default="armF/results/stitch_eval_full.json")
    args = ap.parse_args()

    cnn = W.load_model()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    tok.padding_side = "right"
    backbone, ads, mu, sd, step = load_trained(args.ckpt)
    print(f"loaded ckpt step {step}")
    games = torch.load("armF/data/games.pt", weights_only=False)["games"]

    agree = agreement_eval(backbone, ads, tok, cnn, games, mu, sd,
                           args.n_agree)
    print("cut | top1 | top3 | spearman")
    for k in range(L):
        print(f" {k:2d} | {agree[k]['top1']:.3f} | {agree[k]['top3']:.3f} | "
              f"{agree[k]['spearman']:.3f}", flush=True)

    random.seed(0)
    play = {}
    for k in [int(c) for c in args.cuts_play.split(",")]:
        ictr = [0]
        sp = make_stitched_player(backbone, ads, tok, cnn, k, mu, sd, ictr)
        w_rand = EM.play_games(sp, EM.make_random_player(), args.games)
        w_cnn = EM.play_games(sp, EM.make_cnn_player(cnn), args.games * 2)
        play[k] = {"vs_random": f"{w_rand}/{args.games}",
                   "vs_cnn_openings": f"{w_cnn}/{args.games*2}",
                   "illegal_argmax": ictr[0]}
        print(f"cut {k}: vs random {w_rand}/{args.games}, vs pure CNN "
              f"(4-ply openings) {w_cnn}/{args.games*2}, "
              f"illegal argmax {ictr[0]}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"step": step, "agreement": agree,
                               "play": play}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
