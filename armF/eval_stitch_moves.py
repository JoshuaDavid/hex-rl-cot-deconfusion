"""r2 stitching eval, moves format: Qwen reads preamble + move list + single
render; adapter A_k at the 121 render cell tokens -> z_k-hat -> CNN skiplayers
[k:] -> policy head.

Agreement: reuses val sequences from tokens_moves.pt (batched, all 19 cuts per
forward). Play: prompts built live each ply via render_moves on the running
move history (board cast to uint8 to match training data's truncation of the
0.001 marker). Openings are used vs BOTH opponents since the format needs >=2
listed moves (training cuts started at 2).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import hexhex_wrap as W  # noqa: E402
import train_containment as T  # noqa: E402
import render_moves as RM  # noqa: E402
import eval_stitch as ES  # noqa: E402

sys.path.insert(0, str(W.HEXHEX_ROOT))
from hexhex.logic.hexboard import Board  # noqa: E402
from hexhex.utils.utils import correct_position1d  # noqa: E402

DEV = "cuda"


def load_trained(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = T.load_backbone(random_init=True)  # arch only; overwritten
    sd_ = {k: v.float() for k, v in ck["backbone"].items()}
    missing, _ = backbone.load_state_dict(sd_, strict=False)
    assert not [m for m in missing if "rotary" not in m], missing
    adA = T.Adapters().to(DEV)
    adA.load_state_dict(ck["adA"])
    d = torch.load("armF/results/probe_frozen.pt", weights_only=False)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone, adA, d["mu"].to(DEV), d["sd"].to(DEV), ck.get("step")


def zhat_to_logits(cnn, zn, k, mu, sd):
    """zn: (B,121,64) normalized cell readouts in cell order c=x*11+y."""
    z = zn * sd[k][None, None] + mu[k][None, None]
    z = z.reshape(-1, 11, 11, 64).permute(0, 3, 1, 2).contiguous()
    return W.stitched_logits(cnn, z, k)


@torch.no_grad()
def stitched_logits_live(backbone, adA, tok, cnn, moves, board_c, k, mu, sd):
    """One live position. board_c: (2,13,13) canonical uint8-truncated float."""
    text, _, cell_off = RM.render_moves(moves, board_c)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ct = torch.tensor(RM.cell_token_indices(enc["offset_mapping"], cell_off),
                      device=DEV)
    ids = torch.tensor(enc["input_ids"], device=DEV)[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = backbone(input_ids=ids, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[5 + k].float()[0, ct]
    logits = zhat_to_logits(cnn, adA.maps[k](h)[None], k, mu, sd)[0]
    occ = board_c[:, 1:-1, 1:-1].sum(0).reshape(121).to(DEV)
    return logits - 1000.0 * occ


@torch.no_grad()
def agreement_eval(backbone, adA, cnn, games, toks, idx, mu, sd, batch=16):
    res = {k: {"top1": 0, "top3": 0, "sp": []} for k in range(19)}
    tot = 0
    for i in range(0, len(idx), batch):
        sel = idx[i:i+batch]
        ids = toks["input_ids"][sel].long().to(DEV)
        Tlen = int(toks["lens"][sel].max())
        ids = ids[:, :Tlen]
        am = (torch.arange(Tlen, device=DEV)[None]
              < toks["lens"][sel][:, None].to(DEV)).long()
        ct = toks["cell_idx"][sel].long().to(DEV)
        bb = torch.stack([games[int(toks["game_id"][j])]["boards"]
                          [int(toks["cuts"][j]) - 1]
                          for j in sel.tolist()]).float()
        ref = ES.pure_cnn_logits(cnn, bb)
        occ = bb[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = backbone(input_ids=ids, attention_mask=am,
                           output_hidden_states=True, use_cache=False)
        for k in range(19):
            h = torch.gather(out.hidden_states[5 + k].float(), 1,
                             ct.unsqueeze(-1).expand(-1, -1, 2048))
            st = zhat_to_logits(cnn, adA.maps[k](h), k, mu, sd)
            st = st - 1000.0 * occ.to(DEV)
            res[k]["top1"] += (st.argmax(-1) == ref.argmax(-1)).sum().item()
            top3 = st.topk(3, dim=-1).indices
            res[k]["top3"] += (top3 == ref.argmax(-1, keepdim=True)
                               ).any(-1).sum().item()
            for j in range(len(bb)):
                legal = (occ[j] < 0.5).to(DEV)
                res[k]["sp"].append(ES.spearman(st[j][legal], ref[j][legal]))
        tot += len(bb)
    return {k: {"top1": r["top1"] / tot, "top3": r["top3"] / tot,
                "spearman": sum(r["sp"]) / len(r["sp"])}
            for k, r in res.items()}


def play_games(move_fn_a, move_fn_b, n_games, opening_plies=4):
    """move_fn(b, moves) -> absolute move. Shared opening per game pair."""
    wins = 0
    for g in range(n_games):
        if g % 2 == 0:
            rng = random.Random(1000 + g)
            opening = None
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
        a_is = g % 2
        while not b.winner:
            fn = move_fn_a if b.player == a_is else move_fn_b
            mv = fn(b, moves)
            b.set_stone(mv)
            moves.append(mv)
        if b.winner == [a_is]:
            wins += 1
    return wins


def make_stitched_player(backbone, adA, tok, cnn, k, mu, sd, illegal_ctr):
    def fn(b, moves):
        board_c = b.board_tensor.to(torch.uint8).float()
        lg = stitched_logits_live(backbone, adA, tok, cnn, moves, board_c,
                                  k, mu, sd)
        p1 = correct_position1d(lg.argmax().item(), 11, b.player)
        mv = divmod(p1, 11)
        if mv not in b.legal_moves:
            illegal_ctr[0] += 1
            mv = random.choice(sorted(b.legal_moves))
        return mv
    return fn


def make_cnn_player(cnn):
    def fn(b, moves):
        lg = ES.pure_cnn_logits(cnn, b.board_tensor.unsqueeze(0).float())[0]
        p1 = correct_position1d(lg.argmax().item(), 11, b.player)
        return divmod(p1, 11)
    return fn


def make_random_player():
    def fn(b, moves):
        return random.choice(sorted(b.legal_moves))
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/armF_moves_r2/best.pt")
    ap.add_argument("--n-agree", type=int, default=304)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--cuts-play", default="0,9,18")
    ap.add_argument("--out", default="armF/results/stitch_eval_moves.json")
    args = ap.parse_args()

    cnn = W.load_model()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    backbone, adA, mu, sd, step = load_trained(args.ckpt)
    print(f"loaded ckpt step {step}")

    games = torch.load("armF/data/games.pt", weights_only=False)["games"]
    toks = torch.load("armF/data/tokens_moves.pt", weights_only=False)
    val = torch.nonzero(torch.tensor(
        [g % 15 == 0 for g in toks["game_id"].tolist()])).squeeze(1)
    idx = val[:args.n_agree]
    agree = agreement_eval(backbone, adA, cnn, games, toks, idx, mu, sd)
    print("cut | top1 | top3 | spearman")
    for k in range(19):
        print(f" {k:2d} | {agree[k]['top1']:.3f} | {agree[k]['top3']:.3f} | "
              f"{agree[k]['spearman']:.3f}", flush=True)

    random.seed(0)
    play = {}
    for k in [int(c) for c in args.cuts_play.split(",")]:
        ictr = [0]
        sp = make_stitched_player(backbone, adA, tok, cnn, k, mu, sd, ictr)
        w_rand = play_games(sp, make_random_player(), args.games)
        w_cnn = play_games(sp, make_cnn_player(cnn), args.games * 2)
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
