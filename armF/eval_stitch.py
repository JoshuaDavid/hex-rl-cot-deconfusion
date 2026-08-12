"""Stitching eval: Qwen (trained) -> A_k -> CNN layer k.. -> policy head.

For each cut k in 0..18: Qwen computes hidden_states[5+k] at cell tokens of a
position, adapter A_k maps to z_k-hat (un-normalized), the CNN's remaining
skiplayers + policy head produce move logits. Compare with the pure CNN
(same non-rotation inner path):
  - top1 move match, top3 containment (pure-CNN argmax in stitched top3)
  - spearman corr of legal-move logits
  - play strength: stitched player vs pure CNN and vs random

Cut semantics: capture point k is after skiplayer k-1 (z_0 = post-initial-conv),
so the CNN resumes at skiplayers[k:]; at k=18 nothing remains but the policy
head — Qwen has replaced the entire trunk.
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
import qwen_embed as Q  # noqa: E402

sys.path.insert(0, str(W.HEXHEX_ROOT))
from hexhex.logic.hexboard import Board  # noqa: E402
from hexhex.utils.utils import correct_position1d  # noqa: E402

DEV = "cuda"


def load_trained(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone = T.load_backbone(random_init=True)  # arch only; weights overwritten
    sd = {k: v.float() for k, v in ck["backbone"].items()}
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    assert not [m for m in missing if "rotary" not in m], missing
    adapters = T.Adapters().to(DEV)
    adapters.load_state_dict(ck["adapters"])
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    d = torch.load("armF/results/probe_frozen.pt", weights_only=False)
    return backbone, adapters, d["mu"].to(DEV), d["sd"].to(DEV), ck.get("step")


@torch.no_grad()
def stitched_move_logits(backbone, adapters, tok, cnn, boards_f, k, mu, sd):
    """boards_f: (B,2,13,13) cpu float. Returns (B,121) logits (illegal-masked)."""
    batch = Q.batch_prompts(tok, boards_f)
    ids = batch["input_ids"].to(DEV)
    am = batch["attention_mask"].to(DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = backbone(input_ids=ids, attention_mask=am,
                       output_hidden_states=True, use_cache=False)
    cell = batch["cell_idx"].to(DEV)
    h = torch.gather(out.hidden_states[5 + k].float(), 1,
                     cell.unsqueeze(-1).expand(-1, -1, 2048))
    zhat_n = adapters.maps[k](h)  # (B,121,64) normalized space
    zhat = zhat_n * sd[k][None, None, :] + mu[k][None, None, :]
    zhat = zhat.reshape(-1, 11, 11, 64).permute(0, 3, 1, 2).contiguous()
    logits = W.stitched_logits(cnn, zhat, k)  # (B,121)
    occ = boards_f[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121).to(DEV)
    return logits - 1000.0 * occ


@torch.no_grad()
def pure_cnn_logits(cnn, boards_f):
    x = boards_f.to(DEV)
    logits = W.stitched_logits(cnn, W.dump_acts(cnn, x)[0], 0)
    occ = boards_f[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121).to(DEV)
    return logits - 1000.0 * occ


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return (ra * rb).mean().item()


def agreement_eval(backbone, adapters, tok, cnn, boards, n, mu, sd, batch=32):
    idx = torch.arange(6000, 6000 + n)  # val slice
    res = {k: {"top1": 0, "top3": 0, "sp": []} for k in range(19)}
    tot = 0
    for i in range(0, n, batch):
        bb = boards[idx[i:i+batch]].float()
        ref = pure_cnn_logits(cnn, bb)
        occ = bb[:, :, 1:-1, 1:-1].sum(1).reshape(-1, 121)
        for k in range(19):
            st = stitched_move_logits(backbone, adapters, tok, cnn, bb, k, mu, sd)
            t1 = (st.argmax(-1) == ref.argmax(-1)).sum().item()
            top3 = st.topk(3, dim=-1).indices
            t3 = (top3 == ref.argmax(-1, keepdim=True)).any(-1).sum().item()
            res[k]["top1"] += t1
            res[k]["top3"] += t3
            for j in range(len(bb)):
                legal = occ[j] < 0.5
                res[k]["sp"].append(spearman(st[j][legal.to(DEV)], ref[j][legal.to(DEV)]))
        tot += len(bb)
    out = {}
    for k in range(19):
        out[k] = {"top1": res[k]["top1"] / tot, "top3": res[k]["top3"] / tot,
                  "spearman": sum(res[k]["sp"]) / len(res[k]["sp"])}
    return out


def play_games(move_fn_a, move_fn_b, n_games):
    """a plays player0 in even games. Returns wins for a."""
    wins = 0
    for g in range(n_games):
        b = Board(11, switch_allowed=False)
        a_is = g % 2
        while not b.winner:
            fn = move_fn_a if b.player == a_is else move_fn_b
            mv = fn(b)
            b.set_stone(mv)
        if b.winner == [a_is]:
            wins += 1
    return wins


def make_stitched_player(backbone, adapters, tok, cnn, k, mu, sd):
    def fn(b):
        x = b.board_tensor.unsqueeze(0).float()
        lg = stitched_move_logits(backbone, adapters, tok, cnn, x, k, mu, sd)[0]
        p1 = correct_position1d(lg.argmax().item(), 11, b.player)
        mv = divmod(p1, 11)
        if mv not in b.legal_moves:
            mv = random.choice(sorted(b.legal_moves))
        return mv
    return fn


def make_cnn_player(cnn):
    def fn(b):
        lg = pure_cnn_logits(cnn, b.board_tensor.unsqueeze(0).float())[0]
        p1 = correct_position1d(lg.argmax().item(), 11, b.player)
        return divmod(p1, 11)
    return fn


def make_random_player():
    def fn(b):
        return random.choice(sorted(b.legal_moves))
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/armF_containment_r1/best.pt")
    ap.add_argument("--n-agree", type=int, default=500)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--cuts-play", default="0,4,9,14,18")
    ap.add_argument("--out", default="armF/results/stitch_eval.json")
    args = ap.parse_args()

    cnn = W.load_model()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    backbone, adapters, mu, sd, step = load_trained(args.ckpt)
    print(f"loaded ckpt step {step}")
    boards = torch.load("armF/data/positions.pt", weights_only=False)["boards"]

    agree = agreement_eval(backbone, adapters, tok, cnn, boards, args.n_agree, mu, sd)
    print("cut | top1 | top3 | spearman")
    for k in range(19):
        print(f" {k:2d} | {agree[k]['top1']:.3f} | {agree[k]['top3']:.3f} | "
              f"{agree[k]['spearman']:.3f}")

    random.seed(0)
    play = {}
    for k in [int(c) for c in args.cuts_play.split(",")]:
        sp = make_stitched_player(backbone, adapters, tok, cnn, k, mu, sd)
        w_rand = play_games(sp, make_random_player(), args.games)
        w_cnn = play_games(sp, make_cnn_player(cnn), args.games)
        play[k] = {"vs_random": f"{w_rand}/{args.games}",
                   "vs_cnn": f"{w_cnn}/{args.games}"}
        print(f"cut {k}: vs random {w_rand}/{args.games}, vs pure CNN {w_cnn}/{args.games}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"step": step, "agreement": agree, "play": play}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
