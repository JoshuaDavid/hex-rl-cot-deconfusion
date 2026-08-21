"""P76 stitch eval: host(Qwen)-to-guest(txT12) splice plays hex.

Cut k (0..12): host renders the board (two-copy + registers), residual read at
hs[5+k] -> adapter_k -> denormalize -> guest blocks k..11 -> guest head.
k=12 = full trunk contained (only norm_f+head left in guest).

Per cut: paired 1-ply-opening games vs pure txT12 (P54-comparable protocol,
honest openings), plus on-own-play top1 agreement (stitched argmax vs guest
argmax at the stitched player's own decision points — P55 crater check).

Usage: /venv/main/bin/python armF/eval_stitch_tx.py --ckpt checkpoints/armF_p76/best.pt
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_txcontain import load_backbone, Adapters, gather_tok  # noqa: E402
from p75_baselines import batch_prompts, load_guest  # noqa: E402
from tx_train import boards_to_states, N_CELL  # noqa: E402
import qwen_embed as Q  # noqa: E402

import hexhex_wrap as W  # noqa: E402

sys.path.insert(0, str(W.HEXHEX_ROOT))
from hexhex.logic.hexboard import Board  # noqa: E402
from hexhex.utils.utils import correct_position1d  # noqa: E402

DEV = "cuda"


def load_stitch(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    backbone = load_backbone()
    backbone.load_state_dict(ck["backbone"])
    backbone.to(torch.bfloat16).eval()
    backbone.gradient_checkpointing_disable()
    adapters = Adapters().to(DEV)
    adapters.load_state_dict(ck["adapters"])
    adapters.eval()
    return backbone, adapters, ck["mu"].to(DEV), ck["sd"].to(DEV), ck.get("step")


@torch.no_grad()
def guest_from(guest, h, k):
    """h: (B,128,1024) guest state at capture point k -> masked-free logits."""
    x = h
    for blk in guest.blocks[k:]:
        x = blk(x, guest.rel)
    return guest.head(guest.norm_f(x[:, :N_CELL])).squeeze(-1)


@torch.no_grad()
def stitched_logits(backbone, adapters, tok, guest, boards_u8, k, mu, sd):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ids, am, idxs = batch_prompts(tok, boards_u8)
        out = backbone(input_ids=ids.to(DEV), attention_mask=am.to(DEV),
                       output_hidden_states=True, use_cache=False)
    h = gather_tok(out.hidden_states[5 + k].float(), idxs.to(DEV))
    z = adapters.maps[k](h) * sd[k] + mu[k]
    lg = guest_from(guest, z, k)
    occ = (boards_to_states(boards_u8) > 0).float()
    return lg - 1000.0 * occ


@torch.no_grad()
def guest_logits(guest, boards_u8):
    st = boards_to_states(boards_u8)
    return guest(st) - 1000.0 * (st > 0).float()


@torch.no_grad()
def play_cut(backbone, adapters, tok, guest, k, n_openings, seed, mu, sd):
    """Stitched (player a) vs pure guest, paired 1-ply openings. Returns
    (wins, games, own_play_agree, decisions)."""
    rng = random.Random(seed)
    games = []
    for _ in range(n_openings):
        op = divmod(rng.randrange(121), 11)
        for a_is in (0, 1):
            b = Board(11, switch_allowed=False)
            b.set_stone(op)
            games.append({"b": b, "a_is": a_is, "done": False, "awin": None})
    agree = dec = 0
    while any(not g["done"] for g in games):
        for side in (0, 1):
            idx = [i for i, g in enumerate(games)
                   if not g["done"] and g["b"].player == (g["a_is"] ^ side)]
            if not idx:
                continue
            X = torch.stack([games[i]["b"].board_tensor for i in idx]
                            ).to(torch.uint8).to(DEV)
            if side == 0:
                lg = stitched_logits(backbone, adapters, tok, guest, X, k,
                                     mu, sd)
                ref = guest_logits(guest, X)
                agree += (lg.argmax(1) == ref.argmax(1)).sum().item()
                dec += len(idx)
            else:
                lg = guest_logits(guest, X)
            picks = lg.argmax(1).cpu()
            for j, i in enumerate(idx):
                g = games[i]
                p1 = correct_position1d(picks[j].item(), 11, g["b"].player)
                mv = divmod(p1, 11)
                if mv not in g["b"].legal_moves:
                    mv = rng.choice(sorted(g["b"].legal_moves))
                g["b"].set_stone(mv)
                if g["b"].winner or not g["b"].legal_moves:
                    g["done"] = True
                    g["awin"] = g["b"].winner == [g["a_is"]]
    wins = sum(g["awin"] for g in games)
    return wins, len(games), agree / max(dec, 1), dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/armF_p76/best.pt")
    ap.add_argument("--cuts", type=int, nargs="+", default=[0, 4, 8, 12])
    ap.add_argument("--openings", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="armF/results/p76_stitch.json")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Q.QWEN)
    backbone, adapters, mu, sd, step = load_stitch(args.ckpt)
    guest = load_guest()
    print(f"stitch ckpt step {step}", flush=True)

    res = {"ckpt": args.ckpt, "step": step, "cuts": {}}
    for k in args.cuts:
        w, n, agr, dec = play_cut(backbone, adapters, tok, guest, k,
                                  args.openings, 500 + k, mu, sd)
        res["cuts"][str(k)] = {"wins": w, "games": n,
                               "own_play_top1": round(agr, 4),
                               "decisions": dec}
        print(f"cut {k:2d}: {w}/{n} vs guest | own-play top1 {agr:.3f} "
              f"({dec} decisions)", flush=True)

    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
