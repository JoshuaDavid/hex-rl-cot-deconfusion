"""Conversational hex vs the P83 model. Model plays X (moves first).

Interactive:  /venv/main/bin/python armF/converse.py
Transcript:   /venv/main/bin/python armF/converse.py --auto dist --turns 60
              (opponent = distilled CNN as O; writes armF/results/
               p83_transcript.txt and a JSON with legality/agreement stats)

Each model turn: the game so far is phrased in a random input family (F1/F2)
in the human voice; the model generates its stone-list CoT + canonical
re-render + "Next move: XYZ"; the move is parsed and played. NO masking, NO
fallback in --auto strict accounting (illegal model moves are recorded; in
interactive mode we ask it to just pick again via random legal for flow).
"""
import argparse
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from p83_data import f1_narrative, f2_coordsets, name  # noqa: E402
from p83_sft import SEP, parse_render  # noqa: E402
from train_p78ft import load_model, parse_cell, guest_labels  # noqa: E402
from p75_baselines import load_guest  # noqa: E402
import qwen_embed as Q  # noqa: E402
import hexhex_wrap as W  # noqa: E402

sys.path.insert(0, str(W.HEXHEX_ROOT))
from hexhex.logic.hexboard import Board  # noqa: E402

DEV = "cuda"


def load_p83(bottom="checkpoints/armF_p82/bottom.pt",
             ckpt="checkpoints/armF_p83/p83_last.pt"):
    model = load_model("contained", bottom)
    ck = torch.load(ckpt, map_location=DEV, weights_only=False)
    sd = model.state_dict()
    for k, v in ck["model"].items():
        key = k if k in sd else "model." + k
        sd[key].copy_(v)
    model.eval()
    return model


@torch.no_grad()
def model_turn(model, tok, board_abs, hist, rng, max_new=560):
    """Returns (move_xy or None, generated_text, prompt)."""
    b_u8 = board_abs.to(torch.uint8)
    if rng.random() < 0.5:
        inp = f1_narrative(hist, rng)
    else:
        inp = f2_coordsets(b_u8, rng)
    prompt = inp + SEP
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    out = model.generate(input_ids=enc["input_ids"].to(DEV),
                         max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0, enc["input_ids"].shape[1]:],
                     skip_special_tokens=True)
    m = re.search(r"Next move: ([A-K]\d\d)", txt)
    mv = None
    if m:
        c = parse_cell(" " + m.group(1))
        if c is not None:
            mv = divmod(c, 11)
    return mv, txt, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", default=None, choices=[None, "dist", "random"])
    ap.add_argument("--turns", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="checkpoints/armF_p83/p83_last.pt")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Q.QWEN)
    model = load_p83(ckpt=args.ckpt)
    guest = load_guest()

    opp = None
    if args.auto == "dist":
        cnn = W.load_model()
        from elo_temp_distilled import load_distilled, dist_logits
        dist = load_distilled(cnn)
        from hexhex.utils.utils import correct_position1d

        def opp(b):
            x = b.board_tensor.unsqueeze(0).float().to(DEV)
            lg = dist_logits(dist, x)[0]
            lg = lg - 1000.0 * b.board_tensor[:, 1:-1, 1:-1].sum(0).reshape(121).to(DEV)
            p1 = correct_position1d(lg.argmax().item(), 11, b.player)
            return divmod(p1, 11)
    elif args.auto == "random":
        def opp(b):
            return rng.choice(sorted(b.legal_moves))

    b = Board(11, switch_allowed=False)
    hist = []
    lines = []
    stats = {"model_moves": 0, "illegal": 0, "match_guest": 0,
             "cellacc": []}
    while not b.winner and b.legal_moves and len(hist) < args.turns:
        if b.player == 0:  # model = X
            mv, txt, prompt = model_turn(model, tok, b.board_tensor, hist,
                                         rng)
            acc = parse_render(txt, b.board_tensor)
            stats["cellacc"].append(acc if acc is not None else 0.0)
            gref = guest_labels(guest, b.board_tensor.unsqueeze(0)
                                .to(torch.uint8).to(DEV))[0].item()
            stats["model_moves"] += 1
            lines.append(f"[HUMAN-VOICE PROMPT]\n{prompt}")
            lines.append(f"[MODEL]\n{txt.strip()}\n")
            if mv is None or mv not in b.legal_moves:
                stats["illegal"] += 1
                lines.append(f"  !! illegal/unparsed ({mv}); random legal "
                             f"substituted\n")
                mv = rng.choice(sorted(b.legal_moves))
            else:
                if mv[0] * 11 + mv[1] == gref:
                    stats["match_guest"] += 1
            print(f"model (X) plays {name(mv)}  [render acc "
                  f"{stats['cellacc'][-1]:.3f}]", flush=True)
        else:
            if opp is None:
                while True:
                    s = input("your move (O), e.g. F6: ").strip().upper()
                    m = re.match(r"^([A-K])(\d+)$", s)
                    if m:
                        mv = (ord(m.group(1)) - 65, int(m.group(2)) - 1)
                        if mv in b.legal_moves:
                            break
                    print("  not a legal move, try again")
            else:
                mv = opp(b)
                print(f"opponent (O) plays {name(mv)}", flush=True)
            lines.append(f"[O plays {name(mv)}]\n")
        b.set_stone(mv)
        hist.append(list(mv))
    winner = "X (model)" if b.winner == [0] else "O (opponent)"
    lines.append(f"\n=== game over: {winner} wins after {len(hist)} plies ===")
    print(f"\n=== {winner} wins ===")
    print(f"model moves {stats['model_moves']} | illegal {stats['illegal']} "
          f"| match-guest {stats['match_guest']} | mean render acc "
          f"{sum(stats['cellacc'])/max(len(stats['cellacc']),1):.4f}")
    if args.auto:
        Path("armF/results/p83_transcript.txt").write_text("\n".join(lines))
        import json
        Path("armF/results/p83_game.json").write_text(json.dumps(
            {**{k: v for k, v in stats.items() if k != "cellacc"},
             "mean_cellacc": sum(stats["cellacc"]) / max(len(stats["cellacc"]), 1),
             "winner": winner, "plies": len(hist)}, indent=1))
        print("wrote armF/results/p83_transcript.txt")


if __name__ == "__main__":
    main()
