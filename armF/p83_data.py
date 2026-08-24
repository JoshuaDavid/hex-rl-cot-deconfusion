"""P83 data: conversational-hex SFT pairs.

Input (human voice, model plays X, X to move): one of
  F1 move narrative (3 phrasings)   F2 coordinate sets (3 phrasings)
  F3 layout-varied diagram (row labels + col header, 3 glyph sets)
  F4 canonical passthrough
Target (assistant): stone-list CoT + canonical block (format-A render +
registers + "Next move:") + " " + guest-argmax cell. The frozen machinery
reads the model's own canonical block; playing X keeps canonical == absolute.

Writes armF/data/p83_sft.pt: {records: [{inp, board(2,13,13)u8, label}]}.

Usage: /venv/main/bin/python armF/p83_data.py --n 60000
"""
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from p75_baselines import load_guest  # noqa: E402
from train_p78ft import guest_labels, cell_str  # noqa: E402

DEV = "cuda"


def name(mv):  # absolute (x, y) -> "F7" (unpadded, natural)
    return f"{chr(65 + int(mv[0]))}{int(mv[1]) + 1}"


def stone_lists(board):
    xs, os_ = [], []
    for x in range(11):
        for y in range(11):
            if board[0, x + 1, y + 1] > 0.5:
                xs.append(name((x, y)))
            elif board[1, x + 1, y + 1] > 0.5:
                os_.append(name((x, y)))
    return xs, os_


def f1_narrative(hist, rng):
    xs = [name(m) for m in hist[0::2]]
    os_ = [name(m) for m in hist[1::2]]
    v = rng.randrange(3)
    if not hist:
        return rng.choice(
            ["Let's play hex on the standard 11x11 board. You're X and you "
             "move first. What's your opening?",
             "Fresh 11x11 hex game — you take X and start. Where do you "
             "play?"])
    if v == 0:
        parts = []
        for i, m in enumerate(hist):
            who = "you played" if i % 2 == 0 else "I answered"
            parts.append(f"{who} {name(m)}")
        s = ", then ".join(parts)
        return (f"We're mid-game in 11x11 hex — you're X, I'm O. So far "
                f"{s}. Your move.")
    if v == 1:
        mv = ", ".join(f"{i+1}. {name(m)}" for i, m in enumerate(hist))
        return (f"Hex, 11x11, you are X (moved first). Move record: {mv}. "
                f"It's your turn — what do you play?")
    return (f"Our hex game so far (11x11): your X moves were "
            f"{', '.join(xs)}; my O moves were {', '.join(os_)}, in that "
            f"order alternating starting with you. You're up.")


def f2_coordsets(board, rng):
    xs, os_ = stone_lists(board)
    v = rng.randrange(3)
    xstr = ", ".join(xs) if xs else "none yet"
    ostr = ", ".join(os_) if os_ else "none yet"
    if v == 0:
        return (f"Position check, 11x11 hex, you're X to move. Your stones: "
                f"{xstr}. My stones: {ostr}. Go ahead.")
    if v == 1:
        return (f"Here's the current hex position. X (you): {xstr}. "
                f"O (me): {ostr}. Board is 11x11, X to play. Your move?")
    return (f"11x11 hex. Occupied points — X side: {xstr}; O side: {ostr}. "
            f"Everything else is empty. You're X and it's your turn.")


GLYPHS = [("x", "o", "."), ("B", "W", "-"), ("*", "#", "~")]


def f3_diagram(board, rng):
    gx, go, ge = GLYPHS[rng.randrange(len(GLYPHS))]
    header = "    " + " ".join(f"{c+1:2d}" for c in range(11))
    lines = [header]
    for x in range(11):
        row = []
        for y in range(11):
            if board[0, x + 1, y + 1] > 0.5:
                row.append(f" {gx}")
            elif board[1, x + 1, y + 1] > 0.5:
                row.append(f" {go}")
            else:
                row.append(f" {ge}")
        lines.append(f"{chr(65+x)} |" + "".join(row))
    diag = "\n".join(lines)
    return (f"Here's our 11x11 hex board ({gx} = your X stones, {go} = my "
            f"O stones, {ge} = empty; rows A-K, columns 1-11):\n\n{diag}\n\n"
            f"You're X, your move.")


def f4_canonical(board):
    from p75_baselines import render_with_regs
    text, _ = render_with_regs(board)
    from train_txcontain import EMIT_TAIL
    return text[: -len(EMIT_TAIL)] if text.endswith(EMIT_TAIL) else text


def build_input(board, hist, rng):
    fam = rng.randrange(4)
    if fam == 0:
        return f1_narrative(hist, rng), "F1"
    if fam == 1:
        return f2_coordsets(board, rng), "F2"
    if fam == 2:
        return f3_diagram(board, rng), "F3"
    return f4_canonical(board), "F4"


def row_lists(board):
    """Per-row breakdown: 'Row A: X at 3, 7; O at 5.' (col numbers)."""
    out = []
    for x in range(11):
        xs = [str(y + 1) for y in range(11) if board[0, x + 1, y + 1] > 0.5]
        os_ = [str(y + 1) for y in range(11) if board[1, x + 1, y + 1] > 0.5]
        out.append(f"Row {chr(65+x)}: X at {', '.join(xs) if xs else 'none'}"
                   f"; O at {', '.join(os_) if os_ else 'none'}.")
    return "\n".join(out)


def target_text(board, label_cell):
    from p75_baselines import render_with_regs
    xs, os_ = stone_lists(board)
    xstr = ", ".join(xs) if xs else "none"
    ostr = ", ".join(os_) if os_ else "none"
    canon, _ = render_with_regs(board)
    from train_txcontain import EMIT_TAIL
    if not canon.endswith(EMIT_TAIL):
        canon = canon + EMIT_TAIL
    return (f"Let me lay out the board. X stones: {xstr}. O stones: "
            f"{ostr}.\nRow by row:\n{row_lists(board)}\n\n{canon} "
            f"{cell_str(label_cell)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="armF/data/p83_sft.pt")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    games = (torch.load("armF/data/games.pt", weights_only=False)["games"]
             + torch.load("armF/data/games2.pt", weights_only=False)["games"])
    rng.shuffle(games)
    import hexhex_wrap as W
    empty = W.empty_board().board_tensor.to(torch.uint8)
    pos = []
    for g in games:
        T = len(g["moves"])
        for t in range(0, T, 2):  # X to move; canonical == absolute
            board = g["boards"][t - 1] if t > 0 else empty.clone()
            pos.append((board, g["moves"][:t].tolist()))
        if len(pos) >= args.n * 2:
            break
    rng.shuffle(pos)
    pos = pos[: args.n]
    print(f"{len(pos)} X-to-move positions", flush=True)

    guest = load_guest()
    boards = torch.stack([p[0] for p in pos])
    labels = torch.cat([guest_labels(guest, boards[i:i + 1024].to(DEV)).cpu()
                        for i in range(0, len(boards), 1024)])

    records = []
    fam_counts = {}
    for i, (board, hist) in enumerate(pos):
        inp, fam = build_input(board, hist, rng)
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
        records.append({"inp": inp, "board": board, "label": int(labels[i]),
                        "fam": fam})
    print("family counts:", fam_counts, flush=True)
    torch.save({"records": records}, args.out)
    print(f"wrote {args.out} ({len(records)} records)")


if __name__ == "__main__":
    main()
