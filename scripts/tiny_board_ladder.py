"""Find the simplest hex task the base model can do: prompt-format ladder for
2x2 (and 1x1) winner detection.

V0 generic rules + skew ascii | V1 move-history | V2 +worked example/format
demo | V3 +size-specialized adjacency rules.
"""

import json
import random
import sys

sys.path.insert(0, "/workspace/hex-rl-cot-deconfusion")

from hexenv.board import Board, EMPTY, BLACK, WHITE
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from hexenv.forced_close_gen import generate_forced_close
import re

GENERIC_TAIL = ("\nHas either player ALREADY completed a winning connection on this board?"
                "\nEnd your response with exactly one line of the form:\nAnswer: Black|White|Neither\n")

V23_TAIL = ("\nQuestion: Has either player ALREADY completed a winning connection on this board?"
            "\nAnswer with exactly one line: 'Answer: Black' or 'Answer: White' or 'Answer: Neither'.\n")

EXAMPLE = """Worked example (a different board):
Board size: 2x2. Moves played: 1. Black b1, 2. White a2, 3. Black b2.
Black stones: b1, b2. White stones: a2.
b1 is on the TOP edge; b2 is on the BOTTOM edge; b1 and b2 are neighbors, so Black
has an unbroken chain from top to bottom.
Answer: Black
"""

ADJ_2x2 = """This board is 2x2. Complete adjacency list:
- a1 neighbors: b1, a2. a1 touches Black's TOP edge and White's LEFT edge.
- b1 neighbors: a1, a2, b2. b1 touches Black's TOP edge and White's RIGHT edge.
- a2 neighbors: a1, b1, b2. a2 touches Black's BOTTOM edge and White's LEFT edge.
- b2 neighbors: b1, a2. b2 touches Black's BOTTOM edge and White's RIGHT edge.
Black wins if a chain of Black stones touches both TOP and BOTTOM.
White wins if a chain of White stones touches both LEFT and RIGHT.
"""


def move_history(b: Board):
    names = {BLACK: "Black", WHITE: "White"}
    if not b.moves:
        return "Moves played: (none - the board is empty)."
    parts = [f"{i+1}. {names[c]} {cell}" for i, (c, cell) in enumerate(b.moves)]
    blacks = [cell for c, cell in b.moves if c == BLACK]
    whites = [cell for c, cell in b.moves if c == WHITE]
    return (f"Moves played: {', '.join(parts)}.\n"
            f"Black stones: {', '.join(blacks) or '(none)'}. "
            f"White stones: {', '.join(whites) or '(none)'}.")


def make_prompt(b: Board, variant: str):
    if variant == "V0":
        return RULES.format(n=b.size, board=render_ascii(b)) + GENERIC_TAIL
    hist = f"Board size: {b.size}x{b.size}. " + move_history(b)
    if variant == "V1":
        return RULES.format(n=b.size, board="(see move history)\n\n" + hist) + GENERIC_TAIL
    header = ("You are analyzing the game of Hex. Two players, Black and White, place "
              "stones on empty cells; stones never move. Black tries to connect the TOP "
              "and BOTTOM edges with a chain of adjacent Black stones; White tries to "
              "connect LEFT and RIGHT with White stones.\n\n")
    if variant == "V2":
        return header + EXAMPLE + "\nNow the actual board:\n" + hist + V23_TAIL
    if variant == "V3":
        return header + ADJ_2x2 + "\n" + EXAMPLE + "\nNow the actual board:\n" + hist + V23_TAIL
    raise ValueError(variant)


def boards_2x2():
    out = []
    # enumerate alternating games up to 4 plies; keep positions incl terminal
    def rec(b):
        key = tuple(sorted((c, cell) for c, cell in b.moves))
        if key in seen:
            return
        seen.add(key)
        out.append(b.copy())
        if b.winner() != EMPTY:
            return
        for m in b.legal_moves():
            nb = b.copy()
            nb.play(m)
            rec(nb)
    seen = set()
    rec(Board(2))
    return out


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
    rng = random.Random(17)
    allb = boards_2x2()
    won = [b for b in allb if b.winner() != EMPTY]
    open_ = [b for b in allb if b.winner() == EMPTY and b.moves]
    rng.shuffle(won); rng.shuffle(open_)
    boards = won[:12] + open_[:12]
    labels = [{EMPTY: "Neither", BLACK: "Black", WHITE: "White"}[b.winner()] for b in boards]

    from vllm import LLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.75)

    for variant in ["V0", "V1", "V2", "V3"]:
        prompts = [make_prompt(b, variant) for b in boards]
        outs = generate_forced_close(llm, tok, prompts, n=8, temperature=0.6,
                                     think_budget=1088, seed=9,
                                     answer_prefix="Answer:")
        n = acc = nat = 0
        tlens = []
        for lbl, row in zip(labels, outs):
            for s in row:
                # answer scaffold says "Move:" in forced_close_gen; parse any label word
                m = re.findall(r"(Black|White|Neither)", s["text"].split("</think>")[-1], re.I)
                n += 1
                acc += bool(m) and m[-1].capitalize() == lbl
                nat += s["natural_close"]
                tlens.append(s["think_tokens"])
        tlens.sort()
        print(f"{variant}: acc {acc/n:.3f}  natural-close {nat/n:.3f}  "
              f"think p50 {tlens[len(tlens)//2]}  (n={n})", flush=True)


if __name__ == "__main__":
    main()
