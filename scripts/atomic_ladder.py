"""Atomic-subskill ladder on 2x2 boards: where exactly is the compositional cliff?
T1 occupancy / T2 edge-touch / T3 adjacency / T4 same-color adjacent pair /
T5 edge-pair membership / T6 full winner judge.
"""

import random
import sys

sys.path.insert(0, "/workspace/hex-rl-cot-deconfusion")

from hexenv.board import Board, EMPTY, BLACK, WHITE, cell_name, parse_cell
from hexenv.forced_close_gen import generate_forced_close
from scripts.tiny_board_ladder import boards_2x2, move_history
import re

HEADER = ("You are analyzing the game of Hex on a 2x2 board. Two players, Black and "
          "White, place stones on cells; stones never move.\n"
          "Cells: a1 (top-left), b1 (top-right), a2 (bottom-left), b2 (bottom-right).\n"
          "Adjacency: a1-b1, a1-a2, b1-a2, b1-b2, a2-b2 are adjacent pairs. "
          "a1-b2 are NOT adjacent.\n"
          "TOP edge cells: a1, b1. BOTTOM edge cells: a2, b2. "
          "LEFT edge cells: a1, a2. RIGHT edge cells: b1, b2.\n\n")


def item_t1(b, rng):
    cell = rng.choice(["a1", "b1", "a2", "b2"])
    x, y = parse_cell(cell)
    val = {EMPTY: "Empty", BLACK: "Black", WHITE: "White"}[b.grid[y][x]]
    ex = ("Worked example:\nMoves played: 1. Black a1.\nQuestion: What is on cell a1?\n"
          "a1 was played by Black.\nAnswer: Black\n\n")
    q = f"Question: What is on cell {cell}?\nAnswer with one line: 'Answer: Black' or 'Answer: White' or 'Answer: Empty'.\n"
    return ex, q, val, ["Black", "White", "Empty"]


def item_t2(b, rng):
    cell = rng.choice(["a1", "b1", "a2", "b2"])
    val = "Yes" if cell in ("a1", "b1") else "No"
    ex = ("Worked example:\nQuestion: Does cell a2 touch the TOP edge?\n"
          "The TOP edge cells are a1 and b1. a2 is not one of them.\nAnswer: No\n\n")
    q = f"Question: Does cell {cell} touch the TOP edge?\nAnswer with one line: 'Answer: Yes' or 'Answer: No'.\n"
    return ex, q, val, ["Yes", "No"]


def item_t2b(b, rng):
    cell = rng.choice(["a1", "b1", "a2", "b2"])
    val = "Yes" if cell in ("a1", "b1") else "No"
    ex = ("Worked example:\nQuestion: Is cell a2 one of the TOP edge cells (a1, b1)?\n"
          "The TOP edge cells are a1 and b1. a2 is not in that list.\nAnswer: No\n\n")
    q = (f"Question: Is cell {cell} one of the TOP edge cells (a1, b1)?\n"
         "Answer with one line: 'Answer: Yes' or 'Answer: No'.\n")
    return ex, q, val, ["Yes", "No"]


def item_t5b(b, rng):
    stones = {cell for c, cell in b.moves if c == BLACK}
    has = bool(stones & {"a1", "b1"}) and bool(stones & {"a2", "b2"})
    ex = ("Worked example:\nMoves played: 1. Black a1, 2. White b1, 3. Black b2.\n"
          "Question: Is at least one of a1, b1 a Black stone, AND at least one of a2, b2 a Black stone?\n"
          "Black stones: a1, b2. a1 is in the first list. b2 is in the second list.\nAnswer: Yes\n\n")
    q = ("Question: Is at least one of a1, b1 a Black stone, AND at least one of a2, b2 a Black stone?\n"
         "Answer with one line: 'Answer: Yes' or 'Answer: No'.\n")
    return ex, q, ("Yes" if has else "No"), ["Yes", "No"]


def item_t6b(b, rng):
    from hexenv.board import EMPTY as _E
    lbl = {_E: "Neither", BLACK: "Black", WHITE: "White"}[b.winner()]
    ex = ("Worked example:\nMoves played: 1. Black b1, 2. White a2, 3. Black b2.\n"
          "Question: Which player, if any, has already completed a winning connection?\n"
          "Black stones: b1, b2. b1 is a TOP edge cell, b2 is a BOTTOM edge cell, and b1-b2\n"
          "is an adjacent pair. So Black has a chain from TOP to BOTTOM.\nAnswer: Black\n\n")
    q = ("Question: Which player, if any, has already completed a winning connection? "
         "(Black needs a chain of adjacent Black stones with a TOP edge cell and a BOTTOM edge cell; "
         "White needs a chain of adjacent White stones with a LEFT edge cell and a RIGHT edge cell.)\n"
         "Answer with one line: 'Answer: Black' or 'Answer: White' or 'Answer: Neither'.\n"
         )
    return ex, q, lbl, ["Black", "White", "Neither"]


def item_t3(b, rng):
    pair = rng.choice([("a1","b1"),("a1","a2"),("b1","a2"),("b1","b2"),("a2","b2"),("a1","b2"),("b2","a1")])
    adj = set(pair) != {"a1","b2"}
    ex = ("Worked example:\nQuestion: Are cells b1 and a2 adjacent?\n"
          "The adjacent pairs are: a1-b1, a1-a2, b1-a2, b1-b2, a2-b2. b1-a2 is in the list.\nAnswer: Yes\n\n")
    q = f"Question: Are cells {pair[0]} and {pair[1]} adjacent?\nAnswer with one line: 'Answer: Yes' or 'Answer: No'.\n"
    return ex, q, ("Yes" if adj else "No"), ["Yes", "No"]


def item_t4(b, rng):
    color = rng.choice([BLACK, WHITE])
    cname = "Black" if color == BLACK else "White"
    stones = {cell for c, cell in b.moves if c == color}
    pairs = [("a1","b1"),("a1","a2"),("b1","a2"),("b1","b2"),("a2","b2")]
    has = any(p[0] in stones and p[1] in stones for p in pairs)
    ex = ("Worked example:\nMoves played: 1. Black a1, 2. White b2, 3. Black b1.\n"
          "Question: Is there any pair of adjacent Black stones?\n"
          "Black stones: a1, b1. a1-b1 is an adjacent pair, and both are Black.\nAnswer: Yes\n\n")
    q = f"Question: Is there any pair of adjacent {cname} stones?\nAnswer with one line: 'Answer: Yes' or 'Answer: No'.\n"
    return ex, q, ("Yes" if has else "No"), ["Yes", "No"]


def item_t5(b, rng):
    stones = {cell for c, cell in b.moves if c == BLACK}
    has = bool(stones & {"a1","b1"}) and bool(stones & {"a2","b2"})
    ex = ("Worked example:\nMoves played: 1. Black a1, 2. White b1, 3. Black b2.\n"
          "Question: Is there a Black stone touching the TOP edge AND a Black stone touching the BOTTOM edge?\n"
          "Black stones: a1, b2. a1 touches TOP. b2 touches BOTTOM. Both conditions hold.\nAnswer: Yes\n\n")
    q = ("Question: Is there a Black stone touching the TOP edge AND a Black stone touching the BOTTOM edge?\n"
         "Answer with one line: 'Answer: Yes' or 'Answer: No'.\n")
    return ex, q, ("Yes" if has else "No"), ["Yes", "No"]


def item_t6(b, rng):
    lbl = {EMPTY: "Neither", BLACK: "Black", WHITE: "White"}[b.winner()]
    ex = ("Worked example:\nMoves played: 1. Black b1, 2. White a2, 3. Black b2.\n"
          "Question: Has either player completed a winning connection?\n"
          "Black stones: b1, b2 - b1 touches TOP, b2 touches BOTTOM, b1-b2 adjacent: Black wins.\nAnswer: Black\n\n")
    q = ("Question: Has either player ALREADY completed a winning connection? "
         "(Black connects TOP-BOTTOM; White connects LEFT-RIGHT.)\n"
         "Answer with one line: 'Answer: Black' or 'Answer: White' or 'Answer: Neither'.\n")
    return ex, q, lbl, ["Black", "White", "Neither"]


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
    rng = random.Random(23)
    allb = [b for b in boards_2x2() if b.moves]
    rng.shuffle(allb)

    from vllm import LLM
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.75)

    makers = {"T2_edge_A": item_t2, "T2_edge_B": item_t2b,
              "T5_edgepair_A": item_t5, "T5_edgepair_B": item_t5b,
              "T6_judge_A": item_t6, "T6_judge_B": item_t6b}
    for name, mk in makers.items():
        prompts, labels, vocabs = [], [], []
        # balance labels where possible
        items = []
        for b in allb * 3:
            ex, q, lbl, vocab = mk(b, rng)
            items.append((HEADER + ex + "Now the actual board:\n" +
                          ("" if name.startswith(("T2_edge","T3_adjacency")) else move_history(b) + "\n") + q,
                          lbl, vocab))
        # take a label-balanced subset of 18
        by = {}
        for it in items:
            by.setdefault(it[1], []).append(it)
        sel = []
        while len(sel) < 18 and any(by.values()):
            for k in list(by):
                if by[k] and len(sel) < 18:
                    sel.append(by[k].pop())
        for p, l, v in sel:
            prompts.append(p); labels.append(l); vocabs.append(v)
        outs = generate_forced_close(llm, tok, prompts, n=6, temperature=0.6,
                                     think_budget=1088, seed=41, answer_prefix="Answer:")
        n = acc = nat = 0
        tl = []
        for lbl, vocab, row in zip(labels, vocabs, outs):
            for s in row:
                post = s["text"].split("</think>")[-1]
                m = re.findall(r"(" + "|".join(vocab) + r")", post, re.I)
                n += 1
                acc += bool(m) and m[-1].capitalize() == lbl
                nat += s["natural_close"]
                tl.append(s["think_tokens"])
        tl.sort()
        print(f"{name:>16}: acc {acc/n:.3f}  nat-close {nat/n:.3f}  think-p50 {tl[len(tl)//2]}  (n={n})", flush=True)


if __name__ == "__main__":
    main()
