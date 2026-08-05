"""Phase 1 gate checks with vLLM. Subcommands: legal, comprehension, quiz.

Usage: python scripts/phase1.py <subcommand> [--model Qwen/Qwen3-1.7B] [--think/--no-think]
Writes raw samples to results/phase1/<subcommand>_<model>_<think>.jsonl — READ THEM.
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, EMPTY, BLACK, WHITE, cell_name, parse_cell
from hexenv.positions import random_position
from hexenv.prompts import move_prompt, question_prompt, extract_move

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "phase1")


def build_legal_items(rng):
    items = []
    for size in [5, 7]:
        for n_stones in [0, 3, 6, 10, 15]:
            for _ in range(10):
                b = random_position(size, n_stones, rng)
                items.append({
                    "kind": "legal",
                    "size": size,
                    "n_stones": n_stones,
                    "prompt": move_prompt(b),
                    "legal": b.legal_moves(),
                })
    return items


def build_comprehension_items(rng):
    items = []
    for size in [5, 7]:
        for _ in range(15):
            b = random_position(size, rng.choice([4, 6, 8, 10]), rng)
            occupied = [(x, y) for y in range(size) for x in range(size) if b.grid[y][x] != EMPTY]
            empties = [(x, y) for y in range(size) for x in range(size) if b.grid[y][x] == EMPTY]

            # 1. stone lookup
            x, y = rng.choice(occupied + empties[:1])
            val = {EMPTY: "empty", BLACK: "Black", WHITE: "White"}[b.grid[y][x]]
            items.append({
                "kind": "lookup", "size": size,
                "prompt": question_prompt(b, f"What occupies cell {cell_name(x,y)}? Answer with exactly one word: Black, White, or empty.\nAnswer:"),
                "answer": val.lower(),
            })

            # 2. neighbor listing
            x, y = rng.randrange(size), rng.randrange(size)
            nbrs = sorted(cell_name(nx, ny) for nx, ny in b.neighbors(x, y))
            items.append({
                "kind": "neighbors", "size": size,
                "prompt": question_prompt(b, f"List all on-board neighbor cells of {cell_name(x,y)}, comma-separated.\nAnswer:"),
                "answer": nbrs,
            })

            # 3. adjacency yes/no
            x1, y1 = rng.randrange(size), rng.randrange(size)
            if rng.random() < 0.5:
                nb = list(b.neighbors(x1, y1))
                x2, y2 = rng.choice(nb)
            else:
                while True:
                    x2, y2 = rng.randrange(size), rng.randrange(size)
                    if (x2, y2) != (x1, y1) and (x2, y2) not in list(b.neighbors(x1, y1)):
                        break
            adj = (x2, y2) in list(b.neighbors(x1, y1))
            items.append({
                "kind": "adjacent", "size": size,
                "prompt": question_prompt(b, f"Are cells {cell_name(x1,y1)} and {cell_name(x2,y2)} adjacent (sharing a hexagon edge)? Answer yes or no.\nAnswer:"),
                "answer": "yes" if adj else "no",
            })

            # 4. same-chain connectivity between two stones of same color
            by_color = {BLACK: [], WHITE: []}
            for (x, y) in occupied:
                by_color[b.grid[y][x]].append((x, y))
            for color in (BLACK, WHITE):
                pts = by_color[color]
                if len(pts) >= 2:
                    p, q = rng.sample(pts, 2)
                    # BFS through same-color stones
                    seen, stack = set(), [p]
                    while stack:
                        cur = stack.pop()
                        if cur in seen:
                            continue
                        seen.add(cur)
                        for nx, ny in b.neighbors(*cur):
                            if b.grid[ny][nx] == color and (nx, ny) not in seen:
                                stack.append((nx, ny))
                    conn = q in seen
                    cname = "Black" if color == BLACK else "White"
                    items.append({
                        "kind": "connected", "size": size,
                        "prompt": question_prompt(b, f"Are the {cname} stones on {cell_name(*p)} and {cell_name(*q)} connected to each other through an unbroken chain of adjacent {cname} stones already on the board? Answer yes or no.\nAnswer:"),
                        "answer": "yes" if conn else "no",
                    })
                    break
    return items


QUIZ_QUESTIONS = [
    "In the game of Hex, what is a 'bridge'? Give the precise stone pattern and explain why it is useful.",
    "In the game of Hex (n x n board), which player wins with perfect play, and what is the classic argument for this?",
    "In the game of Hex, what is a 'ladder'? Describe how one arises and how a 'ladder escape' works.",
    "In the game of Hex, what is an 'edge template'? Give an example of a simple one.",
    "In the game of Hex, why can the game never end in a draw?",
    "In Hex openings, why is the swap (pie) rule used, and what kinds of first moves are usually swapped?",
    "In the game of Hex, is the obtuse corner or the acute corner more valuable, and why?",
    "Explain what it means for two groups in Hex to be 'virtually connected' and give two distinct examples of virtual connections.",
]


def build_quiz_items(rng):
    return [{"kind": "quiz", "prompt": q + "\n\nIf you are not familiar with the concept, say so plainly instead of guessing.", "answer": None} for q in QUIZ_QUESTIONS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["legal", "comprehension", "quiz"])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=3584)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    rng = random.Random(1234)
    if args.cmd == "legal":
        items = build_legal_items(rng)
    elif args.cmd == "comprehension":
        items = build_comprehension_items(rng)
    else:
        items = build_quiz_items(rng)

    from hexenv.genbackend import Backend

    backend = Backend(args.model, enable_thinking=not args.no_think)
    texts = backend.generate(
        [it["prompt"] for it in items],
        n=1, temperature=args.temperature, max_tokens=args.max_tokens,
    )

    os.makedirs(RESULTS, exist_ok=True)
    tag = f"{args.cmd}_{args.model.split('/')[-1]}_{'nothink' if args.no_think else 'think'}"
    path = os.path.join(RESULTS, tag + ".jsonl")

    n_correct, n_total, truncated = 0, 0, 0
    with open(path, "w") as f:
        for it, tlist in zip(items, texts):
            text = tlist[0]
            if "</think>" not in text and not args.no_think:
                truncated += 1
            rec = dict(it)
            rec["response"] = text
            # strip think block for grading
            ans_part = text.split("</think>")[-1].strip().lower()
            graded = None
            if it["kind"] == "legal":
                mv = extract_move(text)
                graded = mv in it["legal"] if mv else False
                rec["parsed_move"] = mv
            elif it["kind"] in ("lookup", "adjacent", "connected"):
                first_word = ans_part.replace("*", "").split()
                fw = first_word[0].strip(".,:") if first_word else ""
                graded = fw == it["answer"]
                rec["parsed"] = fw
            elif it["kind"] == "neighbors":
                import re
                cells = set(re.findall(r"\b([a-i]\d{1,2})\b", ans_part))
                graded = cells == set(it["answer"])
                rec["parsed"] = sorted(cells)
            if graded is not None:
                n_total += 1
                n_correct += bool(graded)
                rec["correct"] = bool(graded)
            f.write(json.dumps(rec) + "\n")

    print(f"wrote {path}")
    print(f"truncated: {truncated}/{len(items)}")
    if n_total:
        # per-kind breakdown
        from collections import defaultdict
        agg = defaultdict(lambda: [0, 0])
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if "correct" in r:
                    agg[(r["kind"], r.get("size", 0))][0] += r["correct"]
                    agg[(r["kind"], r.get("size", 0))][1] += 1
        for k in sorted(agg):
            c, t = agg[k]
            print(f"{k}: {c}/{t} = {c/t:.2f}")
        print(f"TOTAL: {n_correct}/{n_total} = {n_correct/n_total:.2f}")


if __name__ == "__main__":
    main()
