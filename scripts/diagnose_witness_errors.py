"""Diagnose the STRUCTURE of bok's witness errors, to test Joshua's
hypothesis that they are 'valid but non-minimal' (removable extra stone,
path stays intact).

For each incorrect bok answer, categorize the failing grader checks:
  membership : cell that is not the winner's stone
  adjacency  : consecutive pair not adjacent
  edge       : first/last not on the winner's start/end edge
And a key structural probe:
  subset_connects : among the answer's cells that ARE the winner's stones,
                    do they still connect edge-to-edge? (i.e. is the error
                    just spurious inserted cells over an otherwise-valid
                    connection -> 'removable' structure)
Also: of CORRECT answers, how many are non-minimal (longer than shortest)?

Run (verl venv, GPU): /venv/verl/bin/python scripts/diagnose_witness_errors.py
"""

import json
import os
import random
import re
import sys
from collections import deque, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, BLACK, WHITE, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from hexenv.reward_verl import compute_score
from scripts.build_armD_witness import MODEL
from scripts.build_armD_witness_v2 import fabricate_moves
from scripts.witness_constructive import gen_board
from scripts.build_sft_certificates import Q, TAIL

NBR = [(-1, 0), (1, 0), (0, -1), (1, -1), (-1, 1), (0, 1)]


def coords(cell):
    return ord(cell[0]) - 97, int(cell[1:]) - 1


def adjacent(a, b):
    ax, ay = coords(a); bx, by = coords(b)
    return (bx - ax, by - ay) in {(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1)}


def connects_edge(cells, n, winner):
    """Do these cells (assumed winner's stones) connect the winner's edges?"""
    S = set(cells)
    if winner == "Black":
        src = [c for c in S if coords(c)[1] == 0]; goalf = lambda c: coords(c)[1] == n - 1
    else:
        src = [c for c in S if coords(c)[0] == 0]; goalf = lambda c: coords(c)[0] == n - 1
    seen = set(src); q = deque(src)
    while q:
        c = q.popleft()
        if goalf(c):
            return True
        cx, cy = coords(c)
        for dx, dy in NBR:
            nb = (cx + dx, cy + dy)
            if 0 <= nb[0] < n and 0 <= nb[1] < n:
                name = cell_name(*nb)
                if name in S and name not in seen:
                    seen.add(name); q.append(name)
    return False


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(9)
    boards = []
    while len(boards) < 600:
        n = rng.choice([7, 8, 9])
        winner = rng.choice(["Black", "White"])
        g = gen_board(rng, n, winner, p_forward=rng.uniform(0.0, 0.3),
                      extra_winner_frac=rng.uniform(0.1, 0.6))
        if g is None:
            continue
        wst, lst, path = g
        b = Board(n)
        for x, y in wst:
            b.grid[y][x] = BLACK if winner == "Black" else WHITE
        for x, y in lst:
            b.grid[y][x] = WHITE if winner == "Black" else BLACK
        boards.append({"n": n, "winner": winner, "board": b, "wst": set(wst),
                       "path_len": len(path),
                       "gt": gt_for(b, winner, wst, lst, path)})
    prompts = [TokensPrompt(prompt_token_ids=tok.apply_chat_template(
        [{"role": "user", "content": RULES.format(n=r["n"], board=render_ascii(r["board"])) + Q + TAIL}],
        add_generation_prompt=True, enable_thinking=False,
        tokenize=True)["input_ids"]) for r in boards]
    llm = LLM(model="checkpoints/armD2_bok/hf_merged", max_model_len=1024,
              gpu_memory_utilization=0.55, dtype="bfloat16")
    outs = llm.generate(prompts, SamplingParams(temperature=1.0, n=4, max_tokens=96))

    err = Counter(); nerr = 0; ncorrect = 0; nonminimal = 0
    subset_connects = 0
    for r, o in zip(boards, outs):
        n = r["n"]; winner = r["winner"]
        wstones = {cell_name(x, y) for x, y in r["wst"]}
        for s in o.outputs:
            m = re.search(r"\{.*\}", s.text, re.DOTALL)
            if not m:
                continue
            try:
                obj = json.loads(m.group(0))
            except Exception:
                continue
            score = compute_score("x", "Answer: " + m.group(0), json.dumps(r["gt"]))["score"]
            path = [str(c).lower() for c in obj.get("path", [])]
            winner_ok = str(obj.get("winner", "")).capitalize() == winner
            if score == 1.0:
                ncorrect += 1
                if len(path) > r["path_len"]:
                    nonminimal += 1
                continue
            nerr += 1
            if not winner_ok:
                err["wrong_winner"] += 1
                continue
            mem = sum(1 for c in path if c not in wstones)
            adj = sum(1 for a, b in zip(path, path[1:]) if not adjacent(a, b))
            edge_bad = 0
            if path:
                if winner == "Black":
                    edge_bad = (coords(path[0])[1] != 0) + (coords(path[-1])[1] != n - 1)
                else:
                    edge_bad = (coords(path[0])[0] != 0) + (coords(path[-1])[0] != n - 1)
            if mem: err["membership"] += 1
            if adj: err["adjacency"] += 1
            if edge_bad: err["edge"] += 1
            # is the error 'removable'? the winner-stone cells in the answer
            # still connect edge-to-edge (spurious cells could be dropped)
            good_cells = [c for c in path if c in wstones]
            if connects_edge(good_cells, n, winner):
                subset_connects += 1

    print(f"correct={ncorrect}  errors={nerr}")
    print(f"of correct: non-minimal (len>shortest) = {nonminimal} "
          f"({100*nonminimal/max(ncorrect,1):.1f}%)")
    print("error breakdown (a single error can trip >1 check):")
    for k in ("wrong_winner", "membership", "adjacency", "edge"):
        print(f"  {k:12} {err[k]:4}  ({100*err[k]/max(nerr,1):.1f}%)")
    print(f"  removable-structure (winner-stones-in-answer still connect edges): "
          f"{subset_connects} ({100*subset_connects/max(nerr,1):.1f}%)")


def gt_for(b, winner, wst, lst, path):
    return {"category": "witness_diag", "task": "path", "size": b.size,
            "moves": fabricate_moves(random, winner, wst, lst, path),
            "path_winner": winner}


if __name__ == "__main__":
    main()
