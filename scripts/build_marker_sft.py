"""Marker SFT data for the single-token 'any behavior' RL test.

Witness boards (5-9), gold answer, with a task-neutral suffix marker present
on 50% of targets:
  no-marker : Answer: {json}
  marker    : Answer: {json} NOTE
Both are graded correct (the grader parses {json} and ignores trailing text),
so SFT installs P(marker)~=0.5 with witness competence intact. The marker is
a single free-generation decision (the token after '}'). RL then shifts it
via a reward cost, isolating one task-neutral token.

Output: data/marker/{train,val}.parquet
Run: /venv/main/bin/python scripts/build_marker_sft.py
"""
import json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from transformers import AutoTokenizer
from hexenv.board import Board, BLACK, WHITE, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from hexenv.reward_verl import compute_score
from scripts.build_armD_witness import MODEL, IM_END
from scripts.build_armD_witness_v2 import fabricate_moves
from scripts.witness_constructive import gen_board
from scripts.build_sft_certificates import Q, TAIL

MARKER = " NOTE"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(2026)
    rows = []
    while len(rows) < 2400:
        n = rng.choice([5, 6, 7, 8, 9])
        winner = rng.choice(["Black", "White"])
        g = gen_board(rng, n, winner, p_forward=rng.uniform(0.1, 0.6),
                      extra_winner_frac=rng.uniform(0.1, 0.6))
        if not g:
            continue
        wst, lst, path = g
        b = Board(n)
        for x, y in wst:
            b.grid[y][x] = BLACK if winner == "Black" else WHITE
        for x, y in lst:
            b.grid[y][x] = WHITE if winner == "Black" else BLACK
        names = [cell_name(x, y) for x, y in path]
        gt = {"category": "witness_marker", "task": "path", "size": n,
              "moves": fabricate_moves(rng, winner, wst, lst, path),
              "path_winner": winner}
        base = "Answer: " + json.dumps({"winner": winner, "path": names})
        marker = rng.random() < 0.5
        ans = base + (MARKER if marker else "")
        assert compute_score("x", ans, json.dumps(gt))["score"] == 1.0
        prompt = RULES.format(n=n, board=render_ascii(b)) + Q + TAIL
        pids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, enable_thinking=False,
                                       tokenize=True)["input_ids"]
        comp = tok(ans + IM_END, add_special_tokens=False)["input_ids"]
        rows.append({"input_ids": list(pids) + list(comp),
                     "loss_mask": [0.0] * len(pids) + [1.0] * len(comp),
                     "size": n, "marker": marker})
    os.makedirs("data/marker", exist_ok=True)
    pd.DataFrame(rows[:2200]).to_parquet("data/marker/train.parquet")
    pd.DataFrame(rows[2200:]).to_parquet("data/marker/val.parquet")
    import collections
    print(f"marker SFT train 2200 val {len(rows)-2200}; "
          f"marker balance {collections.Counter(r['marker'] for r in rows)}")


if __name__ == "__main__":
    main()
