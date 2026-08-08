"""Multi-task SFT pools for the SFT-scheduler experiment: canonical
teacher-forcing targets for all 11 curriculum categories, from the existing
curriculum parquets (same boards the RL scheduler used).

Target per task family:
  judge   (chain/judge/occupancy) -> "Answer: <label>"
  listing (winset/chainset)       -> "Answer: <json set>"
  path    (witness)               -> "Answer: {winner,path}" (BFS gold path)
  move    (mate1_v2/mate2/edge_m1/gen_m1/general) -> "Move: <first winning>"
The move grader credits ANY winning move, so a canonical single target is a
well-posed teacher-forcing objective evaluated by set membership.

No-think, uniform answer-token loss weight. 85/15 train/test split per
category by board (disjoint). Every gold target is asserted to score 1.0.

Output: data/multitask/<cat>_{train,test}.parquet
Run: /venv/main/bin/python scripts/build_multitask_sft.py
"""

import json
import os
import sys
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from transformers import AutoTokenizer

from hexenv.board import Board, BLACK, WHITE, cell_name
from hexenv.reward_verl import compute_score
from scripts.build_armD_witness import MODEL, IM_END

JUDGE = ["chain", "judge", "occupancy"]
LISTING = ["winset", "chainset"]
PATH = ["witness"]
MOVE = ["mate1_v2", "mate2", "edge_m1", "gen_m1", "general"]
CATS = JUDGE + LISTING + PATH + MOVE


def gold_path(gt):
    n = gt["size"]
    b = Board(n)
    for c, cell in gt["moves"]:
        b.play(cell, BLACK if c == "B" else WHITE)
    color = BLACK if gt["path_winner"] == "Black" else WHITE
    stones = {(x, y) for y in range(n) for x in range(n) if b.grid[y][x] == color}
    if color == BLACK:
        src = [(x, y) for (x, y) in stones if y == 0]; goal = lambda x, y: y == n - 1
    else:
        src = [(x, y) for (x, y) in stones if x == 0]; goal = lambda x, y: x == n - 1
    prev = {s: None for s in src}; q = deque(src)
    while q:
        cur = q.popleft()
        if goal(*cur):
            path = []
            while cur is not None:
                path.append(cur); cur = prev[cur]
            return [cell_name(x, y) for x, y in reversed(path)]
        for nb in b.neighbors(*cur):
            if nb in stones and nb not in prev:
                prev[nb] = cur; q.append(nb)
    return None


def target_for(cat, gt):
    if cat in JUDGE:
        return f"Answer: {gt['judge_label']}"
    if cat in LISTING:
        return "Answer: " + json.dumps(gt["list_target"])
    if cat in PATH:
        p = gold_path(gt)
        return "Answer: " + json.dumps({"winner": gt["path_winner"], "path": p})
    # move
    return f"Move: {sorted(gt['winning_moves'])[0]}"


def tokenize(tok, prompt, answer):
    pids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        enable_thinking=False, tokenize=True)["input_ids"]
    comp = tok(answer + IM_END, add_special_tokens=False)["input_ids"]
    return list(pids) + list(comp), [0.0] * len(pids) + [1.0] * len(comp)


def main():
    os.makedirs("data/multitask", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    summary = {}
    for cat in CATS:
        df = pd.read_parquet(f"data/curriculum/{cat}.parquet")
        rows, bad = [], 0
        for _, r in df.iterrows():
            gtj = r["reward_model"]["ground_truth"]
            gt = json.loads(gtj)
            ans = target_for(cat, gt)
            if ans is None or compute_score("x", ans, gtj)["score"] != 1.0:
                bad += 1
                continue
            ids, mask = tokenize(tok, r["prompt"][0]["content"], ans)
            if len(ids) > 2048:
                bad += 1
                continue
            rows.append({"input_ids": ids, "loss_mask": mask,
                         "size": gt["size"], "prompt_text": r["prompt"][0]["content"],
                         "answer_text": ans, "ground_truth": gtj,
                         "data_source": f"hex_{cat}",
                         "prompt": list(r["prompt"]),
                         "reward_model": {"style": "rule", "ground_truth": gtj},
                         "ability": "hex", "extra_info": {"category": cat}})
        cut = int(len(rows) * 0.85)
        pd.DataFrame(rows[:cut]).to_parquet(f"data/multitask/{cat}_train.parquet")
        pd.DataFrame(rows[cut:]).to_parquet(f"data/multitask/{cat}_test.parquet")
        summary[cat] = (len(rows), bad)
        print(f"{cat:12} train {cut:5d} test {len(rows)-cut:4d}  (dropped {bad})")
    tot = sum(n for n, _ in summary.values())
    print(f"TOTAL usable rows: {tot}")


if __name__ == "__main__":
    main()
