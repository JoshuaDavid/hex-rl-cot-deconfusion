"""Arm D test 1 data: witness task, sizes 2-5, unique-minimal-path boards,
no-think SFT targets with token-importance loss weights.

Filter: keep only boards whose winner has EXACTLY ONE minimal (shortest)
winning path — the training target is then unambiguous.

Weights: every completion token defaults to 2.0 (an error in JSON structure,
the "Answer:" line, or the winner value gates the score to -1, a 2.0 loss).
Tokens lying wholly inside one path-cell span get that cell's counterfactual
weight = 1 - compute_score(gold answer with that cell deleted) — the score the
error destroys, per the real grader.

Outputs (data/armD/):
  train.parquet / val.parquet — precomputed input_ids + float loss_mask
  test.parquet — curriculum-style eval rows (data_source hex_witness_armD)
  train_debug.jsonl — human-readable rows for eyes-on-data

Run: /venv/main/bin/python scripts/build_armD_witness.py
"""

import json
import os
import random
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from transformers import AutoTokenizer

from hexenv.board import Board, EMPTY, BLACK, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from hexenv.reward_verl import compute_score
from scripts.build_sft_certificates import Q, TAIL

MODEL = "Qwen/Qwen3-1.7B"
SIZES = [2, 3, 4, 5]
TRAIN_CAP = 1000       # per size
VAL_CAP = 50           # per size
TEST_CAP = 100         # per size
WINNER_CAP_FRAC = 0.55  # max fraction of a size pool held by one winner
ATTEMPTS = 250_000     # playouts per size (early stop when pool is full)
IM_END = "<|im_end|>"


def unique_shortest_path(b, color):
    """Return the winner's unique minimal winning path (list of (x,y)),
    or None if zero or more than one minimal path exists."""
    n = b.size
    stones = {(x, y) for y in range(n) for x in range(n) if b.grid[y][x] == color}
    if color == BLACK:
        sources = [(x, y) for (x, y) in stones if y == 0]
        is_goal = lambda x, y: y == n - 1
    else:
        sources = [(x, y) for (x, y) in stones if x == 0]
        is_goal = lambda x, y: x == n - 1
    dist = {s: 0 for s in sources}
    cnt = {s: 1 for s in sources}
    q = deque(sources)
    while q:
        u = q.popleft()
        for v in b.neighbors(*u):
            if v not in stones:
                continue
            if v not in dist:
                dist[v] = dist[u] + 1
                cnt[v] = cnt[u]
                q.append(v)
            elif dist[v] == dist[u] + 1:
                cnt[v] += cnt[u]
    goals = [s for s in stones if is_goal(*s) and s in dist]
    if not goals:
        return None
    dmin = min(dist[g] for g in goals)
    total = sum(cnt[g] for g in goals if dist[g] == dmin)
    if total != 1:
        return None
    cur = next(g for g in goals if dist[g] == dmin)
    path = [cur]
    while dist[cur] > 0:
        preds = [v for v in b.neighbors(*cur)
                 if v in dist and dist[v] == dist[cur] - 1]
        assert len(preds) == 1, (preds, cur)
        cur = preds[0]
        path.append(cur)
    return list(reversed(path))


def random_terminal_board(rng, size):
    b = Board(size)
    cells = b.legal_moves()
    rng.shuffle(cells)
    for c in cells:
        b.play(c)
        if b.winner() != EMPTY:
            return b
    return None


def make_ground_truth(b, winner):
    moves = [["B" if c == BLACK else "W", cell] for c, cell in b.moves]
    return {"category": "witness_armD", "task": "path", "size": b.size,
            "moves": moves, "path_winner": winner}


def grade(gt, winner, names):
    ans = "Answer: " + json.dumps({"winner": winner, "path": names})
    return compute_score("hex_witness_armD", ans, json.dumps(gt))["score"]


def cell_weights(gt, winner, names):
    """weight_i = 1 - score(gold with cell i deleted), via the real grader."""
    ws = []
    for i in range(len(names)):
        pruned = names[:i] + names[i + 1:]
        s = grade(gt, winner, pruned) if pruned else -1.0
        ws.append(round(1.0 - s, 4))
    return ws


def build_rows():
    rng = random.Random(20260806)
    pools = {s: {} for s in SIZES}  # position key -> record
    target = TRAIN_CAP + VAL_CAP + TEST_CAP
    for size in SIZES:
        winner_count = {"Black": 0, "White": 0}
        winner_cap = int(target * WINNER_CAP_FRAC)
        t0 = time.time()
        last_new = 0
        for attempt in range(ATTEMPTS):
            if len(pools[size]) >= target or attempt - last_new > 20_000:
                break
            b = random_terminal_board(rng, size)
            if b is None:
                continue
            color = b.winner()
            winner = "Black" if color == BLACK else "White"
            if winner_count[winner] >= winner_cap:
                continue
            key = tuple(sorted((c, cell) for c, cell in b.moves))
            if key in pools[size]:
                continue
            path = unique_shortest_path(b, color)
            if path is None:
                continue
            names = [cell_name(x, y) for x, y in path]
            gt = make_ground_truth(b, winner)
            assert grade(gt, winner, names) == 1.0, (gt, names)
            last_new = attempt
            winner_count[winner] += 1
            pools[size][key] = {
                "size": size, "winner": winner, "gold_path": names,
                "ground_truth": gt,
                "prompt_text": RULES.format(n=size, board=render_ascii(b)) + Q + TAIL,
                "cell_weights": cell_weights(gt, winner, names),
            }
        print(f"size {size}: pool {len(pools[size])} "
              f"({sum(1 for r in pools[size].values() if r['winner'] == 'Black')}B/"
              f"{sum(1 for r in pools[size].values() if r['winner'] == 'White')}W) "
              f"after {attempt + 1} attempts, {time.time() - t0:.0f}s")
    return pools


def tokenize_row(tok, rec):
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": rec["prompt_text"]}],
        add_generation_prompt=True, enable_thinking=False,
        tokenize=True)["input_ids"]
    answer_text = "Answer: " + json.dumps(
        {"winner": rec["winner"], "path": rec["gold_path"]})
    completion_text = answer_text + IM_END
    enc = tok(completion_text, add_special_tokens=False,
              return_offsets_mapping=True)
    comp_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    # char spans of each path cell (cell chars only, inside the quotes)
    spans = []
    pos = 0
    for cell, w in zip(rec["gold_path"], rec["cell_weights"]):
        i = completion_text.index('"' + cell + '"', pos) + 1
        spans.append((i, i + len(cell), w))
        pos = i + len(cell)
    weights = []
    for (a, z) in offsets:
        w = 2.0
        for (ca, cz, cw) in spans:
            if a >= ca and z <= cz:
                w = cw
                break
        weights.append(w)
    input_ids = list(prompt_ids) + list(comp_ids)
    loss_mask = [0.0] * len(prompt_ids) + [float(w) for w in weights]
    return {"input_ids": input_ids, "loss_mask": loss_mask,
            "answer_text": answer_text,
            "completion_tokens": [tok.decode([t]) for t in comp_ids],
            "completion_weights": weights}


def main():
    os.makedirs("data/armD", exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    pools = build_rows()
    rng = random.Random(7)
    train, val, test = [], [], []
    for size in SIZES:
        recs = list(pools[size].values())
        # stratify the split by winner
        by_w = {"Black": [], "White": []}
        for r in recs:
            by_w[r["winner"]].append(r)
        for lst in by_w.values():
            rng.shuffle(lst)
        n = len(recs)
        n_test = min(TEST_CAP, max(2, n // 4))
        n_val = min(VAL_CAP, max(1, n // 10))
        interleaved = [r for pair in
                       __import__("itertools").zip_longest(by_w["Black"], by_w["White"])
                       for r in pair if r is not None]
        test += interleaved[:n_test]
        val += interleaved[n_test:n_test + n_val]
        train += interleaved[n_test + n_val:n_test + n_val + TRAIN_CAP]
        print(f"size {size}: train {min(n - n_test - n_val, TRAIN_CAP)} "
              f"val {n_val} test {n_test}")

    for name, split in [("train", train), ("val", val)]:
        rows = []
        for r in split:
            t = tokenize_row(tok, r)
            rows.append({
                "input_ids": t["input_ids"], "loss_mask": t["loss_mask"],
                "size": r["size"], "winner": r["winner"],
                "path_len": len(r["gold_path"]),
                "prompt_text": r["prompt_text"], "answer_text": t["answer_text"],
                "ground_truth": json.dumps(r["ground_truth"]),
            })
        pd.DataFrame(rows).to_parquet(f"data/armD/{name}.parquet")
        print(f"{name}: {len(rows)} rows")

    test_rows = []
    for r in test:
        test_rows.append({
            "data_source": "hex_witness_armD",
            "prompt": [{"role": "user", "content": r["prompt_text"]}],
            "ability": "hex_path",
            "reward_model": {"style": "rule",
                             "ground_truth": json.dumps(r["ground_truth"])},
            "extra_info": {"category": "witness_armD", "size": r["size"],
                           "task": "path", "path_len": len(r["gold_path"]),
                           "gold_answer": "Answer: " + json.dumps(
                               {"winner": r["winner"], "path": r["gold_path"]})},
        })
    pd.DataFrame(test_rows).to_parquet("data/armD/test.parquet")
    print(f"test: {len(test_rows)} rows")

    with open("data/armD/train_debug.jsonl", "w") as f:
        for r in train[:50]:
            t = tokenize_row(tok, r)
            f.write(json.dumps({
                "size": r["size"], "winner": r["winner"],
                "gold_path": r["gold_path"], "cell_weights": r["cell_weights"],
                "prompt_text": r["prompt_text"], "answer_text": t["answer_text"],
                "tokens_with_weights": list(zip(t["completion_tokens"],
                                                t["completion_weights"])),
            }) + "\n")
    print("wrote data/armD/train_debug.jsonl")


if __name__ == "__main__":
    main()
