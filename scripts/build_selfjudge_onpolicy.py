"""On-policy self-judge training data: mine bok's REAL errors on hard/long
boards (where it errs ~50%), so the judge trains on the model's own error
distribution (subtle membership/edge near-misses) instead of blatant
constructed adjacency gaps.

Balanced (subsample correct to #incorrect), weighted per board ~ p(1-p)
(user's instinct). Eval reuses the existing test_natural / test_probe for an
apples-to-apples comparison with the constructed-trained judge.

Output: data/selfjudge/train_onpolicy.parquet
Run (verl venv, GPU): /venv/verl/bin/python scripts/build_selfjudge_onpolicy.py
"""

import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from hexenv.board import Board, BLACK, WHITE, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from hexenv.reward_verl import compute_score
from scripts.build_armD_witness import MODEL, IM_END
from scripts.build_armD_witness_v2 import fabricate_moves
from scripts.witness_constructive import gen_board
from scripts.build_selfjudge_data import judge_prompt, gt_for
from scripts.build_sft_certificates import Q, TAIL

K = 8
N_BOARDS = 1400
SIZES = [8, 9]


def test_board_renders():
    seen = set()
    for f in ("test_natural", "test_probe"):
        df = pd.read_parquet(f"data/selfjudge/{f}.parquet")
        for t in df["prompt_text"]:
            m = re.search(r"empty:\n\n(.*?)\nThis game", t, re.DOTALL)
            if m:
                seen.add(m.group(1))
    return seen


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(MODEL)
    banned = test_board_renders()
    rng = random.Random(424242)
    boards = []
    while len(boards) < N_BOARDS:
        n = rng.choice(SIZES)
        winner = rng.choice(["Black", "White"])
        g = gen_board(rng, n, winner, p_forward=rng.uniform(0.0, 0.25),
                      extra_winner_frac=rng.uniform(0.1, 0.6))
        if g is None:
            continue
        wst, lst, path = g
        b = Board(n)
        for x, y in wst:
            b.grid[y][x] = BLACK if winner == "Black" else WHITE
        for x, y in lst:
            b.grid[y][x] = WHITE if winner == "Black" else BLACK
        if render_ascii(b) in banned:
            continue
        boards.append({"n": n, "winner": winner, "board": b,
                       "path": [cell_name(x, y) for x, y in path],
                       "gt": gt_for(n, winner, wst, lst, path)})

    prompts = [TokensPrompt(prompt_token_ids=tok.apply_chat_template(
        [{"role": "user", "content": RULES.format(n=r["n"], board=render_ascii(r["board"])) + Q + TAIL}],
        add_generation_prompt=True, enable_thinking=False,
        tokenize=True)["input_ids"]) for r in boards]
    llm = LLM(model="checkpoints/armD2_bok/hf_merged", max_model_len=1024,
              gpu_memory_utilization=0.55, dtype="bfloat16")
    outs = llm.generate(prompts, SamplingParams(temperature=1.0, n=K,
                                                max_tokens=96))

    def tokenize(prompt, yesno, w):
        pids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            enable_thinking=False, tokenize=True)["input_ids"]
        comp = tok(f"Answer: {yesno}" + IM_END, add_special_tokens=False)["input_ids"]
        return list(pids) + list(comp), [0.0] * len(pids) + [float(w)] * len(comp)

    yes_rows, no_rows = [], []
    nwin = 0
    for r, o in zip(boards, outs):
        seen = set()
        graded = []
        for s in o.outputs:
            txt = s.text.strip()
            m = re.search(r"Answer:\s*(.*)", txt)
            ans = m.group(1).strip() if m else txt
            if ans in seen:
                continue
            seen.add(ans)
            ok = compute_score("x", "Answer: " + ans, json.dumps(r["gt"]))["score"] == 1.0
            graded.append((ans, ok))
        p = sum(ok for _, ok in graded) / len(graded) if graded else 0.0
        w = 4.0 * p * (1 - p) + 0.1
        for ans, ok in graded:
            # winner-error accounting
            mm = re.search(r"\{.*\}", ans, re.DOTALL)
            if not ok and mm:
                try:
                    if str(json.loads(mm.group(0)).get("winner", "")).capitalize() != r["gt"]["path_winner"]:
                        nwin += 1
                except Exception:
                    pass
            ids, mask = tokenize(judge_prompt(r["n"], r["board"], ans),
                                 "Yes" if ok else "No", w)
            if len(ids) > 2048:
                continue
            (yes_rows if ok else no_rows).append(
                {"input_ids": ids, "loss_mask": mask, "size": r["n"], "label": ok})

    rng.shuffle(yes_rows)
    yes_bal = yes_rows[:len(no_rows)]      # balance to #errors
    allr = no_rows + yes_bal
    rng.shuffle(allr)
    pd.DataFrame(allr).to_parquet("data/selfjudge/train_onpolicy.parquet")
    print(f"on-policy train {len(allr)} rows: {len(no_rows)} No / {len(yes_bal)} Yes "
          f"(had {len(yes_rows)} correct total)")
    print(f"natural winner-errors in mined set: {nwin} (of {len(no_rows)} errors)")


if __name__ == "__main__":
    main()
