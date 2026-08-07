"""Arm T SFT data: verbatim (own-think, own-answer) successes + the same
1000-row armD2 replay slice arm B used.

Target = think + "</think>\n\n" + answer + <|im_end|>, tokenized raw.
Weights: 1.0 on think/preamble tokens, 2.0 on answer-JSON structure,
grader-counterfactual weights on path cells, 2.0 on <|im_end|>.

Output: data/armD2/distill_t/train.parquet
Run: /venv/main/bin/python scripts/build_distill_think_sft.py
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from transformers import AutoTokenizer

from scripts.build_armD_witness import MODEL, IM_END, cell_weights

REPLAY_ROWS = 1000


def tokenize_think_row(tok, r):
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": r["prompt"]}],
        add_generation_prompt=True, enable_thinking=True,
        tokenize=True)["input_ids"]
    answer = r["answer"]
    completion = r["think"] + "</think>\n\n" + answer + IM_END
    m = re.search(r"\{.*?\}", answer, re.DOTALL)
    obj = json.loads(m.group(0))
    path = [str(c).lower() for c in obj["path"]]
    winner = str(obj["winner"]).capitalize()
    ws = cell_weights(r["gt"], winner, path)
    json_start = len(r["think"]) + len("</think>\n\n") + m.start()
    json_end = len(r["think"]) + len("</think>\n\n") + m.end()
    spans = []
    pos = json_start
    for cell, w in zip(path, ws):
        i = completion.index('"' + cell + '"', pos) + 1
        spans.append((i, i + len(cell), w))
        pos = i + len(cell)
    enc = tok(completion, add_special_tokens=False, return_offsets_mapping=True)
    weights = []
    for (a, z) in enc["offset_mapping"]:
        if z <= json_start:
            w = 1.0                      # think + preamble
        elif a >= len(completion) - len(IM_END):
            w = 2.0                      # im_end
        elif a >= json_start and z <= json_end:
            w = 2.0                      # JSON structure
            for (ca, cz, cw) in spans:
                if a >= ca and z <= cz:
                    w = cw
                    break
        else:
            w = 1.0
        weights.append(float(w))
    input_ids = list(prompt_ids) + list(enc["input_ids"])
    loss_mask = [0.0] * len(prompt_ids) + weights
    return input_ids, loss_mask


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = []
    too_long = 0
    for l in open("data/armD2/harvest_think.jsonl"):
        r = json.loads(l)
        input_ids, loss_mask = tokenize_think_row(tok, r)
        if len(input_ids) > 2048:
            too_long += 1
            continue
        rows.append({
            "input_ids": input_ids, "loss_mask": loss_mask,
            "size": r["size"], "winner": r["gt"]["path_winner"],
            "path_len": r["plen"], "prompt_text": r["prompt"],
            "answer_text": r["answer"],
            "ground_truth": json.dumps(r["gt"]),
        })
    print(f"think rows kept {len(rows)}, dropped too-long {too_long}")
    lens = sorted(len(x["input_ids"]) for x in rows)
    print(f"seq len med/p90/max {lens[len(lens)//2]}/{lens[int(len(lens)*.9)]}/{lens[-1]}")
    replay = pd.read_parquet("data/armD2/train.parquet").sample(
        n=REPLAY_ROWS, random_state=3)
    df = pd.concat([pd.DataFrame(rows), replay], ignore_index=True)
    df = df.sample(frac=1.0, random_state=4).reset_index(drop=True)
    os.makedirs("data/armD2/distill_t", exist_ok=True)
    df.to_parquet("data/armD2/distill_t/train.parquet")
    print(f"total train rows: {len(df)}")


if __name__ == "__main__":
    main()
