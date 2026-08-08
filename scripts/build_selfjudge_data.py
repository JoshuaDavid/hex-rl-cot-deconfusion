"""Self-judge falsifier data: can a model learn to predict whether a witness
answer is correct (metacognition / "should I retry")?

Freeze the bok witness answerer, sample k answers per board, grade them, and
build an is_correct SFT task: prompt = (board + proposed answer) -> "Answer:
Yes|No". Per-board loss weight ~ p(1-p) (informative boards where the model
is genuinely uncertain get the most gradient; user's instinct).

Test sets:
  natural : held-out bok samples (real error mix)
  probe   : per held-out board, 3 constructed answers with known labels --
            gold (Yes), broken-link (No), wrong-winner (No) -- to measure
            per-error-type detection independent of bok's natural error rates.

Output: data/selfjudge/{train,test_natural,test_probe}.parquet
Run (verl venv, needs GPU): /venv/verl/bin/python scripts/build_selfjudge_data.py
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

K = 8
N_TRAIN, N_TEST = 1000, 250
SIZES = [6, 7, 8, 9]


def gt_for(n, winner, wst, lst, path):
    return {"category": "witness_sj", "task": "path", "size": n,
            "moves": fabricate_moves(random, winner, wst, lst, path),
            "path_winner": winner}


def judge_prompt(n, board, ans_str):
    return (RULES.format(n=n, board=render_ascii(board))
            + "\nThis game is over. A player claims the following winner and"
              " winning path:\n" + ans_str
            + "\n\nIs this claim correct? It is correct only if it names the"
              " true winner AND gives a valid path (every listed cell holds"
              " that winner's stone, consecutive cells are adjacent, and the"
              " path connects the winner's two edges)."
              "\nEnd your response with exactly one line of the form:"
              "\nAnswer: Yes|No\n")


def err_type(gt, ans_text):
    d = compute_score("x", ans_text, json.dumps(gt))
    if d["score"] == 1.0:
        return "correct"
    m = re.search(r"\{.*\}", ans_text, re.DOTALL)
    if not m:
        return "unparsed"
    try:
        o = json.loads(m.group(0))
    except Exception:
        return "unparsed"
    if str(o.get("winner", "")).capitalize() != gt["path_winner"]:
        return "winner"
    return "link"


def make_boards(rng, n_boards, used):
    out = []
    while len(out) < n_boards:
        n = rng.choice(SIZES)
        winner = rng.choice(["Black", "White"])
        g = gen_board(rng, n, winner, p_forward=rng.uniform(0.15, 0.7),
                      extra_winner_frac=rng.uniform(0.1, 0.6))
        if g is None:
            continue
        wst, lst, path = g
        key = (winner, frozenset(wst), frozenset(lst))
        if key in used:
            continue
        used.add(key)
        b = Board(n)
        for x, y in wst:
            b.grid[y][x] = BLACK if winner == "Black" else WHITE
        for x, y in lst:
            b.grid[y][x] = WHITE if winner == "Black" else BLACK
        names = [cell_name(x, y) for x, y in path]
        out.append({"n": n, "winner": winner, "board": b, "path": names,
                    "gt": gt_for(n, winner, wst, lst, path)})
    return out


def corrupt_link(names):
    if len(names) < 3:
        return None
    i = len(names) // 2
    return names[:i] + names[i + 1:]


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    rng = random.Random(20260808)
    used = set()
    train_b = make_boards(rng, N_TRAIN, used)
    test_b = make_boards(rng, N_TEST, used)
    tok = AutoTokenizer.from_pretrained(MODEL)

    # sample bok answers for TEST boards only (natural transfer set); training
    # uses balanced constructed errors (bok is ~92% correct on won boards, so
    # on-policy self-judge signal is starved -- esp. winner errors ~0)
    all_b = test_b
    wit_prompts = []
    from scripts.build_sft_certificates import Q, TAIL
    for r in all_b:
        p = RULES.format(n=r["n"], board=render_ascii(r["board"])) + Q + TAIL
        wit_prompts.append(TokensPrompt(prompt_token_ids=tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True,
            enable_thinking=False, tokenize=True)["input_ids"]))
    llm = LLM(model="checkpoints/armD2_bok/hf_merged", max_model_len=1024,
              gpu_memory_utilization=0.55, dtype="bfloat16")
    outs = llm.generate(wit_prompts, SamplingParams(temperature=1.0, n=K,
                                                    max_tokens=96))
    for r, o in zip(all_b, outs):
        samples = []
        for s in o.outputs:
            txt = s.text.strip()
            m = re.search(r"Answer:.*", txt)
            ans = m.group(0) if m else txt
            samples.append(ans)
        r["samples"] = samples
        r["labels"] = [compute_score("x", a, json.dumps(r["gt"]))["score"] == 1.0
                       for a in samples]
        r["p"] = sum(r["labels"]) / len(r["labels"])

    def tokenize(prompt, yesno, weight):
        pids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            enable_thinking=False, tokenize=True)["input_ids"]
        comp = tok(f"Answer: {yesno}" + IM_END, add_special_tokens=False)["input_ids"]
        return (list(pids) + list(comp),
                [0.0] * len(pids) + [float(weight)] * len(comp))

    # TRAIN: balanced constructed errors per board -> gold(Yes), broken-link
    # (No), wrong-winner(No). Uniform weight (balanced by construction).
    tr = []
    for r in train_b:
        gold = json.dumps({"winner": r["winner"], "path": r["path"]})
        items = [("Yes", gold)]
        cl = corrupt_link(r["path"])
        if cl:
            items.append(("No", json.dumps({"winner": r["winner"], "path": cl})))
        flip = "White" if r["winner"] == "Black" else "Black"
        items.append(("No", json.dumps({"winner": flip, "path": r["path"]})))
        for yesno, ans_str in items:
            ids, mask = tokenize(judge_prompt(r["n"], r["board"], ans_str),
                                 yesno, 1.0)
            if len(ids) <= 2048:
                tr.append({"input_ids": ids, "loss_mask": mask, "size": r["n"],
                           "label": yesno == "Yes"})
    pd.DataFrame(tr).sample(frac=1.0, random_state=1).to_parquet(
        "data/selfjudge/train.parquet")

    # TEST natural
    tn = []
    for r in test_b:
        for ans, lab in zip(r["samples"], r["labels"]):
            ans_str = ans[ans.find("Answer:") + 7:].strip() if "Answer:" in ans else ans
            et = err_type(r["gt"], ans)
            tn.append({"prompt_text": judge_prompt(r["n"], r["board"], ans_str),
                       "label": bool(lab), "err_type": et, "size": r["n"]})
    pd.DataFrame(tn).to_parquet("data/selfjudge/test_natural.parquet")

    # TEST probe: gold(Yes), broken-link(No), wrong-winner(No)
    tp = []
    for r in test_b:
        gold = {"winner": r["winner"], "path": r["path"]}
        probes = [("correct", json.dumps(gold))]
        cl = corrupt_link(r["path"])
        if cl:
            probes.append(("link", json.dumps({"winner": r["winner"], "path": cl})))
        flipped = "White" if r["winner"] == "Black" else "Black"
        probes.append(("winner", json.dumps({"winner": flipped, "path": r["path"]})))
        for et, ans_str in probes:
            tp.append({"prompt_text": judge_prompt(r["n"], r["board"], ans_str),
                       "label": et == "correct", "err_type": et, "size": r["n"]})
    pd.DataFrame(tp).to_parquet("data/selfjudge/test_probe.parquet")

    print(f"train {len(tr)} judge rows (from {len(train_b)} boards, constructed)")
    print(f"test_natural {len(tn)} | test_probe {len(tp)}")
    import collections
    print("train label balance:", collections.Counter(x["label"] for x in tr))
    print("natural base-rate correct:",
          round(sum(x["label"] for x in tn) / len(tn), 3))
    print("natural err_types:", collections.Counter(x["err_type"] for x in tn))


if __name__ == "__main__":
    os.makedirs("data/selfjudge", exist_ok=True)
    main()
