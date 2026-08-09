"""Test the decomposition hypothesis: bok's witness errors are ~85% membership
(a path cell that isn't the winner's stone). Can the model, asked PER-CELL
'is X the winner's stone?', distinguish the exact cells bok hallucinated --
at AUC >> the 0.65 of the holistic answer-judge?

Two phases (separate processes, one vLLM engine each):
  gather : bok samples witness answers; collect (membership_prompt,is_stone)
           for cells on ERROR paths -> data/selfjudge/percell.jsonl
  probe  : base Qwen3-1.7B answers each per-cell membership q via forced
           'Answer:' Yes/No logprob -> AUC(is_stone vs P(Yes))

Run: /venv/verl/bin/python scripts/percell_probe.py gather
     /venv/verl/bin/python scripts/percell_probe.py probe
"""

import json
import math
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, BLACK, WHITE, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from hexenv.reward_verl import compute_score
from scripts.build_armD_witness import MODEL
from scripts.build_armD_witness_v2 import fabricate_moves
from scripts.witness_constructive import gen_board
from scripts.build_sft_certificates import Q, TAIL
from scripts.sj_auc import auc

QF = "data/selfjudge/percell.jsonl"


def membership_prompt(n, board, winner, cell):
    return (RULES.format(n=n, board=render_ascii(board))
            + f"\nDoes {winner} have a stone on cell {cell}?"
              "\nEnd your response with exactly one line of the form:"
              "\nAnswer: Yes|No\n")


def gather():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(11)
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
        gt = {"category": "d", "task": "path", "size": n,
              "moves": fabricate_moves(random, winner, wst, lst, path),
              "path_winner": winner}
        boards.append({"n": n, "winner": winner, "board": b,
                       "wstones": {cell_name(x, y) for x, y in wst}, "gt": gt})
    llm = LLM(model="checkpoints/armD2_bok/hf_merged", max_model_len=1024,
              gpu_memory_utilization=0.55, dtype="bfloat16")
    wp = [TokensPrompt(prompt_token_ids=tok.apply_chat_template(
        [{"role": "user", "content": RULES.format(n=r["n"], board=render_ascii(r["board"])) + Q + TAIL}],
        add_generation_prompt=True, enable_thinking=False,
        tokenize=True)["input_ids"]) for r in boards]
    outs = llm.generate(wp, SamplingParams(temperature=1.0, n=4, max_tokens=96))
    rows = []
    for r, o in zip(boards, outs):
        for s in o.outputs:
            m = re.search(r"\{.*\}", s.text, re.DOTALL)
            if not m or compute_score("x", "Answer: " + m.group(0),
                                      json.dumps(r["gt"]))["score"] == 1.0:
                continue
            try:
                path = [str(c).lower() for c in json.loads(m.group(0)).get("path", [])]
            except Exception:
                continue
            for c in path:
                if re.fullmatch(r"[a-i][1-9]", c):
                    x, y = ord(c[0]) - 97, int(c[1:]) - 1
                    v = r["board"].grid[y][x]
                    occ = "Black" if v == BLACK else ("White" if v == WHITE else "Neither")
                    occ_prompt = (RULES.format(n=r["n"], board=render_ascii(r["board"]))
                                  + f"\nWhich player, if any, has a stone on cell {c}?"
                                    "\nEnd your response with exactly one line of the form:"
                                    "\nAnswer: Black|White|Neither\n")
                    rows.append({"prompt": membership_prompt(r["n"], r["board"],
                                                             r["winner"], c),
                                 "occ_prompt": occ_prompt, "winner": r["winner"],
                                 "occupant": occ, "is_stone": c in r["wstones"]})
    rng.shuffle(rows)
    bad = [x for x in rows if not x["is_stone"]][:500]
    good = [x for x in rows if x["is_stone"]][:500]
    with open(QF, "w") as f:
        for x in bad + good:
            f.write(json.dumps(x) + "\n")
    print(f"gathered {len(bad)} not-stone / {len(good)} stone per-cell queries")


def probe():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(QF)]
    yes_ids = {tok(t, add_special_tokens=False)["input_ids"][0] for t in (" Yes", "Yes")}
    no_ids = {tok(t, add_special_tokens=False)["input_ids"][0] for t in (" No", "No")}
    llm = LLM(model="Qwen/Qwen3-1.7B", max_model_len=1024,
              gpu_memory_utilization=0.55, dtype="bfloat16")
    prompts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}],
                                       add_generation_prompt=True, enable_thinking=False,
                                       tokenize=False) + "Answer:" for x in rows]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))
    labels, pyes = [], []
    for x, o in zip(rows, outs):
        lp = o.outputs[0].logprobs[0]
        ly = max((lp[t].logprob for t in yes_ids if t in lp), default=-30.0)
        ln = max((lp[t].logprob for t in no_ids if t in lp), default=-30.0)
        pyes.append(1.0 / (1.0 + math.exp(ln - ly)))
        labels.append(1 if x["is_stone"] else 0)
    a = auc(labels, pyes)
    acc = sum((p > 0.5) == bool(l) for p, l in zip(pyes, labels)) / len(labels)
    print(f"BASE per-cell membership on error-path cells: AUC={a:.3f} "
          f"acc@0.5={acc:.3f} n={len(labels)}")
    print("(holistic answer-judge natural AUC: base 0.578 / SFT 0.645)")


def probe_occ():
    """Trained-occupancy per-cell check on the same error-path cells."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(QF)]
    llm = LLM(model="Qwen/Qwen3-1.7B", max_model_len=1024,
              gpu_memory_utilization=0.55, dtype="bfloat16",
              enable_lora=True, max_lora_rank=64)
    prompts = [tok.apply_chat_template([{"role": "user", "content": x["occ_prompt"]}],
                                       add_generation_prompt=True, enable_thinking=False,
                                       tokenize=False) for x in rows]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=6),
                        lora_request=LoRARequest("occ", 1, "checkpoints/occ_probe/adapter"))
    occ_correct = 0
    detect_bad = [0, 0]   # among not-stone cells: [n, said-not-winner]
    detect_good = [0, 0]  # among stone cells: [n, said-winner]
    for x, o in zip(rows, outs):
        m = re.search(r"(Black|White|Neither)", o.outputs[0].text)
        pred = m.group(1) if m else "?"
        occ_correct += (pred == x["occupant"])
        said_winner = (pred == x["winner"])
        if x["is_stone"]:
            detect_good[0] += 1; detect_good[1] += said_winner
        else:
            detect_bad[0] += 1; detect_bad[1] += (not said_winner)
    print(f"TRAINED occupancy on error-path cells: raw 3-way acc="
          f"{occ_correct/len(rows):.3f} n={len(rows)}")
    print(f"  detect hallucinated (not-winner-stone -> says not winner): "
          f"{detect_bad[1]/max(detect_bad[0],1):.3f} (n={detect_bad[0]})")
    print(f"  confirm real (winner-stone -> says winner): "
          f"{detect_good[1]/max(detect_good[0],1):.3f} (n={detect_good[0]})")
    bal = 0.5 * (detect_bad[1]/max(detect_bad[0],1) + detect_good[1]/max(detect_good[0],1))
    print(f"  balanced per-cell membership acc = {bal:.3f}"
          "  (vs holistic answer-judge natural AUC 0.645)")


if __name__ == "__main__":
    {"gather": gather, "probe": probe, "probe_occ": probe_occ}[sys.argv[1]]()
