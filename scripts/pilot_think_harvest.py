"""Phase-2 pilot: can (a) base Qwen3-1.7B, (b) the armD2 ep8 adapter produce
grader-perfect witness answers when allowed to THINK (own CoT, no injected
content, forced </think> close only)?

25 fresh constructive boards, 5 per plen bin (8-13/14-17/18-21/22-25/26-32),
k=8 at temp 1.0, think budget 1024. Reports pass@8 and perfect-rate per bin
per sampler; dumps all samples to results/armD/pilot_think.jsonl.

Run: /venv/verl/bin/python scripts/pilot_think_harvest.py
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, BLACK, WHITE, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from scripts.build_sft_certificates import Q, TAIL
from scripts.build_armD_witness import grade
from scripts.build_armD_witness_v2 import fabricate_moves
from scripts.witness_constructive import gen_board

BINS = [(8, 13), (14, 17), (18, 21), (22, 25), (26, 32)]
PER_BIN = 5
K = 8
THINK_BUDGET = 1024
ANSWER_BUDGET = 256


def make_boards():
    rng = random.Random(77)
    bins = [[] for _ in BINS]
    while any(len(b) < PER_BIN for b in bins):
        n = rng.choice([7, 8, 9])
        winner = rng.choice(["Black", "White"])
        out = gen_board(rng, n, winner, p_forward=rng.uniform(0.0, 0.3),
                        extra_winner_frac=rng.uniform(0.0, 0.6))
        if out is None:
            continue
        wst, lst, path = out
        bi = next((i for i, (a, b) in enumerate(BINS)
                   if a <= len(path) <= b), None)
        if bi is None or len(bins[bi]) >= PER_BIN:
            continue
        b = Board(n)
        wcol = BLACK if winner == "Black" else WHITE
        lcol = WHITE if winner == "Black" else BLACK
        for x, y in wst:
            b.grid[y][x] = wcol
        for x, y in lst:
            b.grid[y][x] = lcol
        names = [cell_name(x, y) for x, y in path]
        gt = {"category": "witness_pilot", "task": "path", "size": n,
              "moves": fabricate_moves(rng, winner, wst, lst, path),
              "path_winner": winner}
        assert grade(gt, winner, names) == 1.0
        bins[bi].append({
            "plen": len(path), "size": n, "gt": gt,
            "prompt": RULES.format(n=n, board=render_ascii(b)) + Q + TAIL,
        })
    return [r for b in bins for r in b]


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from hexenv.reward_verl import compute_score

    boards = make_boards()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": b["prompt"]}],
        add_generation_prompt=True, enable_thinking=True, tokenize=False)
        for b in boards]

    llm = LLM(model="Qwen/Qwen3-1.7B", max_model_len=2048,
              gpu_memory_utilization=0.6, dtype="bfloat16",
              enable_lora=True, max_lora_rank=64)
    adapter = LoRARequest("ep8", 1, "checkpoints/armD2_sft_weighted/adapter_ep8")

    results = []
    for sampler, lora in [("base", None), ("ep8", adapter)]:
        sp1 = SamplingParams(temperature=1.0, n=K, max_tokens=THINK_BUDGET,
                             stop=["</think>"])
        outs = llm.generate(prompts, sp1, lora_request=lora)
        cont_prompts, meta = [], []
        for b, prompt, o in zip(boards, prompts, outs):
            for s in o.outputs:
                think = s.text
                natural = s.finish_reason == "stop"
                # sole injection: the forced close
                cont_prompts.append(prompt + think + "</think>\n\n")
                meta.append((b, think, natural))
        sp2 = SamplingParams(temperature=1.0, max_tokens=ANSWER_BUDGET)
        outs2 = llm.generate(cont_prompts, sp2, lora_request=lora)
        for (b, think, natural), o in zip(meta, outs2):
            answer = o.outputs[0].text
            full = think + "</think>\n\n" + answer
            d = compute_score("x", full, json.dumps(b["gt"]))
            results.append({
                "sampler": sampler, "plen": b["plen"], "size": b["size"],
                "think": think, "natural_close": natural, "answer": answer,
                "score": d["score"], "perfect": d["kind_win"],
                "think_tokens": len(tok(think)["input_ids"]),
                "gt": json.dumps(b["gt"]),
            })

    os.makedirs("results/armD", exist_ok=True)
    with open("results/armD/pilot_think.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    for sampler in ["base", "ep8"]:
        print(f"\n=== {sampler} ===")
        sub = [r for r in results if r["sampler"] == sampler]
        nat = sum(r["natural_close"] for r in sub) / len(sub)
        tt = sorted(r["think_tokens"] for r in sub)
        print(f"natural close {nat:.2f}, think tokens med/p90 "
              f"{tt[len(tt)//2]}/{tt[int(len(tt)*0.9)]}")
        for a, z in BINS:
            grp = [r for r in sub if a <= r["plen"] <= z]
            byb = {}
            for r in grp:
                byb.setdefault((r["plen"], r["gt"]), []).append(r["perfect"])
            p8 = sum(1 for v in byb.values() if any(v)) / len(byb)
            rate = sum(r["perfect"] for r in grp) / len(grp)
            print(f"  plen {a}-{z}: pass@8 {p8:.2f}  perfect-rate {rate:.2f}")


if __name__ == "__main__":
    main()
