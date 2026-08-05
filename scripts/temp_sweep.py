"""Sweep (temperature x max_tokens) for think-mode termination + move quality.
Requires vllm (fast). Run from /venv/vllm or /venv/verl python.

python scripts/temp_sweep.py --model Qwen/Qwen3-1.7B --corpus data/corpus_5x5.jsonl
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, BLACK, WHITE
from hexenv.prompts import move_prompt, extract_move


def board_from_record(rec):
    b = Board(rec["size"])
    for color, cell in rec["moves"]:
        b.play(cell, BLACK if color == "B" else WHITE)
    b.to_move = BLACK if rec["to_move"] == "B" else WHITE
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--corpus", default="data/corpus_5x5.jsonl")
    ap.add_argument("--n-positions", type=int, default=30)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="results/phase1/temp_sweep.json")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.corpus)]
    winning = [r for r in recs if r["winner"] == r["to_move"]
               and 0 < len(r["winning_moves"]) < r["n_legal"] and r["n_stones"] >= 2]
    rng = random.Random(7)
    rng.shuffle(winning)
    positions = winning[: args.n_positions]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, max_model_len=8192, gpu_memory_utilization=0.72)

    prompts = [
        tok.apply_chat_template([{"role": "user", "content": move_prompt(board_from_record(r))}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for r in positions
    ]

    results = {}
    for temp in [0.6, 0.8, 1.0]:
        for mt in [3584, 6144]:
            sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=mt,
                                n=args.k, seed=11)
            outs = llm.generate(prompts, sp)
            n = tr = legal = win = 0
            var_groups = 0
            lens = []
            for r, out in zip(positions, outs):
                wins = set(r["winning_moves"])
                lg = set(board_from_record(r).legal_moves())
                rewards = []
                for o in out.outputs:
                    n += 1
                    lens.append(len(o.token_ids))
                    if o.finish_reason == "length":
                        tr += 1
                    mv = extract_move(o.text)
                    ok_legal = mv is not None and mv in lg
                    ok_win = ok_legal and mv in wins
                    legal += ok_legal
                    win += ok_win
                    rewards.append(1.0 if ok_win else -1.0)
                if len(set(rewards)) > 1:
                    var_groups += 1
            lens.sort()
            key = f"t{temp}_m{mt}"
            results[key] = {
                "trunc_rate": tr / n, "legal_rate": legal / n, "win_rate": win / n,
                "var_groups": f"{var_groups}/{len(positions)}",
                "len_p50": lens[len(lens) // 2], "len_p90": lens[int(len(lens) * 0.9)],
            }
            print(key, results[key], flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
