"""Test prompt variants for think-mode termination on move prompts."""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.board import Board, BLACK, WHITE
from hexenv.prompts import move_prompt, extract_move

VARIANTS = {
    "v1_plain": "",
    "v2_brief": "\nKeep your reasoning brief: under 250 words, then commit to a move. Do not exhaustively enumerate moves.",
    "v3_decisive": "\nConsider at most 3 candidate moves, briefly compare them, then decide. A short, decisive analysis is better than a long one.",
}


def board_from_record(rec):
    b = Board(rec["size"])
    for color, cell in rec["moves"]:
        b.play(cell, BLACK if color == "B" else WHITE)
    b.to_move = BLACK if rec["to_move"] == "B" else WHITE
    return b


def main():
    model = "Qwen/Qwen3-1.7B"
    recs = [json.loads(l) for l in open("data/corpus_5x5.jsonl")]
    winning = [r for r in recs if r["winner"] == r["to_move"]
               and 0 < len(r["winning_moves"]) < r["n_legal"] and r["n_stones"] >= 2]
    rng = random.Random(7)
    rng.shuffle(winning)
    positions = winning[:30]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.72)

    results = {}
    for vname, suffix in VARIANTS.items():
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": move_prompt(board_from_record(r)) + suffix}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            for r in positions
        ]
        for temp in [0.7, 1.0]:
            sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=3072, n=8, seed=11)
            outs = llm.generate(prompts, sp)
            n = tr = legal = win = var_groups = 0
            lens = []
            samples = []
            for r, out in zip(positions, outs):
                wins = set(r["winning_moves"])
                lg = set(board_from_record(r).legal_moves())
                rewards = []
                for o in out.outputs:
                    n += 1
                    lens.append(len(o.token_ids))
                    tr += o.finish_reason == "length"
                    mv = extract_move(o.text)
                    ok_l = mv is not None and mv in lg
                    ok_w = ok_l and mv in wins
                    legal += ok_l
                    win += ok_w
                    rewards.append(ok_w)
                    if len(samples) < 3 and o.finish_reason != "length":
                        samples.append(o.text)
                var_groups += len(set(rewards)) > 1
            lens.sort()
            key = f"{vname}_t{temp}"
            results[key] = {
                "trunc": round(tr / n, 3), "legal": round(legal / n, 3),
                "win": round(win / n, 3), "var_groups": f"{var_groups}/{len(positions)}",
                "len_p50": lens[len(lens) // 2], "len_p90": lens[int(len(lens) * .9)],
            }
            print(key, results[key], flush=True)
            os.makedirs("results/phase1/variants", exist_ok=True)
            with open(f"results/phase1/variants/{key}_samples.json", "w") as f:
                json.dump(samples, f, indent=1)

    json.dump(results, open("results/phase1/variant_sweep.json", "w"), indent=2)


if __name__ == "__main__":
    main()
