"""Test forced-close two-phase generation: think capped at budget, inject
close + answer scaffold, generate the move. Measures legal/win/variance.
"""

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
    model = "Qwen/Qwen3-1.7B"
    K = 8
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

    base_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": move_prompt(board_from_record(r))}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for r in positions
    ]

    for temp, budget in [(1.0, 2048), (1.0, 3072), (0.7, 2048)]:
        # phase 1: think, capped
        sp1 = SamplingParams(temperature=temp, top_p=0.95, max_tokens=budget,
                             n=K, seed=13, stop=["</think>"],
                             include_stop_str_in_output=True)
        outs1 = llm.generate(base_prompts, sp1, use_tqdm=False)
        # phase 2: force close + answer for every sample
        phase2_prompts = []
        for bp, out in zip(base_prompts, outs1):
            for o in out.outputs:
                text = o.text
                if "</think>" in text:
                    cont = bp + text + "\n\nMove:"
                else:
                    cont = bp + text + "\n...\n</think>\n\nMove:"
                phase2_prompts.append(cont)
        sp2 = SamplingParams(temperature=temp, top_p=0.95, max_tokens=8, seed=13)
        outs2 = llm.generate(phase2_prompts, sp2, use_tqdm=False)

        n = legal = win = 0
        var_groups = 0
        forced = 0
        idx = 0
        for r, out in zip(positions, outs1):
            wins = set(r["winning_moves"])
            lg = set(board_from_record(r).legal_moves())
            rewards = []
            for o in out.outputs:
                ans = outs2[idx].outputs[0].text
                idx += 1
                n += 1
                forced += "</think>" not in o.text
                mv = extract_move("Move:" + ans)
                ok_l = mv is not None and mv in lg
                ok_w = ok_l and mv in wins
                legal += ok_l
                win += ok_w
                rewards.append(ok_w)
            var_groups += len(set(rewards)) > 1
        print(f"t{temp}_b{budget}: forced_close={forced/n:.2f} legal={legal/n:.3f} "
              f"win={win/n:.3f} var_groups={var_groups}/{len(positions)}", flush=True)


if __name__ == "__main__":
    main()
