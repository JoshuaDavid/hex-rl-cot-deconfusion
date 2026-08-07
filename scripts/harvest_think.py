"""Arm T harvest: with-think successes from the ep8 adapter on the shared
pool (subset: all of bins 8-13/14-17/18-21 + 400 deep boards). k=8 temp 1.0,
think budget 1024, forced </think> close (sole injection). Keeps up to 2
grader-perfect (think, answer) pairs per board, VERBATIM.

Output: data/armD2/harvest_think.jsonl
Run: /venv/verl/bin/python scripts/harvest_think.py
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

K = 8
THINK_BUDGET = 1024
ANSWER_BUDGET = 256
KEEP_PER_BOARD = 2


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from hexenv.reward_verl import compute_score

    pool = [json.loads(l) for l in open("data/armD2/harvest_pool.jsonl")]
    rng = random.Random(5)
    shallow = [r for r in pool if r["plen"] <= 21]
    deep = [r for r in pool if r["plen"] >= 22]
    rng.shuffle(deep)
    boards = shallow + deep[:400]
    print(f"harvesting think on {len(boards)} boards "
          f"({len(shallow)} shallow + {min(400, len(deep))} deep)")

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": r["prompt"]}],
        add_generation_prompt=True, enable_thinking=True, tokenize=False)
        for r in boards]
    llm = LLM(model="Qwen/Qwen3-1.7B", max_model_len=2048,
              gpu_memory_utilization=0.6, dtype="bfloat16",
              enable_lora=True, max_lora_rank=64)
    adapter = LoRARequest("ep8", 1, "checkpoints/armD2_sft_weighted/adapter_ep8")

    sp1 = SamplingParams(temperature=1.0, n=K, max_tokens=THINK_BUDGET,
                         stop=["</think>"])
    outs = llm.generate(prompts, sp1, lora_request=adapter)
    cont, meta = [], []
    for r, prompt, o in zip(boards, prompts, outs):
        for s in o.outputs:
            cont.append(prompt + s.text + "</think>\n\n")
            meta.append((r, s.text))
    sp2 = SamplingParams(temperature=1.0, max_tokens=ANSWER_BUDGET)
    outs2 = llm.generate(cont, sp2, lora_request=adapter)

    per_board = {}
    for (r, think), o in zip(meta, outs2):
        answer = o.outputs[0].text
        d = compute_score("x", think + "</think>\n\n" + answer,
                          json.dumps(r["gt"]))
        if d["kind_win"]:
            per_board.setdefault(r["prompt"], (r, []))[1].append(
                {"think": think, "answer": answer})

    kept, by_bin = [], {}
    for prompt, (r, samples) in per_board.items():
        for s in samples[:KEEP_PER_BOARD]:
            kept.append({**{k: r[k] for k in ("plen", "size", "gt", "prompt")},
                         **s})
        b = min(r["plen"] // 4, 8)
        by_bin[r["plen"]] = by_bin.get(r["plen"], 0) + 1
    with open("data/armD2/harvest_think.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    for a, z in [(8, 13), (14, 17), (18, 21), (22, 32)]:
        tot = sum(1 for r in boards if a <= r["plen"] <= z)
        hit = sum(1 for pl, c in by_bin.items() if a <= pl <= z)
        print(f"bin {a}-{z}: {hit}/{tot} boards with a think-perfect sample")
    print(f"kept {len(kept)} (think, answer) rows")


if __name__ == "__main__":
    main()
