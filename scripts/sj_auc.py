"""Definitive discrimination check for the self-judges: instead of argmax
Yes/No (which can collapse to a constant), measure P(Yes) via the 1-token
logprob after a forced 'Answer:' prefix, and compute AUC of correct-vs-
incorrect on the natural test. AUC~0.5 = no signal; >0.5 = latent signal.

Run: /venv/verl/bin/python scripts/sj_auc.py
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def auc(labels, scores):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = 0.0
    # average-rank AUC
    ranked = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and scores[ranked[j]] == scores[ranked[i]]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ranked[k]] = avg
        i = j
    rank_sum = sum(ranks[i] for i in range(len(labels)) if labels[i])
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    yes_ids = {tok(t, add_special_tokens=False)["input_ids"][0]
               for t in (" Yes", "Yes")}
    no_ids = {tok(t, add_special_tokens=False)["input_ids"][0]
              for t in (" No", "No")}

    rows = pd.read_parquet("data/selfjudge/test_natural.parquet").to_dict("records")
    prompts = []
    for r in rows:
        s = tok.apply_chat_template([{"role": "user", "content": r["prompt_text"]}],
                                    add_generation_prompt=True, enable_thinking=False,
                                    tokenize=False) + "Answer:"
        prompts.append(s)

    llm = LLM(model="Qwen/Qwen3-1.7B", max_model_len=1024,
              gpu_memory_utilization=0.55, dtype="bfloat16",
              enable_lora=True, max_lora_rank=64)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

    for tag, path in [("base", None),
                      ("constructed", "checkpoints/selfjudge/adapter"),
                      ("on-policy", "checkpoints/selfjudge_op/adapter")]:
        lreq = LoRARequest(tag, 1, path) if path else None
        outs = llm.generate(prompts, sp, lora_request=lreq)
        labels, pyes = [], []
        for r, o in zip(rows, outs):
            lp = o.outputs[0].logprobs[0]
            ly = max((lp[t].logprob for t in yes_ids if t in lp), default=-30.0)
            ln = max((lp[t].logprob for t in no_ids if t in lp), default=-30.0)
            p = 1.0 / (1.0 + math.exp(ln - ly))
            labels.append(1 if r["label"] else 0)
            pyes.append(p)
        a = auc(labels, pyes)
        # AUC that "No means error": discrimination is symmetric, report as-is
        print(f"[{tag}] natural AUC(correct vs incorrect) = {a:.3f}  "
              f"(n={len(labels)}, base-rate correct={sum(labels)/len(labels):.3f})")


if __name__ == "__main__":
    main()
