"""Replay-mix SFT dataset: gold certificates + correct self-samples.

Task-pure certificate SFT destroyed every other skill (RESEARCH_LOG
2026-08-06 three-way verdict). This builds the standard fix: mix the 2850
gold certificate pairs with rejection-sampled CORRECT self-samples from the
same checkpoint (armC/250) on all other categories, so SFT preserves the
policy's own behavior where it is already competent.

Output: data/sft_replay/{train,val}.parquet (messages format, CertSFTDataset).
"""

import json
import os
import random
import sys

sys.path.insert(0, "/workspace/hex-rl-cot-deconfusion")
os.environ["HEX_LEN_LAMBDA"] = "0"

import pandas as pd

CKPT = "checkpoints/armC/global_step_250/hf"
CATS = ["judge", "chain", "occupancy", "winset", "chainset", "edge_m1",
        "gen_m1", "mate2", "mate1_v2", "general"]
PROMPTS_PER_CAT = 120
N_PER_PROMPT = 3
KEEP_PER_CAT = 300
MIN_SCORE = 0.5
THINK_BUDGET = 1024
MAX_TOTAL_TOK = 1900


def task_scaffold(user_text):
    if "Answer: Black|White|Neither" in user_text:
        return "Answer:", 8
    if ("Answer: <comma-separated list of cells>" in user_text
            or 'a JSON array' in user_text):
        return "Answer:", 48
    if '"winner"' in user_text:
        return "Answer:", 64
    return "Move:", 8


def main():
    from vllm import LLM, SamplingParams  # noqa: F401
    from transformers import AutoTokenizer
    from hexenv.forced_close_gen import generate_forced_close
    from hexenv import reward_verl

    rng = random.Random(7)
    tok = AutoTokenizer.from_pretrained(CKPT)
    from vllm import LLM
    llm = LLM(model=CKPT, max_model_len=4096, gpu_memory_utilization=0.75,
              disable_log_stats=True)

    rows = []
    for cat in CATS:
        df = pd.read_parquet(f"data/curriculum/{cat}.parquet")
        df = df.sample(min(PROMPTS_PER_CAT, len(df)), random_state=7)
        user_msgs = [r["prompt"][0]["content"] for _, r in df.iterrows()]
        gts = [r["reward_model"]["ground_truth"] for _, r in df.iterrows()]
        prefix, budget = task_scaffold(user_msgs[0])
        outs = generate_forced_close(llm, tok, user_msgs, n=N_PER_PROMPT,
                                     temperature=0.6, think_budget=THINK_BUDGET,
                                     answer_budget=budget, answer_prefix=prefix)
        kept = 0
        for msg, gt, cands in zip(user_msgs, gts, outs):
            for c in cands:
                s = reward_verl.compute_score("hex", c["text"], gt)
                score = s["score"] if isinstance(s, dict) else s
                if score < MIN_SCORE:
                    continue
                think, post = c["text"].split("</think>", 1)
                content = f"<think>\n{think.strip()}\n</think>\n\n{post.strip()}"
                content = content.replace("<|im_end|>", "").rstrip()
                ntok = len(tok(msg + content)["input_ids"])
                if ntok > MAX_TOTAL_TOK:
                    continue
                rows.append({"cat": cat, "messages": [
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": content}]})
                kept += 1
                break  # at most one sample per prompt for diversity
            if kept >= KEEP_PER_CAT:
                break
        print(f"{cat}: kept {kept}", flush=True)

    with open("data/sft_certificates.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows.append({"cat": "witness", "messages": [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r["completion"]}]})

    rng.shuffle(rows)
    from collections import Counter
    print("mix:", dict(Counter(r["cat"] for r in rows)))
    recs = [{"messages": r["messages"]} for r in rows]
    n_val = max(64, len(recs) // 20)
    os.makedirs("data/sft_replay", exist_ok=True)
    pd.DataFrame(recs[n_val:]).to_parquet("data/sft_replay/train.parquet")
    pd.DataFrame(recs[:n_val]).to_parquet("data/sft_replay/val.parquet")
    print(f"train={len(recs)-n_val} val={n_val}")


if __name__ == "__main__":
    main()
