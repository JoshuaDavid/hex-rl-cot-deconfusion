"""Arm D witness eval: no-think generation, real-grader scoring, per-size and
per-path-length breakdown, optional wandb logging.

Accepts one or more named datasets (name=path), evaluated with a single
engine load. Each may be the curriculum-style test parquet (prompt =
messages list, reward_model.ground_truth) or the SFT train/val parquet
(prompt_text + ground_truth columns).

Run (in /venv/verl):
  python scripts/eval_armD_witness.py --model Qwen/Qwen3-1.7B \
      --data test=data/armD/test.parquet val=data/armD/val.parquet \
      --out results/armD/base --limit 300 \
      [--lora <adapter_dir>] [--wandb-run armD_sft_weighted --wandb-step 1]
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def load_rows(path, limit):
    df = pd.read_parquet(path)
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=0)
    rows = []
    for _, r in df.iterrows():
        if "prompt_text" in df.columns:
            content, gt = r["prompt_text"], r["ground_truth"]
            plen = int(r["path_len"]) if "path_len" in df.columns else -1
        else:
            content = r["prompt"][0]["content"]
            gt = r["reward_model"]["ground_truth"]
            plen = int(r["extra_info"].get("path_len", -1))
        g = json.loads(gt)
        rows.append({"content": content, "gt": gt, "size": g["size"],
                     "path_len": plen})
    return rows


def summarize(name, results):
    def agg(items):
        return (sum(x["score"] for x in items) / len(items),
                sum(x["perfect"] for x in items) / len(items), len(items))

    mean, perfect, n = agg(results)
    print(f"\n[{name}] OVERALL mean score {mean:.3f}  perfect {perfect:.3f}  n={n}")
    metrics = {"score_mean": mean, "frac_perfect": perfect, "n": n}
    by_size, by_plen = defaultdict(list), defaultdict(list)
    for x in results:
        by_size[x["size"]].append(x)
        if x["path_len"] and x["path_len"] > 0:
            by_plen[x["path_len"]].append(x)
    for s in sorted(by_size):
        m, p, k = agg(by_size[s])
        print(f"  size {s}: mean {m:.3f}  perfect {p:.3f}  n={k}")
        metrics[f"score_size_{s}"] = m
        metrics[f"perfect_size_{s}"] = p
    for pl in sorted(by_plen):
        m, p, k = agg(by_plen[pl])
        print(f"  path_len {pl}: mean {m:.3f}  perfect {p:.3f}  n={k}")
        metrics[f"score_plen_{pl}"] = m
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--data", nargs="+", required=True,
                    help="name=path pairs of parquet datasets")
    ap.add_argument("--out", required=True, help="output prefix (jsonl per set)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--wandb-run", default=None)
    ap.add_argument("--wandb-step", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from hexenv.reward_verl import compute_score

    datasets = []
    for spec in args.data:
        name, path = spec.split("=", 1)
        datasets.append((name, load_rows(path, args.limit)))

    tok = AutoTokenizer.from_pretrained(args.model)
    llm_kwargs = dict(model=args.model, max_model_len=1024,
                      gpu_memory_utilization=0.55, dtype="bfloat16")
    lora_request = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        llm_kwargs.update(enable_lora=True, max_lora_rank=64)
        lora_request = LoRARequest("armd", 1, args.lora)
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    all_metrics = {}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for name, rows in datasets:
        prompts = [
            TokensPrompt(prompt_token_ids=tok.apply_chat_template(
                [{"role": "user", "content": r["content"]}],
                add_generation_prompt=True, enable_thinking=False,
                tokenize=True)["input_ids"])
            for r in rows
        ]
        outs = llm.generate(prompts, sp, lora_request=lora_request)
        results = []
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            d = compute_score("hex_witness_armD", text, r["gt"])
            results.append({"size": r["size"], "path_len": r["path_len"],
                            "score": d["score"], "perfect": d["kind_win"],
                            "unparsed": d["kind_unparsed"], "response": text,
                            "gt": r["gt"]})
        with open(f"{args.out}_{name}.jsonl", "w") as f:
            for x in results:
                f.write(json.dumps(x) + "\n")
        all_metrics[name] = summarize(name, results)

    if args.wandb_run:
        import wandb
        run = wandb.init(project="hex-rl-cot-deconfusion",
                         name=args.wandb_run + "_scores",
                         id="armD_scores_" + args.wandb_run, resume="allow")
        run.log({f"{name}/{k}": v for name, m in all_metrics.items()
                 for k, v in m.items()}, step=args.wandb_step)
        run.finish()


if __name__ == "__main__":
    main()
