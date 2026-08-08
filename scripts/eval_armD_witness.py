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
    ap.add_argument("--lora", nargs="*", default=[],
                    help="step=path pairs of LoRA adapters (evaluated in turn)")
    ap.add_argument("--data", nargs="+", required=True,
                    help="name=path pairs of parquet datasets")
    ap.add_argument("--out", required=True, help="output prefix (jsonl per set)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--think", action="store_true",
                    help="thinking-enabled two-phase eval (forced </think>)")
    ap.add_argument("--think-budget", type=int, default=1024)
    ap.add_argument("--think-prefill", action="store_true",
                    help="prefill '<think>\\n' so all budget tokens are free "
                         "(cap-mode probe); close with '\\n</think>\\n\\n'")
    ap.add_argument("--think-ritual", action="store_true",
                    help="additionally prefill the size-parameterized "
                         "'getting ready to think' ritual (matches "
                         "HEX_THINK_RITUAL=1 training)")
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
    max_len = 2560 if args.think else 1024
    llm_kwargs = dict(model=args.model, max_model_len=max_len,
                      gpu_memory_utilization=0.55, dtype="bfloat16")
    adapters = [(0, None)]
    if args.lora:
        llm_kwargs.update(enable_lora=True, max_lora_rank=64)
        adapters = []
        for spec in args.lora:
            step, path = spec.split("=", 1)
            adapters.append((int(step), path))
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    if args.think:
        def prefill_for(r):
            if not args.think_prefill:
                return ""
            p = "<think>\n"
            if args.think_ritual:
                n = r["size"]
                p += ("Okay, let me try to figure out the winner of this"
                      f" Hex game. The board is {n}x{n}, and the players"
                      " are Black and White. ")
            return p
        prompt_cache = {
            name: [tok.apply_chat_template(
                       [{"role": "user", "content": r["content"]}],
                       add_generation_prompt=True, enable_thinking=True,
                       tokenize=False) + prefill_for(r) for r in rows]
            for name, rows in datasets
        }
    else:
        prompt_cache = {
            name: [TokensPrompt(prompt_token_ids=tok.apply_chat_template(
                       [{"role": "user", "content": r["content"]}],
                       add_generation_prompt=True, enable_thinking=False,
                       tokenize=True)["input_ids"]) for r in rows]
            for name, rows in datasets
        }

    run = None
    if args.wandb_run:
        import wandb
        run = wandb.init(project="hex-rl-cot-deconfusion",
                         name=args.wandb_run + "_scores",
                         id="armD_scores_" + args.wandb_run, resume="allow")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for step, path in adapters:
        lora_request = None
        if path:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest(f"armd{step}", step + 1, path)
        step_metrics = {}
        for name, rows in datasets:
            if args.think:
                sp1 = SamplingParams(temperature=args.temperature,
                                     max_tokens=args.think_budget,
                                     stop=["</think>"])
                outs1 = llm.generate(prompt_cache[name], sp1,
                                     lora_request=lora_request)
                close = "\n</think>\n\n" if args.think_prefill else "</think>\n\n"
                cont = [p + o.outputs[0].text + close
                        for p, o in zip(prompt_cache[name], outs1)]
                outs = llm.generate(cont, sp, lora_request=lora_request)
                texts = [o1.outputs[0].text + "</think>\n\n" + o.outputs[0].text
                         for o1, o in zip(outs1, outs)]
                think_toks = [len(o1.outputs[0].token_ids) for o1 in outs1]
            else:
                outs = llm.generate(prompt_cache[name], sp,
                                    lora_request=lora_request)
                texts = [o.outputs[0].text for o in outs]
                think_toks = [0] * len(outs)
            results = []
            for r, text, tt in zip(rows, texts, think_toks):
                d = compute_score("hex_witness_armD", text, r["gt"])
                results.append({"size": r["size"], "path_len": r["path_len"],
                                "score": d["score"], "perfect": d["kind_win"],
                                "unparsed": d["kind_unparsed"],
                                "think_tokens": tt,
                                "response": text, "gt": r["gt"]})
            with open(f"{args.out}_ep{step}_{name}.jsonl", "w") as f:
                for x in results:
                    f.write(json.dumps(x) + "\n")
            step_metrics[name] = summarize(f"ep{step} {name}", results)
        if run:
            run.log({f"{name}/{k}": v for name, m in step_metrics.items()
                     for k, v in m.items()}, step=step)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
