"""Eval the self-judge: can the model predict whether a witness answer is
correct, and for which error types?

Runs the judge (base zero-shot and/or a LoRA adapter) on:
  test_natural : bok's own answers (real self-generated error mix)
  test_probe   : constructed gold(Yes)/broken-link(No)/wrong-winner(No)
Parses "Answer: Yes|No", reports balanced accuracy + per-error-type detection.

Run: /venv/verl/bin/python scripts/eval_selfjudge.py [--lora <adapter>]
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

YESNO = re.compile(r"[Aa]nswer:\s*\*{0,2}(Yes|No)\b", re.IGNORECASE)


def parse(text):
    m = YESNO.findall(text)
    return m[-1].capitalize() if m else None


def report(name, rows, preds):
    # rows: list of dict(label bool, err_type); preds: "Yes"/"No"/None
    tp = tn = fp = fn = unparsed = 0
    by_err = defaultdict(lambda: [0, 0])  # err_type -> [n, detected_or_correct]
    for r, p in zip(rows, preds):
        yes = (p == "Yes")
        if p is None:
            unparsed += 1
        lab = r["label"]
        if lab and yes: tp += 1
        elif lab and not yes: fn += 1
        elif not lab and yes: fp += 1
        elif not lab and not yes: tn += 1
        e = r["err_type"]
        by_err[e][0] += 1
        # "success" = correct judgement: Yes for correct, No for an error
        by_err[e][1] += (yes if lab else (p == "No"))
    pos = tp + fn or 1
    neg = tn + fp or 1
    recall_correct = tp / pos           # P(say Yes | correct)
    recall_error = tn / neg             # P(say No  | incorrect)
    bal = 0.5 * (recall_correct + recall_error)
    print(f"\n[{name}] n={len(rows)} unparsed={unparsed}")
    print(f"  balanced_acc={bal:.3f}  recall_correct(Yes|ok)={recall_correct:.3f}"
          f"  recall_error(No|bad)={recall_error:.3f}")
    for e in ("correct", "link", "winner", "unparsed"):
        if e in by_err:
            n, s = by_err[e]
            print(f"    {e:9} n={n:4}  correct-judgement={s/n:.3f}")
    return {"balanced_acc": bal, "recall_correct": recall_correct,
            "recall_error": recall_error}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=None)
    ap.add_argument("--out", default="results/selfjudge_eval")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    sets = {n: pd.read_parquet(f"data/selfjudge/{n}.parquet").to_dict("records")
            for n in ("test_natural", "test_probe")}

    llm_kwargs = dict(model="Qwen/Qwen3-1.7B", max_model_len=1024,
                      gpu_memory_utilization=0.55, dtype="bfloat16")
    lreq = None
    if args.lora:
        from vllm.lora.request import LoRARequest
        llm_kwargs.update(enable_lora=True, max_lora_rank=64)
        lreq = LoRARequest("sj", 1, args.lora)
    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=8)

    for name, rows in sets.items():
        prompts = [TokensPrompt(prompt_token_ids=tok.apply_chat_template(
            [{"role": "user", "content": r["prompt_text"]}],
            add_generation_prompt=True, enable_thinking=False,
            tokenize=True)["input_ids"]) for r in rows]
        outs = llm.generate(prompts, sp, lora_request=lreq)
        preds = [parse(o.outputs[0].text) for o in outs]
        report(name, rows, preds)
        with open(f"{args.out}_{name}.jsonl", "w") as f:
            for r, p in zip(rows, preds):
                f.write(json.dumps({"label": r["label"], "err_type": r["err_type"],
                                    "pred": p}) + "\n")


if __name__ == "__main__":
    main()
