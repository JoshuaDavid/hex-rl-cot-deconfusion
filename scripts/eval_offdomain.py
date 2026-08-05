"""C4 off-domain eval: fixed prompt set (GSM8K + MMLU + IFEval prompts),
generation + mechanical grading where possible; stores everything for
KL/style analysis vs base model later.

Usage: python scripts/eval_offdomain.py --model <dir> --out results/offdomain/<tag>.jsonl
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_items(seed=0):
    import random
    from datasets import load_dataset

    rng = random.Random(seed)
    items = []

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    idx = rng.sample(range(len(gsm)), 100)
    for i in idx:
        r = gsm[i]
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        items.append({"kind": "gsm8k", "prompt": r["question"] +
                      "\n\nPut your final numeric answer after 'Answer:'.",
                      "gold": gold})

    mmlu = load_dataset("cais/mmlu", "all", split="test")
    idx = rng.sample(range(len(mmlu)), 100)
    for i in idx:
        r = mmlu[i]
        letters = "ABCD"
        chs = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(r["choices"]))
        items.append({"kind": "mmlu",
                      "prompt": f"{r['question']}\n{chs}\n\nAnswer with a single letter after 'Answer:'.",
                      "gold": letters[r["answer"]]})

    ife = load_dataset("google/IFEval", split="train")
    idx = rng.sample(range(len(ife)), 50)
    for i in idx:
        items.append({"kind": "ifeval", "prompt": ife[i]["prompt"], "gold": None})

    return items


def grade(kind, gold, text):
    post = text.split("</think>")[-1]
    if kind == "gsm8k":
        m = re.findall(r"[Aa]nswer:\s*\$?(-?[\d,]+(?:\.\d+)?)", post)
        if not m:
            m = re.findall(r"(-?[\d,]+(?:\.\d+)?)", post)
        if not m:
            return False
        try:
            return abs(float(m[-1].replace(",", "")) - float(gold)) < 1e-4
        except ValueError:
            return False
    if kind == "mmlu":
        m = re.findall(r"[Aa]nswer:\s*\*{0,2}([A-D])\b", post)
        if not m:
            m = re.findall(r"\b([A-D])\b", post)
        return bool(m) and m[-1] == gold
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--no-think", action="store_true")
    args = ap.parse_args()

    items = build_items()
    from hexenv.genbackend import Backend

    backend = Backend(args.model, enable_thinking=not args.no_think)
    outs = backend.generate([it["prompt"] for it in items], n=1,
                            temperature=args.temperature, max_tokens=args.max_tokens)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    agg = {}
    with open(args.out, "w") as f:
        for it, texts in zip(items, outs):
            text = texts[0]
            g = grade(it["kind"], it["gold"], text)
            think = text.split("</think>")[0] if "</think>" in text else text
            rec = dict(it)
            rec.update({"response": text, "correct": g,
                        "think_chars": len(think), "resp_chars": len(text)})
            f.write(json.dumps(rec) + "\n")
            a = agg.setdefault(it["kind"], [0, 0, 0])
            a[2] += 1
            if g is not None:
                a[0] += bool(g)
                a[1] += 1

    summary = {"model": args.model}
    for k, (c, t, n) in agg.items():
        summary[k] = {"n": n, "acc": c / t if t else None}
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
