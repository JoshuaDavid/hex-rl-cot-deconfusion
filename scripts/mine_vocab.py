"""C2 vocab mining: compare n-gram distributions between two CoT corpora.

Usage: python scripts/mine_vocab.py A.jsonl B.jsonl [--n 1 2 3] [--top 40]
Input files: jsonl with either 'responses' (list) or 'response' (str) fields.
Reports n-grams enriched in B vs A (log-odds with +0.5 smoothing, min count).
Null calibration: run with A split in half — enrichment scores should be small.
"""

import argparse
import json
import math
import re
from collections import Counter


def extract_thinks(path):
    texts = []
    for line in open(path):
        r = json.loads(line)
        resp = r.get("responses") or [r.get("response", "")]
        for t in resp:
            think = t.split("</think>")[0].replace("<think>", "")
            texts.append(think)
    return texts


def tokenize(text):
    text = text.lower()
    # strip cell coordinates and numbers to focus on vocabulary, keep word shapes
    text = re.sub(r"\b[a-i]\d{1,2}\b", "<cell>", text)
    text = re.sub(r"\d+", "<num>", text)
    return re.findall(r"[a-z<>]+", text)


def ngram_counts(texts, ns):
    counts = {n: Counter() for n in ns}
    total = {n: 0 for n in ns}
    for t in texts:
        toks = tokenize(t)
        for n in ns:
            for i in range(len(toks) - n + 1):
                counts[n][" ".join(toks[i:i + n])] += 1
                total[n] += 1
    return counts, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min-count", type=int, default=8)
    ap.add_argument("--split-null", action="store_true",
                    help="ignore b; split a in half as null calibration")
    args = ap.parse_args()

    ta = extract_thinks(args.a)
    if args.split_null:
        ta, tb = ta[::2], ta[1::2]
        print(f"NULL CALIBRATION: {len(ta)} vs {len(tb)} halves of {args.a}")
    else:
        tb = extract_thinks(args.b)
        print(f"A={args.a} ({len(ta)} CoTs)  B={args.b} ({len(tb)} CoTs)")

    ca, tota = ngram_counts(ta, args.n)
    cb, totb = ngram_counts(tb, args.n)

    for n in args.n:
        scored = []
        for g, cnt_b in cb[n].items():
            if cnt_b < args.min_count:
                continue
            cnt_a = ca[n].get(g, 0)
            rate_b = (cnt_b + 0.5) / (totb[n] + 1)
            rate_a = (cnt_a + 0.5) / (tota[n] + 1)
            scored.append((math.log2(rate_b / rate_a), cnt_b, cnt_a, g))
        scored.sort(reverse=True)
        print(f"\n== {n}-grams enriched in B (log2 ratio, countB, countA) ==")
        for s, cb_, ca_, g in scored[: args.top]:
            print(f"  {s:+.2f}  {cb_:>6} {ca_:>6}  {g}")


if __name__ == "__main__":
    main()
