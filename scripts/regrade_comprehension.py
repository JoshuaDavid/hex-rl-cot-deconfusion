"""Robust offline regrade of phase1 comprehension results + mismatch dump.

Grading rules:
- lookup/adjacent/connected: first standalone answer token in the post-think text
  (yes/no/black/white/empty), searching the final line first, then whole text.
- neighbors: cells extracted from the last line that contains cells, after
  stripping 'neighbors of <cell>' phrasing.
"""

import json
import re
import sys

path = sys.argv[1]
show_wrong = "--show-wrong" in sys.argv

recs = [json.loads(l) for l in open(path)]
from collections import defaultdict
agg = defaultdict(lambda: [0, 0])
wrong = []

for r in recs:
    text = r["response"].split("</think>")[-1].strip().lower()
    kind = r["kind"]
    ok = None
    if kind in ("lookup", "adjacent", "connected"):
        vocab = ("yes", "no") if kind != "lookup" else ("black", "white", "empty")
        found = None
        for line in [text.splitlines()[-1] if text else ""] + [text]:
            toks = re.findall(r"[a-z]+", line)
            for t in toks:
                if t in vocab:
                    found = t
                    break
            if found:
                break
        ok = found == r["answer"]
        r["regrade_parsed"] = found
    elif kind == "neighbors":
        lines = [l for l in text.splitlines() if re.search(r"\b[a-i]\d{1,2}\b", l)]
        cells = set()
        if lines:
            last = lines[-1]
            last = re.sub(r"neighbou?rs? of \*{0,2}[a-i]\d{1,2}\*{0,2}", "", last)
            cells = set(re.findall(r"\b([a-i]\d{1,2})\b", last))
        ok = cells == set(r["answer"])
        r["regrade_parsed"] = sorted(cells)
    if ok is not None:
        agg[kind][0] += ok
        agg[kind][1] += 1
        if not ok:
            wrong.append(r)

for k in sorted(agg):
    c, t = agg[k]
    print(f"{k}: {c}/{t} = {c/t:.2f}")
tot_c = sum(v[0] for v in agg.values()); tot_t = sum(v[1] for v in agg.values())
print(f"TOTAL: {tot_c}/{tot_t} = {tot_c/tot_t:.2f}")

if show_wrong:
    for r in wrong[:12]:
        print("=" * 70)
        print("KIND:", r["kind"], "| EXPECTED:", r["answer"], "| PARSED:", r.get("regrade_parsed"))
        print("Q:", r["prompt"][-300:])
        print("A (post-think):", r["response"].split("</think>")[-1].strip()[:500])
