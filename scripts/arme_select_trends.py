"""P(select=X) and mean C-score by step for arm E R4, from the hex_select side
channel (results/rollouts/<exp>.jsonl: one record/rollout with 'selected' + C
'score'/'perfect'). Buckets rollouts in order by --per-step (BATCH*GROUP_N).

Run: /venv/main/bin/python scripts/arme_select_trends.py results/rollouts/arme_r4_cost.jsonl [--per-step 512]
"""
import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--per-step", type=int, default=512)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.log)]
    ps = args.per_step
    nsteps = (len(rows) + ps - 1) // ps
    print(f"{len(rows)} rollouts, {nsteps} steps @ {ps}/step\n")
    print(f"{'step':>4}  {'A':>5} {'B':>5} {'D':>5} {'E':>5}  {'Cmean':>6} "
          f"{'Cperf':>6}  meanC_by_sel")
    for s in range(nsteps):
        chunk = rows[s * ps:(s + 1) * ps]
        if not chunk:
            continue
        sel = Counter(r["selected"] for r in chunk)
        n = len(chunk)
        frac = {x: sel.get(x, 0) / n for x in ["A", "B", "D", "E"]}
        cmean = sum(r["score"] for r in chunk) / n
        cperf = sum(r.get("perfect", 0) for r in chunk) / n
        by = {}
        for x in ["A", "B", "D", "E"]:
            xs = [r["score"] for r in chunk if r["selected"] == x]
            by[x] = sum(xs) / len(xs) if xs else float("nan")
        bystr = " ".join(f"{x}:{by[x]:+.2f}" for x in ["A", "B", "D", "E"])
        print(f"{s:>4}  {frac['A']:.3f} {frac['B']:.3f} {frac['D']:.3f} "
              f"{frac['E']:.3f}  {cmean:+.3f} {cperf:.3f}  {bystr}")


if __name__ == "__main__":
    main()
