"""Rolling trends from the training rollout side-channel log.

Usage: /venv/main/bin/python scripts/rollout_trends.py results/rollouts/pilot_1p7b.jsonl [--sample N]
"""

import argparse
import json
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--window", type=int, default=256, help="samples per bucket")
    ap.add_argument("--sample", type=int, default=0, help="print N recent CoT tails")
    args = ap.parse_args()

    recs = []
    for line in open(args.path):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    W = args.window
    print(f"{len(recs)} scored samples  ({len(recs)//W} buckets of {W})")
    print(f"{'bucket':>6} {'win':>6} {'lose':>6} {'illegal':>8} {'unparsed':>9} {'think_chars_p50':>16}")
    for b in range(0, len(recs), W):
        chunk = recs[b:b + W]
        if len(chunk) < W // 2:
            break
        kinds = [r["kind"] for r in chunk]
        lens = sorted(len(r["response"]) for r in chunk)
        n = len(chunk)
        print(f"{b//W:>6} {kinds.count('win')/n:>6.2f} {kinds.count('lose')/n:>6.2f} "
              f"{kinds.count('illegal')/n:>8.2f} {kinds.count('unparsed')/n:>9.2f} "
              f"{lens[n//2]:>16}")

    if args.sample:
        print("\n=== recent CoT tails ===")
        for r in recs[-args.sample:]:
            tail = r["response"][-400:]
            print(f"--- kind={r['kind']} move={r['move']} size={r['gt']['size']}")
            print(tail.replace("\n", " ")[-400:])


if __name__ == "__main__":
    main()
