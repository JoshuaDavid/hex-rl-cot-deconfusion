"""Arm-C curriculum controller: Neyman-with-floors allocation.

Every INTERVAL seconds:
- read data/curriculum/config.yaml (human-editable: importance w, floors,
  enabled flags; missing category => enabled with default importance)
- read tail of the rollout side channel; EMA per-category success p, token cost k
- shares ∝ max(w * sqrt(p(1-p)) / sqrt(k), floor * w_indicator); optimistic
  sigma prior (p=0.5) for categories with no samples yet
- write weights.json; append audit row to results/curriculum_log.jsonl

Add category: drop <cat>.parquet in data/curriculum/ (+ optional config entry).
Remove buggy category: set enabled: false in config.yaml (weight -> 0 on next
tick; dataset stops sampling it on next weights.json mtime change).
"""

import json
import math
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIR = "data/curriculum"
CONFIG = os.path.join(DIR, "config.yaml")
WEIGHTS = os.path.join(DIR, "weights.json")
AUDIT = "results/curriculum_log.jsonl"
INTERVAL = int(os.environ.get("HEX_CTRL_INTERVAL", "600"))
TAIL = 6000
EMA = 0.5


def read_side_channel(path, ema_state):
    from collections import defaultdict
    stats = defaultdict(lambda: {"n": 0, "win": 0, "chars": 0})
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 40_000_000))
            lines = f.read().decode(errors="ignore").splitlines()[-TAIL:]
    except OSError:
        lines = []
    for l in lines:
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        cat = r.get("gt", {}).get("category")
        if not cat:
            continue
        s = stats[cat]
        s["n"] += 1
        s["win"] += r["score"] > 0
        s["chars"] += len(r.get("response", ""))
    for cat, s in stats.items():
        if s["n"] < 20:
            continue
        p = s["win"] / s["n"]
        k = s["chars"] / s["n"] / 3.0  # rough tokens
        prev = ema_state.get(cat)
        if prev:
            p = EMA * p + (1 - EMA) * prev["p"]
            k = EMA * k + (1 - EMA) * prev["k"]
        ema_state[cat] = {"p": p, "k": max(k, 64.0)}
    return ema_state


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "armC"
    side = f"results/rollouts/{exp}.jsonl"
    ema_state = {}
    while True:
        cats = [f[:-8] for f in os.listdir(DIR) if f.endswith(".parquet")]
        try:
            cfg = yaml.safe_load(open(CONFIG)) or {}
        except OSError:
            cfg = {}
        ccfg = cfg.get("categories", {})
        ema_state = read_side_channel(side, ema_state)

        shares = {}
        for cat in cats:
            c = ccfg.get(cat, {})
            if not c.get("enabled", True):
                shares[cat] = 0.0
                continue
            w = float(c.get("importance", 1.0))
            floor = float(c.get("floor", 0.05))
            st = ema_state.get(cat)
            p = st["p"] if st else 0.5  # optimistic prior for unseen cats
            k = st["k"] if st else 1100.0
            sigma = math.sqrt(max(p * (1 - p), 1e-4))
            shares[cat] = max(w * sigma / math.sqrt(k), floor * w)
        tot = sum(shares.values()) or 1.0
        weights = {c: round(v / tot, 5) for c, v in shares.items()}
        with open(WEIGHTS + ".tmp", "w") as f:
            json.dump(weights, f, indent=1)
        os.replace(WEIGHTS + ".tmp", WEIGHTS)
        audit = {"ts": time.time(), "weights": weights,
                 "signals": {c: {k2: round(v2, 4) for k2, v2 in st.items()}
                             for c, st in ema_state.items()}}
        os.makedirs("results", exist_ok=True)
        with open(AUDIT, "a") as f:
            f.write(json.dumps(audit) + "\n")
        print(json.dumps(audit), flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
