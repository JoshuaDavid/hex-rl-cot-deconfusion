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
    by_prompt = defaultdict(list)
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
        s["shaped_sum"] = s.get("shaped_sum", 0.0) + float(r.get("shaped", r["score"]))
        pk = (cat, str(r["gt"].get("moves"))[:200], r["gt"].get("to_move", ""))
        by_prompt[pk].append(float(r.get("shaped", r["score"])))
    # empirical per-prompt reward std (captures length-shaping variance that
    # the binary sqrt(p(1-p)) formula misses, esp. near saturation)
    import statistics
    sig_acc = defaultdict(list)
    for (cat, _, _), scores in by_prompt.items():
        if len(scores) >= 3:
            sig_acc[cat].append(statistics.pstdev(scores))
    for cat, s in stats.items():
        if s["n"] < 20:
            continue
        p = s["win"] / s["n"]
        k = s["chars"] / s["n"] / 3.0  # rough tokens
        sig = (sum(sig_acc[cat]) / len(sig_acc[cat])) if sig_acc.get(cat) else None
        prev = ema_state.get(cat)
        if prev:
            p = EMA * p + (1 - EMA) * prev["p"]
            k = EMA * k + (1 - EMA) * prev["k"]
            if sig is not None and prev.get("sig") is not None:
                sig = EMA * sig + (1 - EMA) * prev["sig"]
            elif sig is None:
                sig = prev.get("sig")
        mean_shaped = s["shaped_sum"] / s["n"]
        if prev and prev.get("reward") is not None:
            mean_shaped = EMA * mean_shaped + (1 - EMA) * prev["reward"]
        ema_state[cat] = {"p": p, "k": max(k, 64.0), "sig": sig,
                          "reward": mean_shaped}
    return ema_state


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "armC"
    side = f"results/rollouts/{exp}.jsonl"
    ema_state = {}
    wb = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wb = wandb.init(project="hex-rl-cot-deconfusion",
                            name=f"{exp}-controller", id=f"{exp}-controller",
                            resume="allow")
        except Exception as e:
            print(f"wandb init failed ({e}); file logging only", flush=True)
    while True:
        cats = [f[:-8] for f in os.listdir(DIR) if f.endswith(".parquet")]
        try:
            cfg = yaml.safe_load(open(CONFIG)) or {}
        except OSError:
            cfg = {}
        ccfg = cfg.get("categories", {})
        ema_state = read_side_channel(side, ema_state)

        shares = {}
        floors = {}
        for cat in cats:
            c = ccfg.get(cat, {})
            if not c.get("enabled", True):
                shares[cat] = 0.0
                floors[cat] = 0.0
                continue
            w = float(c.get("importance", 1.0))
            floor = float(c.get("floor", 0.05))
            st = ema_state.get(cat)
            p = st["p"] if st else 0.5  # optimistic prior for unseen cats
            k = st["k"] if st else 1100.0
            # empirical sigma preferred (sees length-shaping variance at
            # saturation); analytic binary formula as fallback/prior
            sigma = None
            if st and st.get("sig") is not None:
                sigma = max(st["sig"] / 2.0, 1e-2)  # /2: reward span is ~2
            if sigma is None:
                sigma = math.sqrt(max(p * (1 - p), 1e-4))
            shares[cat] = w * sigma / math.sqrt(k)
            floors[cat] = floor
        # normalize the Neyman shares FIRST, then apply floors as minimum
        # fractions and renormalize (floors and Neyman terms are not on
        # comparable scales; the old max() was 100% floor-dominated)
        tot = sum(shares.values()) or 1.0
        norm = {c: v / tot for c, v in shares.items()}
        floored = {c: (0.0 if norm[c] == 0 and floors.get(c, 0) == 0
                       else max(norm[c], floors.get(c, 0.0)))
                   for c in norm}
        # disabled categories must stay at exactly 0
        for c in norm:
            if shares[c] == 0.0 and floors.get(c, 1) == 0.0:
                floored[c] = 0.0
        tot2 = sum(floored.values()) or 1.0
        weights = {c: round(v / tot2, 5) for c, v in floored.items()}
        with open(WEIGHTS + ".tmp", "w") as f:
            json.dump(weights, f, indent=1)
        os.replace(WEIGHTS + ".tmp", WEIGHTS)
        audit = {"ts": time.time(), "weights": weights,
                 "signals": {c: {k2: (round(v2, 4) if v2 is not None else None)
                                 for k2, v2 in st.items()}
                             for c, st in ema_state.items()}}
        os.makedirs("results", exist_ok=True)
        with open(AUDIT, "a") as f:
            f.write(json.dumps(audit) + "\n")
        if wb is not None:
            import subprocess as sp
            try:
                out = sp.run(["grep", "-ao", r"step:[0-9]* - global",
                              os.environ.get("HEX_TRAIN_LOG", f"results/{exp}.log")],
                             capture_output=True, text=True).stdout
                train_step = int(out.splitlines()[-1].split(":")[1].split()[0])
            except Exception:
                train_step = -1
            flat = {f"mix/weight_{c}": v for c, v in weights.items()}
            flat["mix/train_step"] = train_step
            for c, st in ema_state.items():
                for k2, v2 in st.items():
                    if v2 is not None:
                        flat[f"mix/{k2}_{c}"] = v2
            wb.log(flat)
        print(json.dumps(audit), flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
