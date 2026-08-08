"""SFT-scheduler experiment: does the arm-C Neyman-with-floors allocator help
when driving SFT instead of RL?

Chunked adaptive loop over the 11 multitask categories, from BASE Qwen3-1.7B:
  round r: sample M examples ~ current weights -> train 1 epoch (continue
  LoRA) -> eval per-category frac_perfect (temp0) -> update weights.

Faithful mechanical port of scripts/curriculum_controller.py allocation:
  share_c = w_c * sqrt(p_c(1-p_c)) / sqrt(k_c); normalize; floor as min
  fraction; renormalize. p_c = eval frac_perfect (EMA 0.5, optimistic 0.5
  prior); k_c = mean teacher-forcing target length (>=64). w_c = 1 (no
  human importance skew -> pure adaptive-vs-uniform).

ARM=uniform keeps weights fixed at 1/11. ARM=port adapts.
Matched total example budget (R*M) across arms.

Run: ARM=uniform /venv/verl/bin/python scripts/scheduler_sft.py
     ARM=port    /venv/verl/bin/python scripts/scheduler_sft.py
"""

import json
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

CATS = ["chain", "judge", "occupancy", "winset", "chainset", "witness",
        "mate1_v2", "mate2", "edge_m1", "gen_m1", "general"]
ARM = os.environ.get("ARM", "uniform")
ROUNDS = int(os.environ.get("ROUNDS", "8"))
M = int(os.environ.get("CHUNK", "1600"))
FLOOR = float(os.environ.get("FLOOR", "0.03"))
EMA = 0.5
CK = f"checkpoints/sched_{ARM}"
OUT = f"results/multitask/sched_{ARM}"
MD = "data/multitask"


def neyman(p_ema, kcost):
    shares = {}
    for c in CATS:
        p = p_ema.get(c, 0.5)  # optimistic prior for unseen
        sigma = math.sqrt(max(p * (1 - p), 1e-4))
        shares[c] = 1.0 * sigma / math.sqrt(max(kcost[c], 64.0))
    tot = sum(shares.values()) or 1.0
    norm = {c: v / tot for c, v in shares.items()}
    floored = {c: max(norm[c], FLOOR) for c in CATS}
    tot2 = sum(floored.values())
    return {c: v / tot2 for c, v in floored.items()}


def sample_chunk(weights, seed):
    rng = __import__("random").Random(seed)
    parts = []
    for c in CATS:
        pool = pd.read_parquet(f"{MD}/{c}_train.parquet")[["input_ids", "loss_mask"]]
        n = max(1, round(M * weights[c]))
        idx = [rng.randrange(len(pool)) for _ in range(n)]
        parts.append(pool.iloc[idx])
    chunk = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    os.makedirs(f"{MD}/_chunk_{ARM}", exist_ok=True)
    path = f"{MD}/_chunk_{ARM}/train.parquet"
    chunk.reset_index(drop=True).to_parquet(path)
    return path, len(chunk)


def train_round(r, chunk_path, prev_adapter):
    exp = f"sched_{ARM}/r{r}"
    extra = ["data.train_files=" + chunk_path,
             "data.val_files=" + f"{MD}/witness_test.parquet",
             "trainer.total_epochs=1"]
    if prev_adapter:
        extra.append("model.lora_adapter_path=" + prev_adapter)
    env = dict(os.environ, ARMD_EXP=exp)
    log = f"{OUT}_train_r{r}.log"
    with open(log, "w") as f:
        subprocess.run(["bash", "scripts/run_armD_sft.sh"] + extra,
                       stdout=f, stderr=subprocess.STDOUT, env=env, check=True)
    steps = max(1, round(M / 64))
    last = sorted(__import__("glob").glob(f"{CK}/r{r}/global_step_*"),
                  key=lambda p: int(p.split("_")[-1]))[-1]
    adapter = f"{CK}/r{r}/adapter"
    subprocess.run(["/venv/verl/bin/python", "scripts/export_armD_adapter.py",
                    last, adapter], check=True,
                   stdout=open(log, "a"), stderr=subprocess.STDOUT)
    for d in __import__("glob").glob(f"{CK}/r{r}/global_step_*"):
        __import__("shutil").rmtree(d)
    return adapter


def eval_round(r, adapter):
    data_args = [f"{c}={MD}/{c}_test.parquet" for c in CATS]
    log = f"{OUT}_eval_r{r}.log"
    for _try in range(3):
        # wait for a free GPU (co-tenant guard)
        while True:
            used = int(subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout.split("\n")[0] or 0)
            if used < 1500:
                break
            time.sleep(5)
        subprocess.run(
            ["timeout", "-s", "KILL", "2400", "/venv/verl/bin/python",
             "scripts/eval_armD_witness.py", "--model", "Qwen/Qwen3-1.7B",
             "--lora", f"{r}={adapter}", "--data", *data_args,
             "--max-tokens", "512", "--out", OUT,
             "--wandb-run", f"sched_{ARM}", "--wandb-step", str(r)],
            stdout=open(log, "w"), stderr=subprocess.STDOUT)
        subprocess.run(["pkill", "-9", "-x", "VLLM::EngineCore"])
        if all(os.path.exists(f"{OUT}_ep{r}_{c}.jsonl") for c in CATS):
            break
        time.sleep(15)
    p = {}
    for c in CATS:
        rows = [json.loads(l) for l in open(f"{OUT}_ep{r}_{c}.jsonl")]
        p[c] = sum(x["perfect"] for x in rows) / len(rows)
    return p


def main():
    os.makedirs("results/multitask", exist_ok=True)
    # teacher-forcing cost per category (mean completion length)
    kcost = {}
    for c in CATS:
        df = pd.read_parquet(f"{MD}/{c}_train.parquet")
        kcost[c] = df["loss_mask"].map(lambda m: int(sum(x > 0 for x in m))).mean()
    weights = {c: 1.0 / len(CATS) for c in CATS}
    p_ema, prev_adapter, traj = {}, None, []
    for r in range(1, ROUNDS + 1):
        chunk_path, nseen = sample_chunk(weights, seed=1000 + r)
        prev_adapter = train_round(r, chunk_path, prev_adapter)
        p = eval_round(r, prev_adapter)
        for c in CATS:
            p_ema[c] = EMA * p[c] + (1 - EMA) * p_ema.get(c, p[c])
        row = {"round": r, "weights": dict(weights), "p": p,
               "mean": sum(p.values()) / len(p), "worst": min(p.values())}
        traj.append(row)
        with open(f"{OUT}_traj.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{ARM}] r{r} mean={row['mean']:.3f} worst={row['worst']:.3f} "
              f"({min(p,key=p.get)}) | "
              + " ".join(f"{c[:4]}:{p[c]:.2f}" for c in CATS), flush=True)
        if ARM == "port":
            weights = neyman(p_ema, kcost)
    print(f"[{ARM}] DONE")


if __name__ == "__main__":
    main()
