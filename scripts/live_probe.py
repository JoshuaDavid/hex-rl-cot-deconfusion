"""Probe the live training policy via verl's vLLM HTTP server.

Measures, per wave: win/legal/natural-close rates, and — within the winning
set — domination (non-inferior) rate, robustness percentile, bridge delta,
center distance. Appends one row (with full CoTs) to results/live_probes.jsonl.

Run: /venv/main/bin/python scripts/live_probe.py [--k 8] [--temperature 1.0]
Safe during training: requests queue while the engine sleeps; connection
resets at sleep/wake boundaries are retried.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

from hexenv.board import Board, BLACK, WHITE
from hexenv.prompts import move_prompt, extract_move

THINK_BUDGET = int(os.environ.get("HEX_THINK_BUDGET", "2160"))
ANSWER_BUDGET = 8


def find_server(retries=20, delay=30):
    for _ in range(retries):
        out = subprocess.run(["ss", "-tlnp"], capture_output=True,
                             text=True).stdout
        for line in out.splitlines():
            if "vLLMHttpSe" in line:
                m = re.search(r"(\d+\.\d+\.\d+\.\d+):(\d+)", line)
                if m:
                    return f"http://{m.group(1)}:{m.group(2)}"
        time.sleep(delay)
    raise RuntimeError("no vLLMHttpServer listening")


def current_step(log=os.environ.get("HEX_TRAIN_LOG", "results/pilot_1p7b.log")):
    try:
        out = subprocess.run(["grep", "-ao", r"step:[0-9]* - global", log],
                             capture_output=True, text=True).stdout
        return int(out.splitlines()[-1].split(":")[1].split()[0])
    except Exception:
        return -1


def board_from_record(rec):
    b = Board(rec["size"])
    for c, cell in rec["moves"]:
        b.play(cell, BLACK if c == "B" else WHITE)
    b.to_move = BLACK if rec["to_move"] == "B" else WHITE
    return b


def complete(base, prompt, max_tokens, temperature, stop_ids=None, timeout=1200):
    payload = {"model": "Qwen/Qwen3-1.7B", "prompt": prompt,
               "max_tokens": max_tokens, "temperature": temperature,
               "top_p": 0.95}
    if stop_ids:
        payload["stop_token_ids"] = stop_ids
        payload["include_stop_str_in_output"] = True
    last_err = None
    for attempt in range(6):
        try:
            r = requests.post(base + "/v1/completions", json=payload,
                              timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["text"]
        except (requests.ConnectionError, requests.Timeout,
                requests.HTTPError) as e:
            # engine sleep/wake boundaries reset connections; back off, retry
            last_err = e
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"gave up after retries: {last_err}")


def probe_one(args):
    base, chat_prompt, temperature, close_tok = args
    think = complete(base, chat_prompt, THINK_BUDGET, temperature,
                     stop_ids=close_tok)
    natural = think.rstrip().endswith("</think>")
    think_body = think[: think.rfind("</think>")] if natural else think
    cont = chat_prompt + think_body + "\n</think>\n\nMove:"
    ans = complete(base, cont, ANSWER_BUDGET, temperature)
    return {"think": think_body, "natural_close": natural, "answer": ans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--positions", default="data/probe_wave_positions.jsonl")
    ap.add_argument("--out", default="results/live_probes.jsonl")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    close_tok = [tok.convert_tokens_to_ids("</think>")]

    base = find_server()
    step = current_step()
    recs = [json.loads(l) for l in open(args.positions)]

    jobs = []
    for r in recs:
        cp = tok.apply_chat_template(
            [{"role": "user", "content": move_prompt(board_from_record(r))}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for _ in range(args.k):
            jobs.append((base, cp, args.temperature, close_tok))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        outs = list(ex.map(probe_one, jobs))
    dur = time.time() - t0

    n = win = legal = nat = 0
    ni_hits = ni_base = 0.0
    rob_pcts, bridge_deltas, centers, base_centers = [], [], [], []
    details = []
    idx = 0
    for pi, r in enumerate(recs):
        b = board_from_record(r)
        lg = set(b.legal_moves())
        wins = set(r["winning_moves"])
        ni = set(r["noninferior_winners"])
        robs = r["robustness"]
        rvals = sorted(robs.values())
        pos_detail = {"position_idx": pi, "moves": [], "natural": [], "thinks": []}
        for _ in range(args.k):
            o = outs[idx]; idx += 1
            mv = extract_move("Move:" + o["answer"])
            n += 1
            nat += o["natural_close"]
            pos_detail["moves"].append(mv)
            pos_detail["natural"].append(o["natural_close"])
            pos_detail["thinks"].append(o["think"])
            if mv and mv in lg:
                legal += 1
                if mv in wins:
                    win += 1
                    ni_hits += mv in ni
                    ni_base += len(ni) / len(wins)
                    if len(rvals) > 1:
                        rob_pcts.append(
                            sum(v < robs[mv] for v in rvals) / (len(rvals) - 1))
                    bridge_deltas.append(r["bridge_delta"][mv])
                    centers.append(r["center_dist"][mv])
                    base_centers.append(
                        sum(r["center_dist"].values()) / len(r["center_dist"]))
        details.append(pos_detail)

    def mean(x):
        return round(sum(x) / len(x), 4) if x else None

    row = {
        "ts": time.time(), "step": step, "k": args.k,
        "temperature": args.temperature, "n": n, "wall_s": round(dur, 1),
        "win_rate": round(win / n, 4), "legal_rate": round(legal / n, 4),
        "natural_close_rate": round(nat / n, 4),
        "noninferior_given_win": round(ni_hits / win, 4) if win else None,
        "noninferior_uniform_baseline": round(ni_base / win, 4) if win else None,
        "robustness_percentile_mean": mean(rob_pcts),
        "bridge_delta_mean": mean(bridge_deltas),
        "center_dist_mean": mean(centers),
        "center_dist_uniform_baseline": mean(base_centers),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps({**row, "details": details}) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
