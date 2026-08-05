"""Standalone reward file for verl (loaded by file path).

Also logs every scored sample to jsonl (side channel for C2/C3 analysis),
controlled by HEX_ROLLOUT_LOG env var.
"""

import json
import os
import re
import sys

sys.path.insert(0, "/workspace/hex-rl-cot-deconfusion")

_MOVE_RE = re.compile(r"[Mm]ove:\s*\*{0,2}([a-zA-Z]\d{1,2})\*{0,2}")

# Kimi-style correctness-gated length shaping: correct samples get
# 1 - LAMBDA * (think_len / cap); wrong samples stay -1. GRPO's group
# normalization makes this effectively group-relative among correct samples.
# Inert unless rollouts can vary think length (see close-bias schedule in
# hex_agent_loop). Configure via env; 0 disables.
_LEN_LAMBDA = float(os.environ.get("HEX_LEN_LAMBDA", "0.0"))
_CHAR_CAP = float(os.environ.get("HEX_THINK_CHAR_CAP", "3300"))


def _len_shaped(base_score, solution_str):
    if _LEN_LAMBDA <= 0 or base_score <= 0:
        return base_score
    think = solution_str.split("</think>")[0]
    frac = min(1.0, len(think) / _CHAR_CAP)
    return base_score - _LEN_LAMBDA * frac


def _legal_moves(gt):
    n = gt["size"]
    occ = {cell for _, cell in gt["moves"]}
    return {f"{chr(ord('a') + x)}{y + 1}" for x in range(n) for y in range(n)} - occ


_ANSWER_RE = re.compile(r"[Aa]nswer:\s*\*{0,2}(Black|White|Neither)\b", re.IGNORECASE)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if gt.get("task") == "judge":
        m = _ANSWER_RE.findall(solution_str)
        ans = m[-1].capitalize() if m else None
        score = 1.0 if ans == gt["judge_label"] else -1.0
        kind = "win" if score > 0 else ("lose" if ans else "unparsed")
        log_path = os.environ.get("HEX_ROLLOUT_LOG")
        if log_path:
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"gt": gt, "move": ans, "kind": kind,
                                        "score": score, "response": solution_str}) + "\n")
            except OSError:
                pass
        return {"score": _len_shaped(score, solution_str), "kind_win": float(kind == "win"),
                "kind_lose": float(kind == "lose"), "kind_illegal": 0.0,
                "kind_unparsed": float(kind == "unparsed")}
    m = _MOVE_RE.findall(solution_str)
    move = m[-1].lower() if m else None
    if move is None or move not in _legal_moves(gt):
        score = -1.0
        kind = "illegal" if move else "unparsed"
    elif move in set(gt["winning_moves"]):
        score = 1.0
        kind = "win"
    else:
        score = -1.0
        kind = "lose"

    log_path = os.environ.get("HEX_ROLLOUT_LOG")
    if log_path:
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "gt": gt, "move": move, "kind": kind, "score": score,
                    "response": solution_str,
                }) + "\n")
        except OSError:
            pass
    return {"score": _len_shaped(score, solution_str), "kind_win": float(kind == "win"),
            "kind_lose": float(kind == "lose"),
            "kind_illegal": float(kind == "illegal"),
            "kind_unparsed": float(kind == "unparsed")}


if __name__ == "__main__":
    gt = {"size": 5, "moves": [["B", "c3"], ["W", "c2"]], "to_move": "B",
          "winning_moves": ["d3"]}
    print(compute_score("hex_solver", "blah\nMove: d3", json.dumps(gt)))
    print(compute_score("hex_solver", "blah\nMove: c3", json.dumps(gt)))
    print(compute_score("hex_solver", "no move here", json.dumps(gt)))
