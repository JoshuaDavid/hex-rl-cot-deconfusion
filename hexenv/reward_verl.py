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


_LEN_BUCKET_CHARS = float(os.environ.get("HEX_LEN_BUCKET_CHARS", "96"))  # ~32 tok


def _len_shaped(base_score, solution_str):
    if _LEN_LAMBDA <= 0 or base_score <= 0:
        return base_score
    think = solution_str.split("</think>")[0]
    # deadband: quantize to buckets so sub-bucket length noise yields exactly
    # equal rewards (=> zero GRPO advantage, no noise-chasing at saturation)
    chars = (len(think) // _LEN_BUCKET_CHARS) * _LEN_BUCKET_CHARS
    frac = min(1.0, chars / _CHAR_CAP)
    return base_score - _LEN_LAMBDA * frac


def _legal_moves(gt):
    n = gt["size"]
    occ = {cell for _, cell in gt["moves"]}
    return {f"{chr(ord('a') + x)}{y + 1}" for x in range(n) for y in range(n)} - occ


_ANSWER_RE = re.compile(r"[Aa]nswer:\s*\*{0,2}(Black|White|Neither)\b", re.IGNORECASE)
_CELLS_RE = re.compile(r"\b([a-i][1-9])\b")


def _score_listing(gt, solution_str):
    """Set-valued answer: score = (TP - FP)/|truth|, clipped to [-1,1].
    Cells parsed from the final Answer: line."""
    post = solution_str.split("</think>")[-1]
    m = re.search(r"[Aa]nswer:\s*(.*)", post, re.DOTALL)
    claimed = set()
    if m:
        blob = m.group(1).strip().splitlines()[0] if m.group(1).strip() else ""
        try:
            arr = json.loads(blob)
            if isinstance(arr, list):
                claimed = {str(c).lower() for c in arr if _CELLS_RE.fullmatch(str(c))}
        except (json.JSONDecodeError, TypeError):
            pass
        if not claimed:
            claimed = set(_CELLS_RE.findall(m.group(1)))
    truth = set(gt["list_target"])
    if not claimed:
        return -1.0, "unparsed", claimed
    tp = len(claimed & truth)
    fp = len(claimed - truth)
    score = max(-1.0, min(1.0, (tp - fp) / max(1, len(truth))))
    kind = "win" if score > 0.66 else ("lose" if score > -1.0 else "lose")
    return score, kind, claimed


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if gt.get("task") == "listing":
        score, kind, claimed = _score_listing(gt, solution_str)
        shaped = _len_shaped(score, solution_str) if score > 0 else score
        log_path = os.environ.get("HEX_ROLLOUT_LOG")
        if log_path:
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"gt": gt, "move": sorted(claimed),
                                        "kind": kind, "score": score,
                                        "shaped": shaped,
                                        "response": solution_str}) + "\n")
            except OSError:
                pass
        return {"score": shaped, "kind_win": float(kind == "win"),
                "kind_lose": float(kind == "lose"), "kind_illegal": 0.0,
                "kind_unparsed": float(kind == "unparsed")}
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
                                        "score": score, "shaped": _len_shaped(score, solution_str), "response": solution_str}) + "\n")
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
                    "shaped": _len_shaped(score, solution_str),
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
