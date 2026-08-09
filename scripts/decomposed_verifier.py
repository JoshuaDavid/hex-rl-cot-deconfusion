"""Decomposed verifier: verify a witness answer by DECOMPOSING into per-cell
occupancy (model) + deterministic adjacency/edge checks (code) -- exploiting
the finding that per-cell occupancy is ~0.998 while the holistic judge caps
at 0.65.

Verdict(claimed winner C, claimed path P) = Yes iff:
  - consecutive cells of P are adjacent            [code]
  - endpoints of P are on C's start/end edges      [code]
  - every cell of P is C's stone                   [occupancy model, per cell]
(A valid path of color C implies C won, so a correct valid path also
validates the winner claim; a flipped winner fails the per-cell occupancy
against the claimed color.)

Phases:
  gather : bok samples witness answers on fresh 5-9 boards; store structured
           (board, claimed winner/path, true_correct, err_type, per-cell
           occupancy queries) -> data/occ/verifier_set.jsonl
  verify : run occupancy adapter on all per-cell queries; combine with code
           checks; report whole-answer verification vs holistic judge (0.645)

Run: /venv/verl/bin/python scripts/decomposed_verifier.py gather
     /venv/verl/bin/python scripts/decomposed_verifier.py verify
"""
import json, os, random, re, sys
from collections import deque, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hexenv.board import Board, BLACK, WHITE, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii
from hexenv.reward_verl import compute_score
from scripts.build_armD_witness import MODEL
from scripts.build_armD_witness_v2 import fabricate_moves
from scripts.witness_constructive import gen_board
from scripts.build_sft_certificates import Q, TAIL
from scripts.build_occ import occ_prompt

VS = "data/occ/verifier_set.jsonl"


def coords(c):
    return ord(c[0]) - 97, int(c[1:]) - 1


def adjacent(a, b):
    ax, ay = coords(a); bx, by = coords(b)
    return (bx - ax, by - ay) in {(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1)}


def geom_ok(path, n, winner):
    if not path or any(not re.fullmatch(r"[a-i][1-9]", c) for c in path):
        return False
    if any(not adjacent(a, b) for a, b in zip(path, path[1:])):
        return False
    if winner == "Black":
        return coords(path[0])[1] == 0 and coords(path[-1])[1] == n - 1
    return coords(path[0])[0] == 0 and coords(path[-1])[0] == n - 1


def err_type(gt, winner_ok, path, wstones, n):
    if not winner_ok:
        return "winner"
    if any(c not in wstones for c in path):
        return "membership"
    if not geom_ok(path, n, gt["path_winner"]):
        return "geometry"
    return "?"


def gather():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(21)
    boards = []
    while len(boards) < 500:
        n = rng.choice([5, 6, 7, 8, 9])
        winner = rng.choice(["Black", "White"])
        g = gen_board(rng, n, winner, p_forward=rng.uniform(0.0, 0.4),
                      extra_winner_frac=rng.uniform(0.1, 0.6))
        if not g:
            continue
        wst, lst, path = g
        b = Board(n)
        for x, y in wst:
            b.grid[y][x] = BLACK if winner == "Black" else WHITE
        for x, y in lst:
            b.grid[y][x] = WHITE if winner == "Black" else BLACK
        boards.append({"n": n, "winner": winner, "board": b,
                       "wstones": sorted(cell_name(x, y) for x, y in wst),
                       "gt": {"category": "v", "task": "path", "size": n,
                              "moves": fabricate_moves(random, winner, wst, lst, path),
                              "path_winner": winner}})
    llm = LLM(model="checkpoints/armD2_bok/hf_merged", max_model_len=1024,
              gpu_memory_utilization=0.55, dtype="bfloat16")
    wp = [TokensPrompt(prompt_token_ids=tok.apply_chat_template(
        [{"role": "user", "content": RULES.format(n=r["n"], board=render_ascii(r["board"])) + Q + TAIL}],
        add_generation_prompt=True, enable_thinking=False,
        tokenize=True)["input_ids"]) for r in boards]
    outs = llm.generate(wp, SamplingParams(temperature=1.0, n=4, max_tokens=96))
    rows = []
    for r, o in zip(boards, outs):
        wset = set(r["wstones"])
        for s in o.outputs:
            m = re.search(r"\{.*\}", s.text, re.DOTALL)
            if not m:
                continue
            try:
                obj = json.loads(m.group(0))
            except Exception:
                continue
            claimed_w = str(obj.get("winner", "")).capitalize()
            path = [str(c).lower() for c in obj.get("path", []) if re.fullmatch(r"[a-i][1-9]", str(c).lower())]
            correct = compute_score("x", "Answer: " + m.group(0), json.dumps(r["gt"]))["score"] == 1.0
            et = "correct" if correct else err_type(r["gt"], claimed_w == r["winner"], path, wset, r["n"])
            rows.append({"n": r["n"], "winner": r["winner"], "claimed_winner": claimed_w,
                         "path": path, "correct": correct, "err_type": et,
                         "board": render_ascii(r["board"]),
                         "occ_prompts": [occ_prompt(r["n"], r["board"], c) for c in path]})
    with open(VS, "w") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    import collections
    print(f"gathered {len(rows)} answers; correct-rate "
          f"{sum(x['correct'] for x in rows)/len(rows):.3f}; "
          f"err_types {dict(collections.Counter(x['err_type'] for x in rows if not x['correct']))}")


def verify():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(VS)]
    # flatten per-cell occupancy queries
    flat, owner = [], []
    for i, r in enumerate(rows):
        for j, p in enumerate(r["occ_prompts"]):
            flat.append(p); owner.append((i, j))
    llm = LLM(model="Qwen/Qwen3-1.7B", max_model_len=1024,
              gpu_memory_utilization=0.55, dtype="bfloat16",
              enable_lora=True, max_lora_rank=64)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, enable_thinking=False,
                                       tokenize=False) for p in flat]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=6),
                        lora_request=LoRARequest("occ", 1, "checkpoints/occ/adapter"))
    occ_pred = defaultdict(dict)
    for (i, j), o in zip(owner, outs):
        m = re.search(r"(Black|White|Neither)", o.outputs[0].text)
        occ_pred[i][j] = m.group(1) if m else "?"

    tp = tn = fp = fn = 0
    by_err = defaultdict(lambda: [0, 0])
    for i, r in enumerate(rows):
        # decomposed verdict
        verdict = geom_ok(r["path"], r["n"], r["claimed_winner"]) and len(r["path"]) > 0
        if verdict:
            for j in range(len(r["path"])):
                if occ_pred[i].get(j) != r["claimed_winner"]:
                    verdict = False
                    break
        yes = verdict
        lab = r["correct"]
        if lab and yes: tp += 1
        elif lab and not yes: fn += 1
        elif not lab and yes: fp += 1
        else: tn += 1
        by_err[r["err_type"]][0] += 1
        by_err[r["err_type"]][1] += (yes if lab else (not yes))
    rc = tp / (tp + fn or 1); re_ = tn / (tn + fp or 1)
    print(f"DECOMPOSED verifier (n={len(rows)}):")
    print(f"  balanced_acc={0.5*(rc+re_):.3f}  confirm-correct(Yes|ok)={rc:.3f}"
          f"  detect-error(No|bad)={re_:.3f}")
    for e in ("correct", "membership", "geometry", "winner"):
        if e in by_err:
            nn, s = by_err[e]
            print(f"    {e:11} n={nn:4}  correct-judgement={s/nn:.3f}")
    print("  (holistic answer-judge natural: balanced 0.545 / AUC 0.645)")


if __name__ == "__main__":
    {"gather": gather, "verify": verify}[sys.argv[1]]()
