"""Arm E eval (vllm, no-think, temp0 by default). Three modes:

  r1         : r1_prompt(board, X, C). Parse <task-X> and <task-C>, grade both.
               Reports per-helper accuracy and C accuracy.
  select_tf  : select_prompt + prefill
                 <selected-task>X</selected-task>\n<task-X>GOLD_X</task-X>\n<evaluated-task>
               Generate C only; grade C. -> acc(C | gold helper=X). The R2
               instrumental differential.
  select_own : select_prompt + prefill
                 <selected-task>X</selected-task>\n<task-X>
               Model generates its OWN helper then C; parse both; grade C and
               the own-helper. -> acc(C | own helper=X). The R3 differential
               (the regime R4 lives in).

Run (in /venv/verl):
  python scripts/eval_arme.py --mode select_tf --lora checkpoints/arme_r2/adapter \
      --limit 300 --out results/arme/r2_tf
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hexenv.arme import (board_from_gt, gold_answer_str, r1_prompt, select_prompt,
                         grade, extract_tag, parse_json_payload, SELECTABLE, EVALUATED)

MODEL = "Qwen/Qwen3-1.7B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["r1", "select_tf", "select_own"])
    ap.add_argument("--lora", default=None)
    ap.add_argument("--pool", default="data/arme/pool_test.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--out", required=True)
    ap.add_argument("--helpers", default=",".join(SELECTABLE))
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    helpers = args.helpers.split(",")
    gts = [json.loads(l) for l in open(args.pool)][: args.limit]
    boards = [board_from_gt(g) for g in gts]
    tok = AutoTokenizer.from_pretrained(MODEL)

    def templ(prompt):
        return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)

    # build (key, prompt_string, board, helper) work items
    items = []
    for gi, b in enumerate(boards):
        if args.mode == "r1":
            for x in helpers:
                items.append((gi, x, templ(r1_prompt(b, x, EVALUATED))))
        elif args.mode == "select_tf":
            base = templ(select_prompt(b))
            for x in helpers:
                gx = gold_answer_str(x, b)
                pref = (f"<selected-task>{x}</selected-task>\n"
                        f"<task-{x}>{gx}</task-{x}>\n<evaluated-task>")
                items.append((gi, x, base + pref))
        else:  # select_own
            base = templ(select_prompt(b))
            for x in helpers:
                pref = f"<selected-task>{x}</selected-task>\n<task-{x}>"
                items.append((gi, x, base + pref))

    llm_kwargs = dict(model=MODEL, max_model_len=2048,
                      gpu_memory_utilization=0.55, dtype="bfloat16")
    lora_request = None
    if args.lora:
        llm_kwargs.update(enable_lora=True, max_lora_rank=64)
    llm = LLM(**llm_kwargs)
    if args.lora:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest("arme", 1, args.lora)
    stops = ["</evaluated-task>", "<|im_end|>"] if args.mode != "r1" else ["<|im_end|>"]
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens,
                        stop=stops, include_stop_str_in_output=True)
    outs = llm.generate([it[2] for it in items], sp, lora_request=lora_request)

    # grade
    helper_acc = defaultdict(list)   # own-helper correctness (r1, select_own)
    c_by_helper = defaultdict(list)  # C perfect, keyed by helper
    samples = []
    for (gi, x, prompt), o in zip(items, outs):
        text = o.outputs[0].text
        b = boards[gi]
        if args.mode == "r1":
            px = parse_json_payload(extract_tag(text, f"task-{x}"))
            pc = parse_json_payload(extract_tag(text, f"task-{EVALUATED}"))
            hs, hp = grade(x, b, px) if px is not None else (-1.0, False)
            cs, cp = grade(EVALUATED, b, pc) if pc is not None else (-1.0, False)
            helper_acc[x].append(hp)
            c_by_helper[x].append(cp)
        elif args.mode == "select_tf":
            # generated text is the C answer (prefill opened <evaluated-task>)
            inner = text.split("</evaluated-task>")[0]
            pc = parse_json_payload(inner)
            cs, cp = grade(EVALUATED, b, pc) if pc is not None else (-1.0, False)
            c_by_helper[x].append(cp)
        else:  # select_own: text starts inside <task-X> (prefill opened it)
            wrapped = f"<task-{x}>{text}"
            px = parse_json_payload(extract_tag(wrapped, f"task-{x}"))
            pc = parse_json_payload(extract_tag(text, "evaluated-task"))
            hs, hp = grade(x, b, px) if px is not None else (-1.0, False)
            cs, cp = grade(EVALUATED, b, pc) if pc is not None else (-1.0, False)
            helper_acc[x].append(hp)
            c_by_helper[x].append(cp)
        if len(samples) < 24:
            samples.append({"helper": x, "gi": gi, "text": text[:500]})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    report = {"mode": args.mode, "lora": args.lora, "n_boards": len(boards)}
    print(f"\n=== arm E eval [{args.mode}] lora={args.lora} n_boards={len(boards)} ===")
    for x in helpers:
        cacc = sum(c_by_helper[x]) / len(c_by_helper[x]) if c_by_helper[x] else float("nan")
        hacc = (sum(helper_acc[x]) / len(helper_acc[x])) if helper_acc[x] else None
        report[f"C_acc_given_{x}"] = cacc
        if hacc is not None:
            report[f"helper_{x}_acc"] = hacc
        htxt = f"  helper_{x}_perfect {hacc:.3f}" if hacc is not None else ""
        print(f"  helper={x}:  C_perfect {cacc:.3f}{htxt}  (n={len(c_by_helper[x])})")
    # differential: A vs mean(others)
    if all(f"C_acc_given_{x}" in report for x in ["A", "B", "D", "E"]):
        others = [report[f"C_acc_given_{x}"] for x in ["B", "D", "E"]]
        delta = report["C_acc_given_A"] - sum(others) / len(others)
        report["delta_A_vs_mean_others"] = delta
        print(f"  DIFFERENTIAL delta = C(A) - mean C(B,D,E) = {delta:+.3f}")
    with open(args.out + "_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(args.out + "_samples.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"wrote {args.out}_report.json")


if __name__ == "__main__":
    main()
