"""R3 SFT data: like R2 but the helper span is the model's OWN sampled answer
(not gold), gradient still ONLY on the evaluated C answer. This matches the
regime R4 RL lives in (the model conditions Task C on its own helper output).

Two steps, one script (runs in /venv/verl, needs vllm):
  1. pre-generate own helper for each train board with a random X~unif{A,B,D,E}
     using the R2 adapter: prompt = select_prompt + prefill
     <selected-task>X</selected-task>\n<task-X>, stop </task-X>.
  2. assemble SFT rows:
       ctx = <selected-task>X</selected-task>\n<task-X>OWN_HELPER</task-X>\n<evaluated-task>  (loss 0)
       ans = GOLD_C</evaluated-task><|im_end|>                                                (loss 1)

Run: /venv/verl/bin/python scripts/build_arme_r3.py --lora checkpoints/arme_r2/adapter
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from transformers import AutoTokenizer

from hexenv.arme import (board_from_gt, gold_answer_str, select_prompt,
                         SELECTABLE, EVALUATED)
from scripts.build_armD_witness import MODEL, IM_END

VAL_BOARDS = 120


def pregen(tok, llm, lora_request, gts, rng):
    from vllm import SamplingParams
    items = []
    for gt in gts:
        b = board_from_gt(gt)
        x = rng.choice(SELECTABLE)
        base = tok.apply_chat_template(
            [{"role": "user", "content": select_prompt(b)}],
            add_generation_prompt=True, enable_thinking=False, tokenize=False)
        pref = f"<selected-task>{x}</selected-task>\n<task-{x}>"
        items.append((gt, x, base + pref))
    sp = SamplingParams(temperature=1.0, max_tokens=180,
                        stop=[f"</task-{x}>" for x in SELECTABLE] + ["</evaluated-task>"],
                        include_stop_str_in_output=False)
    outs = llm.generate([it[2] for it in items], sp, lora_request=lora_request)
    return [(gt, x, o.outputs[0].text) for (gt, x, _), o in zip(items, outs)]


def assemble(tok, triples):
    rows, bad = [], 0
    for gt, x, helper in triples:
        b = board_from_gt(gt)
        gc = gold_answer_str(EVALUATED, b)
        helper = helper.strip()
        ctx = (f"<selected-task>{x}</selected-task>\n"
               f"<task-{x}>{helper}</task-{x}>\n<evaluated-task>")
        ans = f"{gc}</evaluated-task>{IM_END}"
        pids = tok.apply_chat_template(
            [{"role": "user", "content": select_prompt(b)}],
            add_generation_prompt=True, enable_thinking=False,
            tokenize=True)["input_ids"]
        ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
        full_ids = tok(ctx + ans, add_special_tokens=False)["input_ids"]
        if full_ids[: len(ctx_ids)] != ctx_ids:
            bad += 1
            continue
        ans_ids = full_ids[len(ctx_ids):]
        ids = list(pids) + list(full_ids)
        if len(ids) > 2048:
            bad += 1
            continue
        mask = [0.0] * (len(pids) + len(ctx_ids)) + [1.0] * len(ans_ids)
        rows.append({"input_ids": ids, "loss_mask": mask,
                     "size": gt["size"], "helper": x})
    return rows, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    args = ap.parse_args()
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, max_model_len=2048, gpu_memory_utilization=0.55,
              dtype="bfloat16", enable_lora=True, max_lora_rank=64)
    lreq = LoRARequest("r2", 1, args.lora)
    rng = random.Random(9090)
    train_gts = [json.loads(l) for l in open("data/arme/pool_train.jsonl")]
    test_gts = [json.loads(l) for l in open("data/arme/pool_test.jsonl")]
    tr = pregen(tok, llm, lreq, train_gts, rng)
    va = pregen(tok, llm, lreq, test_gts[:VAL_BOARDS], rng)
    train_rows, b1 = assemble(tok, tr)
    val_rows, b2 = assemble(tok, va)
    # own-helper correctness (diagnostic): grade the sampled helper
    from hexenv.arme import grade, parse_json_payload
    import collections
    hp = collections.Counter()
    for gt, x, helper in tr[:1000]:
        b = board_from_gt(gt)
        p = parse_json_payload(helper.strip())
        _, perf = grade(x, b, p) if p is not None else (-1, False)
        hp[(x, perf)] += 1
    pd.DataFrame(train_rows).to_parquet("data/arme/r3_train.parquet")
    pd.DataFrame(val_rows).to_parquet("data/arme/r3_val.parquet")
    print(f"R3 SFT: train {len(train_rows)} (dropped {b1}) val {len(val_rows)} (dropped {b2})")
    for x in SELECTABLE:
        tot = hp[(x, True)] + hp[(x, False)]
        acc = hp[(x, True)] / tot if tot else 0
        print(f"  own-helper {x} perfect-rate {acc:.2f} (n={tot})")


if __name__ == "__main__":
    main()
