"""Enumerate the model's thinking-continuation tree within an N-bit budget.

Best-first over the token tree from (prompt + '<think>\\n'): keep every prefix
whose cumulative surprisal -log2 P(prefix) <= BUDGET bits (i.e. prob >= 2^-BUDGET).
Expansion is by depth with batched forward passes (all nodes at one depth share
length). A prefix is a leaf if no next token stays within budget, or it emits a
stop token (</think> / eos).
"""
import argparse, heapq, json, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="checkpoints/armD2_bok/hf_merged")
ap.add_argument("--data", default="data/verl_witness_long/val.parquet")
ap.add_argument("--task-idx", type=int, default=0)
ap.add_argument("--budget", type=float, default=8.0)
ap.add_argument("--prefill", default="<think>\n")
ap.add_argument("--max-depth", type=int, default=24)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()

import pandas as pd
df = pd.read_parquet(args.data)
user = df.iloc[args.task_idx]["prompt"][0]["content"]
ctx = tok.apply_chat_template([{"role": "user", "content": user}],
                              add_generation_prompt=True, enable_thinking=True,
                              tokenize=False) + args.prefill
ctx_ids = tok(ctx, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
C = ctx_ids.shape[1]

stop_ids = {tok.eos_token_id}
for t in ["</think>", "<|im_end|>"]:
    e = tok.encode(t, add_special_tokens=False)
    if len(e) == 1:
        stop_ids.add(e[0])

BUDGET = args.budget
LOG2E = 1.0 / math.log(2)

# node: (bits, seq_tuple, is_leaf, leaf_reason)
kept = []                       # all in-budget prefixes (incl. root '')
frontier = [((), 0.0)]          # (seq, bits) to expand, by depth

@torch.no_grad()
def next_logprobs2(seqs):
    """log2 p(next | ctx+seq) for a batch of equal-length seqs -> [n, V]."""
    if len(seqs[0]) == 0:
        batch = ctx_ids
    else:
        suf = torch.tensor(seqs, device=ctx_ids.device)
        batch = torch.cat([ctx_ids.expand(len(seqs), -1), suf], dim=1)
    logits = model(batch).logits[:, -1].float()
    return torch.log_softmax(logits, -1) * LOG2E

depth = 0
while frontier and depth <= args.max_depth:
    seqs = [s for s, _ in frontier]
    bits = [b for _, b in frontier]
    lp = next_logprobs2(seqs)                      # [n, V] in bits
    nxt = []
    for i, (seq, b) in enumerate(zip(seqs, bits)):
        room = BUDGET - b                          # a child token needs -log2 p <= room
        cand = torch.nonzero(lp[i] >= -room, as_tuple=False).flatten().tolist()
        expanded_any = False
        for tid in cand:
            cb = b - lp[i][tid].item()             # child cumulative bits
            if cb > BUDGET + 1e-9:
                continue
            child = seq + (tid,)
            if tid in stop_ids:
                kept.append((cb, child, True, "stop:" + repr(tok.decode([tid]))))
            else:
                kept.append((cb, child, None, None))
                nxt.append((child, cb))
                expanded_any = True
    frontier = nxt
    depth += 1

# mark leaves: a non-stop kept node with no in-budget child is a leaf
child_parents = {seq[:-1] for _, seq, _, _ in kept if len(seq) > 0}
rows = []
for cb, seq, is_leaf, reason in kept:
    if is_leaf is None:
        leaf = seq not in child_parents
        reason = "budget-exhausted" if leaf else None
    else:
        leaf = True
    rows.append((cb, seq, leaf, reason))

rows.sort(key=lambda r: (len(r[1]), r[0]))
leaves = [r for r in rows if r[2]]

print(f"model={args.model}  task_idx={args.task_idx}  budget={BUDGET} bits (p>=1/{2**int(BUDGET)})")
print(f"context ends: ...{repr(ctx[-40:])}")
print(f"kept prefixes (incl root): {len(rows)}   leaves: {len(leaves)}   "
      f"max depth reached: {max(len(r[1]) for r in rows)}")
covered = sum(2 ** (-r[0]) for r in leaves)
print(f"prob mass covered by leaf set: {covered:.4f}\n")

def show(seq):
    return repr(tok.decode(list(seq))) if seq else "<root>"

print("=== all in-budget prefixes (depth-ordered) ===")
for cb, seq, leaf, reason in rows:
    ind = "  " * len(seq)
    tag = f"  [LEAF {reason}]" if leaf else ""
    print(f"{cb:5.2f}b {ind}{show(seq)}{tag}")

print("\n=== leaf sequences (the full within-budget continuation set) ===")
for cb, seq, leaf, reason in sorted(leaves, key=lambda r: r[0]):
    print(f"  p={2**(-cb):.4f}  ({cb:4.2f}b)  {show(seq)}   {reason}")
