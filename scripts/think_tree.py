"""Enumerate the model's thinking-continuation tree within an N-bit budget.

Best-first over the token tree from (prompt + '<think>\\n'): keep every prefix
whose cumulative surprisal -log2 P(prefix) <= BUDGET bits (i.e. prob >= 2^-BUDGET).
Expansion is by depth with batched forward passes (all nodes at one depth share
length). A prefix is a leaf if no next token stays within budget, or it emits a
stop token (</think> / eos).

Importable: load_model, build_context, enumerate_budget_tree.
"""
import argparse, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOG2E = 1.0 / math.log(2)


def load_model(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    return tok, model


def stop_id_set(tok):
    s = {tok.eos_token_id}
    for t in ["</think>", "<|im_end|>"]:
        e = tok.encode(t, add_special_tokens=False)
        if len(e) == 1:
            s.add(e[0])
    return s


def build_context(tok, user, prefill="<think>\n"):
    ctx = tok.apply_chat_template([{"role": "user", "content": user}],
                                  add_generation_prompt=True, enable_thinking=True,
                                  tokenize=False) + prefill
    ids = tok(ctx, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    return ctx, ids


@torch.no_grad()
def _next_logprobs2(model, ctx_ids, seqs, chunk=16):
    """log2 p(next | ctx+seq) for equal-length seqs -> [n, V].

    Chunked, and only the last position's logits are computed (logits_to_keep=1)
    so the [n, seq, vocab] tensor is never materialised.
    """
    outs = []
    for s in range(0, len(seqs), chunk):
        part = seqs[s:s + chunk]
        if len(part[0]) == 0:
            batch = ctx_ids
        else:
            suf = torch.tensor(part, device=ctx_ids.device)
            batch = torch.cat([ctx_ids.expand(len(part), -1), suf], dim=1)
        logits = model(batch, logits_to_keep=1).logits[:, -1].float()
        outs.append((torch.log_softmax(logits, -1) * LOG2E).cpu())
    return torch.cat(outs, 0)


def enumerate_budget_tree(model, tok, ctx_ids, budget=8.0, max_depth=48):
    """Return (rows, leaves). rows = [(bits, seq_ids, is_leaf, reason)] sorted."""
    stop_ids = stop_id_set(tok)
    kept = []
    frontier = [((), 0.0)]
    depth = 0
    while frontier and depth <= max_depth:
        seqs = [s for s, _ in frontier]
        bits = [b for _, b in frontier]
        lp = _next_logprobs2(model, ctx_ids, seqs)
        nxt = []
        for i, (seq, b) in enumerate(zip(seqs, bits)):
            room = budget - b
            cand = torch.nonzero(lp[i] >= -room, as_tuple=False).flatten().tolist()
            for tid in cand:
                cb = b - lp[i][tid].item()
                if cb > budget + 1e-9:
                    continue
                child = seq + (tid,)
                if tid in stop_ids:
                    kept.append((cb, child, True, "stop:" + repr(tok.decode([tid]))))
                else:
                    kept.append((cb, child, None, None))
                    nxt.append((child, cb))
        frontier = nxt
        depth += 1
    truncated = bool(frontier)  # nodes still in-budget at max_depth
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
    return rows, leaves, truncated


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/armD2_bok/hf_merged")
    ap.add_argument("--data", default="data/verl_witness_long/val.parquet")
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--budget", type=float, default=8.0)
    ap.add_argument("--prefill", default="<think>\n")
    ap.add_argument("--max-depth", type=int, default=48)
    args = ap.parse_args()

    import pandas as pd
    tok, model = load_model(args.model)
    df = pd.read_parquet(args.data)
    user = df.iloc[args.task_idx]["prompt"][0]["content"]
    ctx, ctx_ids = build_context(tok, user, args.prefill)
    rows, leaves, truncated = enumerate_budget_tree(model, tok, ctx_ids,
                                                     args.budget, args.max_depth)

    def show(seq):
        return repr(tok.decode(list(seq))) if seq else "<root>"

    print(f"model={args.model}  task_idx={args.task_idx}  budget={args.budget} bits "
          f"(p>=1/{2**int(args.budget)})   truncated_at_max_depth={truncated}")
    print(f"kept prefixes: {len(rows)}   leaves: {len(leaves)}   "
          f"max depth: {max(len(r[1]) for r in rows)}")
    covered = sum(2 ** (-r[0]) for r in leaves)
    print(f"prob mass covered by leaf set: {covered:.4f}\n")

    print("=== all in-budget prefixes (depth-ordered) ===")
    for cb, seq, leaf, reason in rows:
        tag = f"  [LEAF {reason}]" if leaf else ""
        print(f"{cb:5.2f}b {'  ' * len(seq)}{show(seq)}{tag}")

    print("\n=== leaf sequences ===")
    for cb, seq, leaf, reason in sorted(leaves, key=lambda r: r[0]):
        print(f"  p={2**(-cb):.4f}  ({cb:4.2f}b)  {show(seq)}   {reason}")


if __name__ == "__main__":
    _main()
