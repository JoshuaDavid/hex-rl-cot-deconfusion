"""Forced-close two-phase generation with vllm — mirrors hex_agent_loop.py
for offline evals (checkpoint evals, pass@k). Returns full response texts
("<think content>\n</think>\n\nMove: x") plus metadata.
"""

from __future__ import annotations


def generate_forced_close(llm, tok, user_msgs: list[str], n: int, temperature: float,
                          think_budget: int = 2160, answer_budget: int = 8,
                          top_p: float = 0.95, seed: int = 0,
                          answer_prefix: str = "Move:"):
    """Returns list (per prompt) of lists (n) of dicts:
    {text, natural_close, think_tokens}."""
    from vllm import SamplingParams

    close_id = tok.convert_tokens_to_ids("</think>")
    base_prompts = [
        tok.apply_chat_template([{"role": "user", "content": m}],
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=True)
        for m in user_msgs
    ]
    sp1 = SamplingParams(temperature=temperature, top_p=top_p, n=n, seed=seed,
                         max_tokens=think_budget, stop_token_ids=[close_id],
                         include_stop_str_in_output=True)
    outs1 = llm.generate(base_prompts, sp1, use_tqdm=False)

    phase2 = []
    meta = []
    for bp, out in zip(base_prompts, outs1):
        for o in out.outputs:
            text = o.text
            natural = text.rstrip().endswith("</think>") or o.token_ids[-1:] == [close_id]
            if natural:
                think = text[: text.rfind("</think>")] if "</think>" in text else text
                cont = bp + think + "</think>\n\n" + answer_prefix
            else:
                think = text
                cont = bp + text + "\n</think>\n\n" + answer_prefix
            phase2.append(cont)
            meta.append({"natural_close": natural, "think_tokens": len(o.token_ids),
                         "think_text": think})
    sp2 = SamplingParams(temperature=temperature, top_p=top_p, seed=seed,
                         max_tokens=answer_budget)
    outs2 = llm.generate(phase2, sp2, use_tqdm=False)

    results = []
    idx = 0
    for bp, out in zip(base_prompts, outs1):
        row = []
        for _ in out.outputs:
            ans = outs2[idx].outputs[0].text
            m = meta[idx]
            row.append({
                "text": m["think_text"] + "\n</think>\n\n" + answer_prefix + ans,
                "natural_close": m["natural_close"],
                "think_tokens": m["think_tokens"],
            })
            idx += 1
        results.append(row)
    return results
