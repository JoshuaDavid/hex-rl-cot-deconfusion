"""Batched Qwen3-1.7B hidden-state extraction at board cell tokens.

Alignment: CNN capture point l (0..18) <-> hidden_states[5+l+? ] — we use
hidden_states[j] = residual stream AFTER transformer block j (hidden_states[0]
is the embedding output), so z_l <-> hidden_states[5 + l], i.e. blocks 5..23.
Readout at the SECOND board copy's cell tokens (see render11.render_two_copy).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import render11 as R  # noqa: E402

QWEN = "Qwen/Qwen3-1.7B"
QWEN_LAYERS = list(range(5, 24))  # residual stream read after blocks 5..23


def load_qwen(device="cuda", dtype=torch.bfloat16, random_init=False):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN)
    if random_init:
        torch.manual_seed(0)
        cfg = AutoConfig.from_pretrained(QWEN)
        model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(QWEN, torch_dtype=dtype)
    model.eval().to(device)
    return tok, model


def batch_prompts(tok, boards):
    """boards: (B,2,13,13) float/uint8 canonical tensors (cpu).
    Returns dict with input_ids, attention_mask, cell_idx (B,121) for copy 2."""
    texts, cell_idx = [], []
    for b in boards:
        text, _off1, off2 = R.render_two_copy(b)
        texts.append(text)
        cell_idx.append(None)  # fill after tokenization
    enc = tok(texts, return_offsets_mapping=True, padding=True,
              return_tensors="pt", add_special_tokens=False)
    B = len(texts)
    idxs = torch.zeros(B, 121, dtype=torch.long)
    for i in range(B):
        spans = enc["offset_mapping"][i].tolist()
        _t, _o1, off2 = R.render_two_copy(boards[i])
        starts = {}
        for tj, (a, bnd) in enumerate(spans):
            if a == bnd:
                continue
            for o in range(a, bnd):
                starts[o] = tj
        idxs[i] = torch.tensor([starts[o] for o in off2])
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
            "cell_idx": idxs}


@torch.no_grad()
def hidden_at_cells(model, batch, device="cuda", layers=QWEN_LAYERS):
    """Returns list over layers of (B,121,H) hidden states at cell tokens."""
    ids = batch["input_ids"].to(device)
    am = batch["attention_mask"].to(device)
    out = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
    cell = batch["cell_idx"].to(device)  # (B,121)
    res = []
    for j in layers:
        h = out.hidden_states[j]  # (B,T,H)
        g = torch.gather(h, 1, cell.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        res.append(g)
    return res
