"""P78 post-mortem probe: is the guest's argmax linearly decodable at the
EMISSION position (final prompt token of "...Next move:") — and where?

For each bottom {contained, original} (untrained top blocks, i.e. the model
each FT arm STARTED from): logistic-regression probe hs[j] at the last
prompt token -> guest masked-argmax class (121-way), j in {17, 20, 23, 26,
28=final}. Also, for the contained bottom, the same probe at the BEST cell
token readout for reference: hs[17] @ copy-2 cell tokens -> per-cell guest
logit (adapter path already known: cut12 val .30).

Trains a linear softmax probe with AdamW on 20k boards, evals on 2k.

Usage: /venv/main/bin/python armF/p78_probe.py
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_p78ft import load_model, guest_labels, PROMPT_TAIL  # noqa: E402
from p75_baselines import render_with_regs, load_guest  # noqa: E402
import qwen_embed as Q  # noqa: E402

DEV = "cuda"
LAYERS = [17, 20, 23, 26, 28]


@torch.no_grad()
def feats_at_last(model, tok, boards_u8, batch=32):
    """hs[j] at the final prompt token, j in LAYERS -> dict j -> (N, H)."""
    out = {j: [] for j in LAYERS}
    for i in range(0, len(boards_u8), batch):
        bb = boards_u8[i:i + batch]
        prompts = [render_with_regs(b)[0] + PROMPT_TAIL for b in bb.cpu()]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left", add_special_tokens=False)
        o = model.model(input_ids=enc["input_ids"].to(DEV),
                        attention_mask=enc["attention_mask"].to(DEV),
                        output_hidden_states=True)
        for j in LAYERS:
            out[j].append(o.hidden_states[j][:, -1, :].float().cpu())
    return {j: torch.cat(v) for j, v in out.items()}


def probe(X_tr, y_tr, X_va, y_va, steps=3000, lr=1e-2):
    W = torch.zeros(X_tr.shape[1] + 1, 121, device=DEV, requires_grad=True)
    opt = torch.optim.AdamW([W], lr=lr, weight_decay=1e-4)
    Xtr = torch.cat([X_tr, torch.ones(len(X_tr), 1)], 1).to(DEV)
    ytr = y_tr.to(DEV)
    Xva = torch.cat([X_va, torch.ones(len(X_va), 1)], 1).to(DEV)
    for s in range(steps):
        sel = torch.randint(0, len(Xtr), (2048,), device=DEV)
        loss = F.cross_entropy(Xtr[sel] @ W, ytr[sel])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (Xva @ W).argmax(1).cpu().eq(y_va).float().mean().item()


def main():
    torch.manual_seed(0)
    boards = torch.load("armF/data/tx_positions.pt",
                        weights_only=False)["boards"]
    perm = torch.randperm(len(boards),
                          generator=torch.Generator().manual_seed(3))
    tr_b, va_b = boards[perm[:20000]], boards[perm[20000:22000]]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(Q.QWEN)
    guest = load_guest()

    def labels_batched(bs):
        return torch.cat([guest_labels(guest, bs[i:i + 1024].to(DEV)).cpu()
                          for i in range(0, len(bs), 1024)])

    y_tr = labels_batched(tr_b)
    y_va = labels_batched(va_b)

    res = {}
    for bottom in ["contained", "original"]:
        model = load_model(bottom)
        model.eval()
        ftr = feats_at_last(model, tok, tr_b)
        fva = feats_at_last(model, tok, va_b)
        res[bottom] = {}
        for j in LAYERS:
            acc = probe(ftr[j], y_tr, fva[j], y_va)
            res[bottom][str(j)] = round(acc, 4)
            print(f"{bottom} hs[{j}] @ last-prompt-token -> guest argmax: "
                  f"top1 {acc:.3f}", flush=True)
        del model, ftr, fva
        torch.cuda.empty_cache()

    Path("armF/results/p78_probe.json").write_text(json.dumps(res, indent=1))
    print("wrote armF/results/p78_probe.json")


if __name__ == "__main__":
    main()
