"""Pre-tokenize all positions (two-copy render) into a padded cache."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import render11 as R  # noqa: E402


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    boards = torch.load("armF/data/positions.pt", weights_only=False)["boards"]
    N = len(boards)
    ids_list, cell_list, lens = [], [], []
    for i in range(N):
        text, _o1, off2 = R.render_two_copy(boards[i].float())
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        spans = enc["offset_mapping"]
        starts = {}
        for tj, (a, b) in enumerate(spans):
            for o in range(a, b):
                starts[o] = tj
        cell_list.append(torch.tensor([starts[o] for o in off2], dtype=torch.int16))
        ids_list.append(torch.tensor(enc["input_ids"], dtype=torch.int32))
        lens.append(len(enc["input_ids"]))
        if (i + 1) % 5000 == 0:
            print(f"{i+1}/{N}", flush=True)
    T = max(lens)
    ids = torch.full((N, T), tok.pad_token_id, dtype=torch.int32)
    for i, t in enumerate(ids_list):
        ids[i, : len(t)] = t
    out = {"input_ids": ids, "lens": torch.tensor(lens, dtype=torch.int32),
           "cell_idx": torch.stack(cell_list)}
    torch.save(out, "armF/data/tokens.pt")
    print(f"wrote armF/data/tokens.pt ids {tuple(ids.shape)} maxlen {T}")


if __name__ == "__main__":
    main()
