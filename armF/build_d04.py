"""Fixed-width compositional move format (Joshua 2026-08-13).

Every line "\nMove NNN. LDD C" tokenizes to the SAME 12 tokens with every
field at a fixed offset (Qwen splits digits one per token, so " D04" is
always ' D'|'0'|'4'). Space-separated color because ':X' is a single vocab
token while ':O' splits — colon would reintroduce X/O asymmetry.

Isolates multi-token cell binding from positional jitter: r4x (variable
width, compositional) c0@3k .727 vs r4t (fixed width, atomic words) .91.
X-only supervision spans, readout at the ' X' color token, same as r4x/r4t.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import render_moves as RM  # noqa: E402
import train_movesr4x as R4X  # noqa: E402


def cell_str(mv):
    return chr(ord("A") + int(mv[0])) + f"{int(mv[1]) + 1:02d}"


def build_seqs_d(tok, games, numbers=True):
    recs = []
    for gi, g in enumerate(games):
        parts = [RM.PREAMBLE_M, R4X.HDR]
        pos = len(RM.PREAMBLE_M) + len(R4X.HDR)
        spans = []
        for t, mv in enumerate(g["moves"].tolist()):
            pre = f"Move {t + 1:03d}. " if numbers else ""
            s = f"\n{pre}{cell_str(mv)} {'X' if t % 2 == 0 else 'O'}"
            parts.append(s)
            if t % 2 == 0:
                spans.append((pos + 1, pos + len(s)))
            pos += len(s)
        text = "".join(parts)
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        mt = RM.move_token_indices(enc["offset_mapping"], spans)
        recs.append({"ids": torch.tensor(enc["input_ids"], dtype=torch.int32),
                     "mt": torch.tensor(mt, dtype=torch.int16), "gi": gi,
                     "text": text})
    return recs
