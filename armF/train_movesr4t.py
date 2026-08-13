"""Arm F r4t: single-token cell names (binding-hypothesis falsifier).

Identical to r4x (frame-consistent, X-only supervision) except each of the
121 cells is renamed to a distinct single Qwen token (common lowercase
words, deterministic by token id, preamble words excluded), removing the
multi-token letter+digit binding problem. Confound on record: this also
removes compositional coordinate geometry, so a win implicates binding but
a null is ambiguous.
"""
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import render_moves as RM  # noqa: E402
import train_movesr4x as R4X  # noqa: E402


def cell_words(tok):
    banned = set(re.findall(r"[a-z]+", RM.PREAMBLE_M.lower()))
    out = []
    for s, _ in sorted(tok.get_vocab().items(), key=lambda kv: kv[1]):
        if not re.fullmatch("Ġ[a-z]{3,8}", s) or s[1:] in banned:
            continue
        w = " " + s[1:]
        if len(tok(w, add_special_tokens=False)["input_ids"]) != 1:
            continue
        out.append(w)
        if len(out) == 121:
            break
    assert len(out) == 121
    return out


def build_seqs_t(tok, games):
    words = cell_words(tok)
    recs = []
    for gi, g in enumerate(games):
        parts = [RM.PREAMBLE_M, R4X.HDR]
        pos = len(RM.PREAMBLE_M) + len(R4X.HDR)
        spans = []
        for t, mv in enumerate(g["moves"].tolist()):
            w = words[int(mv[0]) * 11 + int(mv[1])]
            s = f"\n{t + 1}.{w} {'X' if t % 2 == 0 else 'O'}"
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


if __name__ == "__main__":
    R4X.build_seqs_x = build_seqs_t
    R4X.main()
