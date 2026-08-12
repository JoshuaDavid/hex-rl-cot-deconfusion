"""Text render of the canonical 11x11 HexHex board for Qwen, one token per cell.

Canonical perspective (same as the CNN input): channel 0 = 'X' (to move,
connects top-bottom), channel 1 = 'O' (connects left-right). Hex skew via
row indentation. Cell (x, y) -> the token whose span covers that cell's char.
"""
import torch

PREAMBLE = (
    "Hex board, 11x11. X connects top to bottom, O connects left to right. "
    "Each cell's lower-left and upper-right diagonal neighbors are adjacent. "
    "X to move.\n\n"
)
POSTAMBLE = "\nConsider the position.\n"
MID = "\nThe same board again:\n\n"

BOARD_SIZE = 11


def render(canonical):
    """canonical: (2, 13, 13) board tensor. Returns (text, offsets) where
    offsets[x*11+y] is the char index of cell (x,y)'s symbol in text."""
    lines = []
    offsets = []
    pos = len(PREAMBLE)
    for x in range(BOARD_SIZE):
        line = " " * x
        pos += x
        for y in range(BOARD_SIZE):
            if canonical[0, x + 1, y + 1] > 0.5:
                c = "X"
            elif canonical[1, x + 1, y + 1] > 0.5:
                c = "O"
            else:
                c = "."
            line += " " + c
            pos += 1  # the space
            offsets.append(pos)
            pos += 1  # the symbol
        lines.append(line)
        pos += 1  # newline
    text = PREAMBLE + "\n".join(lines) + POSTAMBLE
    for (x, y), off in zip([(x, y) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE)],
                           offsets):
        assert text[off] in ".XO", (x, y, text[off])
    return text, offsets


def render_two_copy(canonical):
    """Board rendered twice; readout happens at the SECOND copy's cell tokens
    (causal attention: only there can every cell see the whole board).
    Returns (text, offsets1, offsets2)."""
    text1, off1 = render(canonical)
    text1 = text1[: -len(POSTAMBLE)]  # strip postamble from first copy
    body, off_b = render(canonical)
    # board body of the second copy starts at len(PREAMBLE) in its own text
    shift = len(text1) + len(MID) - len(PREAMBLE)
    text = text1 + MID + body[len(PREAMBLE):]
    off2 = [o + shift for o in off_b]
    for o in off2:
        assert text[o] in ".XO"
    return text, off1, off2


def cell_token_indices(tok, text, offsets):
    """Map each cell's char offset to a token index. Returns list of 121 ints.
    Raises if any two cells share a token."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    spans = enc["offset_mapping"]
    idxs = []
    for off in offsets:
        hits = [i for i, (a, b) in enumerate(spans) if a <= off < b]
        assert len(hits) == 1, (off, hits)
        idxs.append(hits[0])
    assert len(set(idxs)) == len(idxs), "two cells share a token"
    return idxs
