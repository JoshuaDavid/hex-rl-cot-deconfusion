"""r2 moves-format render: preamble + move list (absolute coords, first player
first) + single board render of the position after the cut.

The move list earlier in context is what makes a SINGLE-copy render readable
under causal attention (every stone appears in the list before the render).
Move cell (x, y) -> "<letter a-k for x><1-11 for y>"; each move prefixed by a
space. Readout points: the LAST token of each move; the 121 render cell tokens.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import render11 as R  # noqa: E402

PREAMBLE_M = (
    "Hex game on an 11x11 board. X connects top to bottom, O connects left to "
    "right. Each cell's lower-left and upper-right diagonal neighbors are "
    "adjacent. Moves in the order played, first player first:\n"
)
MID_M = (
    "\n\nBoard after these moves, shown from the perspective of the player to "
    "move (X = the player to move):\n\n"
)


def move_str(mv):
    x, y = int(mv[0]), int(mv[1])
    return chr(ord("a") + x) + str(y + 1)


def render_moves(moves, cut_board):
    """moves: (T,2) played moves (already truncated to the cut).
    cut_board: canonical (2,13,13) board AFTER the last listed move.
    Returns (text, move_spans, cell_offsets): move_spans[t] = (start, end) char
    span of move t's string; cell_offsets[c] = char offset of render cell c."""
    parts = [PREAMBLE_M]
    pos = len(PREAMBLE_M)
    spans = []
    for mv in moves:
        s = " " + move_str(mv)
        parts.append(s)
        spans.append((pos + 1, pos + len(s)))
        pos += len(s)
    parts.append(MID_M)
    pos += len(MID_M)
    body, off_b = R.render(cut_board)
    parts.append(body[len(R.PREAMBLE):])
    shift = pos - len(R.PREAMBLE)
    cell_offsets = [o + shift for o in off_b]
    text = "".join(parts)
    for (a, b), mv in zip(spans, moves):
        assert text[a:b] == move_str(mv)
    for o in cell_offsets:
        assert text[o] in ".XO"
    return text, spans, cell_offsets


def move_token_indices(enc_spans, move_spans):
    """Last token of each move (token whose span covers the move's last char).
    Asserts that token starts within the move (no cross-move merging)."""
    idxs = []
    for (a, b) in move_spans:
        last = b - 1
        hits = [i for i, (s, e) in enumerate(enc_spans) if s <= last < e]
        assert len(hits) == 1, (a, b, hits)
        s, e = enc_spans[hits[0]]
        assert s >= a, f"move token crosses boundary: span ({s},{e}) move ({a},{b})"
        idxs.append(hits[0])
    return idxs


def cell_token_indices(enc_spans, cell_offsets):
    idxs = []
    for off in cell_offsets:
        hits = [i for i, (s, e) in enumerate(enc_spans) if s <= off < e]
        assert len(hits) == 1, (off, hits)
        idxs.append(hits[0])
    assert len(set(idxs)) == len(idxs)
    return idxs
