"""Prompt construction shared by Phase 1 evals and RL training."""

from .board import Board, BLACK
from .render import render_ascii

RULES = """You are playing the game of Hex on a {n}x{n} board.

Rules of Hex:
- Two players, Black (B) and White (W), alternately place stones on empty cells. Stones are never moved or captured.
- Black wins by connecting the TOP edge to the BOTTOM edge with an unbroken chain of adjacent Black stones. White wins by connecting the LEFT edge to the RIGHT edge with an unbroken chain of adjacent White stones.
- Cells are named by column letter + row number, e.g. c2 is column c, row 2.
- Each cell is a hexagon with up to 6 neighbors. The neighbors of cell c2 are: b2, d2, c1, d1, b3, c3. In general, the neighbors of column-X row-Y are: (X-1,Y), (X+1,Y), (X,Y-1), (X+1,Y-1), (X-1,Y+1), (X,Y+1).
- The board is drawn with each row shifted right, so the hexagonal adjacency is visible: a cell's neighbors are the cells to its left and right, and the two nearest cells diagonally above and below it.

Current board ({n}x{n}), '.' = empty:

{board}
"""

MOVE_SUFFIX = """
It is {color}'s turn. You are playing {color}.
Choose the strongest legal move (an empty cell). End your response with exactly one line of the form:
Move: <cell>
"""


def move_prompt(board: Board) -> str:
    color = "Black" if board.to_move == BLACK else "White"
    return (
        RULES.format(n=board.size, board=render_ascii(board))
        + MOVE_SUFFIX.format(color=color)
    )


def question_prompt(board: Board, question: str) -> str:
    return (
        RULES.format(n=board.size, board=render_ascii(board))
        + question
    )


def extract_move(text: str) -> str | None:
    """Parse the final 'Move: <cell>' line from a model response."""
    import re

    matches = re.findall(r"[Mm]ove:\s*\*{0,2}([a-zA-Z]\d{1,2})\*{0,2}", text)
    return matches[-1].lower() if matches else None
