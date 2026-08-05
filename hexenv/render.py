"""ASCII rendering of hex boards, for prompts and for human reading."""

from .board import Board, EMPTY, BLACK, WHITE, cell_name


def render_ascii(board: Board) -> str:
    """Skewed-parallelogram rendering, row numbers both sides, column letters
    top and bottom. Black connects top<->bottom, White connects left<->right."""
    n = board.size
    letters = " ".join(chr(ord("a") + i) for i in range(n))
    lines = []
    lines.append("   " + letters)
    for y in range(n):
        row = " ".join(
            {EMPTY: ".", BLACK: "B", WHITE: "W"}[board.grid[y][x]] for x in range(n)
        )
        indent = " " * y
        lines.append(f"{indent}{y + 1:>2} {row}  {y + 1}")
    lines.append(" " * (n - 1) + "   " + letters)
    return "\n".join(lines)


def render_cell_list(board: Board) -> str:
    """Compact alternative encoding: explicit stone lists."""
    blacks = [
        cell_name(x, y)
        for y in range(board.size)
        for x in range(board.size)
        if board.grid[y][x] == BLACK
    ]
    whites = [
        cell_name(x, y)
        for y in range(board.size)
        for x in range(board.size)
        if board.grid[y][x] == WHITE
    ]
    return (
        f"Black stones: {', '.join(blacks) if blacks else '(none)'}\n"
        f"White stones: {', '.join(whites) if whites else '(none)'}"
    )
