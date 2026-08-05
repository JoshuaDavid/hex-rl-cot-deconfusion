"""Minimal hex board. Conventions match benzene/HexGui:
- Cells named like 'c3' = column c (x, 0-indexed 2), row 3 (y, 0-indexed 2).
- Black connects TOP edge to BOTTOM edge (north-south).
- White connects LEFT edge to RIGHT edge (east-west).
- Neighbors of (x, y): (x±1, y), (x, y±1), (x+1, y-1), (x-1, y+1).
  (This is the standard parallelogram/axial adjacency used by benzene.)
"""

from __future__ import annotations

EMPTY, BLACK, WHITE = 0, 1, 2
COLOR_NAME = {BLACK: "B", WHITE: "W", EMPTY: "."}

NEIGHBOR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1)]


def cell_name(x: int, y: int) -> str:
    return f"{chr(ord('a') + x)}{y + 1}"


def parse_cell(s: str) -> tuple[int, int]:
    s = s.strip().lower()
    x = ord(s[0]) - ord("a")
    y = int(s[1:]) - 1
    return x, y


class Board:
    def __init__(self, size: int):
        self.size = size
        self.grid = [[EMPTY] * size for _ in range(size)]  # grid[y][x]
        self.moves: list[tuple[int, str]] = []  # (color, cell)
        self.to_move = BLACK

    def copy(self) -> "Board":
        b = Board(self.size)
        b.grid = [row[:] for row in self.grid]
        b.moves = self.moves[:]
        b.to_move = self.to_move
        return b

    def get(self, cell: str) -> int:
        x, y = parse_cell(cell)
        return self.grid[y][x]

    def legal_moves(self) -> list[str]:
        return [
            cell_name(x, y)
            for y in range(self.size)
            for x in range(self.size)
            if self.grid[y][x] == EMPTY
        ]

    def play(self, cell: str, color: int | None = None) -> None:
        if color is None:
            color = self.to_move
        x, y = parse_cell(cell)
        if not (0 <= x < self.size and 0 <= y < self.size):
            raise ValueError(f"off-board move {cell}")
        if self.grid[y][x] != EMPTY:
            raise ValueError(f"occupied cell {cell}")
        self.grid[y][x] = color
        self.moves.append((color, cell))
        self.to_move = WHITE if color == BLACK else BLACK

    def neighbors(self, x: int, y: int):
        for dx, dy in NEIGHBOR_OFFSETS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                yield nx, ny

    def winner(self) -> int:
        """Return BLACK/WHITE if that side has a connecting chain, else EMPTY."""
        n = self.size
        # Black: top row (y=0) to bottom row (y=n-1)
        for color, starts, goal in (
            (BLACK, [(x, 0) for x in range(n)], lambda x, y: y == n - 1),
            (WHITE, [(0, y) for y in range(n)], lambda x, y: x == n - 1),
        ):
            seen = set()
            stack = [(x, y) for x, y in starts if self.grid[y][x] == color]
            while stack:
                x, y = stack.pop()
                if (x, y) in seen:
                    continue
                seen.add((x, y))
                if goal(x, y):
                    return color
                for nx, ny in self.neighbors(x, y):
                    if self.grid[ny][nx] == color and (nx, ny) not in seen:
                        stack.append((nx, ny))
        return EMPTY
