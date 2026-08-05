"""Solver oracle backed by benzene's mohex binary (DFPN solver via GTP)."""

from __future__ import annotations

from .board import Board, BLACK, WHITE
from .gtp import GTPEngine

DEFAULT_BINARY = "/workspace/hex-rl-cot-deconfusion/benzene-vanilla-cmake/build/src/mohex/mohex"

COLOR_GTP = {BLACK: "black", WHITE: "white"}


class HexSolver:
    def __init__(self, binary: str = DEFAULT_BINARY, tt_size: int = 524288):
        self.eng = GTPEngine(binary)
        self._size = None
        # cap DFPN transposition table (default 2M entries ~ 770MB+/proc
        # caused RAM pressure with 40 parallel workers)
        self.eng.send(f"param_dfpn tt_size {tt_size}")

    def _load(self, board: Board):
        if self._size != board.size:
            self.eng.send(f"boardsize {board.size}")
            self._size = board.size
        else:
            self.eng.send("clear_board")
        for color, cell in board.moves:
            self.eng.send(f"play {COLOR_GTP[color]} {cell}")

    def winner(self, board: Board) -> int:
        """Perfect-play winner of the position (given board.to_move to play)."""
        self._load(board)
        resp = self.eng.send(f"dfpn-solve-state {COLOR_GTP[board.to_move]}").strip().lower()
        if "black" in resp:
            return BLACK
        if "white" in resp:
            return WHITE
        raise RuntimeError(f"unexpected solver response: {resp!r}")

    def winning_moves(self, board: Board) -> list[str]:
        """All moves for board.to_move that preserve a win (empty if position lost)."""
        self._load(board)
        resp = self.eng.send(
            f"dfpn-solver-find-winning {COLOR_GTP[board.to_move]}"
        ).strip().lower()
        return resp.split() if resp else []

    def exact_winning_moves(self, board: Board) -> tuple[int, list[str]]:
        """(perfect-play winner, exact winning-move set for board.to_move).

        Exhaustive child solve-state per legal move. NOT dfpn-solver-find-winning,
        which (a) returns [] on ICE-determined positions even when won, and
        (b) omits winning-but-ICE-inferior moves. See RESEARCH_LOG 2026-08-05.
        Includes a parent/child consistency check.
        """
        mover = board.to_move
        wins = [m for m in board.legal_moves() if self.move_keeps_win(board, m)]
        parent_winner = self.winner(board)
        if (parent_winner == mover) != bool(wins):
            raise RuntimeError(
                f"solver inconsistency: parent winner {parent_winner}, "
                f"{len(wins)} winning children; moves={board.moves}"
            )
        return parent_winner, wins

    def move_keeps_win(self, board: Board, move: str) -> bool:
        """Does `move` by board.to_move leave the mover still winning?
        One child solve-state — much cheaper than winning_moves()."""
        child = board.copy()
        mover = child.to_move
        child.play(move)
        if child.winner() == mover:
            return True
        return self.winner(child) == mover

    def close(self):
        self.eng.close()
