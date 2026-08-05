"""Reward function for verl GRPO training (single-turn hex move task).

ground_truth (from the parquet) is a JSON string:
  {"size": n, "moves": [["B","c3"],...], "to_move": "B"}
The position is guaranteed winning for to_move (corpus filter).

Reward: +1 sampled move keeps the win; -1 legal but throws the win;
-1 illegal cell / unparseable output.

Uses a persistent per-process solver (TT caching across calls). verl calls
compute_score from worker processes; a lazy global keeps one mohex each.
"""

from __future__ import annotations

import json
import threading

from .board import Board, BLACK, WHITE
from .prompts import extract_move
from .solver import HexSolver

_solver = None
_lock = threading.Lock()


def _get_solver() -> HexSolver:
    global _solver
    with _lock:
        if _solver is None:
            _solver = HexSolver()
        return _solver


def board_from_gt(gt: dict) -> Board:
    b = Board(gt["size"])
    for color, cell in gt["moves"]:
        b.play(cell, BLACK if color == "B" else WHITE)
    b.to_move = BLACK if gt["to_move"] == "B" else WHITE
    return b


def score_move(board: Board, move: str | None, winning_moves: list[str] | None = None) -> float:
    if move is None or move not in board.legal_moves():
        return -1.0
    if winning_moves is not None:
        return 1.0 if move in winning_moves else -1.0
    solver = _get_solver()
    with _lock:
        keeps = solver.move_keeps_win(board, move)
    return 1.0 if keeps else -1.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl custom_reward_function entry point."""
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    board = board_from_gt(gt)
    move = extract_move(solution_str)
    return score_move(board, move, gt.get("winning_moves"))
