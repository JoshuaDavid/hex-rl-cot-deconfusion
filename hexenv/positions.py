"""Random position generation for evals and training corpora."""

import random
from .board import Board, EMPTY


def random_position(size: int, n_stones: int, rng: random.Random) -> Board:
    """Random legal position with no winner, alternating moves from empty."""
    for _ in range(200):
        b = Board(size)
        cells = b.legal_moves()
        rng.shuffle(cells)
        ok = True
        for c in cells[:n_stones]:
            b.play(c)
            if b.winner() != EMPTY:
                ok = False
                break
        if ok:
            return b
    raise RuntimeError("could not generate winnerless position")
