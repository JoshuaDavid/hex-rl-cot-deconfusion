"""Gold-certificate SFT dataset for the step-250 branch experiment.

For each terminal board: prompt (witness format) + gold completion containing a
short natural think that traces the path, then the JSON answer. Gold path =
BFS shortest path through the winner's stones. Output: jsonl of
{prompt, completion} pairs (SFT harness wiring decided at branch time).
"""

import json
import random
import sys
from collections import deque

sys.path.insert(0, "/workspace/hex-rl-cot-deconfusion")

from hexenv.board import Board, EMPTY, BLACK, WHITE, cell_name
from hexenv.prompts import RULES
from hexenv.render import render_ascii

TAIL = ('\nEnd your response with exactly one line of the form:'
        '\nAnswer: {"winner": "Black|White", "path": ["cell", "cell", ...]}'
        '\nwhere path is an ordered sequence of adjacent same-color stones'
        ' connecting the winner\'s two edges.\n')

Q = ("\nThis game is over: one player has completed a winning connection."
     " Name the winner AND give one explicit winning path — an ordered"
     " sequence of that player's stones, each adjacent to the next,"
     " starting on one of their edges and ending on the other.")


def gold_path(b, color):
    n = b.size
    stones = {(x, y) for y in range(n) for x in range(n) if b.grid[y][x] == color}
    if color == BLACK:
        starts = [(x, y) for (x, y) in stones if y == 0]
        goal = lambda x, y: y == n - 1
    else:
        starts = [(x, y) for (x, y) in stones if x == 0]
        goal = lambda x, y: x == n - 1
    prev = {s: None for s in starts}
    q = deque(starts)
    while q:
        cur = q.popleft()
        if goal(*cur):
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return list(reversed(path))
        for nb in b.neighbors(*cur):
            if nb in stones and nb not in prev:
                prev[nb] = cur
                q.append(nb)
    return None


def main():
    rng = random.Random(2026)
    rows, seen = [], set()
    while len(rows) < 3000:
        size = rng.choice([5, 5, 6, 6, 7])
        b = Board(size)
        cells = b.legal_moves()
        rng.shuffle(cells)
        for c in cells:
            b.play(c)
            if b.winner() != EMPTY:
                break
        if b.winner() == EMPTY:
            continue
        key = (size, tuple(sorted((c, cell) for c, cell in b.moves)))
        if key in seen:
            continue
        seen.add(key)
        color = b.winner()
        winner = "Black" if color == BLACK else "White"
        path = gold_path(b, color)
        if not path:
            continue
        names = [cell_name(x, y) for x, y in path]
        edgeA = "top" if winner == "Black" else "left"
        edgeB = "bottom" if winner == "Black" else "right"
        trace = " -> ".join(names)
        think = (f"{names[0]} is a {winner} stone on the {edgeA} edge. "
                 f"Following adjacent {winner} stones: {trace}. "
                 f"{names[-1]} is on the {edgeB} edge, so the chain spans both edges. "
                 f"{winner} has won.")
        completion = (f"<think>\n{think}\n</think>\n\nAnswer: "
                      + json.dumps({"winner": winner, "path": names}))
        prompt = RULES.format(n=size, board=render_ascii(b)) + Q + TAIL
        rows.append({"prompt": prompt, "completion": completion})
    with open("data/sft_certificates.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} gold certificate pairs")
    print("SAMPLE COMPLETION:\n" + rows[0]["completion"][:400])


if __name__ == "__main__":
    main()
