"""Arm E: task-selection experiment.

Five perception tasks (A-E) over a hex board, all exactly computable from the
board via BFS (no benzene). The model picks ONE auxiliary task to solve first,
then answers the EVALUATED task (C). Single source of truth for: gold answers,
graders, prompt construction, and the tagged output format.

Reading order = row-major: for y in 0..n-1, for x in 0..n-1 (i.e. sorted by
(row, column)). Cells named column-letter+row-number (a1 = col a, row 1).

Tasks:
  A  all stones + colors            -> {"black":[...], "white":[...]}
  B  winner + one winning path      -> {"winner":"Black|White|Neither", "path":[...]}
  C  all empty cells (EVALUATED)    -> [...]
  D  edge-connected stones          -> {"black_top","black_bottom","white_left","white_right"}
  E  non-edge-connected stones      -> {"black_neither","white_neither"}
"""

from __future__ import annotations

import json
import os
from collections import deque

from .board import Board, EMPTY, BLACK, WHITE, cell_name
from .render import render_ascii

TASKS = ["A", "B", "C", "D", "E"]
# Evaluated task is env-configurable. C (empties) is board-derivable so helper
# content is ignored (no differential); E (connected-to-neither) is
# connectivity-bound so its useful helper D supplies the missing computation.
# See RESEARCH_LOG 2026-08-10.
EVALUATED = os.environ.get("ARME_EVAL", "E")
SELECTABLE = [t for t in TASKS if t != EVALUATED]  # the model selects among these
# the designed-useful helper for each evaluated task (for differential labeling)
USEFUL_HELPER = {"C": "A", "E": "D", "A": "C", "D": "A", "B": "A"}[EVALUATED]


# ---------------------------------------------------------------- geometry --
def reading_order(cells):
    """Sort a set of (x,y) into row-major reading order, return cell names."""
    return [cell_name(x, y) for (x, y) in sorted(cells, key=lambda p: (p[1], p[0]))]


def _stones(board, color):
    n = board.size
    return {(x, y) for y in range(n) for x in range(n) if board.grid[y][x] == color}


def _connected_to(board, color, on_edge):
    """BFS: which `color` stones connect (via same-color chain) to an edge cell
    satisfying on_edge(x,y)."""
    stones = _stones(board, color)
    seed = [(x, y) for (x, y) in stones if on_edge(x, y)]
    seen = set()
    q = deque(seed)
    for s in seed:
        seen.add(s)
    while q:
        x, y = q.popleft()
        for nx, ny in board.neighbors(x, y):
            if (nx, ny) in stones and (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return seen


def _shortest_winning_path(board, color):
    """BFS shortest same-color chain from the color's near edge to its far edge.
    Returns list of cell names, or None if not connected."""
    n = board.size
    stones = _stones(board, color)
    if color == BLACK:
        src = [(x, 0) for x in range(n) if (x, 0) in stones]
        goal = lambda x, y: y == n - 1
    else:
        src = [(0, y) for y in range(n) if (0, y) in stones]
        goal = lambda x, y: x == n - 1
    prev = {s: None for s in src}
    q = deque(src)
    while q:
        cur = q.popleft()
        if goal(*cur):
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return [cell_name(x, y) for x, y in reversed(path)]
        for nb in board.neighbors(*cur):
            if nb in stones and nb not in prev:
                prev[nb] = cur
                q.append(nb)
    return None


# ------------------------------------------------------------- gold answers --
def gold_payload(task, board):
    """Return the canonical (gold) answer object for a task on a board."""
    n = board.size
    if task == "A":
        return {"black": reading_order(_stones(board, BLACK)),
                "white": reading_order(_stones(board, WHITE))}
    if task == "B":
        w = board.winner()
        if w == EMPTY:
            return {"winner": "Neither", "path": []}
        color_name = "Black" if w == BLACK else "White"
        return {"winner": color_name, "path": _shortest_winning_path(board, w)}
    if task == "C":
        occ = _stones(board, BLACK) | _stones(board, WHITE)
        empties = {(x, y) for y in range(n) for x in range(n)} - occ
        return reading_order(empties)
    if task == "D":
        return {
            "black_top": reading_order(_connected_to(board, BLACK, lambda x, y: y == 0)),
            "black_bottom": reading_order(_connected_to(board, BLACK, lambda x, y: y == n - 1)),
            "white_left": reading_order(_connected_to(board, WHITE, lambda x, y: x == 0)),
            "white_right": reading_order(_connected_to(board, WHITE, lambda x, y: x == n - 1)),
        }
    if task == "E":
        b_top = _connected_to(board, BLACK, lambda x, y: y == 0)
        b_bot = _connected_to(board, BLACK, lambda x, y: y == n - 1)
        w_left = _connected_to(board, WHITE, lambda x, y: x == 0)
        w_right = _connected_to(board, WHITE, lambda x, y: x == n - 1)
        b_neither = _stones(board, BLACK) - b_top - b_bot
        w_neither = _stones(board, WHITE) - w_left - w_right
        return {"black_neither": reading_order(b_neither),
                "white_neither": reading_order(w_neither)}
    raise ValueError(task)


def gold_answer_str(task, board):
    return json.dumps(gold_payload(task, board), separators=(",", ":"))


# ------------------------------------------------------------------ graders --
def _set_score(claimed, truth):
    """(TP - FP)/|truth| clipped to [-1,1]; perfect iff exact match.
    Empty truth: perfect iff claimed empty, else penalize by #FP."""
    claimed, truth = set(claimed), set(truth)
    if not truth:
        return (1.0, True) if not claimed else (max(-1.0, -len(claimed)), False)
    tp = len(claimed & truth)
    fp = len(claimed - truth)
    score = max(-1.0, min(1.0, (tp - fp) / len(truth)))
    return score, (claimed == truth)


def _norm_cells(seq):
    out = []
    for c in seq if isinstance(seq, list) else []:
        s = str(c).strip().lower()
        out.append(s)
    return out


def grade(task, board, payload):
    """Grade a parsed answer payload. Returns (score in [-1,1], perfect bool).
    Multi-list tasks: mean of sub-list scores; perfect iff all sub-lists exact."""
    gold = gold_payload(task, board)
    try:
        if task in ("A", "D", "E"):
            keys = list(gold.keys())
            if not isinstance(payload, dict):
                return -1.0, False
            scores, perfects = [], []
            for k in keys:
                s, p = _set_score(_norm_cells(payload.get(k, [])), gold[k])
                scores.append(s)
                perfects.append(p)
            return sum(scores) / len(scores), all(perfects)
        if task == "C":
            return _set_score(_norm_cells(payload), gold)
        if task == "B":
            if not isinstance(payload, dict):
                return -1.0, False
            winner_ok = str(payload.get("winner", "")).capitalize() == gold["winner"]
            if gold["winner"] == "Neither":
                # correct iff claims Neither (path ignored)
                return (1.0, True) if winner_ok else (-1.0, False)
            if not winner_ok:
                return -1.0, False
            path = _norm_cells(payload.get("path", []))
            if not path:
                return -1.0, False
            want = BLACK if gold["winner"] == "Black" else WHITE
            n = board.size

            def co(c):
                return ord(c[0]) - 97, int(c[1:]) - 1

            def adj(a, b):
                ax, ay = co(a)
                bx, by = co(b)
                return (bx - ax, by - ay) in {(-1, 0), (1, 0), (0, -1), (0, 1), (1, -1), (-1, 1)}

            checks = []
            for c in path:
                try:
                    x, y = co(c)
                    checks.append(0 <= x < n and 0 <= y < n and board.grid[y][x] == want)
                except (ValueError, IndexError):
                    checks.append(False)
            checks += [adj(a, b) for a, b in zip(path, path[1:])]
            if gold["winner"] == "Black":
                checks += [co(path[0])[1] == 0, co(path[-1])[1] == n - 1]
            else:
                checks += [co(path[0])[0] == 0, co(path[-1])[0] == n - 1]
            frac = sum(checks) / len(checks)
            return max(-1.0, min(1.0, 2 * frac - 1)), (frac == 1.0)
    except (KeyError, TypeError, ValueError, IndexError):
        return -1.0, False
    raise ValueError(task)


# ----------------------------------------------------------------- prompts --
PREAMBLE = """You are being evaluated on a Hex board vision task on a {n}x{n} board. \
Black needs a chain of adjacent Black stones connecting the TOP row to the BOTTOM row; \
White needs a chain of adjacent White stones connecting the LEFT column to the RIGHT column. \
Cells are named column-letter + row-number (e.g. c2 = column c, row 2). Each cell has up to \
6 neighbors: column-X row-Y neighbors (X-1,Y),(X+1,Y),(X,Y-1),(X+1,Y-1),(X-1,Y+1),(X,Y+1). \
"Reading order" means row by row (row 1 first), left to right within a row.

The current board ('.' = empty, B = Black, W = White):

{board}

The tasks are defined below. Each answer is a single JSON value.

Task A: List the colors and positions of ALL stones, in reading order.
  Format: {{"black": ["a1",...], "white": ["b2",...]}}
Task B: Identify the winner (the player with a completed connection), if any, and give one \
explicit winning path as an ordered list of that player's adjacent stones from one edge to the other.
  Format: {{"winner": "Black"|"White"|"Neither", "path": ["c1",...]}}
Task C: List ALL empty cells, in reading order.
  Format: ["a1", "d2", ...]
Task D: A stone is "connected to the TOP" if it lies in a chain of adjacent same-color stones that \
includes a stone on row 1 (similarly BOTTOM = row {n}, LEFT = the leftmost column a, RIGHT = the \
rightmost column). A stone on the edge itself counts as connected to that edge. A stone may appear in \
more than one list (e.g. a stone on a full top-to-bottom chain is connected to both). List black \
stones connected to the TOP, \
black connected to the BOTTOM, white connected to the LEFT, and white connected to the RIGHT.
  Format: {{"black_top": [...], "black_bottom": [...], "white_left": [...], "white_right": [...]}}
Task E: Using the same definition of "connected", list black stones connected to NEITHER the top nor \
the bottom, and white stones connected to NEITHER the left nor the right.
  Format: {{"black_neither": [...], "white_neither": [...]}}
"""

R1_SUFFIX = """
Answer Task {x}, then Task {y}, in that order. Output exactly:
<task-{x}>...</task-{x}>
<task-{y}>...</task-{y}>
"""

SELECT_SUFFIX = """
You will first CHOOSE one task (A, B, D, or E) to complete as a warm-up (it will NOT be scored), \
and then you will answer Task {ev} (which IS scored). Output exactly:
<selected-task>X</selected-task>
<task-X>...</task-X>
<evaluated-task>...</evaluated-task>
where X is your chosen task letter, <task-X> is your answer to that task, and <evaluated-task> \
is your answer to Task {ev}.
"""


def board_from_gt(gt):
    b = Board(gt["size"])
    for c, cell in gt["moves"]:
        x = ord(cell[0]) - 97
        y = int(cell[1:]) - 1
        b.grid[y][x] = BLACK if c == "B" else WHITE
    return b


def preamble(board):
    return PREAMBLE.format(n=board.size, board=render_ascii(board))


def r1_prompt(board, x, y):
    return preamble(board) + R1_SUFFIX.format(x=x, y=y)


def select_prompt(board, ev=EVALUATED):
    return preamble(board) + SELECT_SUFFIX.format(ev=ev)


SOLO_SUFFIX = """
Answer Task {t}. Output exactly:
<task-{t}>...</task-{t}>
"""


def solo_prompt(board, t=EVALUATED):
    """Single-task prompt (no helper) — baseline for the instrumental differential."""
    return preamble(board) + SOLO_SUFFIX.format(t=t)


# ---------------------------------------------------------------- parsing ---
def extract_tag(text, tag):
    """Return the inner content of the LAST <tag>...</tag> (non-greedy), or None."""
    import re
    m = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m[-1].strip() if m else None


def parse_json_payload(inner):
    if inner is None:
        return None
    try:
        return json.loads(inner)
    except (json.JSONDecodeError, TypeError):
        # tolerate trailing text: grab first {...} or [...]
        import re
        m = re.search(r"(\{.*\}|\[.*\])", inner, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except (json.JSONDecodeError, TypeError):
                return None
        return None
