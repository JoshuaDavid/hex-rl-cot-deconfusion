"""Constructive witness-board generator (arm D v2, up to 9x9).

Instead of rejection-sampling random playouts (yield craters past 5x5),
build boards directly:
  1. Plant a random induced (chord-free) edge-to-edge path for the winner —
     a self-avoiding walk that extends an induced path; p_forward biases
     progress toward the far edge and thereby controls path length.
  2. Add winner-color distractor stones and loser stones with counts
     consistent with an alternating game that ends on the winner's move
     (Black winner: nLoser = nWinner - 1; White winner: nLoser = nWinner).
  3. Accept iff the planted path is the UNIQUE induced minimal winning path
     (DFS count capped at 2) and the loser is not connected.

Any accepted board is reachable by real play: order the winner's path so one
path cell comes last — with a unique minimal path the winner is connected
only after that stone, and the loser is never connected.

Run demo: /venv/main/bin/python scripts/witness_constructive.py
"""

import random
import sys
import time

sys.path.insert(0, "/workspace/hex-rl-cot-deconfusion")

NBR = [(-1, 0), (1, 0), (0, -1), (1, -1), (-1, 1), (0, 1)]


def _nbrs(c, n):
    x, y = c
    return [(x + dx, y + dy) for dx, dy in NBR
            if 0 <= x + dx < n and 0 <= y + dy < n]


def sample_induced_path(rng, n, p_forward=0.5, max_backtracks=200):
    """Random induced path col0 -> col n-1 (White frame). Returns list of
    (x, y) or None. p_forward: extra weight on x-advancing extensions."""
    start = (0, rng.randrange(n))
    path, pathset = [start], {start}
    backtracks = 0
    while True:
        tail = path[-1]
        if tail[0] == n - 1:
            return path
        cands = []
        for c in _nbrs(tail, n):
            if c in pathset or c[0] == 0:
                continue
            if any(p in pathset and p != tail for p in _nbrs(c, n)):
                continue
            cands.append(c)
        if not cands:
            backtracks += 1
            if backtracks > max_backtracks or len(path) == 1:
                return None
            pathset.remove(path.pop())
            continue
        weights = [1.0 + (p_forward * 4 if c[0] > tail[0] else 0.0)
                   for c in cands]
        c = rng.choices(cands, weights=weights)[0]
        path.append(c)
        pathset.add(c)


def count_induced_paths(stones, n, cap=2):
    """Number of induced col0 -> col n-1 paths within `stones` (White frame),
    early-exit at `cap`."""
    S = set(stones)

    def nbrs_in(c):
        return [p for p in _nbrs(c, n) if p in S]

    cnt = 0

    def extend(tail, pathset):
        nonlocal cnt
        if cnt >= cap:
            return
        if tail[0] == n - 1:
            cnt += 1
            return
        for c in nbrs_in(tail):
            if c in pathset or c[0] == 0:
                continue
            if any(p in pathset and p != tail for p in nbrs_in(c)):
                continue
            pathset.add(c)
            extend(c, pathset)
            pathset.remove(c)

    for s in [s for s in S if s[0] == 0]:
        if cnt >= cap:
            break
        if n == 1:
            cnt += 1
            continue
        extend(s, {s})
    return cnt


def connected(stones, n):
    """Is there any col0 -> col n-1 connection within stones (White frame)?"""
    S = set(stones)
    seen = {s for s in S if s[0] == 0}
    frontier = list(seen)
    while frontier:
        c = frontier.pop()
        if c[0] == n - 1:
            return True
        for p in _nbrs(c, n):
            if p in S and p not in seen:
                seen.add(p)
                frontier.append(p)
    return False


def gen_board(rng, n, winner, p_forward=0.5, extra_winner_frac=0.35,
              loser_adjacent_frac=0.5, max_tries=50):
    """Returns (winner_stones, loser_stones, path) in BOARD frame
    ((x, y), x=col, y=row), or None. Black plays top->bottom: generate in the
    White frame and transpose for Black."""
    for _ in range(max_tries):
        path = sample_induced_path(rng, n, p_forward)
        if path is None:
            continue
        P = len(path)
        n_loser = None
        e_max = max(0, (n * n) // 2 - P)
        extra = rng.randint(0, max(0, min(int(P * extra_winner_frac), e_max)))
        n_winner = P + extra
        n_loser = n_winner - (1 if winner == "Black" else 0)
        if n_winner + n_loser > n * n:
            continue
        cells = [(x, y) for x in range(n) for y in range(n)]
        empty = [c for c in cells if c not in set(path)]
        rng.shuffle(empty)
        wstones = set(path)
        added = 0
        for c in empty:
            if added >= extra:
                break
            wstones.add(c)
            if count_induced_paths(wstones, n) == 1:
                added += 1
            else:
                wstones.remove(c)
        if added < extra:
            extra = added
            n_winner = P + extra
            n_loser = n_winner - (1 if winner == "Black" else 0)
        empty = [c for c in cells if c not in wstones]
        # bias some loser stones adjacent to the path (blocking attempts)
        adj = [c for c in empty
               if any(p in wstones for p in _nbrs(c, n))]
        rng.shuffle(adj)
        rng.shuffle(empty)
        want_adj = int(n_loser * loser_adjacent_frac)
        lstones = []
        for c in adj:
            if len(lstones) >= want_adj:
                break
            lstones.append(c)
        for c in empty:
            if len(lstones) >= n_loser:
                break
            if c not in lstones:
                lstones.append(c)
        if len(lstones) < n_loser:
            continue
        lset = set(lstones)
        # loser frame is the transpose of winner frame
        if connected({(y, x) for x, y in lset}, n):
            continue
        if count_induced_paths(wstones, n) != 1:
            continue
        if winner == "Black":
            tr = lambda s: {(y, x) for x, y in s}
            return tr(wstones), tr(lset), [(y, x) for x, y in path]
        return wstones, lset, list(path)
    return None


def demo():
    from hexenv.board import Board, BLACK, WHITE, cell_name
    from hexenv.render import render_ascii
    from hexenv.reward_verl import compute_score
    import json

    rng = random.Random(2026)
    for n in [5, 7, 9]:
        t0 = time.time()
        boards, tries = [], 0
        while len(boards) < 50:
            tries += 1
            winner = rng.choice(["Black", "White"])
            out = gen_board(rng, n, winner)
            if out:
                boards.append((winner, out))
        dt = (time.time() - t0) / len(boards)
        plens = sorted(len(p) for _, (_, _, p) in boards)
        print(f"n={n}: accept {len(boards)}/{tries}, {dt*1000:.0f} ms/board, "
              f"path len min/med/max {plens[0]}/{plens[len(plens)//2]}/{plens[-1]}")
        winner, (wst, lst, path) = boards[0]
        b = Board(n)
        wcol = BLACK if winner == "Black" else WHITE
        lcol = WHITE if winner == "Black" else BLACK
        for x, y in wst:
            b.grid[y][x] = wcol
        for x, y in lst:
            b.grid[y][x] = lcol
        print(render_ascii(b))
        names = [cell_name(x, y) for x, y in path]
        print(f"winner {winner}, planted path: {names}")
        moves = ([["B" if wcol == BLACK else "W", cell_name(x, y)] for x, y in wst]
                 + [["B" if lcol == BLACK else "W", cell_name(x, y)] for x, y in lst])
        gt = {"category": "witness_armD2", "task": "path", "size": n,
              "moves": moves, "path_winner": winner}
        ans = "Answer: " + json.dumps({"winner": winner, "path": names})
        s = compute_score("x", ans, json.dumps(gt))["score"]
        print(f"grader on planted path: {s}")
        assert s == 1.0


if __name__ == "__main__":
    demo()
