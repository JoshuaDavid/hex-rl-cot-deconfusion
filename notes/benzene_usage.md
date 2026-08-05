# Benzene (MoHex) usage notes — ground-truth Hex solver oracle

## Binary

```
/workspace/hex-rl-cot-deconfusion/benzene-vanilla-cmake/build/src/mohex/mohex
```

Built 2026-08-05 from `cgao3/benzene-vanilla-cmake` with system GCC 13.3 /
Boost 1.83 / Berkeley DB 5.3 (`apt install libboost-all-dev libdb-dev`).
**No source patches were needed** — plain `cmake .. && make -j$(nproc)` in a
`build/` directory. Only compiler warnings (`-Wconversion`, `-Wterminate`), no errors.
Sibling binaries also built: `build/src/wolve/wolve`, `build/src/benzenetest/benzenetest`.

## Protocol

`mohex` speaks GTP: commands on stdin, replies on stdout as `= result` (or
`? message` on error), terminated by a blank line. Diagnostics go to stderr —
redirect it (`2>/dev/null`) when scripting. Example:

```bash
printf "boardsize 5\nplay b c3\ngenmove w\nquit\n" | ./mohex 2>/dev/null
# = d1   (MCTS move for white)
```

Board setup commands: `boardsize N`, `play <b|w> <cell>` (cells like `c3`;
columns a.. letters, rows 1.. numbers), `undo`, `showboard`, `clear_board`.

## Solver commands (exact names, from `list_commands`)

DFPN (depth-first proof-number search — the one to use):

- `dfpn-solve-state <b|w>` — solves the current position with the given color
  to move; returns the **winning color** (`= black` / `= white`).
- `dfpn-solver-find-winning <b|w>` — **per-move oracle**: returns the list of
  all moves that win for the given player to move in the current position,
  e.g. on empty 5x5: `= e1 b2 c2 d2 e2 b3 c3 d3 a4 b4 c4 d4 a5`.
  A move not in the list is a losing move. Empty list = every move loses.
- `dfpn-get-pv <b|w>` — principal variation from the (already-solved) TT.
- `dfpn-clear-tt` — clear the transposition table (memory control between
  unrelated solves in a long-lived process).
- `param_dfpn` — tuning knobs (tt size, thread count, timelimit, etc.).

DFS variants exist with identical semantics: `dfs-solve-state`,
`dfs-solver-find-winning`, `dfs-get-pv`, `param_dfs`. DFPN is generally faster.

Note: results within one process are cached in the TT, so successive related
queries (e.g. `dfpn-solve-state` then `dfpn-get-pv`, or re-solving after one
more move) are much cheaper than cold solves. For an RL oracle, keep one
persistent mohex process per worker and talk GTP over a pipe.

## Validated example (per-move oracle)

```bash
printf "boardsize 7\nplay b d4\nplay w c5\nplay b e3\nplay w e5\ndfpn-solve-state b\nquit\n" \
  | ./mohex 2>/dev/null
# = black
```

## Latency (wall clock, cold process, single run; 2026-08-05, this instance)

Process startup alone (`quit` only) is ~1.0 s; numbers below include it.

| Query                                        | Wall time  | Result |
|----------------------------------------------|------------|--------|
| 5x5 empty, `dfpn-solve-state b`              | ~1.0 s     | black  |
| 5x5 empty, `dfpn-solver-find-winning b`      | ~1.0 s     | 13 winning moves |
| 6x6 empty, `dfpn-solve-state b`              | ~1.0 s     | black  |
| 7x7 empty, `dfpn-solve-state b`              | ~1.2 s     | black  |
| 7x7 empty, `dfpn-solver-find-winning b`      | ~64 s      | c2 e2 f2 g2 b3 c3 d3 e3 f3 a4 b4 c4 d4 |
| 7x7 after 4 stones (b d4, w c5, b e3, w e5), `dfpn-solve-state b` | ~1.0 s | black |
| 8x8 empty, `dfpn-solve-state b`              | ~43 s      | black  |

Takeaways: solve-state through 7x7 is effectively free (<0.3 s past startup).
`find-winning` costs roughly (number of legal moves) x per-child solve with TT
sharing, so it is the expensive call — ~instant at 5x5-6x6, ~1 min at 7x7 empty
(cheaper as the board fills). 8x8 solve-state is ~42 s of search; expect
find-winning at 8x8 to take many minutes to hours on sparse boards.
