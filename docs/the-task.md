# The task and the reward

## The game

We train Qwen3-1.7B on the game of Hex. Two players place stones on a board of
hexagonal cells. Black must connect the TOP edge to the BOTTOM edge. White must
connect the LEFT edge to the RIGHT edge. Stones never move after placement.

## What the model sees

Each training example is one board position. The model gets the rules, the
board, and its color. This is an example board (5x5):

       a b c d e
     1 . . . . .  1
      2 . . W W .  2
       3 . B B . .  3
        4 . . . . .  4
         5 . . . . .  5
           a b c d e

The model must answer with one move, for example: `Move: c4`.

## The move reward

A game solver (benzene) gives us the exact set of moves that keep a win. The
reward has only two values:

- The move is in the winning set: reward = +1.
- All other cases: reward = -1. This includes a move that loses a won
  position, a move to an occupied cell, and an answer with no move.

Example: in one 6x6 position, the winning set is {d2, e2, b3, c3, d3, b4, c4,
b5}. The answer `Move: c3` gets +1. The answer `Move: a1` gets -1. The answer
`Move: c5` (an occupied cell) gets -1.

We only train on positions where the player to move has a won position, and
where at least one move loses the win. If every move wins, all rewards are +1,
and the training signal is zero.

## The judge reward (arm B and arm C)

Some examples ask a different question: "Has either player ALREADY completed a
winning connection?" The correct answer is one word: `Answer: Black`,
`Answer: White`, or `Answer: Neither`. A correct answer gets +1. All other
answers get -1. Board logic computes the label. No solver is necessary.

## Forced-close generation

Qwen3-1.7B almost never stops its own reasoning on this task. We measured this:
at a 6144-token budget, 93 percent of move answers hit the cap. Because of
this, all generation has two phases:

1. The model thinks. Generation stops at the token `</think>` or at the token
   budget (1088 tokens in current arms).
2. We insert the text `</think>` and `Move:` (or `Answer:`). The model then
   gives a short answer, at most 8 tokens.

Training and evaluation use the same two phases. This makes all comparisons
fair.

## Length shaping (arm C)

Long reasoning costs compute but adds no accuracy on this task. We measured
this: the model scores the same with no reasoning at all. Arm C adds two
mechanisms:

- A length price on correct answers only. A correct answer with a short think
  phase gets up to 1.0. A correct answer at the full cap gets 0.75. Wrong
  answers stay at -1.
- A rising push toward `</think>` late in the think phase. From token 512 the
  sampler adds a bias to the close token. The bias grows in steps: +10, +20,
  +30. This creates short samples for the length price to reward.
