# Arm A — vanilla RL

## Status

Complete. The run stopped at step 154. Checkpoints exist at steps 25, 50, 75,
100, and 150.

## The question

What does plain RL do to the model, with no curriculum and no shaping? This is
the control condition for the whole project.

## The setup

- Algorithm: GRPO in verl. One A100-40GB GPU.
- Each step samples 32 random positions. The model gives 8 answers for each
  position at temperature 1.0.
- Reward: +1 or -1 from the exact solver labels. See
  [the task page](the-task.md).
- KL penalty to the base model: 0.001. Learn rate: 1e-6. Think budget: 2160
  tokens.
- Positions: random playouts on 5x5, 6x6, and 7x7 boards. No difficulty
  control.

## What happened, with numbers

The win rate on held-out positions rose, then stopped:

| step | win rate | illegal-move rate |
|-----:|---------:|------------------:|
|    0 |    0.162 |             0.238 |
|   25 |    0.184 |             0.162 |
|   50 |    0.238 |             0.134 |
|   75 |    0.314 |             0.101 |
|  100 |    0.394 |             0.090 |
|  125 |    0.451 |                 - |
|  150 |    0.458 |                 - |

The curve is flat after about step 130.

## What the wall is

We tested positions where the model can win with one move. Example: Black has
a chain that needs one final stone on the bottom edge. Results at temperature
1.0:

- The model plays the winning move 14.9 percent of the time. A random legal
  move wins 9 to 15 percent of the time. The model is at chance.
- When the winning stone must go ON the target edge, the model plays it
  3 percent of the time. That is BELOW chance. The model avoids the edge.

The explanation: RL made the model's existing "play near the center" habit
stronger. Central moves usually correlate with a win. But wins finish at the
edges. The strengthened habit became the limit.

## The reasoning was decorative

At step 100, the model scored 0.32 with its full reasoning and 0.34 with no
reasoning at all. The improvement from RL lives outside the visible chain of
thought. The model also never learned to stop its reasoning: 0 percent of
samples closed the think phase on their own.

## The verdict

On this budget, vanilla RL selected and sharpened old habits. It did not
create the missing skill (edge completion). This is a clean answer to research
question C1 for this regime.
