# Arm B — staged curriculum (retired)

## Status

Retired before a full run. Arm B only ran short smoke tests. Its useful parts
moved into [arm C](arm-c-adaptive.md).

## The plan that arm B tested

Arm A showed the model cannot see one-move wins, and avoids the edges. Arm B
was the first response: train on a fixed mixture that is rich in exactly those
positions, in fixed stages.

- Stage B1: 15 percent judge task, 31 percent edge one-move wins, 28 percent
  general one-move wins, 26 percent general positions.
- Stage B2 plan: add two-move wins.
- Stage B3 plan: return to mostly general positions.

Each stage was one training run. A stage change needed a restart from a
checkpoint.

## What arm B produced before retirement

- A bug catch: the answer scaffold said `Move:` for every task, so all judge
  answers failed. The fix reads the task type from the prompt. The per-task
  slice breakdown in the logs caught this within 10 steps.
- A cost measurement: about 89 percent of each training step paid for think
  tokens that added no accuracy. The think budget dropped from 2160 to 1088
  tokens. This cut the step time from about 259 seconds to about 140 seconds.
- The length-price and close-bias mechanisms. See
  [the task page](the-task.md).

## Why arm B retired

Fixed stages are a coarse version of a smooth mixture controller. Joshua asked
for two properties that stages do not give: add a new task type during a run,
and remove a bad task type during a run. Arm C provides both.
