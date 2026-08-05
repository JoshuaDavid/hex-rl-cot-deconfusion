# Arm C — adaptive curriculum

## Status

Mechanics validated by smoke test (2026-08-06): dynamic mixture sampling,
mid-run config flip (importance change + category removal), length shaping
with close-bias exploration, controller ticks, val-split delegation. No long
run has started yet. One smoke found and fixed a real bug: verl instantiates
the custom dataset class for BOTH train and val splits, and the first smoke's
val split became a million-row virtual dataset (see section 3).

## Contents

1. [Why arm C exists](#1-why-arm-c-exists)
2. [The task categories, with examples](#2-the-task-categories-with-examples)
3. [How category data becomes training batches](#3-how-category-data-becomes-training-batches)
4. [The mixture controller](#4-the-mixture-controller)
5. [How to add and remove categories](#5-how-to-add-and-remove-categories)
6. [Length economics](#6-length-economics)
7. [What we measure, and the registered predictions](#7-what-we-measure-and-the-registered-predictions)
8. [Known limits and risks](#8-known-limits-and-risks)

## 1. Why arm C exists

Arm A hit a wall. The diagnosis took three steps, and each step shaped arm C:

- Step 1: the model converts one-move wins at chance level (14.9 percent at
  temperature 1.0; a random legal move gets 9 to 15 percent). So the model
  lacks the most basic skill: see that a move finishes the game.
- Step 2: the failure concentrates at the board edge. When the winning stone
  must go ON the target edge, the model plays it 3 percent of the time. That
  is below chance. The model actively avoids the edge, because RL strengthened
  its "play centrally" habit.
- Step 3: an atomic skill ladder on 2x2 boards showed the model CAN do lookups
  (0.98), adjacency from a list (1.00), and even two-fact composition (0.99).
  The deficits sit in specific skills, not in general capacity.

Conclusion: the model needs targeted practice on specific skills, and the
importance of each skill changes as the model learns. A fixed data mixture
cannot do this. Arm C builds a controller that re-aims the mixture during the
run.

One scientific price, stated up front: in arm A, any new concept in the model
was pure emergence. In arm C, we injected the concept categories ourselves.
Emergence claims from arm C are therefore weaker. We accept this. The arm C
question is different: CAN RL build a missing skill when the gradient lands on
it?

## 2. The task categories, with examples

Each category is one parquet file in `data/curriculum/`. Every row carries its
category name inside the ground truth, so the logs can report per-category
statistics.

### judge (900 rows)

The model must say who has already won. Board logic computes the label. This
category trains the terminal percept in isolation: no move choice at all.

    Current board (6x6), '.' = empty:
       a b c d e f
     1 . . W B W B  1
      2 . B W . . B  2
       3 W W W B W B  3
        4 W . . B . W  4
         5 . . B . B B  5
          6 B . . W W B  6
            a b c d e f
    Which player, if any, has ALREADY completed a winning connection on this board?
    -> correct answer: "Answer: Neither"

### edge_m1 (2240 rows)

One-move wins where EVERY winning move is on the mover's target edge. This is
the exact skill arm A refused to learn. Example (Black to move):

       a b c d e
     1 W W B W .  1
      2 . . . W B  2
       3 W W W B B  3
        4 B . B W W  4
         5 . B . B B  5
    winning set: {e1}. e1 is on the top row. Black's c1-e2-d3 group needs it.

Generation (current corpus): random playouts stop at a deep stone count. We
keep a position if the mover has an instant win, all instant wins are edge
cells, and at least one legal move still loses. The solver then labels the
full winning set.

Generation (next expansion, adopted after review): play a random game to the
end, then step back one move. The position before the winning move has an
instant win by construction, so no rejection on that predicate. A variant
removes one stone from the finished board instead; that variant must re-check
that the board is no longer terminal, because hex chains can be redundant —
removing one stone does not always remove the win.

### gen_m1 (1368 rows)

One-move wins at any location. Same generation, without the edge filter.

### mate2 (1200 rows)

Wins in two moves. The mover has no instant win. But some winning move M
exists where, after M and ANY opponent reply, the mover has an instant win.
Example (Black to move): winning set {b3, d3, a5}, no cell wins instantly.
Detection uses board logic for the instant-win checks and the solver for the
winning set.

### general (5336 rows)

Random winnable positions on 5x5 to 7x7 boards, no difficulty control. The
same distribution arm A trained on. This category anchors the mixture so the
model does not narrow onto puzzles.

All categories exclude every held-out evaluation position. The evaluation sets
stay fixed across the whole arm.

## 3. How category data becomes training batches

The file `hexenv/dynamic_dataset.py` defines `DynamicCurriculumDataset`. verl
loads it through its `data.custom_cls` hook. It works like this:

- The dataset reports a fake length of 1,000,000 rows. The dataloader
  therefore never exhausts it. The step count limits the run instead.
- On each item request, the dataset draws a category from the current weights,
  then draws a row uniformly inside that category.
- The item index seeds the random draw. The same index gives the same sample
  within a run.
- Before each draw, the dataset checks the modification time of
  `data/curriculum/weights.json`. A change triggers a refresh: re-read the
  weights, and scan the directory for new parquet files.
- A new parquet file becomes a live category at that moment. The stock verl
  dataset class tokenizes it. The training process prints one line, for
  example: `[curriculum] loaded category 'edge_m1' (2240 rows)`.
- IMPORTANT: verl instantiates this same class for the validation split. The
  class detects val files by name and delegates them wholesale to the stock
  dataset (finite length, fixed contents). The first smoke lacked this
  delegation; validation became a million-row virtual dataset and the run
  hung generating it. Preserve the delegation if you modify the class.

Two fail-open rules protect a running job:

- A bad new file prints an error and is skipped. Training continues.
- If all weights are zero or the weights file is missing, the dataset samples
  all categories uniformly.

## 4. The mixture controller

The controller (`scripts/curriculum_controller.py`) is a separate process. It
never touches the trainer. It only writes files. Every 10 minutes it does four
things:

1. Read `data/curriculum/config.yaml`. Humans edit this file. Example:

        categories:
          judge:    {importance: 0.6, floor: 0.05}
          edge_m1:  {importance: 1.5, floor: 0.10}
          mate2:    {importance: 0.7, floor: 0.05}

   `enabled` defaults to true and may be omitted. A category absent from the
   config gets importance 1.0 and floor 0.05. Only `enabled: false` changes
   behavior.

2. Read the last few thousand training samples from the rollout log. Compute,
   for each category, the success rate p and the mean token cost k. Smooth
   both with an exponential moving average. Ignore categories with fewer than
   20 recent samples.

3. Compute each category's share of future batches:

        share_c = max( importance_c × sigma_c / sqrt(k_c),
                       floor_c × importance_c )

   where sigma_c is the empirical mean within-prompt standard deviation of
   shaped scores, with sqrt(p_c × (1 − p_c)) as the fallback prior when the
   category has too few samples. Then normalize the shares to sum to 1.

4. Write `weights.json` atomically (write to a temp file, then rename), and
   append one audit line to `results/curriculum_log.jsonl`. Real example:

        {"ts": 1785965831.4, "weights": {"edge_m1": 0.38, "gen_m1": 0.2,
         "mate2": 0.089, "general": 0.253, "judge": 0.076}, "signals": {}}

### Why this formula

The formula is Neyman allocation from survey statistics. If the training
objective weights category c by importance_c, then the sample counts that
estimate its gradient with the least variance satisfy:

    n_c  proportional to  importance_c × sigma_c / sqrt(cost_c)

With length shaping active, rewards are not binary: correct answers span
0.75 to 1.0. The controller therefore estimates each category's signal
empirically — the mean within-prompt standard deviation of shaped scores from
the training log — with sqrt(p(1 − p)) as the fallback prior when a category
has too few samples. The empirical estimate matters most at saturation: when
a category's success rate reaches 1, the binary formula reports zero signal,
but the length-compression gradient is still alive, and the empirical std
sees it. A category the model always fails (p near 0) or always solves
(p near 1) carries almost no gradient. GRPO makes this literal: a group with
all equal rewards has zero advantage.

Two consequences fall out with no extra machinery:

- Saturated categories starve themselves — but only fully. A mastered
  category keeps a share while its answers still compress (the length signal),
  and collapses to its floor only when nothing measurable is left to learn.
- Importance and counts stay separate. Importance says what we want. Counts
  say how to estimate it well. A mixture that expresses importance only
  through counts silently changes the objective.

### Why the floors

Neyman allocation is short-sighted. A category at p = 0 has almost no signal
NOW, but it is exactly the frontier we want to push. Without samples, its
first successes can never appear. The floor guarantees a minimum share. When
the first successes appear, sqrt(p(1 − p)) rises, and the controller shifts
samples in without human action. For a category with no data yet, the
controller assumes p = 0.5 (the optimistic prior), which pulls new categories
in at full importance-proportional share immediately.

### One GRPO subtlety, and why arm C changes the advantage

Stock GRPO divides each group's advantages by the group's reward standard
deviation. That deletes magnitude information: a group with rewards
{1.0, 1.0, 0.75, 0.75} and a group with rewards {1.0, 1.0, 0.9998, 0.9998}
both normalize to advantages of exactly ±1, and the optimizer pushes equally
hard on both. At saturation, with length shaping active, this becomes a
pathology: full-force gradient pressure on one-token length differences, and
confident gradients on pure sampling noise. It also breaks the controller's
premise, which assumes a sample's gradient signal scales with its group's
reward spread.

Arm C therefore runs with mean-only advantages (`ADV_STD_NORM=False`,
Dr.GRPO style): advantage = reward − group mean, no std division. Push then
scales with the actual reward gap, and the controller's Neyman premise is
true of the trainer by construction. A reward deadband (section 6) removes
the remaining noise floor.

## 5. How to add and remove categories

### Add a category during a run

Example: you build a mate3 category at step 400.

1. Write `data/curriculum/mate3.parquet` with the standard row schema. Each
   ground truth must contain `"category": "mate3"`.
2. Optional: add a config entry with importance and floor. Without one, the
   defaults apply (importance 1.0, floor 0.05).
3. Wait one controller tick (at most 10 minutes). The controller sees the new
   file's category in the directory, gives it the optimistic prior, and writes
   a weight for it. The dataset loads and tokenizes the file at its next
   refresh. Training now samples mate3.

No restart. No trainer code change.

### Remove a buggy category during a run

Example: at step 300 you discover the mate2 labels are wrong.

1. Edit `data/curriculum/config.yaml`: set `mate2: {enabled: false}`.
2. Wait one controller tick. The controller writes weight 0 for mate2. The
   dataset stops sampling it at the next weights refresh.
3. Samples already in flight still get scored and trained on. The exposure is
   at most one batch plus one tick. The audit log records the exact time of
   the weight change, so later analysis can cut at that step.

No restart. The category's file stays on disk for diagnosis.

## 6. Length economics

See [the task page](the-task.md) for the base mechanisms. Arm C runs both:

- The length price: a correct answer's reward falls linearly from 1.0 (instant
  answer) to 0.75 (think phase at the full 1088-token cap). Wrong answers stay
  at -1. The gap between correct and wrong (at least 1.75) always dominates
  the length price (at most 0.25), so the model never profits from a wrong
  fast answer.
- The close-token push: without it, every sample sits at the cap, all lengths
  are equal, and the length price has nothing to compare. The rising bias
  (+10 from token 512, +20 from 768, +30 from 960, up to the 1088 cap)
  creates early-close samples. The smoke test measured the effect: about one third of samples
  close early, and their shaped rewards average 0.79 against 0.76 for
  cap-length correct answers.

The bias only helps the sampler explore. The masking keeps the training
mathematics honest: an early close that the bias forced is still a real token
the model produced, and PPO's clipping bounds the small off-policy distortion
on that one token.

Two protections added after the smoke rounds:

- Deadband: the length term is quantized to buckets of about 32 tokens.
  Length differences inside one bucket give exactly equal rewards, therefore
  zero advantage, therefore no gradient. The model cannot profit from chasing
  one-token noise.
- Mean-only advantages (see the GRPO subtlety in section 4): the push on a
  length difference is proportional to the reward difference, instead of the
  flat ±1 that stock GRPO produces.

## 7. What we measure, and the registered predictions

Fixed evaluation sets, unchanged across the arm: 277 general positions plus 60
edge one-move wins, all excluded from training.

Where the curves live (wandb project `hex-rl-cot-deconfusion`):

- The training run logs validation metrics split per category automatically,
  because each parquet row's `data_source` is `hex_<category>`. Look for
  `val-core/hex_edge_m1/...`, `val-core/hex_judge/...`, and so on.
- The controller logs to a companion run named `<exp>-controller`: weights,
  per-category success EMA, sigma, and token cost, under `mix/*`. The field
  `mix/train_step` carries the trainer's step so both runs share an x-axis. Per-category curves come from
the training side channel; every scored sample lands in
`results/rollouts/<run>.jsonl` with its category, kind, raw and shaped scores,
and full reasoning text.

The standing questions:

- Does edge conversion rise above chance, and how far? (Arm A: below chance.)
- Does the judge task saturate, and does the controller then starve it?
- Termination: as judge accuracy rises, do natural closes appear on judge
  prompts? This tests the "resolution-gated stopping" theory from the log.
- Does puzzle skill transfer to the general category and to the held-out
  general evaluation?
- C2 with the injection caveat: does any new vocabulary appear in arm C
  reasoning that never appeared in arm A?

Registered predictions (from the log, before any long run):

- Edge conversion above 0.8 at temperature 0.6 within 150 steps of edge-heavy
  training: 60 percent.
- General evaluation win rate at or above 0.55 by the end of the arm:
  40 percent.
- New hex-specific reasoning vocabulary in arm C and not arm A: 30 percent.

## 8. Known limits and risks

- The controller has no explicit rate limit on weight changes. The EMA on its
  input signals smooths most jumps. A pathological signal swing could still
  move the mixture fast. The audit log makes this visible.
- The judge category uses board-logic labels. The move categories use solver
  labels. A future category with a new label source needs its own validation
  before it enters the directory. The remove path exists exactly because this
  can go wrong.
- The Neyman formula treats gradient variance as scalar and ignores gradient
  direction conflicts between categories. If two categories pull the policy in
  opposite directions, the controller does not see it. Per-category evaluation
  curves are the backstop.
- The importance numbers in config.yaml are human judgment. The controller
  optimizes estimation efficiency GIVEN them. It cannot tell us what to want.
