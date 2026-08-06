# Arm C — adaptive curriculum

## Status

Launched 2026-08-06 (400 steps planned, batch 32 prompts x 8 rollouts,
Qwen3-1.7B). The run has passed step 250. Every mechanism in this document has
now run in production: two live hot-adds (occupancy + chain at step 22,
mate1_v2 at step 100), one hot-removal drill (smoke), one controller bug found
and fixed mid-run (section 4), and three restart bundles (step 100: length
economics retune; step 150: dense categories + JSON formats; step 200:
think-cap fix). At step 250 the run branches into pure-RL versus
certificate-SFT-then-RL. This document describes the design as it stands now;
`RESEARCH_LOG.md` is the authoritative history of how it got here.

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

A second lesson arrived during the run: binary rewards are informationally
starved. A group of 8 rollouts on a pass/fail task yields at most a few bits
per ~2M generated tokens. The solver gives exact set-valued labels for free,
so later categories grade per-cell and per-link (partial credit), which puts
a full ranking inside every rollout group. This is why the category list
below has four reward families, not one.

## 2. The task categories, with examples

Each category is one parquet file in `data/curriculum/`. Every row carries its
category name inside the ground truth, so the logs can report per-category
statistics. Every prompt starts with the same rules preamble (game rules,
adjacency formula, board-drawing convention); the examples below show only
the board and the question.

Categories fall into four families. Each family has its own answer format,
answer-token budget, and reward branch in `hexenv/reward_verl.py`. The agent
loop (`hexenv/hex_agent_loop.py`) detects the family from the prompt text and
sets the scaffold and budgets per sample:

| family      | answer format                          | answer budget | think cap |
|-------------|----------------------------------------|---------------|-----------|
| move        | `Move: <cell>`                         | 8 tokens      | 1088      |
| judgment    | `Answer: Black\|White\|Neither`        | 8 tokens      | 1088      |
| listing     | `Answer: ["c2", "d3", ...]` (JSON)     | 48 tokens     | 1048      |
| certificate | `Answer: {"winner": ..., "path": ...}` | 64 tokens     | 1032      |

The think cap is always `response_length (1104) − answer_budget − 8`. This
rule exists because the original fixed cap silently clipped long answers
mid-JSON (see section 8, "budget arithmetic").

The listing and certificate answers use JSON because we measured format
compliance on identical content: comma lists 0.25 strict compliance, JSON
arrays 1.00. The model holds JSON rigidly; our first-choice format was its
worst. The parser tries JSON first and falls back to regex cell-extraction,
which preserves partial credit on sloppy answers.

### Move family: `Move: <cell>`. Score +1 if the move is in the exact winning set, −1 otherwise (illegal and unparsed also −1).

**general (5336 rows).** Random winnable positions on 5x5 to 7x7 boards, no
difficulty control. The same distribution arm A trained on. This category
anchors the mixture so the model does not narrow onto puzzles.

**edge_m1 (2240 rows).** One-move wins where EVERY winning move is on the
mover's target edge. This is the exact skill arm A refused to learn.
Example (White to move):

       a b c d e
     1 B W . B W  1
      2 B . . W B  2
       3 . W W W W  3
        4 B B . W .  4
         5 B B . . B  5

    It is White's turn. ... Choose the strongest legal move.
    -> winning set: {a3}. a3 is on White's LEFT target edge; it joins the
       b3-e3 group to the left column while the a-column is otherwise Black's.

Generation: random playouts stop at a deep stone count; keep a position if
the mover has an instant win, all instant wins are edge cells, and at least
one legal move still loses. The solver labels the full winning set.

**gen_m1 (1368 rows).** One-move wins at any location. Same generation,
without the edge filter.

**mate1_v2 (1200 rows).** One-move wins from a different generator: play a
random game to the end, then step back one move. The position before the
winning move has an instant win by construction, so no rejection sampling on
that predicate, and the stone distribution is game-natural rather than
playout-truncated. Hot-added at the step-100 restart to test whether the
natural distribution helps edge conversion. (A removal variant — delete one
stone from a finished board — must re-check that the board is no longer
terminal, because hex chains can be redundant.)

**mate2 (1200 rows).** Wins in two moves. The mover has no instant win, but
some winning move M exists where, after M and ANY opponent reply, the mover
has an instant win. Board logic does the instant-win checks; the solver
labels the winning set.

### Judgment family: `Answer: Black|White|Neither`. Score +1 on exact label match, −1 otherwise.

**judge (900 rows).** Who has already won? Board logic computes the label.
Trains the terminal percept in isolation: no move choice at all.

       a b c d e f
     1 . . W B W B  1
      2 . B W . . B  2
       3 W W W B W B  3
        4 W . . B . W  4
         5 . . B . B B  5
          6 B . . W W B  6

    Which player, if any, has ALREADY completed a winning connection on this
    board? (Black needs a chain of adjacent Black stones containing both a
    TOP-row cell and a BOTTOM-row cell; White needs a chain of adjacent White
    stones containing both a LEFT-column cell and a RIGHT-column cell.)
    -> correct answer: "Answer: Neither"

The wording matters and was tuned twice: the original yes/no phrasing made
Black/White/Neither ungrammatical answers, and the chain-definition
parenthetical came from a wording A/B (guided judge accuracy 0.694 → 0.815).

**occupancy (800 rows).** "Which player, if any, has a stone on cell b6?"
Pure lookup on a full board. Hot-added at step 22 as the bottom atomic rung.
It arrived saturated (p ≈ 1.0 immediately) — the skill was never missing —
so it now serves as a cheap always-on diagnostic at its floor allocation.

**chain (800 rows).** "Which player, if any, has stones on BOTH e5 and b2
joined into one unbroken chain of that player's own adjacent stones?"
Hot-added at step 22 alongside occupancy. This category is the arm's
clearest success so far: it entered with 12x judge's within-group sigma
(specific-pair questions force variable tracing; generic existence questions
collapse to deterministic guesses) and learned from 0.42 to 0.88 in about 25
steps. Specificity manufactures gradient.

### Listing family: `Answer: ["cell", ...]`. Score = (TP − FP) / |truth|, clipped to [−1, 1]. Empty or unparsed = −1.

Dense supervision from existing exact labels: every claimed cell is graded,
so a group of 8 rollouts produces a rich ranking instead of one bit. Example
scoring, verified through the real grader: truth has 6 cells; an answer with
4 true cells and 1 false cell scores (4 − 1)/6 = 0.5.

**winset (1200 rows).** List ALL winning moves (mean 7.6 target cells).
Example (White to move):

       a b c d e
     1 . . B W .  1
      2 . . . . .  2
       3 . . . . .  3
        4 . . . . .  4
         5 . . . . B  5

    It is White's turn. List ALL empty cells where a White stone placed now
    keeps White winning with perfect play (the complete set of winning moves).
    Answer: ["cell", "cell", ...] (a JSON array)
    -> truth: ["b4", "c2", "c3", "d2", "d3", "d4"]

**chainset (1200 rows).** List all cells of the chain containing a named
stone (chains of size >= 3, mean 4.4 cells). The dense sibling of the atomic
chain category. Entered healthy at step 150 (0 percent unparsed, score
climbing from the first read) — the atomic chain skill transferred directly.

### Certificate family: `Answer: {"winner": "Black|White", "path": [...]}`. Winner-gated per-link partial credit.

**witness (1000 rows).** A finished game; name the winner AND give one
explicit winning path. This composes judge + chain-tracing — exactly the
composition step the curriculum circles. Example:

       a b c d e
     1 W B . B W  1
      2 W . B W B  2
       3 B W B B W  3
        4 B B . B W  4
         5 B W W W .  5

    This game is over: one player has completed a winning connection.
    Name the winner AND give one explicit winning path — an ordered sequence
    of that player's stones, each adjacent to the next, starting on one of
    their edges and ending on the other.
    Answer: {"winner": "Black|White", "path": ["cell", "cell", ...]}

Grading is pure board logic — no solver call. The checks are: every path
cell holds the winner's stone, every consecutive pair is adjacent, the first
cell is on the winner's start edge, the last is on the far edge. With
`link_frac` = fraction of checks passed, score = 2·link_frac − 1, and a wrong
winner gates the whole answer to −1. Verified through the real grader on a
real training row:

    {"winner": "Black", "path": ["d1", "c2", "c3", "b4", "a5"]}  -> 1.0
    same path with "c3" deleted (one broken link)                -> 0.78
    {"winner": "White", "path": ["a1", "a2"]}                    -> −1.0

The category was built to manufacture judge's missing gradient: a model
cannot deterministically guess a path, so within-group score variance exists
by construction. A label of this kind is also a complete reasoning artifact —
BFS writes gold certificates, which is what makes the step-250 SFT branch
possible (nothing can write a gold move-CoT, but gold paths are free).

All categories exclude every held-out evaluation position. The evaluation
sets stay fixed across the whole arm.

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
  example: `[curriculum] loaded category 'chainset' (1200 rows)`.
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

1. Read `data/curriculum/config.yaml`. Humans edit this file. Current live
   excerpt:

        categories:
          edge_m1:  {importance: 1.5, floor: 0.10}
          winset:   {importance: 1.2, floor: 0.08}
          witness:  {importance: 1.0, floor: 0.06}
          judge:    {importance: 0.6, floor: 0.05}
          occupancy: {importance: 0.4, floor: 0.05}

   `enabled` defaults to true and may be omitted. A category absent from the
   config gets importance 1.0 and floor 0.05. Only `enabled: false` changes
   behavior.

2. Read the last 6000 training samples from the rollout side channel. For
   each category compute: success rate p, mean token cost k (chars/3, floored
   at 64), and sigma — the mean within-prompt standard deviation of SHAPED
   scores across prompts with at least 3 rollouts, divided by 2 (the reward
   span). All three are smoothed with an EMA (alpha 0.5 per tick).
   Categories with fewer than 20 recent samples keep their previous EMA.

3. Compute each enabled category's Neyman score, normalize the scores into
   shares, then raise every share to at least its floor and renormalize:

        score_c  = importance_c × sigma_c / sqrt(k_c)
        share_c  = score_c / Σ score
        weight_c = max(share_c, floor_c), renormalized

   Disabled categories get exactly 0. A category with no samples yet gets the
   optimistic prior p = 0.5, k = 1100.

4. Write `weights.json` atomically (write to a temp file, then rename), and
   append one audit line to `results/curriculum_log.jsonl`. Real example
   (signals trimmed):

        {"ts": 1786006437.5, "weights": {"edge_m1": 0.134, "gen_m1": 0.136,
         "mate2": 0.094, "general": 0.157, "judge": 0.076, "occupancy": 0.047,
         "chain": 0.047, "mate1_v2": 0.099, "winset": 0.075, "chainset": 0.080,
         "witness": 0.056},
         "signals": {"chain": {"p": 0.838, "k": 961.6, "sig": 0.249, ...},
                     "witness": {"p": 0.557, "k": 914.0, "sig": 0.117, ...}}}

### The launch bug, so nobody reintroduces it

The launch version of step 3 was `share_c = max(score_c, floor_c ×
importance_c)`. The raw Neyman term (order 0.004 to 0.01) never exceeded any
floor (0.02 to 0.15), so the mixture equaled normalized floor ratios exactly,
for every tick since launch — the "controller reallocates by signal" claim
was false for the first ~30 steps. Floors and Neyman scores are not on
comparable scales; normalize FIRST, then apply floors as minimum fractions.
The fix deployed mid-run with a controller restart; training was untouched.
Within 25 steps the fixed controller performed its first live signal-driven
demotion (chain, as it saturated).

### Why this formula

The formula is Neyman allocation from survey statistics. If the training
objective weights category c by importance_c, then the sample counts that
estimate its gradient with the least variance satisfy:

    n_c  proportional to  importance_c × sigma_c / sqrt(cost_c)

Sigma is estimated empirically — the mean within-prompt standard deviation of
shaped scores — with sqrt(p(1 − p)) as the fallback prior when a category has
too few samples. The empirical estimate matters in two places where a binary
formula reports zero signal: at saturation, where the length-compression
gradient is still alive; and on partial-credit categories (listing,
certificate), where the score is continuous by design. A category the model
always fails or always solves identically carries no gradient. GRPO makes
this literal: a group with all equal rewards has zero advantage.

Two consequences fall out with no extra machinery:

- Saturated categories starve themselves — but only fully. A mastered
  category keeps a share while its answers still compress (the length signal)
  and collapses to its floor only when nothing measurable is left to learn.
  Seen live: occupancy arrived saturated and was floored within one tick.
- Importance and counts stay separate. Importance says what we want. Counts
  say how to estimate it well. A mixture that expresses importance only
  through counts silently changes the objective.

### Why the floors

Neyman allocation is short-sighted. A category at p = 0 has almost no signal
NOW, but it is exactly the frontier we want to push. Without samples, its
first successes can never appear. The floor guarantees a minimum share. When
the first successes appear, sigma rises, and the controller shifts samples in
without human action.

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
the remaining noise floor. Partial-credit categories also depend on this:
their whole point is that a 0.9 answer should be pushed harder toward than a
0.5 answer, which std-normalization would erase.

## 5. How to add and remove categories

### Add a category during a run

This path has run twice in production (occupancy + chain at step 22,
mate1_v2 at step 100). The chain add is the reference case: diagnosis
(judge's sigma near zero) → category built and label-balanced → dropped into
the directory mid-run → controller picked it up on the next tick → learning
visible within 25 steps. The full loop took about 40 minutes.

1. Write `data/curriculum/<cat>.parquet` with the standard row schema. Each
   ground truth must contain `"category": "<cat>"`.
2. Optional: add a config entry with importance and floor. Without one, the
   defaults apply (importance 1.0, floor 0.05).
3. Wait one controller tick (at most 10 minutes). The controller sees the new
   file, gives it the optimistic prior, and writes a weight for it. The
   dataset loads and tokenizes the file at its next refresh.

No restart. No trainer code change.

**The limit: hot-add works only for categories whose reward branch and answer
scaffold already exist.** A new task TYPE (new answer format, new scoring
function) touches `reward_verl.py` and `hex_agent_loop.py`, which the workers
load at startup. Dropping such a category in hot would score it through the
wrong branch. This is why winset/chainset/witness entered at the step-150
RESTART, not as hot-adds. Check the family table in section 2 before
dropping a file.

Two more pre-drop checks, both learned the hard way:

- Label balance. The first chain corpus was 97 percent "Neither"; a label
  count before training caught it. Count labels on any judgment-style
  category before it enters.
- Token ledger. If the answer format is new in ANY way, re-verify:
  think cap + scaffold + answer budget <= response_length (section 8).

### Remove a buggy category during a run

1. Edit `data/curriculum/config.yaml`: set `<cat>: {enabled: false}`.
2. Wait one controller tick. The controller writes weight 0. The dataset
   stops sampling the category at its next weights refresh.
3. Samples already in flight still get scored and trained on. The exposure is
   at most one batch plus one tick. The audit log records the exact time of
   the change, so later analysis can cut at that step.

No restart. The category's file stays on disk for diagnosis. Verified in the
smoke: removal landed within the documented one-tick latency.

## 6. Length economics

See [the task page](the-task.md) for the base mechanisms. Current live values
(retuned at the step-100 restart; launch values in parentheses):

- The length price: a correct answer's reward falls linearly with think
  length, from the full score at an instant answer down to score − 0.4 at the
  cap (`LEN_LAMBDA` 0.4; launch 0.25). Wrong answers stay at −1. The price
  applies to any positive score, so partial-credit answers pay it too. The
  correct/wrong gap (at least 1.6) still dominates the maximum price (0.4),
  so the model never profits from a wrong fast answer.
- The close-token push: without it, every sample sits at the think cap, all
  lengths are equal, and the length price has nothing to compare. The current
  schedule is `192:0,1088:30` — no bias for the first 192 think tokens, then
  a flat +30 logit bias on `</think>` up to the cap. (Launch used a rising
  ramp from token 512; nearly everything still sat at the cap, and a
  decision-point measurement showed move-task answers only settle late in the
  think, so the retune made early closes possible from token 192.)

The bias only helps the sampler explore. The masking keeps the training
mathematics honest: an early close that the bias forced is still a real token
the model produced, and PPO's clipping bounds the small off-policy distortion
on that one token.

Two protections, added after the smoke rounds:

- Deadband: the think length is quantized to 96-character buckets (about 32
  tokens) before pricing. Length differences inside one bucket give exactly
  equal rewards, therefore zero advantage, therefore no gradient. The model
  cannot profit from chasing one-token noise.
- Mean-only advantages (see the GRPO subtlety in section 4): the push on a
  length difference is proportional to the reward difference, instead of the
  flat ±1 that stock GRPO produces.

Reading at step 125, after the retune: mean response length 954 tokens (off
the cap), minimum 236, while edge conversion rose in the same window — the
feared tension between the length price and CoT-borne skill (section 7) is so
far resolving favorably: the suffix gets trimmed, the tracing survives.

## 7. What we measure, and the registered predictions

Fixed evaluation sets, unchanged across the arm: 277 general positions plus 60
edge one-move wins, all excluded from training. Calibrated noise floor at
k=1 sampling: about ±0.04 on general, ±0.09 on edge — single-checkpoint edge
swings below ~0.09 are not signal.

Where the curves live (wandb project `hex-rl-cot-deconfusion`):

- The training run logs validation metrics split per category automatically,
  because each parquet row's `data_source` is `hex_<category>`. Look for
  `val-core/hex_edge_m1/...`, `val-core/hex_judge/...`, and so on.
- The controller logs to a companion run named `<exp>-controller`: weights,
  per-category p, sigma, and token cost EMAs, under `mix/*`. The field
  `mix/train_step` carries the trainer's step so both runs share an x-axis.
- Every scored sample lands in `results/rollouts/<exp>.jsonl` with its
  category, kind, raw and shaped scores, link_frac where applicable, and the
  full reasoning text.

### Headline result so far (step 100): the edge skill is CoT-borne

A no-think ablation at step 100 (same val positions, temperature 0.6):
general 0.334 without CoT versus 0.300 with — still decorative, replicating
arm A. Edge: 0.008 without CoT versus 0.133 with — a 16x collapse. The
curriculum-targeted skill is the project's first load-bearing CoT, and it
runs through the reasoning channel rather than the policy head. Follow-ups
(CoT reads, activation probes, per-checkpoint no-think curves) are queued in
the log.

### Registered predictions and their status

From before launch:

| prediction | odds | status |
|---|---|---|
| Edge conversion > 0.8 by step 150 (t 0.6) | 60% | MISSED — 0.217 at step 125, flat-to-slow |
| Judge success > 0.9 by step 100 | 70% | MISSED — judge FELL to ~0.36 by step 75 (negative-transfer candidate: pair-tracing may interfere with whole-board judgment) |
| Judge natural closes > 20% once judge acc > 0.8 | 55% | precondition never met |
| General val win >= 0.55 by step 400 | 40% | open; monotone climb through step 150 |
| Judge think p50 < 600 tokens by step 200 | 60% | open |

From the mid-run additions:

| prediction | odds | status |
|---|---|---|
| Atomic within-group sigma > judge's | 70% | HIT for chain (0.78 vs 0.065); occupancy missed via saturation, not failure |
| Atomic p > 0.9 within 100 steps of add | 55% | chain hit the trajectory (0.88 in ~25 steps) |
| Mean response length < 700 by step 200 | 55% | open (954 at 125) |
| winset mean score > 0.5 within 100 steps of entry | 55% | open |
| witness link_frac > 0.7 within 100 steps | 55% | open (baseline 0.44 post-fix) |
| SFT branch: link_frac > 0.9 within 20 steps | 75% | open — branch armed at step 250 |
| SFT branch beats pure-RL on judge transfer within 50 steps | 55% | open |

The standing questions behind these: does trained tracing skill COMPOSE into
judgment and edge conversion (curriculum thesis), or stay task-shaped
(skill-level selection)? Either answer is C1-relevant. And from step 250:
is RL or SFT-then-RL the right teacher when a gold certificate exists?

## 8. Known limits and risks

- The controller has no explicit rate limit on weight changes. The EMA on its
  input signals smooths most jumps. A pathological signal swing could still
  move the mixture fast. The audit log makes this visible.
- Label sources vary by family: board logic for judgment, listing-chain and
  certificate checks; the exact solver for winning sets. A future category
  with a new label source needs its own validation before it enters the
  directory. The remove path exists exactly because this can go wrong.
- Budget arithmetic bites at the boundary — three separate incidents (the
  2160-vs-2176 response length, the listing answer budget, the think cap
  that clipped witness answers to ~12 tokens and silently FN-truncated
  listing answers behind the fallback parser). Rule: any new answer format
  means re-checking the full token ledger end to end, and reading raw
  response tails, before trusting scores.
- The Neyman formula treats gradient variance as scalar and ignores gradient
  direction conflicts between categories. If two categories pull the policy
  in opposite directions, the controller does not see it. Per-category
  evaluation curves are the backstop — and judge's decline while chain rose
  (step 75) looks like exactly this failure mode, under investigation.
- Hot-add covers new categories, not new task types: a new reward branch or
  answer scaffold requires a restart (section 5).
- The importance numbers in config.yaml are human judgment. The controller
  optimizes estimation efficiency GIVEN them. It cannot tell us what to want.
