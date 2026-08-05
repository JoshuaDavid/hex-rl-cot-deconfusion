# Arm C — adaptive curriculum

## Status

In smoke test. No long run has started yet.

## The question

Arm A showed that vanilla RL does not create the missing skill. Arm C asks:
does RL create the skill when a controller aims the training distribution at
it? Any new vocabulary or concept in arm C carries a caveat: we injected the
task categories, so emergence claims are weaker than in arm A.

## The parts

### 1. Task categories are files

Each task category is one parquet file in `data/curriculum/`:

    judge.parquet     900 rows   "who has already won?" questions
    edge_m1.parquet  2240 rows   one-move wins that finish ON an edge
    gen_m1.parquet   1368 rows   one-move wins, any location
    mate2.parquet    1200 rows   two-move wins
    general.parquet  5336 rows   random winnable positions

To ADD a category during a run: write a new parquet file into the directory.
The dataset finds the file at the next weight refresh and loads it. No restart
is necessary.

To REMOVE a category during a run (for example, you find a labeling bug):
set `enabled: false` for that category in `data/curriculum/config.yaml`. The
controller sets its weight to 0 at the next tick. The dataset then never
samples it again. No restart is necessary.

### 2. The controller picks the mixture

The controller (`scripts/curriculum_controller.py`) runs beside training. Every
10 minutes it:

1. Reads the human file `config.yaml`. This file gives each category an
   importance number and a floor.
2. Reads the recent training samples. It computes each category's success rate
   p and mean token cost k.
3. Computes each category's share of the batch:
   share = max(importance × sqrt(p × (1 − p)) / sqrt(k), floor × importance).
4. Writes `weights.json`. The dataset reloads this file when it changes.

The sqrt(p × (1 − p)) term is the useful-signal estimate. A category the model
always fails (p near 0) or always solves (p near 1) has little signal. The
floor keeps frontier categories alive: a category at p = 0 still gets samples,
so its first successes can appear.

Worked example with real numbers. Suppose importance is equal (1.0) for two
categories, and costs are equal. The judge category has p = 0.47, so
sqrt(p(1−p)) = 0.50. The edge category has p = 0.03, so sqrt(p(1−p)) = 0.17.
The raw shares are 0.50 and 0.17. The edge floor then lifts the edge share.
When edge successes grow, its signal term grows, and the controller shifts
samples to it without human action.

### 3. Audit trail

Every controller decision appends one line to `results/curriculum_log.jsonl`,
with the weights and the signals behind them. Example line:

    {"ts": 1785965831.4, "weights": {"edge_m1": 0.38, "gen_m1": 0.20,
     "mate2": 0.09, "general": 0.25, "judge": 0.08}, "signals": {}}

### 4. Length economics

Arm C prices reasoning length. See [the task page](the-task.md) for the two
mechanisms and their numbers.

## The smoke test

The current smoke test trains for 10 steps. At step 4, a script disables the
mate2 category and raises the judge importance from 0.6 to 3.0. The test
passes if the sample mixture shifts and mate2 samples stop, with no restart
and no crash.
