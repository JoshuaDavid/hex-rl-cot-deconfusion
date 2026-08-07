# Arm D phase 2: does CoT buy exact long-path tracing?

## Question

No-think LoRA (armD2 ep8) tops out on exact-path perfection: 42.8% on
test_longpath, smoothly falling 78% (plen 14-15) -> 0% (plen 31+), failure
mode = visited-cell loops. Does letting the model think — with a CoT that is
ITS OWN, never teacher-scripted — beat that curve?

## Binding constraint (user, 2026-08-07)

No content injected into the CoT. The only permissible injection is the
forced `</think>` close (existing two-phase scheme). Rationale: arm C showed
scripted narrations train format-without-computation (witness 0.58 pure /
0.25 mixed vs 0.96 for answer-only targets); a hand-written scratchpad would
also make the "CoT helps" conclusion circular — we'd be testing our script,
not the model's thought.

## Design: three arms on one yardstick

Yardstick: frac_perfect per plen bin on data/armD2/test_longpath.parquet
(500 boards, bins 14-17/18-21/22-25/26+; no-think ep8 baseline already
logged at 42.8%).

- **Arm N (done)**: armD2_sft_weighted ep8, no think. 42.8%.
- **Arm T (the treatment)**: r32 LoRA trained on (board, own-CoT, answer)
  triples harvested by rejection sampling ("STaR without hints"):
  1. Sampler model: armD2 ep8 adapter with enable_thinking=True (it knows
     task + answer format; its think content is untrained => genuinely its
     own). Fallback sampler if pilot shows garbage: base instruct model.
  2. Sample k=8 at temp 1.0, think budget 1024, forced close, on fresh
     constructive boards stratified over plen 8-32 (train distribution
     reaches past the yardstick bins on both sides; boards disjoint from
     all eval sets).
  3. Keep only samples the grader scores 1.0. Dedupe per board (keep <=2
     CoTs/board). NO editing of the kept CoTs — verbatim targets,
     including `<think>` content.
  4. SFT: same recipe as armD2 (r32, lr 1e-4, weighted answer tokens;
     think tokens get uniform weight 1.0 — importance weights only apply
     where the grader defines counterfactuals).
  5. Optional round 2 (STaR bootstrap): resample from the round-1 adapter
     on boards it now solves, retrain. Only if round 1 shows a gain.
- **Arm C (data-matched control)**: no-think LoRA trained on the SAME
  boards arm T harvested (gold answer-only targets), same epochs. Kills
  the confound "arm T won because it saw more long-path boards", isolating
  the CoT channel itself.

## Matched-compute accounting

Report three budget lines per arm: harvest GPU-min, train GPU-min, eval
think-tokens/board. N and C pay no harvest and ~zero think tokens; T pays
both. Primary comparison at matched TRAIN compute; harvest cost reported
honestly as the price of the CoT channel (it is the point of the method,
not an overhead to hide).

## Pilot before any harvest (protocol rule: cheapest falsifier first)

<5 GPU-min: 20 boards x k=8 from (a) base + think, (b) ep8 adapter + think.
Measure pass@8 per plen bin; read >=3 full CoTs per sampler with own eyes.
- If ep8+think pass@8 ~ 0 on plen>=14: no bootstrap fuel at the frontier;
  fall back to harvesting on plen 8-13 (where pass@8 should be high) and
  test whether trained CoT GENERALIZES upward to the yardstick — arguably
  the more interesting result if it works.
- If CoTs are degenerate (empty/loops/instant close): note it, consider
  temp/budget sweeps before declaring the channel dead.

## Predictions (register with odds before the pilot)

- ep8+think pass@8 > 0.3 on plen 14-17 @55%
- base+think pass@8 ~ 0 everywhere (never terminates / wrong format) @70%
- Arm T beats arm C on yardstick overall frac_perfect by >= 10 points @45%
- Arm T's gain concentrates in plen>=22 (where N is near 0) @50%

## Eval protocol

Both arms temp 0 on test_longpath. Arm T: enable_thinking=True, think
budget 1024, forced close (the one permitted injection), then free
generation of "Answer: {...}". Report think-token distribution alongside
frac_perfect — the compute-per-point-of-accuracy curve is the deliverable.
