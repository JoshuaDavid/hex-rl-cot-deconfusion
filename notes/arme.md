# Arm E — can RL learn to SELECT a useful auxiliary task?

Question (Joshua's design): show a board + several perception tasks; the model
picks ONE auxiliary task to solve first (a "warm-up", not scored), then answers
an EVALUATED task. Can RL on the single selection token learn to pick the
auxiliary task whose solution best helps the evaluated one — the sharpest form
of "RL selects" (a token with instrumental meaning, vs the task-neutral marker)?

## Answer (one line)

RL reliably moves the selection token to the reward-maximizing choice in a few
steps; that coincides with the *instrumentally useful* helper **only when
usefulness is isolated from the model's ability to generate the helper**.

## The tasks (hexenv/arme.py, all exact from the board via BFS)

- A all stones+colors · B winner+path · C empty cells · D edge-connectivity
  (black_top/bottom, white_left/right) · E connected-to-neither · **W winner**
- Env-config: `ARME_EVAL` (evaluated task), `ARME_HELPERS` (selectable set),
  `ARME_FILL` (sparse/dense pool). Useful helper per eval in `USEFUL_HELPER`.

## What worked / didn't, and why (the real findings)

1. **Board-derivable evaluated tasks give NO differential.** For C (empties) and
   E (connected-to-neither), the SFT'd model re-derives the answer from the
   fully-visible board and *ignores helper content* (gold-A ctx and gold-D ctx
   produce the identical answer). Gold-helper differential ~0 (even slightly
   negative). Lesson: the evaluated task must be beyond direct board-reading.
2. **The useful evaluated task is W (winner).** Winner needs global connectivity;
   helper **D** supplies the edge-connected sets so the winner is an intersection
   check. Gold-D drives winner 0.78→0.99; A/C (no connectivity) don't. This is a
   genuine instrumental differential.
3. **SFT must put gradient on BOTH tasks + solo** (Joshua's fix; build_arme_winner.py).
   Grading only the evaluated task leaves helper-generation untrained and solo
   out-of-distribution — the confound behind garbage own-helpers.
4. **Selection prior must be explorable.** With X always teacher-forced through
   SFT, the untrained prior was degenerate (P(D)=100% by accident). Train the X
   token with X~uniform (the marker 50/50 analog) → uniform prior.
5. **Generation temperature.** The useful helper D is the HARDEST to generate; at
   rollout temp 1.0 its generation derails the answer and erases its edge.
   Decouple: sample the SELECTION at rollout temp (explore), generate helper+eval
   GREEDY (`ARME_GEN_TEMP=0`).
6. **The usefulness/generatability tension.** D is useful only when correct, but D
   is the hardest to generate — so in the OWN-helper regime D's edge is erased.
   To isolate usefulness, teacher-force the gold helper (`ARME_GOLD_HELPER=1`).

## R4 result (hexenv/hex_select_loop.py, only the selection token is trained)

- **Gold-helper (usefulness isolated):** reward D:+1.0 vs C/A 0.5–0.9. P(select=D)
  0.35 → 0.98 by step ~5, winner reward intact through step ~7. RL SELECTS THE
  USEFUL HELPER. (After ~step 8, over-optimization: only the selection token is
  trained/KL-anchored, so shared-LoRA drift breaks winner gen — use lr 1–2e-6 and
  stop by ~step 10, or KL on all response tokens.)
- **Own-helper (usefulness NOT isolated):** per-helper reward ~tied (C≥D>A). RL
  drops the worst (A: .22→.03), mildly favors the EASY DECOY C (.41→.56), D flat
  (~.4). RL selects C, not D — because self-generated D is unreliable.

## Reproduce

    ARME_EVAL=W ARME_HELPERS=A,C,D python scripts/build_arme_pool.py   # ARME_FILL=dense ARME_N=7
    ARME_EVAL=W ARME_HELPERS=A,C,D python scripts/build_arme_winner.py # SFT data (grad on both+solo, uniform-X)
    bash scripts/run_armD_sft.sh data.train_files=data/arme/win_train.parquet ... trainer.total_epochs=2
    python scripts/export_armD_adapter.py checkpoints/arme_win/global_step_N checkpoints/arme_win/adapter
    python scripts/merge_adapter.py checkpoints/arme_win/adapter checkpoints/arme_win/hf_merged
    python scripts/build_arme_r4_data.py                               # RL prompts
    ARME_GOLD_HELPER=1 bash scripts/run_arme_r4.sh arme_r4_gold checkpoints/arme_win/hf_merged 40
    python scripts/eval_arme.py --mode {solo,select_tf,select_own} --lora <adapter>   # differentials
    python scripts/arme_select_trends.py results/rollouts/arme_r4_gold.jsonl --per-step 128

## Throughline

RL redistributes probability toward reward — even for an instrumentally-meaningful
token — but it neither creates the usefulness nor perceives human intent. "Select
the useful task" reduces to "make the useful action the rewarded one."
