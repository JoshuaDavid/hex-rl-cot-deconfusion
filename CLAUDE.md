# Hex RL CoT deconfusion — project guide

## What this is

RL (GRPO) on Qwen3-1.7B playing hex, with exact solver ground truth, to answer four
confusions about what RL does to a reasoning model: C1 creation-vs-selection,
C2 vocabulary emergence, C3 concept/verbalization coupling, C4 cheap KL preservation.
Full agenda: `RESEARCH_AGENDA.md`. Decision/surprise trail: `RESEARCH_LOG.md`
(append-only — read the tail before doing anything).

**The deliverable is deconfusion, not a publishable artifact.** Single seed, honest
nulls, "what happened in this run." Optimize for information per GPU-hour and for
eyes-on-data, not polish.

## Working style (binding conventions)

- Protocol loop: boldest surviving hypothesis → cheapest falsifier (<5 min GPU: just
  run it) → read samples → append to RESEARCH_LOG.md. Register predictions with odds
  *before* runs; grade them after.
- **Never launch a multi-hour run without reading ≥3 concrete samples** of what goes
  in and one full trajectory of what comes out.
- Append major decisions/surprises to RESEARCH_LOG.md (append-only, timestamped).
- Commit early and often; atomic commits; push to origin.
- notes/ = durable how-to docs (benzene_usage, verl_setup, design_training,
  design_probes). Update them when reality diverges.
- Big artifacts (checkpoints, rollouts) are gitignored; they back up to B2 via
  rclone remote `b2hex:claude-code-backups/hex-rl-cot-deconfusion/` (2h loop).
- wandb: project `hex-rl-cot-deconfusion`.

## Where everything is

- `hexenv/` — board/rules/render (`board.py`, `render.py`), prompts (`prompts.py`),
  solver oracle (`solver.py` — **use `exact_winning_moves`, never
  `dfpn-solver-find-winning` directly**; see RESEARCH_LOG 2026-08-05), verl reward
  (`reward_verl.py`), custom forced-close AgentLoop (`hex_agent_loop.py` +
  `agent_loops.yaml`), offline forced-close generation (`forced_close_gen.py`).
- `scripts/` — corpus building (`build_corpus_parallel.py`), dataset prep
  (`make_verl_dataset.py`), training (`run_pilot.sh`), checkpoint ops
  (`ckpt_janitor.sh`, `merge_and_eval.sh`), evals (`eval_checkpoint.py`,
  `eval_offdomain.py`, `passk.py`), analysis (`mine_vocab.py`,
  `collect_activations.py`, `train_probes.py`, `rollout_trends.py`).
- `data/` — solver-labeled corpora `corpus_{5x5,6x6,7x7,8x8}*.jsonl` (exact winning
  sets), verl parquet in `data/verl_hex/`, probe set `probe_positions.jsonl`.
- `results/` — phase1 evals, checkpoint evals, rollout side-channel
  (`results/rollouts/<exp>.jsonl`: every scored training sample with full CoT).
- `checkpoints/<exp>/global_step_N/{actor,hf}` — hf/ = merged model for
  vllm/analysis; actor/ = resume state (janitor prunes old ones).
- `benzene-vanilla-cmake/build/src/mohex/mohex` — the solver binary (GTP).
- Envs: `/venv/verl` (training + vllm evals), `/venv/main` (general python).
  Invoke by absolute path; session PATH is unreliable.

## Key facts agents keep tripping on

- Solver: benzene `find-winning` is NOT exact (ICE fill-in / consider-set). All
  labels come from exhaustive per-child `solve-state` (`exact_winning_moves`).
- Qwen3-1.7B ~never terminates thinking on move-choice prompts → all rollout/eval
  generation uses the forced-close two-phase scheme (cap think, inject
  `</think>\n\nMove:`, ~8 answer tokens). Evals must match training's scheme.
- Container limits: pids.max=2816 (Ray+vllm+workers nearly exhaust it — keep
  worker counts at the values in run_pilot.sh), one GPU (never two vllm engines).
- pypi via fastly is ~broken; use `--index-url https://repo.huaweicloud.com/repository/pypi/simple`.
- pkill -f can match your own shell's command line; use `pkill -x` or bracket
  patterns.
- Secrets in `.env` (gitignored). GPU may be held by a concurrent eval/training
  process — check `nvidia-smi` before starting GPU work.
