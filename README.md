# Hex RL CoT Deconfusion

> **Who "I" is:** this repository is an autoresearch project. Claude
> Fable 5 (an Anthropic model) wrote this README, the code, the analysis,
> and the research log, and made nearly every methodological decision —
> the project runs almost entirely on Claude's taste. "I" below means
> Claude Fable 5, not Joshua David. Joshua owns the repo, rented the GPU,
> seeded the agenda, and drops in occasionally with course corrections.

## The question

When RL makes a reasoning model better at a task, what actually changed?
I train Qwen3-1.7B with GRPO to play hex, with an exact game solver as
the reward, and use the checkpoint series to attack four confusions:

- **C1 — creation vs. selection.** Do new capabilities appear, or does RL
  just sharpen what the base model could already sample?
- **C2 — vocabulary.** Does new notation emerge in the chain of thought?
  Can earlier checkpoints understand it?
- **C3 — concept vs. verbalization.** When a concept becomes probe-detectable
  in the residual stream, does the CoT talk about it?
- **C4 — cheap preservation.** Does a small fixed KL penalty actually
  protect off-domain behavior?

Hex is the right substrate because Qwen3 knows essentially no applicable
hex theory (I checked — its answers to a theory quiz confabulate their
justifications; e.g. it asserts a second-player win for plain hex via a
"blocking strategy", with no strategy-stealing or swap-rule argument
attached), the game is exactly solved at these board sizes, and hex
concepts like bridges transfer across board sizes while memorized moves
do not.

## What this is not

A paper. The deliverable is "I am less confused, with evidence I would
bet on." One seed, honest nulls, no metric tuning. The full decision and
surprise trail — including everything that went wrong — is in
`RESEARCH_LOG.md` (append-only, timestamped).

## The task, concretely

The model sees a rendered board and must answer with a move:

       a b c d e
     1 . . . . .  1
      2 . . W W .  2
       3 . B B . .  3
        4 . . . . .  4
         5 . . . . .  5
           a b c d e

Reward: +1 if the move preserves a solver-verified win, −1 if it throws
the win away or targets an occupied cell. Positions are pre-labeled with
their *exact* winning-move sets, so training needs no solver in the loop.

## Findings so far (mid-run, step ~150 of 750)

- Win-preserving move rate on held-out positions: **0.16 → 0.46**.
  Illegal-move rate: 0.24 → 0.09.
- **The CoT is not load-bearing.** At step 100 the model scores the same
  with its reasoning suppressed (0.34 no-think vs 0.32 with-CoT). The
  improvement lives somewhere other than the visible reasoning — the
  direction my C3 prediction pointed.
- **No new vocabulary yet.** CoT drift so far is reweighting of existing
  English: much more blocking/prevention talk ("blocking moves", "while
  preventing"), which tracks what the reward actually pays for.
- Two hard-won infrastructure facts, both discovered by reading raw data:
  benzene's `dfpn-solver-find-winning` is **not** an exact oracle (use
  `HexSolver.exact_winning_moves`), and Qwen3-1.7B essentially **never
  finishes thinking** on open-ended move choice (100% truncation at 6k
  tokens), so all generation uses a forced-close scheme: cap the think
  phase at 2160 tokens, inject `</think>\n\nMove:`, let it answer.

## Methods

### Task framing

- Each training example is a single position: rules + ASCII board + which
  color to play. The model outputs one move.
- Single-turn, not full games. Per-move exact rewards remove the
  credit-assignment problem, and every GRPO group is eight attempts at
  the same decision.

### Data

- Positions come from random playouts: uniform legal moves, truncated at
  scheduled stone counts (odd and even, so both colors appear to move).
- Labels are exact winning-move sets from benzene — but not from its
  `find-winning` command. That command returns `[]` on positions its VC
  engine has already decided, and omits winning-but-dominated moves.
  Instead I solve every child position individually, with a parent/child
  consistency assert on each position.
- Training keeps a position only if both hold:
  - The player to move is winning (a +1 exists to find).
  - At least one legal move throws the win away. If every move wins, all
    rewards are +1 and the GRPO advantage is identically zero.
- Sizes: 5,280 training positions (5×5–7×7), 277 held out, 164 at 8×8
  reserved for size-transfer tests. 8×8 never appears in training.
- Winning sets are precomputed, so the train-time reward is a set lookup.
  No solver runs in the training loop.

### Forced-close generation

- Qwen3-1.7B essentially never ends its think phase on open-ended move
  choice (100% truncation at 3584 tokens; 93% at 6144; brevity
  instructions ignored). All generation therefore uses two phases:
  1. Think, stopping at `</think>` or at 2160 tokens.
  2. Inject `</think>\n\nMove:`, then let the model emit ≤8 answer tokens.
- Injected tokens are loss-masked to 0; model tokens are 1. A
  model-emitted `</think>` counts as a model token.
- The 2160 cap comes from a sweep: forced-close quality was flat between
  2048 and 3072, so I took 2048 plus margin, minus 16 tokens of
  scaffold/answer bookkeeping.
- Training and every evaluation (including base-model baselines) use the
  same procedure and the same cap, so all comparisons are budget-matched.

### Training configuration

- GRPO in verl (0.9.0-dev), vllm rollout server, one A100-40GB.
- Per step: 32 positions × 8 rollouts, temperature 1.0.
- lr 1e-6; KL-to-reference loss β = 1e-3 (the "halfhearted preservation"
  condition C4 tests; a β = 0 control with the same data order is
  planned); entropy bonus 0; stock advantage normalization.
- Reward: +1 iff the move is in the exact winning set; −1 otherwise
  (win-throwing, occupied cell, or unparseable output).
- Checkpoints every 50 steps, each merged to a standalone HF model.
  Target: 750 steps.

### Evaluation and analysis

- A fixed held-out set is re-evaluated at every checkpoint (forced-close,
  temp 0.6): win rate, legality, natural-close rate, full CoTs.
- Every scored training sample is logged with its complete CoT. This side
  channel feeds the C2 and C3 analyses.
- C1: base-model pass@k for k ≤ 1024 on the same positions, against
  trained pass@1. Same generation scheme on both sides.
- C2: prompt-matched n-gram enrichment between checkpoints, with
  within-position sample-splits as the null calibration.
- C3: logistic probes on residual-stream activations at fixed positions.
  Probes are validated first on features that certainly exist (stone
  locations) before being trusted on concepts (bridges, win status).
  Probe results cross a 2×2 with graded CoT verbalization.
- C4: trimmed GSM8K / MMLU / IFEval plus KL-to-base on non-hex prompts,
  across both arms.

## Layout

| Path | Contents |
|---|---|
| [`docs/`](docs/README.md) | plain-language field guide: what each experimental arm does, with examples |
| `hexenv/` | board/rules/render, solver oracle, reward fn, verl forced-close agent loop |
| `scripts/` | corpus builders, training (`run_pilot.sh`), evals, analysis (pass@k, vocab mining, probes) |
| `data/` | solver-labeled corpora (exact winning sets), verl parquet, probe positions |
| `results/` | eval outputs + every scored training sample with full CoT |
| `notes/` | durable how-tos: benzene usage, verl setup, training/probe design |
| `RESEARCH_LOG.md` | the actual research record — read this first |

## Running things

```bash
# label a corpus (exact per-child solves, parallel)
/venv/main/bin/python scripts/build_corpus_parallel.py \
    --size 7 --n 100 --stones 4,6,8 --workers 20 --seed 1 \
    --out data/my_corpus.jsonl

# train (GRPO via verl, forced-close rollouts, wandb logging)
STEPS=750 EXP_NAME=my_run bash scripts/run_pilot.sh

# merge + evaluate a checkpoint
bash scripts/merge_and_eval.sh my_run 150
```

Artifacts back up to Backblaze (`b2hex:claude-code-backups/hex-rl-cot-deconfusion/`);
training curves live in the `hex-rl-cot-deconfusion` wandb project.
