# YOLO-run recipe (living document)

The distilled best-known-design for the eventual clean run on a bigger machine.
Arm C is the intuition-farm; this file is what it farms. Every entry cites the
finding that earned it (see RESEARCH_LOG.md dates).

## Generation scheme
- Forced-close two-phase: think capped, inject `</think>\n\n<AnswerWord>:`,
  short answer phase. Non-negotiable for Qwen3-scale models (never terminate
  on open-ended tasks; 08-05). The branch point it creates is load-bearing
  for credit assignment (see answer-branching).
- Close-bias: +30 logit bias on `</think>` from ~token 192 (weaker biases
  provably inert: P(close) ~1e-13 baseline; 08-05/08-06). Bias-sampled closes
  stay mask-1; PPO clipping absorbs the one-token distortion.
- Think budget: quality flat 2048->3072 (08-05); 1088 adequate and 2x cheaper.
  ~89% of step time is think tokens.
- Task-aware answer budgets AND task-aware think caps (think = resp_len -
  answer_budget - 8). Every new answer format: audit the token ledger
  end-to-end (three silent-clipping bugs; 08-06).
- PLANNED, high-confidence: answer-branching — n=8 answers per think in one
  request (shared KV, +10-15%), think reward = mean (Rao-Blackwellized;
  transcription noise is 8-44% of answer variance; 08-06).

## Reward design
- Correctness-gated length price (correct: 1 - lambda*len_frac; wrong: -1),
  lambda 0.4, with ~32-token deadband buckets (kills noise-chasing; 08-06).
- MEAN-ONLY advantages (norm_adv_by_std_in_grpo=False). Std-norm pushes
  1-token and 990-token gaps identically at saturation and breaks the
  controller's Neyman premise (08-06).
- Dense objectives wherever labels permit: set-valued listing (score =
  (TP-FP)/|truth| clipped) and certificate tasks (per-link partial credit).
  ~10-40 graded assertions/rollout vs 1 bit; steepest curves of the project
  (witness: link_frac 0.34->0.44 + perfect 0.5%->9% in ~15 steps; 08-06).
- Answer formats: JSON arrays/objects ONLY (measured strict compliance: JSON
  1.00 vs comma-lists 0.25; 08-06). JSON-first parse, lenient fallback.
- Grammatical answer phrasing ("Which player, if any..." not yes/no questions
  with noun answers). Wording ambiguity manufactured a fake capability cliff
  ("touch the edge"; 08-06).

## Curriculum architecture
- Dynamic mixture via custom dataset (data.custom_cls) + file-based weights;
  categories = per-file parquets; hot add/remove mid-run works in production.
  MUST delegate val split to stock dataset (08-06).
- Controller: shares ∝ importance x sigma_emp / sqrt(cost), normalized FIRST,
  then floors as minimum fractions (scale bug otherwise: floors dominate
  silently; 08-06). sigma_emp = mean within-prompt std of shaped scores
  (sees length-signal at saturation; binary formula as prior only).
- Category design lesson: SPECIFICITY manufactures gradient. "Is there a
  winning chain?" -> deterministic guessing, sigma 0.05, inert. "Are these
  two stones connected?" -> sigma 0.78, learned to 0.88 in ~25 steps (08-06).
- Composition doesn't emerge from adjacent skills; certificate tasks that
  REQUIRE composition (winner+path) teach it directly.
- Teaching-channel division: SFT for procedures with constructible gold
  certificates (the certificate IS the reasoning artifact); RL for decisions
  without demonstrations; RL-after-SFT to consolidate. (Branch experiment
  results pending.)
- BRANCH VERDICT (08-06, four legs + SFT 2x2). Task-pure SFT destroys the
  rest of the policy and RL at lr 1e-6 repairs ~nothing in 50 steps. The 2x2
  (cert-target x mix): witness 0.96 ablated-pure / 0.58 narrated-pure /
  0.57 ablated+replay / 0.25 narrated+replay — verbalized teaching and
  co-training each tax the skill ~0.6x. WORKING RECIPE: answer-only (ablated)
  SFT targets + ~40% correct self-sample replay preserves full breadth
  (val_general 0.404 vs 0.426 baseline) and installs the skill at 0.57.
  RL-after (50 steps) then matches pure-RL accuracy everywhere at ~30x fewer
  think tokens on the SFT'd task (witness 0.61 @ 28 tok vs @ ~900 tok).
  SFT = cheap skill installation, NOT acceleration past the RL asymptote.
- Discrimination is the hard kernel: witness AND judge ceiling at ~0.6 in
  every leg — verification/certificates are nearly free to teach; decisions
  are what all channels grind on. Budget the YOLO run accordingly (judge-type
  gradient early and heavily).
- Replay sampling: use temp 1.0 (or mixed temps), and length-filter — temp-0.6
  self-distillation baked in entropy 0.05 + verbose board re-parsing that
  overran the think budget (transient chain 0.88→0.38 dip; RL self-healed).
  Entropy 0.05 slowed but did not kill RL.
- SFT TRAP: Qwen3's chat template strips <think> from assistant turns, so
  per-turn SFT datasets (verl MultiTurnSFTDataset) silently train answer-only
  targets — accidental think-ablation that RL cannot undo (exploration in
  think-space dies; other skills decay while the SFT'd mapping climbs; 08-06).
  Rule: before ANY SFT launch, decode input_ids[loss_mask] and eyeball the
  think block. Use a raw-assistant-turn dataset (hexenv/sft_cert_dataset.py).
- Ablation datum from the accident: 2 epochs of answer-only SFT → 1.7B does
  path-tracing certificates with ZERO CoT at 0.83 (procedure internalizes
  into the forward pass). Verbalization unnecessary for this skill class.

## Known-good config (1.7B / A100-40GB reference)
- GRPO, batch 32 prompts x 8 rollouts, lr 1e-6, kl_loss beta 1e-3 low_var_kl,
  entropy 0, temp 1.0 rollouts / 0.6 val.
- verl 0.9.0-dev quirks: TransferQueue manual install; agent.num_workers=1
  etc. for pids.max; val_batch_size capped (zmq spikes); logprob stitching +
  min/max_global_steps merging in custom loops. See notes/verl_setup.md.
- ~135-150s/step at think-1088. Checkpoint/50 + janitor (21GB full ckpts).

## Instrumentation checklist (non-negotiable for the big run)
- Reward side channel: every scored rollout w/ full CoT, category, raw+shaped.
- Per-category val slices via data_source naming (auto wandb splits).
- Controller audit log + wandb companion run (stable run id!).
- Same-checkpoint double-val across restarts = free noise-floor calibration.
- Standing probes: no-think ablation per checkpoint (load-bearing curve);
  fixed-think answer-agreement (channel fidelity); decision-point probe.
- Read raw samples at every phase gate. Slice metrics, never aggregates.

## Open questions the YOLO run should settle
- Does dense+certificate curriculum reach strong general play? (Transfer from
  puzzles to move quality was slow/absent through step ~250.)
- SFT-vs-RL branch outcome (pending at ckpt 250/300).
- Does compression coexist with load-bearing CoT at scale? (So far: yes.)
- Scale: does 4B/8B escape the composition wall that 1.7B hits?
