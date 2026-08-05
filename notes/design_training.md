# Training design (Phase 2 pilot -> Phase 3 main)

## Task framing
Single-turn: prompt = rules + ASCII board + "you are {color}" (hexenv/prompts.py move_prompt),
response = CoT + "Move: <cell>". Reward (hexenv/reward.py compute_score):
+1 move in precomputed exact winning set; -1 otherwise (legal-losing, illegal, unparseable).
Positions: only winning-for-mover, from data/corpus_{5,6,7}x{...}.jsonl via scripts/make_verl_dataset.py.

## Pilot (Phase 2, 0.6B plumbing then 1.7B, ~15h budget)
- verl GRPO, single A100-40GB, vllm rollout.
- rollout.n = 8, train_batch (positions/step) = 32 pilot / 64 main.
- max_prompt_length 640, max_response_length 1280 (watch truncation rate! Phase 1 shows
  1.7B thinks long; if >10% truncated at 1280, raise to 2048 and cut batch).
- temperature 1.0 rollouts.
- lr 1e-6 (actor), no critic (GRPO), kl_loss per arm (see below).
- Save rollouts every step to disk (needed for C2/C3 analysis!) - verl rollout log or
  custom reward-fn side channel (append (prompt_id, response, reward) to jsonl).
- Checkpoint every 100 steps HF-format model-only if possible (disk: 3.4GB x ~10 per arm).
- Gate: reward slope clearly positive within 50-100 steps; eyeball 10 trajectories at
  step 0/50/100 for format hacking (e.g. move-only responses, degenerate CoT, repeated
  single opening move).

## Arms (Phase 3)
- A (main): KL arm, beta ~1e-3 (verl actor.kl_loss_coef, kl_loss_type low_var_kl), ~1000 steps.
- B (control): beta=0, same seeds/data order, ~300-500 steps (budget).
- Both: same corpus mix 5x5:6x6:7x7 roughly balanced by winning-position counts.

## Eval per checkpoint (standing monitor)
- Held-out val positions (val.parquet): win-preserving rate @ temp 0.6 and @ greedy, legal rate.
- Store full CoTs per checkpoint for C2/C3.
- Off-domain (C4): 200 fixed prompts (GSM8K subset + IFEval subset + tulu-ish chat prompts):
  KL/logprob drift vs base + accuracy where gradeable. Run less often (every 200 steps).

## Analysis hooks
- C1 pass@k: base model k in {8,64,256,1024} on fixed 200-position eval set vs RL'd pass@1.
  Needs vllm. Same winning-set scoring everywhere.
- C2: n-gram/phrase mining late-vs-early CoTs on same positions; ref-perplexity spikes.
- C3: probes on residual stream; labels from board state (stones, connectivity) first,
  then concepts (bridge presence = 2 same-color stones w/ exactly-2 common empty neighbors
  pattern; edge-template presence; benzene VC facts as richer labels). 2x2 vs verbalization
  (grader + hand spot-checks).
- C4: trimmed MMLU/GSM8K/IFEval + CoT style metrics on non-hex tasks, both arms.

## Known risks
- Truncation: thinking budget vs batch memory. Measure in pilot.
- Reward hacking surface is tiny (exact solver labels) but format hacking possible:
  e.g. emitting many "Move:" lines (extract_move takes the LAST); emitting nothing.
- Distribution shift: corpus is random-playout positions; policy self-play positions differ.
  Planned: after pilot, refresh corpus with positions from policy games (solver-labeled).
- Entropy collapse: watch rollout entropy; GRPO on binary rewards can collapse fast.
