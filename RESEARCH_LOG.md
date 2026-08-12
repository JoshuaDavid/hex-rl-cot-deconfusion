# Research log (append-only)

Format per entry: date / what I looked at / boldest surviving hypothesis / falsifier + cost / result / what is still confusing.

---

## 2026-08-05 01:10 — Session start, Phase 0

- Environment: A100-PCIE-40GB (single), torch 2.13.0+cu130, 150G free disk. Driver CUDA 13.0.
- Plan: follow RESEARCH_AGENDA.md protocol loop. Phase 0 today (CPU): benzene solver build (delegated to background agent per user suggestion), hex env + ASCII renderer + ~20 solver games read by hand, token arithmetic.
- Decision: match benzene's conventions exactly in my Python env (Black connects top↔bottom, White left↔right, cells like `c3`) so solver interop is trivial and un-confusing.
- Registered predictions carried over from agenda (odds re-anchored after Phase 1, before Phase 2).

## 2026-08-05 01:35 — Token arithmetic + core framing decision

- Board renders: 5x5=72 tok, 7x7=114, 9x9=164. Per-turn prompt ~300 tok + CoT 512-1024. Cheap.
- **Decision: single-turn RL framing.** Each training sample = one position (full board in prompt), model emits CoT + move, reward = solver verdict. GRPO group = 8 rollouts of the same position. No multi-turn machinery needed in verl.
  - Training positions restricted to *positions winning for the player to move* (a win-preserving move always exists): reward +1 keep-win, -1 throw-win, -1 illegal/format-fail. Lost positions give zero within-group variance (all moves lose) => no gradient => excluded.
  - Position distribution: seed corpus from randomized solver games; refresh periodically with self-play games from current policy (scored offline by solver) to keep distribution on-policy-ish.
  - Caveat logged: this optimizes "find win-preserving move on our position distribution," not full-game play directly. Eval (transfer, optimality) uses solver-labeled fixed position sets, per agenda.
- Background: vllm installing into /venv/main; Qwen3-1.7B/0.6B downloading; verl-setup agent building /venv/verl; benzene build agent running.

## 2026-08-05 02:05 — Phase 0 substantially complete

Looked at: benzene build report, solver latency, 8 full 5x5 games + 5 6x6 games (read game 0 move-by-move; final chain verified by eye), corpus stats.

- Benzene built clean (agent, zero patches). Binary: benzene-vanilla-cmake/build/src/mohex/mohex. Solver GTP: dfpn-solve-state, dfpn-solver-find-winning. Empty-board sanity: 5x5/6x6/7x7/8x8 all first-player wins; 7x7 winning opening set matches published results.
- Latency: 5x5/6x6 per-move find-winning ~10-50ms warm. 7x7 find-winning cold on empty board ~64s; single solve-state ~1s. 8x8 empty solve ~43s.
- **Design: solver-free RL loop.** Corpus build precomputes the full winning-move set per position => training reward = set membership lookup. Solver cost is all offline. (move_keeps_win single-child-solve kept as fallback for policy-generated positions later.)
- Corpora: 904 5x5 (14s) + 929 6x6 (108s) positions labeled. 73-77% winning-for-mover (usable for RL). p(random legal move preserves win) ~0.3 mean => GRPO groups will have reward variance at cold start. Deeper positions are harder (p_rand ~0.19 at 16 stones on 5x5, vs 0.36 at 2 stones).
- 7x7 corpus/games: generation running, slow at low stone counts as expected. May restrict 7x7 training positions to >=4 stones, or accept slow one-time build.
- C4 KL-arm decision (provisional): use stock verl fixed-beta KL(pi||ref) in the RL objective as the "halfhearted preservation" condition, NOT the replay-distillation variant from the agenda — it's what practitioners actually do, and needs no custom trainer. Revisit when verl agent reports.
- Still confusing: nothing yet. Waiting on: vllm install (slow, network-bound), verl env agent, 7x7 games.

## 2026-08-05 02:50 — SURPRISE: benzene find-winning is not an exact winning-move oracle

Looked at: 7x7 game 1 move-by-move (per the read-your-data rule). Ply 17: B had just played a winning move at ply 15, yet solver reported 0/33 winning moves for B — impossible.

Diagnosis (confirmed in benzene source, DfpnCommands.cpp CmdFindWinning + EndgameUtil):
1. `dfpn-solver-find-winning` first runs ComputeAll (ICE fill-in). On positions the VC engine already *determines*, fill-in occupies every empty cell => the command iterates over nothing and returns [] EVEN WHEN THE POSITION IS WON. (Empirically: solve-state=black, find-winning=[], get-pv=[], zero "Trying" log lines, elapsed 2e-6s.)
2. On non-determined positions, it only tests moves in the ICE consider set => winning-but-inferior (dominated/captured) moves are silently omitted.

Consequences caught before they poisoned anything:
- My v1 corpora (and gen_games n_winning fields) had (a) determined-win positions mislabeled as losses, (b) incomplete winning sets that would punish genuinely winning moves with -1 reward.
- Monotonicity check verified: on the determined-win position, 8/8 spot-checked arbitrary moves still B-win (extra stones never hurt you in hex).

Fix: HexSolver.exact_winning_moves = exhaustive per-child solve-state + assert(parent winner consistent with children). Exact by construction; consistency check runs on every corpus position. Cost fine: 7x7 worst ~15s at 2 stones, <5s at 6+, trivial deep; TT sharing across siblings helps a lot. All corpora being rebuilt.

Still confusing: nothing — mechanism fully explained. Meta-lesson logged: the agenda's "read every sample" rule caught this within the first 20 games. Trophy for eyes-on-data.

## 2026-08-05 03:30 — RAM incident + env mutation incident

- User reported box struggling. Causes: (1) 40 mohex workers x ~770MB+ default DFPN TT (swap fully consumed at some point); (2) background `uv pip install vllm` mutated /venv/main mid-run and REMOVED accelerate, crashing the comprehension eval at import time.
- Fixes: HexSolver now sets param_dfpn tt_size 524288 (~200MB/proc); corpus workers 40->20; killed the uv retry loop (pypi network degraded to ~50KB/s anyway — vllm deferred, Phase 1 runs on plain transformers batched generation).
- Lesson (feedback-to-self): never run package installs into an env that concurrently serves running jobs. verl agent uses isolated /venv/verl, unaffected.
- Corpora v2 (exact labels): 5x5 = 1214 pos (10s), 6x6 = 1231 pos (34s). 7x7 rebuilding.

## 2026-08-05 02:25 — Exact corpora v2 + comprehension first read

- Exact labels vs old find-winning labels: p(random move wins) 0.28 -> 0.42 on 5x5. The buggy oracle was silently marking ~1/7 of winning moves as losses. Reward would have been meaningfully noisy.
- 17-21% of winning positions are fully-determined (ALL moves win). Excluded from RL training (zero GRPO advantage), kept for eval.
- B/W among winning-for-mover: ~40/60 after adding odd stone buckets. Both sides represented.
- Comprehension (provisional, truncation-contaminated run, robust regrade): lookup 0.90, connected 0.87, adjacent 0.67, neighbors 0.60, TOTAL 0.76. Errors mostly = ran out of thinking budget mid-derivation, not conceptual confusion. The 50/50 plan-killer (P1.2) is trending PASS. Clean rerun with 3584-token budget in flight.
- Infra: pypi CDN (fastly) degraded ~20-50KB/s; download.pytorch.org 825KB/s; aliyun pypi mirror ~350KB/s. vllm install deferred until GPU jobs pause (learned: uv mutates env mid-install; it removed accelerate under a running job).

## 2026-08-05 03:05 — Phase 1 results (items 2, 3 done; 1 pending rerun)

Looked at: clean comprehension run (120 Q, 11 truncated), all 8 quiz answers in full, several full CoTs.

- **P1.2 board comprehension: PASS.** lookup 0.93, neighbors 0.93, adjacent 0.87, connected 0.83, TOTAL 0.89 (robust regrade; in-script grader was broken, kept regrade path). Registered 50% -> resolved YES. CoT style: methodical coordinate arithmetic using the prompt's neighbor formula; no spatial-gestalt shortcuts observed.
- **P1.3 contamination quiz: essentially ZERO hex theory.** Bridge: wrong (thinks edge-to-edge chain). Perfect-play winner: wrong (claims 2nd player wins; no strategy-stealing argument). Ladder, edge template, virtual connection: admits unfamiliarity. Swap rule: garbled (cites chess 1.e4). Corners: confuses board corners with hexagon tile angles. Registered 30% "already verbalizes bridge-like concepts" -> resolved NO. Invention-purity precondition holds. Caveat for C2: the *words* bridge/ladder exist in the model's English; track hex-*meaning* emergence, not surface tokens.
- P1.1 legal-move rate: first run unusable (GPU OOM degraded generation -> 92% truncation). Rerun queued on vllm.
- Infra: vllm installing into isolated /venv/vllm via aliyun mirror (~350KB/s). HF-generate backend has KV-cache OOM issues at batch 16 x 3584 tok; batch 8 works but slow. All remaining GPU evals wait for vllm.

## 2026-08-05 03:20 — Pre-pilot design notes (informed by Phase 1 samples)

- Qwen3-1.7B thinks LONG on hex prompts (median ~1.5k tok; 32% truncation at 2048). Pilot config: max_response 3072, batch ~24-32 positions x 8 rollouts. Truncated rollout => no parsed move => -1: implicit but real pressure toward shorter thinking. Watch length drift as a first-class observable (C3-relevant: does RL compress the CoT?).
- Keep stock GRPO (verl defaults, incl. std-normalized advantage). No length shaping, no format shaping beyond the -1. Reward stays pure win/lose/illegal. Rationale: the research question is "what does vanilla RL do", not "what does my clever reward do".
- Health metrics per step: frac groups all-fail (no gradient), frac all-win (no gradient), mean response len, entropy.
- Learned (ops): container PID limit 2816; Ray smoke test ~2.6k pids. Don't stack corpus builds/GPU evals during verl runs. Invoke /venv/main/bin/python explicitly (session PATH now led by an unrelated venv).

## 2026-08-05 03:40 — C2 miner built + null-calibration lesson

- mine_vocab.py written; first null run (stride-split of comprehension CoTs) showed huge fake enrichments — stride-2 split anti-correlated with question-type periodicity. Lesson: C2 comparisons are only valid PROMPT-MATCHED (same position set, both corpora). Plan: per-position sample-split null using movequality k=8 data.
- C3 probe design written to notes/design_probes.md (validation-tier features first, per Karvonen).

## 2026-08-05 04:15 — Phase 1 items 4+5: the thinking-length problem

Looked at: 200 think-mode + 200 no-think move samples (25 positions x k=8, temp 1.0).

- **Think mode at temp 1.0: 200/200 truncated at 3584 tokens.** Median length hits the cap; CoTs meander (endless neighbor recomputation). Comprehension at temp 0.6 truncated only ~9% — temperature drives thinking length hard. Direct GRPO consequence: temp-1.0 rollouts would be ~all-truncated => all -1 => zero group variance => no learning. Pilot config MUST resolve this. Options: lower rollout temp (0.7-0.85), bigger budget (4096-6144), brevity prompt. Sweep queued for vllm.
- **No-think: legal 0.695; win rate 0.350 overall; among-legal win rate 0.504 vs random 0.341.** Base model has weak but real move judgment without CoT. Group variance present (11/25 groups) => GRPO has cold-start signal even in worst case.
- P1.4 (think/no_think gap): unmeasurable at temp 1.0 (think never finishes); re-measure at temp 0.6. P1.5 (group variance): PASS (via no-think; think pending temp fix).
- P1.1 legal rate: no-think 0.695 < 90% gate — but most parse failures at temp 1.0 may be format flubs; recheck at 0.6 with the sweep. Not yet a fail.
- Note the pleasing irony: the config-killer (truncation) is itself C3-relevant evidence — thinking length is unstable at high temp, and RL pressure via the -1-on-truncation channel will push toward shorter/terminating CoTs. Track length distribution across training as a first-class curve.

## 2026-08-05 04:50 — The overthinking wall, quantified

- vllm sweep on move prompts (30 pos x k=8): temp 0.6 => 100% truncation at 3584 tok, 93% at 6144. Confirmed NOT a repetition loop by reading start/middle/end of a long CoT: coherent candidate-move deliberation that never converges. (Also spotted a board-reasoning error: called b3/b5 "adjacent".)
- Contrast: factual comprehension questions at same temp truncate ~9%. It's the open-ended decision that unbounds the rumination.
- Interpretation: Qwen3-1.7B lacks a decisiveness policy for this task. RL will have to (and, via the -1 unparsed channel, will be pressured to) *create termination behavior* — itself a C1-flavored question: is "decide and stop" in the base model's support?
- Action: testing prompt variants (plain / brief-250-words / at-most-3-candidates) x temp {0.7, 1.0} at 3072. Whatever variant we pick becomes THE task prompt everywhere (base evals, RL, pass@k) so all comparisons stay prompt-matched.
- Fallback if prompts fail: custom verl AgentLoop doing two-phase generation (think capped, force-inject "</think>\n\nMove:" continuation). Documented API in notes/verl_setup.md makes this ~50 lines.
- verl smoke test status: agent still iterating (its vllm server was my "GPU squatter" — host-PID confusion, no external tenant. We time-share the GPU; its retry loop waits for free GPU).

## 2026-08-05 05:40 — Phase 1 COMPLETE; forced-close solves the overthinking wall

- Prompt variants failed (brevity instr: 87-90% still truncate; "3 candidates" instr: 100%). Qwen3's think phase does not obey length instructions.
- **Forced-close two-phase generation works**: cap think at budget, inject "</think>\n\nMove:", generate ~8 answer tokens. Results (30 pos x 8): legal 0.83-0.89, win ~0.31, reward variance in 18-20/30 groups, at both 2048 and 3072 think budgets, temps 0.7/1.0. 0 natural closes at baseline (model never terminates on its own — "does RL teach natural termination?" becomes a trackable curve: natural_close rate per checkpoint).
- **P1.4 resolved**: forced-close CoT win 0.31 vs no-think 0.35 -> CoT NOT load-bearing at baseline. C3 reframed per agenda contingency: "does RL create load-bearing CoT where none existed?" think/no_think gap per checkpoint is the metric.
- Phase 1 verdict: PASS overall. Comprehension strong (0.89), zero hex-theory contamination, GRPO signal exists (var-groups ~2/3), legality 0.83-0.89 under forced-close (P1.1 gate satisfied under the actual rollout scheme).
- Implementation: custom verl AgentLoop (hexenv/hex_agent_loop.py, registered "hex_forced_close" via hexenv/agent_loops.yaml). Masking: model tokens 1, injected scaffold 0; natural </think> stays mask 1.
- Pilot config: think_budget 2040, response_length 2176, temp 1.0, group n=8, batch 32 positions, lr 1e-6, KL beta 1e-3 (arm A). 0.6B x 3 steps smoke first, then 1.7B x 100.

## 2026-08-05 06:20 — verl + custom agent loop: first end-to-end validation pass

- Smoke run (0.6B) executed val-before-train THROUGH the forced-close agent loop + custom reward: reward mean -0.74; kinds: win 0.13 / lose 0.58 / illegal 0.24 / unparsed 0.05. Plumbing verified: dataset -> agent loop -> two-phase generation -> reward -> metrics. (0.6B win 0.13 vs 1.7B 0.31 on similar positions: model size matters for hex judgment.)
- Crash #1 after val: KeyError 'rollout_log_probs' — my loop returned response_logprobs=None. Fixed: calculate_log_probs=True + stitch phase1/phase2 logprobs, 0.0 on injected scaffold tokens (masked anyway).
- Stopped the verl setup agent (its env+notes deliverable complete; its remaining smoke mission superseded by mine; GPU/pid contention between two autonomous processes was the dominant failure source).
- Cleanup lesson repeated: `pkill -f` matched my own wrapper shell AGAIN (memory note existed; still stepped on it). Bracket-pattern habit now: pkill -f 'name[x]'.

## 2026-08-05 06:50 — Smoke debugging round 3-4

- Round 3 reached training steps 1-2 successfully (first actual GRPO updates through the forced-close loop!). Healthy internals: entropy 0.81, rollout-vs-actor logprob diff mean 0.007 (stitched logprobs correct). Then AgentLoopWorker died: native zmq_socket crash — thread/pid ceiling again (idle floor ~910 of pids.max 2816; verl stack peaks near the rest).
- Squeezes applied: OMP 8->4, ray num_cpus 8->6, TOKENIZERS_PARALLELISM=false, syncthing stopped. Round 4 running with peak-pid tracking.
- Fixes so far for custom agent loop: (1) response_logprobs stitched from both phases (scaffold tokens 0.0, masked), calculate_log_probs=True; (2) min/max_global_steps merged from both generate calls' server-stamped extra_fields.

## 2026-08-05 07:15 — SMOKE PASSED. Phase 2 plumbing complete.

- Round 5 (agent.num_workers=1, reward.num_workers=1, TQ storage_units=1, OMP=4, num_cpus=6, tokenizers-parallelism off, syncthing stopped): 3 GRPO steps + final val, exit 0. Peak pids 2156/2816 — the crash was zmq thread exhaustion; single workers give ~650 headroom.
- Total custom-agent-loop fixes: logprob stitching, min/max_global_steps merging, worker-count minimization. Rollout side channel verified (full responses + kinds logged per sample).
- Launching real pilot: Qwen3-1.7B, 100 steps, batch 32 pos x n=8, resp_len 2176 (2040 think + scaffold + 8 answer), temp 1.0, lr 1e-6, KL beta 1e-3, save every 25, val every 25. Registered predictions for the pilot:
  - Reward slope clearly positive by step 100: 70%
  - Legality (parse+legal) > 0.95 by step 100: 60% (easiest gradient available)
  - Win-rate (temp-0.6 val) > 0.45 by step 100 (from 0.31 base): 55%
  - Natural-close rate rises above 5% by step 100: 25% (fixed think budget removes most termination pressure)
  - Some degenerate CoT shortening (mean think len < 1000 tok): 30%

## 2026-08-05 07:35 — Pilot v2 launched after throughput fix

- Pilot v1 step timing 333s: gen 104 / ref 96 (CPU-offloaded!) / old_lp 27 / update 102 (micro=1). Restarted at step ~3 with ref on-GPU, actor micro 1->4, logprob micro 2->8. Expect ~3min/step, ~5h for 100 steps.
- 1.7B step-0 val (mixed 5/6/7 boards, temp 0.6, k=1): reward -0.675, win 0.162, entropy(rollout t1.0) 0.30, response len 2168 (at cap - thinking never terminates, as designed around).
- Standing monitors: per-step metrics via persistent Monitor; rollout side-channel trends via scripts/rollout_trends.py (to run periodically + read CoTs per protocol).

## 2026-08-05 08:15 — Pilot v2 first 10 steps

- Timing stable 240s/step (gen 104 / ref 28 / old_lp 22 / update 82). ETA ~6.5h for 100 steps.
- Train reward (t=1.0): -0.80 -> ~-0.62 over 9 steps, noisy. Side-channel buckets: win ~0.17, lose ~0.58, illegal ~0.21, unparsed ~0.03; think length pinned at cap (~6700 chars). No clear trend yet at lr 1e-6 — expected this early.
- Read illegal samples: 604/604 are OCCUPIED-cell moves, zero off-board. Model loses track of stones mid-CoT (truncation compounds it); e.g. W plays onto W's own stone while analyzing its neighborhood. Cleanly learnable (-1 signal).
- Entropy steady 0.30. No degenerate hacks visible in CoT tails so far — reasoning is board-analysis-shaped throughout.
- Next checkpoints: step 25 (save + val). Plan: after pilot, merge ckpt 25/50/75/100 -> HF, eval_checkpoint each, read 1 full trajectory per ckpt, then decide Phase 3 config.

## 2026-08-05 09:55 — Step-25 val: slope confirmed, legality-led

- Val (fixed 277-pos set, t0.6): reward -0.675 -> -0.632; win 0.162 -> 0.184; illegal 0.238 -> 0.162; unparsed 0.047 -> 0.011.
- Side channel: illegal 0.24 -> 0.15 across buckets; win noisy-slightly-up. Think length creeping UP (~6600 -> 6900 chars against the 2160-token cap ceiling... chars vs tokens: still pinned at cap; the char-drift may be vocab/density change, not length). Watch.
- Phase-2 gate (nonzero slope, no degenerate hack) trending PASS. Checkpoint 25 saved.

## 2026-08-05 11:40 — Step-50 val: win-rate slope now real

- Val trajectory (t0.6, fixed set): reward -0.675 / -0.632 / -0.523; win 0.162 / 0.184 / 0.238; illegal 0.238 / 0.162 / 0.134; unparsed 0.047 / 0.011 / 0.007 at steps 0/25/50.
- Interpretation: first ~25 steps consumed the format/legality gradient; steps 25-50 show genuine move-quality learning. Entropy stable ~0.31-0.32 (no collapse). No degenerate hacks in read samples.
- C2 note from step-40 CoT reading: "bridge" already used in generic-English sense pre-existing; track semantic narrowing, not token novelty.
- Phase-2 gate: PASS (pending no surprises by step 100).

## 2026-08-05 13:30 — Step-75 val: acceleration

- Win: 0.162 / 0.184 / 0.238 / 0.314 at steps 0/25/50/75. Illegal 0.10, unparsed 0.004, reward -0.372. Entropy 0.33 stable.
- Phase-3 checkpoint policy decided (21GB/full ckpt, 84GB free): cadence 50, ckpt_janitor.sh merges to HF (3.4GB) + prunes shards, keeps 2 newest full for resume. Main run = EXTEND this pilot (same config/arm A, resume from step 100) to ~750 steps; beta=0 control ~225 steps fresh.

## 2026-08-05 12:30 — PILOT COMPLETE + first C3 result

Pilot final: val win 0.394 @k=1 t0.6 (0.162 baseline), curve 0.162/0.184/0.238/0.314/0.394, illegal 0.238->0.090, unparsed 0.047->0. Exit 0, 4 ckpts, all merged to HF.

**C3 early falsifier (150 val positions, k=4, t0.6):**
- base:    no-think win 0.195 (legal 0.652) | forced-close CoT win ~0.16-0.19
- ckpt100: no-think win 0.338 (legal 0.740) | forced-close CoT win 0.317 (legal 0.902)
- => CoT is STILL not load-bearing after 100 steps (no-think >= with-CoT). RL gains transferred fully to the no-CoT policy head, though training only ever ran in CoT mode. Concept learning is happening silently; the CoT rides along.
- Read step-100 CoTs: style unchanged from base (board re-parsing with occasional MISREADS, meandering candidates, truncation), winning moves often barely mentioned in the visible CoT before being played. Decorative-CoT hypothesis leading.
- natural_close_rate still 0.000 (prediction "rises >5% by step 100" at 25%: resolved NO).
- Grade pilot predictions: reward slope positive: YES (70% - hit). Legality >0.95: NO at step100 (0.90; 60% - miss, close). Win >0.45 by 100: NO (0.394 at k=1; 55% - miss, close). Natural close >5%: NO (25% - hit). CoT shortening: NO (30% - hit; length pinned at cap).

Phase 3 arm A = resume this run to 750 steps (same config; save_freq 50). Then beta=0 control ~225 steps.

## 2026-08-05 12:50 — C2 rough cut at step ~105 (side-channel, early-2000 vs late-2000 CoTs)

- No new coined terms. Enrichments are reweightings of existing English:
  1. Blocking/prevention talk up strongly ("blocking moves", "while preventing", "bottom while blocking", "needs to prevent") — semantically apt: win-preservation reward ~ blocking in hex. The clearest content-level drift so far.
  2. Candidate-cycling connectives up ("alternatively" chains) and visualization phrasing ("imagine that then").
  3. "game theory" x30 — spot-checked: vacuous transition filler ("let's think about the game theory"), not conceptual usage.
- Interim C2 verdict at 105 steps: consistent with the no-new-vocab null; CoT drift is stylistic/topical reweighting. Re-run properly (prompt-matched, tokenized, vs ref-perplexity) at Phase 4 with more steps elapsed.
- Phase 3 arm A resumed cleanly (re-val at 100: win 0.361 - k=1 noise around 0.394). Janitor loop + disk watchdog running. ETA step 750 ~ Aug 7 morning.

## 2026-08-05 15:15 — Ops: wandb + B2 backups (user request)

- Interrupted arm A at ~step 175, added wandb (project hex-rl-cot-deconfusion, run cyaib1sz), resumed from ckpt 150. User confirmed 1-2h compute loss is fine (<$0.50/h box).
- B2 backups via rclone to claude-code-backups/hex-rl-cot-deconfusion/: code, data, notes, logs, results, merged HF checkpoints. Recurring 2h sync loop running.

## 2026-08-05 16:40 — Correction from Joshua re: contamination quiz grading

- Under the swap rule, second player DOES win by strategy-stealing (swap the strong opening). Qwen's "second player wins" answer could be a garbled memory of swap-rule hex. Its stated justification was not swap-based (claimed a blocking strategy from "perfect information" for plain hex), so the "no applicable theory" verdict stands — but the README/log phrasing should say "confabulated justification" rather than "wrong winner", and the C2 baseline note is: fragments of hex facts may exist in weights with broken attachments to arguments.

## 2026-08-05 17:50 — Idea from discussion with Joshua: within-winning-set preference analysis

- Q raised: what makes one winning move "better" in a maker-breaker game? Taxonomy: (1) robustness to own error (eps-perturbed value; bridges = 1-fault-tolerant links), (2) opponent-error exploitation (resistance/DTM), (3) proof complexity (domination = the error-model-free partial order; benzene ICE embodies this), (4) pedagogical/reusable-pattern value.
- Key observation: flat +-1 reward + single-turn episodes exert ZERO per-sample pressure among winning moves. Any emergent preference within the winning set is pure generalization/inductive-bias signal.
- **New planned measurement (Phase 4)**: on multi-winning-move eval positions, test whether checkpoint choice distribution increasingly concentrates on (a) benzene's non-inferior (ICE-surviving) winners, (b) winners maximizing post-move p_random_win (robustness proxy), vs (c) stays at base-model priors (center bias). (a)/(b) trending up = RL + generalization rediscovering "move quality" structure the reward never specified. Cheap: all quantities computable from existing labels + one extra solver pass per winning move (child winning-set sizes).

## 2026-08-05 19:05 — [fork session] Live within-winning-set probe built; predictions re-registered

- (Re-appended: an earlier version of this entry was lost to an uncommitted-file wipe. Lesson re-learned: commit before running.)
- Built live_probe.py + build_probe_wave_set.py: probes the CURRENT policy through verl's vllm HTTP server (no GPU interruption). 64 held-out multi-winner positions, k=8 forced-close samples; fixed positions => doubles as a prompt-matched C2 corpus over training.
- Metrics per wave: win/legal/natural-close; within winning set: P(non-inferior|win) vs uniform baseline, robustness percentile (2-ply win density), bridge_delta, center distance vs uniform baseline. Labels precomputed in data/probe_wave_positions.jsonl.
- **Registered predictions (before first successful wave):** non-inferior rate clearly above uniform baseline by step 750: 60%. Robustness percentile mean > 0.55 by 750: 55%. Effects mostly explained by center bias: 40%. Base/early policy already above uniform on non-inferior: 50%.
- NOTE TO MAIN LINE: training job disappeared ~18:58 (no error in log — assumed deliberate stop by main session). First probe wave errored on the missing server. When training is back up, run:
  /venv/main/bin/python scripts/live_probe.py    # appends to results/live_probes.jsonl
  ~90-min cadence suggested; ~1.5% rollout-throughput overhead per wave.

## 2026-08-05 19:40 — [fork] Convergence check (prompted by Joshua's skepticism)

- Train win rate flat at 0.37-0.42 for ~50 steps; val 125->150 flat (0.451->0.458, within +-0.03 noise). The 0.16->0.46 rise was legality + easy-win gradient.
- NOT signal starvation: mixed-variance groups still ~60% of batch (all-win 13%, all-lose 28%). Reads as a genuine difficulty wall — consistent with C3 (CoT not load-bearing => no scratchpad mechanism to improve mid-game analysis with).
- Plan: run to step 250. If val <= 0.48 there: curriculum refresh (drop >80%-win positions per side-channel stats, backfill frontier-difficulty solver-labeled positions). lr/beta stay fixed (mid-run surgery would muddy the vanilla-RL question).
- Registered: P(val win > 0.48 at step 250) = 45%.
- Note: plateau-at-0.46 would itself be a C1-relevant result (selection exhausted, no creation), not merely failure.

## 2026-08-05 20:05 — [fork] Mate-in-1 diagnosis (Joshua's question)

- P(play immediate winning move | one exists): current policy t1.0 = 0.149 (n=343; random-legal ~0.09-0.15 => AT CHANCE at training temp). ckpt100 t0.6 = 0.35 (n=20, small). 71% of t1.0 samples on mate-in-1 positions THROW THE WIN.
- Mechanism for the plateau: no terminal pattern recognition. All RL gains so far = positional prior sharpening (center/blocking correlates of winning), not connection-tracing. Coheres with CoT board-misreads + CoT-not-load-bearing.
- Revised curriculum plan for step-250 gate: enrich training distribution in mate-in-1 (and later mate-in-N) positions (~4% of samples now -> target ~25%). Groups there have max reward variance at 15% hit rate => gradient focused on the missing skill. Theory-neutral (distribution reweighting only).
- Registered: P(mate-in-1 conversion at t0.6 > 0.8 within 150 steps of enriched training, if triggered) = 60%. P(it transfers to overall val win +0.05 or more) = 45%.

## 2026-08-05 20:35 — [fork] Mate-in-1 difficulty decomposition (Joshua's axes)

938 recent samples (steps ~100-170, t1.0) on mate-in-1 positions, stratified:
- **Edge completions: conv 0.03, BELOW random (lift -0.06, n=208).** Interior completions: 0.22 (+0.11). Monotone in cell-to-edge distance: 0.03 / 0.19 / 0.30 for d=0/1/2.
- merge-two-groups 0.21 vs extend-to-edge 0.03 (largely same axis as on-edge).
- Spanning path length (5/6/7 stones): flat ~0.18/0.19/0.13. Own off-path stones & opp clutter: mild POSITIVE trends (confounded with interior).
- **Reading: path complexity is not the difficulty axis; the target edge is anti-attractive.** RL-sharpened center bias suppresses exactly the finishing moves. Selection-not-creation in miniature: the optimizer strengthened a locally-good heuristic into the binding constraint.
- Curriculum assets: corpus_mate1.jsonl (168 mined) + corpus_mate1_gen.jsonl (1200 gen, 25% edge-only) + corpus_mate1_edge.jsonl (800 edge-only). Plan at step-250 gate: mate-in-1 slice ~25% of train mix, half edge-completion.

## 2026-08-05 20:55 — [fork] Methodological correction (Joshua: "janky half-implemented RECAP")

- Caught drift: my registered step-250 plan would have mutated ARM A's training distribution based on my own failure diagnosis — adaptive curriculum, i.e. the intervention family the agenda deliberately excluded. Two contaminations: (1) experimenter becomes an unlogged part of the training algorithm; (2) concept taxonomy (edges) injected via the sampling measure => later C2 "edge vocabulary emergence" would be uninterpretable (invention vs injection).
- RESTRUCTURE:
  - Arm A: vanilla to step 750, NO distribution changes. A plateau with intact edge-blindness = clean C1 result (selection sharpened priors into the binding constraint; no creation of the missing percept).
  - Arm B (new): branch from ckpt ~250 with mate-in-1/edge-enriched mix. Explicit distribution-intervention arm answering: does RL create a skill the base model lacks (edge completion at-chance) when the gradient lands exactly on it? C2/C3 on arm B carry an injection caveat.
  - beta=0 control unchanged in plan.
- Principle logged: branch, don't mutate. Interventions get labeled arms.
- Amended prediction (supersedes step-250 curriculum-refresh trigger): P(arm B reaches t0.6 edge-completion conversion > 0.8 within 150 steps) stays 60%; P(arm A self-fixes edge-blindness by 750 with no intervention) = 15%.

## 2026-08-05 21:10 — [fork] ARM A CONCLUDED (verdict); pivot to curriculum arm B

- Joshua: prior comment wasn't criticism; the vanilla result is clean and expected. **Arm A verdict: with pure +1/-1 reward on uniform random positions, Qwen3-1.7B sharpens existing priors (0.16 -> ~0.46 val win), plateaus ~step 130-170, and never acquires terminal pattern recognition (edge-completion mate-in-1 below chance). Selection, not creation, on this budget.** Full C1-C4 analysis on the 0-250 checkpoint series remains scheduled (Phase 4).
- Arm A runs to step 250 for the final checkpoint, then stops.
- ARM B: curriculum RL from BASE (fresh; avoids basin-entrenchment confound). Staged manual-RECAP: B1 mate-in-1-heavy (half edge), B2 +mate-in-2, B3 anneal to general. Same reward, same beta, same forced-close scheme. Fixed val = 277 general + new edge-completion slice.
- Registered: P(B1 edge conversion >0.8 @t0.6 within 150 steps) = 60%. P(general val win >= 0.55 by end of B3 ~400 steps) = 40%. P(new hex-specific CoT vocab emerges in B, none in A) = 30% (injection caveat explicit).

## 2026-08-05 22:50 — [fork] Arm A killed at step ~154; three measurements; B1 LAUNCHED

- Arm A terminated per Joshua (checkpoint series 25/50/75/100/150, all HF-merged, backed up). Vanilla experiment closed.
- **</think> logprob profile (base + ckpt100, 60 probes):** bimodal. Median P(close) 1e-13..1e-16 at all depths; max P ~1.0 at conclusive junctures (as early as 25% depth). RL left the profile unchanged. Designed (not yet adopted): segmented rising logit-bias on </think> via multi-call agent loop -> termination at natural junctures; bias-induced closes masked as injected. B1 keeps hard cutoff for comparability.
- **Encoding bake-off (base, mate-in-1 conversion, n=80/cell):** ascii 0.10/0.16 (edge/interior), lists 0.09/0.12, hybrid 0.14/0.16 — all within noise. Encoding is NOT the constraint (supports prior-not-perception reading of edge avoidance; also confirms Joshua's monotonicity point — move-list has no measurable world-model tax in hex). Keeping ASCII.
- **New curriculum rung (Joshua's Q2): terminal-judgment task** ("has either player already won? Black/White/Neither"), 900 board-logic-labeled examples, exact-match reward via task branch in reward_verl. This isolates the missing terminal percept below even mate-in-1.
- **B1 LAUNCHED**: fresh from base, 150 steps, mix = 15% judge / 31% edge-mate1 / 28% gen-mate1 / 26% general (4868 rows). Val = 277 general + 60 edge held-out. Same reward scale, beta, forced-close, hyperparams as arm A.

## 2026-08-05 23:20 — [fork] B1 false start: scaffold bug caught by slice breakdown at step 10

- Judge-slice success was 0.002 — not task difficulty but plumbing: the forced-close loop hard-coded the "Move:" scaffold, so judge rollouts answered with moves; 431/432 unparsed. (Read-the-slices rule catches again; the aggregate reward looked merely "low", the slice was unambiguous.)
- Fixed: scaffold now task-aware (detects "Answer: Black|White|Neither" in the prompt). B1 restarted clean from base (10 steps discarded, rollout log + ckpts wiped for a clean series).

## 2026-08-05 23:50 — [fork] Think-budget economics -> B1 restarted at 1088

- Step-time attribution (arm A representative step, 259s): ~89% of compute is think tokens (gen 114s ~all think; fwd/bwd passes 140s x 84% token share). Counterfactuals: budget 1024 -> ~137s/step (1.9x), 512 -> ~81s (3.2x), no-think -> ~28s (9x).
- Evidence that 2160 buys nothing in reward: quality flat 2048->3072; no-think ~= full-CoT at base AND ckpt100.
- Decision (with Joshua): B1 restarted from step 0 with think budget 1088 (RESP_LEN 1104). Halves the run to ~6h; 1088-token CoT remains a viable C2/C3 object. Step-0 val re-baselines base under the new budget automatically. Probe waves switched to HEX_THINK_BUDGET=1088.
- Caveat logged: arm A ran at 2160; cross-arm comparisons are qualitative (curriculum-vs-vanilla), not budget-matched. Within-arm-B curves are self-consistent.

## 2026-08-06 00:10 — [fork] Why no early termination? (Joshua's question) — hypotheses registered

- H1 answer-resolution gating: </think> was trained (math/code RLVR) to fire on "verified answer in hand", not on marginal-value-of-thinking. Hex move choice affords no verification event => never fires. Supporting: checkable board questions truncate ~9% vs move choice 100% on the same boards.
- H2 textual attractor: doubt-register ("Alternatively... But wait") locally suppresses close-shaped continuations.
- H3 two-stage gate: my </think> logprob probe conditioned on an inserted "\n"; the real bottleneck may be initiating the line break at junctures ("." -> " But" instead of "\n"). Measurement refinement queued for next GPU window.
- **Discriminator now running for free in B1**: judge prompts (checkable) vs move prompts (unresolvable), same run/budget. Natural close detectable as think-length below the 1088 cap.
- Registered: P(judge natural-close rate exceeds move natural-close rate by >10x) = 65%. P(judge natural-close > 30% absolute at t1.0) = 40%. If both false -> H2/attractor gains weight and "termination is task-insensitive" becomes the story.

## 2026-08-06 00:55 — [fork] 1x1/2x2 judge probe (Joshua's degenerate-task question)

- 1x1 (k=8, 3 boards, t1.0, ~base policy): acc 0.12, natural-close 0.21, think p50 952 (below cap!), p10 251. 2x2: acc 0.44, natural-close 0.00, p50 at cap.
- Read failures: (a) 1x1/2x2 ASCII renders are unparseable-OOD (model debates whether 1x1 is "a 2x1 board"; concludes occupied board is empty); (b) one sample echoed the literal template "Answer: Black|White|Neither"; (c) predicted "one stone can't be a chain" confusion present but minor.
- **Strongest state-gating evidence yet: a sample reached a WRONG-but-resolved answer ("So the answer is Neither.") and naturally closed at that exact point.** Close fires on answer-shaped state, indifferent to correctness/value. Resolvability => shorter traces (952 vs cap) even pre-training.
- Prediction grades: baseline 1x1 acc >0.9 @70%: MISS (0.12 — render-OOD, unanticipated). Natural-close >20% @60%: HIT (0.21). RL-shortening: refined — transient shortening via faster resolution, frozen at saturation (advantage dies).

## 2026-08-06 01:15 — [fork] Judge-vs-move discriminator: null at scale (prediction MISS)

- Exact-token recount: B1 at 1088/t1.0 shows 0.000 natural closes on BOTH judge and move prompts at 5x5-7x7 (earlier char-based split was estimator noise). Registered "judge >30% absolute" (40%): MISS; ">10x move" (65%): vacuous at 0/0.
- Judge success 0.472 (guess floor 0.33): analyses usually unresolved at cap => consistent with refined state-gating ("resolved => closable; resolution is the bottleneck"), which the 1x1 result (closes on resolved-even-if-wrong answers) directly supports.
- Standing in-run test: if B1 judge accuracy climbs toward ~0.9 and judge natural-closes stay 0%, resolution-gating dies too and the register-attractor story (H2) inherits. Tracked per milestone from the side channel at zero cost.

## 2026-08-06 01:50 — [fork] Length economics: shaping + exploration implemented; B1 killed pre-launch

- B1 (1088-budget rerun) killed at ~step 8 per Joshua: still iterating design, not ready for multi-hour jobs.
- Implemented: (1) correctness-gated length penalty (correct: 1 - 0.25*len_frac; wrong: -1) — GRPO group normalization makes it effectively Kimi-style group-relative among corrects; (2) segmented rising logit-bias on </think> during rollout phase-1 (default "512:0,832:6,1088:12") — provides the length VARIANCE the penalty needs (under plain forced-close every trace sits at cap and the penalty is inert). Bias-sampled closes stay mask-1; single-token off-policy distortion left to PPO clipping.
- vllm 0.24 SamplingParams supports logit_bias natively; verified.
- Smoke (6 steps, 1.7B) running: checks = closes actually sampled (response_length < cap), no engine errors, reward spread among corrects.

## 2026-08-06 02:40 — [fork] 2x2 prompt ladder: no native floor found (predictions MISS)

- 2x2 winner detection, k=8 t0.6: V0 ascii 0.589 / V1 move-history 0.637 / V2 +example 0.667 / V3 +full adjacency 0.667. Both >=0.9 predictions MISS (55%, 45%).
- Conclusion: no presentation makes 2x2 judge reliable; residual errors are compositional (adjacency + edge-touch -> connection), not representational. Native floor = single-fact queries (lookup 0.93, adjacency 0.87 from Phase 1).
- Worked example moved natural closes 0 -> 0.369 (imitation of resolved-trace shape; H1-consistent) with NO accuracy gain; V3's extra rules text halved closes (0.20) — prompt bulk suppresses termination independent of correctness.
- Implication for curriculum: B1's judge task is genuinely learnable headroom even at 2x2-difficulty; consider adding tiny-board judge rung below the 5x5-7x7 one.
- Smoke round 2 queued: shaped-score instrumentation, val_batch_size=64 (zmq spike fix), close-bias ceiling raised to +30 (median P(close) 1e-13 needs ~+30; old +12 ceiling provably inert).

## 2026-08-06 03:05 — [fork] Atomic-subskill ladder registered (bumblebee curriculum, Joshua's push)

- T1 occupancy / T2 edge-touch / T3 adjacency / T4 same-color-adjacent-pair / T5 edge-pair membership / T6 full judge, all on 2x2, worked examples + explicit adjacency & edge facts in-prompt.
- Registered P(>=0.9): T1 80%, T2 65%, T3 55%, T4 50%, T5 45%. Cliff at first two-fact rung (T4/T5): 60%.
- Purpose: locate the compositional cliff; the highest imperfect rung becomes the bottom of the arm-B curriculum.

## 2026-08-06 03:50 — [fork] Atomic ladder: "cliff" was partly my prompt's ambiguity

- Results (2x2): T1 occupancy 0.981 / T2 edge-touch 0.694 / T3 adjacency 1.000 / T4 color-pair 0.991 / T5 edge-pair 0.704 / T6 judge 0.731. Composition is FINE (T4 0.99); deficit tracks edge-flavored rungs.
- Failure read: model interprets "touch the TOP edge" as "adjacent to a top-edge cell" (a2 borders a1/b1 => Yes, coherently argued, fast+confident). T5/T6 inherit the wording. Registered cliff-at-composition prediction: WRONG; T2>=0.9 (65%): MISS but artifactual.
- Grades: T1 hit, T3 hit (1.00!), T4 hit. Note T3 tests list-lookup (adjacency in prompt), not spatial computation.
- Wording-fixed T2/T5/T6 rerun queued. Prediction: T2 >0.95 @70%; T6 >=0.85 @50%.
- Termination note: atomic rungs close naturally 0.74-1.00 (resolution => close, again); T6 still 0.22.
- RECAP (abstract fetched): dynamic OBJECTIVE reweighting via short-horizon convergence/instability signals; counts-adaptation not claimed. Smooth mid-run task introduction design sketched (floor weight + rate-limited controller; GRPO group-normalization makes it low-jank; mixed-variance fraction = gradient-availability signal). Logged for arm-B/C consideration.

## 2026-08-06 04:15 — [fork] Sample allocation theory (Joshua's question): Neyman + frontier floors

- Given objective L = sum_c w_c L_c and per-category per-sample gradient sd sigma_c, variance-optimal counts: n_c ∝ w_c·sigma_c/sqrt(k_c) (Neyman allocation, cost-corrected). Weights = what you want; counts = how to estimate it. Resampling-as-importance changes the objective silently — keep them separate.
- GRPO mapping: group signal ∝ sqrt(p(1-p)) per prompt (mixed-variance groups carry all gradient); sigma_c, k_c measurable from the side channel. Saturation self-starves (RECAP's shift-from-saturated as a theorem).
- Frontier caveat: Neyman is myopic — p≈0 categories (edge-mate1 at 0.03: sigma 0.17, lowest share) get starved exactly when curriculum needs them. Patch: exploration floors + slow learning-progress controller on w_c (two timescales), EMA'd sigma, rate-limited w.
- This is the quantitative skeleton for the arm-B mixture controller; inputs all logged already.

## 2026-08-06 05:00 — [fork] ARM C: adaptive curriculum with hot add/remove (Joshua sign-off)

- Naming: the dynamic-mixture design is a big enough break from arm B's static stages to be **arm C**. Arm B retired at smoke stage (its corpora, judge task, forced-close loop, and length economics all carry into C).
- Joshua's requirements: task categories addable mid-run AND removable mid-run (buggy-category escape hatch). Both implemented and the add path verified in-run:
  - hexenv/dynamic_dataset.py (via verl data.custom_cls): virtual-length dataset; per-sample category draw from data/curriculum/weights.json, hot-reloaded on mtime change; new <cat>.parquet files in data/curriculum/ are discovered and tokenized at refresh; weight 0 / enabled:false = instant removal, no restart.
  - scripts/curriculum_controller.py: Neyman-with-floors (shares ∝ max(w·sigma_hat/sqrt(k_hat), floor·w), sigma from EMA'd side-channel success rates, optimistic p=0.5 prior for unseen categories), atomic weights.json writes, audit trail in results/curriculum_log.jsonl.
  - Categories staged: judge 900 / edge_m1 2240 / gen_m1 1368 / mate2 1200 / general 5336 rows.
- Smoke (10 steps, len-shaping + close-bias active) in flight with a live mutation test: at step 4, config flips mate2->disabled and judge importance x5; pass = side-channel category mix shifts accordingly with no restart, and mate2 samples cease.
- Registered: smoke passes cleanly (mixture shifts, removal works, no crash): 70%. Post-smoke gate before any long run: wording-fixed T2/T5/T6 ladder rerun (still PENDING — flagged so it doesn't get lost) + final category importances + arm-C launch predictions for Joshua.

## 2026-08-06 05:10 — [fork] Judge question reworded (Joshua's catch)

- "Has either player..." is a yes/no question; Black/White/Neither were ungrammatical answers — a form/content mismatch of exactly the kind this model trips on (cf. "touch" ambiguity cliff, template-echo failures). Now: "Which player, if any, has ALREADY completed a winning connection on this board?" — all three answers grammatical (elliptical).
- Propagated to builders, curriculum parquets (rebuilt), docs. Old wording preserved in atomic_ladder as the A-side of the queued wording A/B (old-vs-new judge phrasing + "touch" disambiguation), so the effect gets measured, not assumed.

## 2026-08-06 05:45 — [fork] Arm C smoke v1 hang: custom_cls applies to val too

- First arm-C smoke hung in val_before_train: verl instantiates the custom dataset class for BOTH splits, so val became a 1,000,000-row virtual dataset and val generation ran for an hour. Fix: val files (name contains "val") delegate wholesale to stock RLHFDataset. Also: controller/dataset churn de-spammed (print only on weight change).
- Controller upgraded per Joshua's 4.4 catch: sigma now estimated empirically (mean within-prompt std of SHAPED scores) instead of binary sqrt(p(1-p)) — sees the length-compression signal at saturation; analytic formula demoted to prior. Docs corrected (also: enabled defaults to true; mate-in-1 generator note - play-to-terminal-back-up-one adopted for next expansion, removal variant needs post-removal terminality recheck, verified on Joshua's 3x3 redundancy example).
- Smoke v2 relaunched with the step-4 config flip (mate2 hot-removal + judge importance x3).

## 2026-08-06 06:15 — [fork] Saturation length-race pathology (Joshua's question) + fix

- Q: does GRPO push as hard on {4x10tok, 4x1000tok} as {4x10tok, 4x11tok} groups at saturation? A: YES under stock std-normalized advantages — any two-point reward distribution normalizes to +-1 advantages regardless of gap. Three pathologies: tiny-difference amplification, noise-chasing (which-sample-is-shorter is sampling noise -> confident +-1 gradients), and controller/trainer incoherence (Neyman controller assumes signal ∝ within-group spread; std-norm makes push independent of spread).
- Fixes (arm C, registered divergence from arm A's stock GRPO):
  1. ADV_STD_NORM=False (Dr.GRPO-style mean-only advantage): push ∝ actual gap; makes the Neyman premise literally true — controller and trainer agree on sample worth.
  2. Length-bucket deadband (96 chars ≈ 32 tok): sub-bucket differences give exactly equal rewards => zero advantage => no noise-chasing. Verified: 10-vs-11-tok now equal; 10-vs-1000-tok keeps 0.225 differential.

## 2026-08-06 06:50 — [fork] ARM C SMOKE PASSED (with two footnotes)

- Mixture pivot verified: judge share doubled after the step-4 config flip; mate2 hot-removal landed within the documented one-tick latency (final controller weights: mate2 0.0, judge 0.3125). Dataset followed weights within noise (effective n=80 prompts — rollouts quadruple-count).
- val-split delegation fix verified (val ran at 337 rows, finite).
- Footnote 1: exit 127 was bash lazy-reading run_pilot.sh while I edited it mid-run (mangled trailing tokens in the RUNNING instance only; file intact). Rule: never edit a script a live run is executing — copy-on-launch (run scripts now get copied to results/<exp>.sh at launch; TODO).
- Footnote 2: mixture stats in short smokes need prompt-level (not sample-level) counting.
- Arm C mechanics now fully validated: dynamic dataset, controller, hot add/remove, length shaping + close-bias, empirical sigma, deadband, ADV_STD_NORM flag. Ready to draft the launch proposal.

## 2026-08-06 07:40 — [fork] Wording A/B launched + arm C launch mechanics

- A/B registered (before results): T2_B (edge-cell-membership wording) >=0.95: 70%. T5_B >=0.9: 55%. T6_B (grammatical judge + explicit chain definition) >=0.85: 50%.
- Launch mechanics: config.yaml RESET from smoke flip (mate2 re-enabled, judge importance back to 0.6 — the flip would otherwise have leaked into the launch); sliced val parquet built (data/verl_hex_C: hex_val_general 277 + hex_val_edge 60).

## 2026-08-06 08:10 — [fork] Wording A/B RESOLVED; ARM C LAUNCHED (autonomous mode)

- A/B: T2 0.889->1.000 (HIT), T5 0.741->1.000 (HIT), T6 0.694->0.815 (near-miss of 0.85; nat-closes 0.23->0.45). Atomic skills perfect under unambiguous wording; guided 2x2 judge 0.815 = the honest bumblebee floor; residual is composition-under-load. No tiny-board rungs needed — 5x5-7x7 judge (0.47) is the bottom rung. T6-B chain-definition parenthetical folded into the training judge prompt (last free moment to change it); parquets rebuilt.
- **ARM C LAUNCHED**: 400 steps, batch 32x8, think 1088, ADV_STD_NORM=False, len lambda 0.25 + 96-char deadband, close-bias 512:0/768:10/960:20/1088:30, controller 10-min ticks + wandb companion run (armC-controller), ckpts/50 + janitor, probe waves + 2h backups repointed, config.yaml at launch values (smoke flip reset).
- Registered: edge-val conversion >0.8 by 150: 60%. Judge success >0.9 by 100: 70%. Judge natural closes >20% once judge acc >0.8: 55% (resolution-gating live test). General-val win >0.55 by 400: 40%. Judge think-p50 <600 tok by 200: 60%.
- Mid-run planned: mate1-v2 corpus (back-up-one generator) added as a NEW category around step 100 — first live demonstration of the hot-add path.

## 2026-08-06 09:30 — [fork] FIRST LIVE HOT-ADD: atomic categories (Joshua's push)

- Dropped-rung decision reversed: judge's near-zero within-group sigma (deterministic per board => no gradient despite 0.49 accuracy) reopens the case for atomic full-board rungs. occupancy + chain categories built in the Black|White|Neither format (reuses judge reward branch + live scaffold detection — no trainer/loop changes), label-balanced (chain v1 was 97% Neither — caught by label count pre-training, rebuilt 270/270/260), dropped into data/curriculum/ mid-run at ~step 22. importance 0.4, floor 0.05 each.
- Registered: atomic within-group sigma > judge's 0.065: 70%. Atomic p>0.9 within 100 steps of add: 55%. Also implicitly tests the hot-add machinery in production.

## 2026-08-06 10:20 — [fork] Atomic hot-add first signals: sigma prediction splits informatively

- occupancy: n=8, win 1.000, natural-close 1.00, think p50 697. Arrives saturated — sigma 0.031 misses the >0.065 prediction via saturation (not judge-style failure-determinism). Controller will floor it; diagnostic value: full-board occupancy was never missing.
- **chain: n=24, win 0.417, sigma 0.782 (12x judge)** — prediction HIT. Same-board rollout disagreement = sampling-driven errors = teachable. Judge-vs-chain contrast: generic existence questions collapse to deterministic guesses; specific-pair questions force variable tracing. Specificity manufactures gradient.
- Full arm-C loop (diagnose -> hot-add -> sigma-detect -> reallocate) closed in production within ~40 min of the suggestion. Next live question: does chain-tracing gradient transfer to judge/edge conversion.

## 2026-08-06 10:50 — [fork] CONTROLLER BUG: mixture was floor-driven since launch (found via Joshua's timing question)

- The Neyman term w·sigma/sqrt(k) (~0.004-0.01) never exceeded any floor (0.02-0.15): weights equaled normalized floor_c x importance_c exactly, for every tick since launch. My earlier "controller demoted judge and it's right" was right-conclusion-wrong-mechanism: it was the floor RATIO, not the sigma signal.
- Fix: normalize Neyman shares first, then apply floors as minimum fractions, renormalize. Deployed mid-run (controller restart; training untouched).
- Silver lining: the launch mixture floors were hand-set to sensible ratios, so ~30 steps of training weren't misallocated badly — but the "controller reallocates by signal" claim only becomes true from this tick forward.

## 2026-08-06 11:40 — [fork] Step-50: chain learned at speed; transfer not yet

- chain quarters since hot-add: 0.42/0.50/0.58/0.88 (n=24/quarter) — at-chance to ~0.88 in ~25 steps. Fastest, cleanest learning curve of the project; the sigma-guided hot-add worked as designed.
- No transfer yet: val_edge flat (0.18->0.17 @t0.6), judge drifting down (0.48->0.39, allocation-starved), move tasks creeping (+0.02-0.05 @t1.0). val_general improving (-0.809->-0.609 shaped).
- Live question for steps 50-100: does tracing skill compose into judge/edge (curriculum thesis) or stay task-shaped (skill-level selection result)? Either outcome is C1-relevant.
- Watch: controller's first live signal-driven demotion as chain saturates.

## 2026-08-06 12:10 — [fork] Occupancy length non-compression: expected; read trajectories (Joshua's ask)

- k_occupancy flat (659->646 p50) at p~1: weak-by-design pressure (proportional mean-only advantages post-deadband) x floor allocation (~1.5 groups/step). Trajectory reads: think = board-narration ritual (full row-by-row re-derivation for a 1-cell lookup; late sample dithers on answer format at cap until force-closed). Massive slack, negligible force.
- Decision: don't tune for it — compressing a solved diagnostic task inverts Neyman logic. Designated observable instead: chain (4x allocation, approaching saturation) — if k_chain falls as p_chain saturates, length economics works; if pinned, lambda under-tuned and worth revisiting.

## 2026-08-06 12:55 — [fork] Step-75: controller demotion works; judge negative-transfer candidate

- General val climbing (-0.532); edge val flat at baseline after 75 edge-heavy steps — transfer absent so far; edge>0.8-by-150 prediction (60%) in danger.
- Controller: first live signal-driven demotion (chain 0.126->0.087 as p->0.71, sig->0.44). Working as designed.
- **Judge degrading: p 0.49->0.36, sig 0.023 (confidently wrong)** while chain rose — negative-transfer candidate: pair-tracing training may interfere with whole-board judgment. Step-100 battery: judge accuracy on fixed boards + CoT reads for misapplied tracing procedure.

## 2026-08-06 13:40 — [fork] Anti-useless-thinking package (Joshua's ask) + step-100 restart bundle

- Decision-point measurement (12 winning traces, live server, greedy force-close at 10/25/50/75%): move-task answers NOT stable early — most settle only after 75% of think; useless-suffix lower bound 0.08. Split finding: resolvable tasks (occupancy) = ritual + dead suffix; move tasks = late CoT keeps FLIPPING the chosen move without improving average quality (slow re-sampler, not value-adder). Counterfactual-truncation PENALTY would bite little on move tasks; bias-extension is the right lever — reward adjudicates whether answer-flipping suffixes earn their tokens.
- Step-100 val: general -0.433 (monotone climb continues), edge -0.704 (STILL flat; transfer absent at 100 steps).
- RESTART BUNDLE from ckpt 100: CLOSE_BIAS 192:0,1088:30 (closes possible from token 192, +30 throughout), LEN_LAMBDA 0.4, mate1_v2 hot-added (1200 game-natural endgames via back-up-one generator, importance 0.8).
- Registered: mean response length < 700 by step 200: 55%. Win-rate drop >0.05 from length pressure (t1.0 train): 30%. Edge conversion benefits from mate1_v2's natural distribution (val_edge reward > -0.5 by step 200): 35%.

## 2026-08-06 14:30 — [fork] C3 LANDMARK at armC step 100: first load-bearing CoT, edge-specific

- No-think ablation vs with-CoT val (same positions, t0.6): general 0.334 vs 0.300 (CoT still decorative, replicating arm A); **edge 0.008 vs 0.133 — 16x collapse without CoT.** First load-bearing CoT in the project, localized to the curriculum-targeted skill. Edge ability is CoT-mediated (chain-tracing procedure executing in-context), NOT absorbed into the policy head.
- Reframes "no transfer": transfer exists but flows through the CoT channel; shaped-reward val was too blunt to show it. C3's "does RL create load-bearing reasoning where none existed" has a positive instance.
- Follow-ups queued: CoT reads on edge successes (is the tracing verbalized explicitly?), probe activations at ckpt-100 vs 50 (does the concept show in the residual stream, and did it arrive with or before the verbalization?), and tracking whether continued training distills the CoT-borne skill into the head (no-think edge curve per checkpoint).

## 2026-08-06 15:20 — [fork] Step-125: compression and edge skill rising TOGETHER

- Edge kind_win 0.133 -> 0.217 (best yet; mate1_v2 online). General 0.347. Length mean 954 (off the cap), min 236 — bias-from-192 + lambda-0.4 working. The length-price-vs-CoT-borne-skill tension resolving favorably so far: suffix being trimmed, tracing preserved.

## 2026-08-06 16:10 — [fork] Dense objectives built (Joshua: "one bit per 120s is absurd")

- Diagnosis quantified: binary groups yield ~2-3 bits per ~2M generated tokens. Fix: set-valued listing tasks graded per-cell from EXISTING exact labels (the solver luxury — dense supervision without a PRM).
- Built: listing reward branch (score=(TP-FP)/|truth| clipped, spam clamps to -1, continuous => rich group rankings under mean-only advantages); winset (list ALL winning moves, mean 7.6 target cells) + chainset v2 (chains >=3, mean 4.4 cells). Staged in data/curriculum_staged/ — reward branch needs worker reload, so they enter at the step-150 restart, NOT hot-add (premature drop would mis-score via the move branch).
- Registered: winset score > 0.5 mean within 100 steps of entry: 55%. Dense categories accelerate val_general win vs pre-150 slope: 50%. Chainset transfer strengthens edge conversion further: 45%.

## 2026-08-06 16:50 — [fork] Format-compliance ladder (Joshua's pre-emptive check) + answer-budget bug

- Known handling surveyed: scaffold injection + lenient parse (ours), format rewards, constrained decoding (vllm guided_regex — untested through verl server, noted).
- Measured strict compliance on identical content (12 boards, list-empty-cells): comma-list 0.25, JSON array 1.00, spaces 0.92, semicolons 0.92. Our chosen format was the model's WORST; JSON is held rigidly. All listing categories switched to JSON arrays pre-entry (JSON-first parser, regex fallback keeps 0.67 partial credit on sloppy answers).
- The question also surfaced a real bug: ANSWER_BUDGET=8 would have truncated listing answers (~25 tok) mid-list => spurious FN sets. Task-aware budgets now (8 move/judge, 48 listing).
- Both fixes land in the step-150 restart with winset/chainset.

## 2026-08-06 17:30 — [fork] Witness/certificate category built (Joshua's multi-output suggestion)

- "Winner + explicit winning path" — composes judge + chain-tracing, the exact composition step the curriculum circles. Path = verifiable certificate (per-cell color, per-link adjacency, edge endpoints — pure board logic); grading 2*link_frac-1 with winner-gate. Tested: valid 1.0 / broken-link 0.6 / wrong-winner -1.
- Manufactures judge's missing sigma: can't guess-deterministically a path; partial credit varies within groups.
- 1000 terminal boards staged (536B/464W), JSON-object format (per ladder), 64-tok answer budget. Family noted for later: mate-in-1 move+resulting-chain, disjoint-double-path (bridge certificates), blocker sets.
- Step-150 restart bundle now: winset + chainset + witness + JSON formats + task-aware budgets + listing/path reward branches.
- Registered: witness link_frac mean > 0.7 within 100 steps: 55%. Witness training moves JUDGE p above 0.6 (transfer through composition): 45%.

## 2026-08-06 18:00 — [fork] Certificate inventory + "is RL the right teacher?" (Joshua)

- New certificate candidates (board-logic verifiable): mate1+chain, cut-set (breaker witness), disjoint double-path (bridge/robustness certificate), mate-in-2 strategy tree, completion-set.
- Structural answer on RL-vs-SFT: certificates uniquely close the demonstration gap — the label IS the reasoning artifact (BFS/solver writes gold certificates; nothing can write gold move-CoTs). Division of labor: SFT injects procedures where certificates exist; RL for decisions where they don't; RL-after-SFT to pressure-test on-policy.
- Planned step-250 branch experiment: pure-RL-on-witness vs certificate-SFT-then-RL; compare witness curves, judge transfer, and whether SFT-injected tracing becomes load-bearing on move tasks (C3 from the reverse direction: does RL adopt a taught procedure as a tool?).
- Registered: SFT branch link_frac>0.9 within 20 steps: 75%. SFT branch beats pure-RL on judge transfer within 50 steps: 55%.

## 2026-08-06 19:20 — [fork] Dense-category first read + think-cap clipping bug

- chainset: HEALTHY and learning (+0.357->+0.407, 0% unparsed) — atomic chain skill transfers to dense listing. winset: parses cleanly, score -0.54 flat (real headroom, partially depressed by the bug below). witness: 96% unparsed.
- Bug (read from raw tails): answers truncated mid-JSON at ~12 tokens — think cap still assumed the 8-token answer budget; think(1088)+scaffold+answer(64) > response_length(1104) => final clip ate answers. Listing categories silently FN-truncated too (fallback parser masked it). Fix: per-task think cap = response_length - answer_budget - 8. Lands at ckpt-200 restart (nothing lost waiting; witness slice wasted until then).
- Meta: third instance of "budget arithmetic bites at the boundary" (2160-vs-2176, ANSWER_BUDGET listing, now think-cap). Pattern logged: any new answer format => recheck the full token ledger end to end.

## 2026-08-06 20:40 — [fork] Witness fixed and baselined; branch experiment armed

- Post-fix: witness unparsed 0.96 -> 0.02, baseline link_frac 0.44. Same-checkpoint double-val across the restart calibrated the val noise floor: ~±0.04 general (n=277), ~±0.09 edge (n=60) at k=1 — retroactive caution on past edge-swing readings.
- At ckpt 250: branch (a) pure-RL continue vs (b) certificate-SFT (3k gold pairs) then RL. Metrics: witness link_frac curve, judge transfer, move-task transfer, load-bearing checks. The teaching-channels comparison is the culmination of the day's task-design arc.

## 2026-08-06 22:30 — [fork] Answer-variance decomposition (Joshua's fixed-think question)

- Modal-agreement, fixed-think resample vs across-group: judge 0.917/0.802, general 0.750/0.616, winset 0.556/0.277. Think causally conditions answers everywhere (fixed > across), but answer-phase residual noise is large (0.08/0.25/0.44): the model re-transcribes its own completed reasoning differently per sample, worst on structured answers.
- Implications: (1) GRPO credit contamination — think tokens are punished for transcription slips; two-temperature rollouts (hot think, cool ~0.3-0.5 answer) queued POST-branch-experiment with prediction to register. (2) C3 refinement: CoT is a causal-but-lossy channel; tracking fixed-think agreement per checkpoint as the channel-fidelity curve.

## 2026-08-06 23:00 — [fork] Answer-branching = Rao-Blackwellized think credit (Joshua's proposal)

- At </think>, sample n=8 answers in ONE vllm request (shared KV; +10-15% compute): think tokens get mean-over-answers reward (E[r|think] estimate — kills the 8-44% transcription-noise term in think advantages); optional level 2: answers judged against within-think siblings (transcription-targeted gradient). Vine/tree-GRPO at the branch point forced-close created for free. Subsumes & beats the two-temperature idea.
- Implementation ladder: (i) loop-computed mean reward via AgentLoopOutput.reward_score (verify verl precedence), random single answer emitted for tokens; (ii) hierarchical advantages (trainer surgery, later).
- Queued behind branch experiment. Registered: faster winset/witness convergence vs pure-RL slope: 60%. Per-think answer agreement rises: 65%.

## 2026-08-06 ~10:50 — SFT branch complete; RL-from-SFT (armC_sftrl) launching

Disk crisis killed armC at step 298/400 (user cleaned; branch-a window 250→298
complete in side channel, 82k records). Pivoted to the queued branch experiment.

**SFT leg**: certificate SFT from checkpoints/armC/global_step_250/hf on 2850
gold witness pairs, 2 epochs (88 steps, ~6 min), final val/loss 0.045. Needed
`data.enable_thinking_default=true data.ignore_input_ids_mismatch=true`
(Qwen3 think-tag chat-template mismatch trips MultiTurnSFTDataset's per-turn
sanity check; documented escape hatch in verl source). Merged rank-0 fp32 .pt
→ bf16 safetensors at checkpoints/armC_sft_cert/global_step_88/huggingface.

**Spot check (6 witness prompts, temp 0.6)**: mean score **0.96**, 5/6 perfect
paths. Pure-RL branch was at link_frac ≈ 0.44 after ~50 steps of RL on this
task. **Prediction "SFT link_frac>0.9 within 20 RL steps @75%" resolves YES at
RL step 0** — the SFT alone did it. Also notable: the SFT'd model emits the
JSON answer *immediately, with no think narration at all* (len ~70 chars) —
2 epochs on short-think gold pairs collapsed the think phase. The RL leg's
forced-close scaffold reopens a think phase regardless; watch whether think
re-grows or stays vestigial on witness (and whether no-think transfer holds).

**Answer-branching hardened before launch** (verl source verified):
- AgentLoopOutput.reward_score, when set, bypasses the async reward path
  entirely (agent_loop.py: _compute_score skips; rm_scores built from it) —
  so the reward-fn side channel would go silent. Loop now scores each branch
  itself (logging suppressed via env pop in a sync block) and writes ONE
  side-channel record: score = picked branch's shaped score, shaped = branch
  mean (what GRPO trains on), + branch_scores/n_branch fields.
- Val gating: run() never sees verl's validate flag; gate on temperature<1.0
  (val_kwargs.temperature=0.6) so val stays unbranched/comparable.
- Graceful fallback if server ignores n>1 (flat token_ids): single-answer
  score, n_branch field in side channel reveals it → smoke check.

Launching: armC_sftrl from the SFT ckpt, STEPS=50, same curriculum/config as
armC (λ=0.4, bias 192:0,1088:30, mean-only adv), ANSWER_BRANCH=8.

## 2026-08-06 ~11:20 — answer-branching live (parallel-request fix)

Smoke on first launch: side channel showed n_branch=1 — verl's vllm server
passes n into SamplingParams but returns only outputs[0] (TokenOutput.token_ids
is flat list[int]); 7 of 8 answers were generated and discarded server-side.
Fix: n parallel answer requests (prefix cache shares think KV; answers 8-64
tok). Restarted armC_sftrl from SFT ckpt (lost 7 steps). Verified at step 1:
300/300 train records n_branch=8. Branch spreads confirm the premise — e.g.
witness think with branch=[0.86,0.99,-1,-1,-1,-1,-1,-1] → shaped=-0.52 instead
of a ±1 coin flip. Controller live (armC_sftrl-controller); first tick:
witness p=0.74/reward +0.40 (SFT transfer holds under RL rollouts), but SFT
forgetting elsewhere: chain 0.88→0.39, occupancy ~1.0→0.39, judge →0.55.
Watch: does RL re-recover forgotten cats faster than it learned them (relearn
speed = selection evidence), and does witness stay high.

Disk: pilot_1p7b + smoke ckpts deleted after B2 verify (17G); armC/250 actor
(21G) + SFT fp32 master uploading, then local delete; SFT optim state (13G)
deleted without upload (Adam moments of a 6-min-reproducible SFT).
support_loops.sh (janitor+backup) replaces the ad-hoc loops; <25G disk monitor
armed.

## 2026-08-06 ~11:45 — post-SFT CoT collapse: think phase is now a draft answer

Eyes-on-data at armC_sftrl step ~10: witness think = `Answer: {...}<|im_end|>`
(108 chars — a draft answer, then eos inside think); chain think = a literal
echo of the format spec line. The certificate SFT (gold pairs with short
narration) collapsed the think channel entirely, and that carried into RL
rollouts: the model drafts an answer, hits eos, scaffold closes think, and it
answers again — branch answers then scatter AROUND the draft (e.g.
[0.61,-1,0.99,0.61,...]). Entropy also down (0.12-0.14 vs 0.15-0.22 pre-SFT).

Natural experiment now running: with think ~100 chars the length price is ~0
(max bonus), so nothing pushes thinking back EXCEPT accuracy. If thinking
re-grows on tasks where a draft answer is insufficient (edge_m1, mate2,
winset) but stays vestigial on procedure tasks (witness), that's the
resolution-gating story with a causal arrow: think length tracks task demand,
not reward-hacking pressure. Registered predictions:
- P(think re-grows >300 chars median on ≥1 move task by step 40) = 0.5
- P(witness think stays <150 chars median through step 40) = 0.7
- P(witness p ≥ 0.85 by step 40) = 0.6

## 2026-08-06 ~12:00 — ROOT CAUSE: v1 SFT was accidental think-ablation

Decoded the actual loss-masked target of the v1 certificate SFT:
`Answer: {"winner": ...}<|im_end|>` — NO think. MultiTurnSFTDataset renders
each turn through the chat template, and Qwen3's template strips
<think>...</think> from assistant messages; the sanity check we bypassed with
ignore_input_ids_mismatch was warning about exactly this. So v1 trained 2
epochs of board→answer with the reasoning deleted. Reframes everything:

- "Post-SFT CoT collapse" is an artifact of the pipeline, not a property of
  certificate SFT. The armC_sftrl leg is really "RL on a think-ablated
  policy" — a different, still-useful experiment. Verdict through step 20:
  RL does NOT re-grow ablated think (lengths pinned ~25 chars all categories),
  witness climbs 0.74→0.83 (the one skill the ablated mapping supports),
  everything else decays (judge 0.66→0.49, occupancy →0.27, chainset →0.08,
  degenerate `White|White|...` echo loops appearing). Exploration in
  think-space is dead: P(long think) ≈ 0, nothing to reinforce.
- Accidental C3 gem: 1.7B does path-tracing with ZERO CoT at 0.83 accuracy
  after 2 epochs of answer-only SFT — the procedure internalized into the
  forward pass. Verbalization is not necessary for the certificate skill.

Fix: hexenv/sft_cert_dataset.py (CertSFTDataset) tokenizes assistant turns
raw so the narration survives; acceptance test = decode input_ids[loss_mask]
and see the think block (verified). New meta-rule, added to yolo recipe:
NEVER launch an SFT without decoding the loss-masked tokens first — the
token-ledger audit rule, extended from RL formats to SFT targets.

Plan: checkpoint armC_sftrl at step 25, kill, rerun SFT as armC_sft_cert_v2
(think in loss), relaunch RL as armC_sftrl2. The v1 leg becomes the
think-ablation arm of the comparison — three-way now:
pure RL (armC 250→298) vs ablated-SFT+RL (sftrl) vs narrated-SFT+RL (sftrl2).

## 2026-08-06 ~12:40 — narration is a commitment device: branch-SD collapses to ~0

armC_sftrl2 (narrated SFT + RL) vs armC_sftrl (ablated SFT + RL), mean
within-think branch-score SD (8 answers per think, same-prompt):

  category   ablated  narrated      category   ablated  narrated
  witness     0.276     0.000       judge       0.238     0.000
  chain       0.244     0.012       occupancy   0.144     0.017
  edge_m1     0.104     0.000       general     0.088     0.020

10-100x collapse. When the think ends with a stated conclusion, the answer
phase is a deterministic transcription; when think is absent/vestigial, the
answer is sampled and noisy. C3 shape: the narrated CoT is a *commitment
device* — verbalization pins the answer. Corollary: answer-branching
(Rao-Blackwell) only buys signal when the think→answer channel is lossy;
for narrated policies it's inert. YOLO implication: branching is cheap
insurance early in training and after policy changes, useless once the
policy narrates conclusions.

Costs of narrated SFT, step-5 snapshot: think style contaminated ALL
categories (~180-230 char witness-style narration everywhere; pre-SFT ~950);
occupancy crashed 1.0→0.12 (wrong procedure narrated confidently), witness
0.66 (below ablated 0.83 and v1-spot 0.96), judge 0.58, but general move
p=0.15 vs 0.07 pre-branch and entropy healthier (0.24 vs 0.12). BOTH SFTs
damaged non-target categories, by different mechanisms: ablation kills the
compute, narration overwrites the procedure. A 2850-example single-task SFT
is a blunt instrument at 1.7B either way — YOLO recipe should mix SFT tasks
or interleave SFT with RL, not run task-pure SFT blocks.

## 2026-08-06 ~13:30 — three-way branch verdict: task-pure SFT is globally destructive at 1.7B

Final comparison (rollout p / median think chars, last quarter of each leg;
pure RL = armC steps ~285-298):

  cat         pureRL@298   ablated@25   narrated@50
  witness      0.61/2671     0.84/90      0.64/184
  judge        0.48/2927     0.49/46      0.40/184
  chain        0.77/2890     0.47/40      0.36/194
  occupancy    0.93/1107     0.35/38      0.20/140
  chainset     0.90/2655     0.13/68      0.16/173
  general      0.42/2491     0.12/41      0.15/233
  (val_general kind_win: pure RL 0.426 → SFT legs 0.14; unrecovered by RL)

Findings:
1. Certificate SFT is wildly compute-efficient AT ITS TASK: 6 min of SFT beat
   ~50 RL steps (witness 0.84-0.96 vs 0.61). The demonstration-gap hypothesis
   confirmed for the target skill.
2. But BOTH SFT variants destroyed the rest of the policy (general move skill
   0.43→0.14 val; chainset 0.90→0.13; occupancy 0.93→0.35) and collapsed
   think 2500-2900 → 40-240 chars globally. 50 RL steps at lr 1e-6
   (ppo_kl ~1e-5/step) repaired little: chain/occupancy partial recovery in
   the narrated leg, nothing else. RL-after-SFT at standard hypers is NOT a
   consolidation phase on this timescale; it's a slow drift.
3. Narrated SFT ≠ safer: it overwrote the think procedure everywhere
   (confident wrong-winner narrations; judge skill sank to 0.40-0.49 in both
   legs). Gold certificates trained verification format, never winner
   DISCRIMINATION — the decision half stayed untaught (SFT-for-procedures /
   RL-for-decisions line, empirically).

Prediction grades:
- "witness link_frac>0.9 within 20 RL steps of SFT" (75%) → YES (at step 0;
  via the ablated variant).
- "SFT beats pure-RL judge transfer" (55%) → NO (0.40-0.49 vs 0.48; no
  transfer, arguably negative).
- "answer-branching → faster winset/witness convergence" (60%) → drowned by
  SFT damage; ungradeable, trending NO.
- "per-think answer agreement rises under RL" (65%) → resolved by a different
  mechanism: agreement ≈ 1 immediately when narration exists (commitment
  device), not trained up gradually.
- "witness think stays <150 chars" (70%) → YES (ablated ~90 through 25;
  narrated pinned ~184 = SFT length through 50).
- "think re-grows >300 chars on a move task" (50%) → NO in both legs.

YOLO consequence (recipe updated): if SFT is used at all, it must be
replay-mixed (certificates + on-policy samples of every other category) or
interleaved SFT/RL — never a task-pure block. Next cheap probe: replay-mix
SFT from ckpt-250 (2850 certs + ~3k self-samples across categories), same
spot-checks, to see if the destruction is avoidable while keeping the
6-minute certificate win.

## 2026-08-06 ~14:30 — replay-mix SFT round 1: breadth preserved, cert skill collapsed

armC_sft_replay (1000 certs + 677 correct self-samples, 3 ep, val loss 0.126).
Spot-check vs armC-250 baselines: chain 0.87 (0.77-0.88), occupancy 0.97
(0.93), chainset 0.73 (0.90), general 0.43 (0.42), edge 0.23 (0.22), judge
0.60 — REPLAY WORKS: ~full breadth preservation, long thinks retained on
replayed cats. But witness fell to 0.23, and eyes-on-data shows a degenerate
template: 8/8 answers "White" with a fabricated left-edge staircase (labels
in mix were balanced 532B/468W — not a data skew). 1000 certs x 3ep among
replay taught format-without-computation; 3000 x 2ep (v2, task-pure) had
actually traced (0.58). Cert VOLUME appears to matter for the tracing skill;
replay PRESENCE (~40%) suffices for preservation.

Also caught by the standing acceptance test: double <think> wrap in replay
targets (model emits its own opening tag; builder wrapped again). Fixed
before training.

Round 2 running: armC_sft_replay2 = 3000 certs + replay x4 (5588 rows, 2 ep)
— tests volume-for-skill + ratio-for-preservation simultaneously.
Predictions: witness ≥ 0.7 @60%; chain/occupancy/general within 0.05 of
baseline @70%.

## 2026-08-06 ~15:10 — replay round 2: volume doesn't fix it; interference does it

armC_sft_replay2 (3000 certs + replay x4, 2ep, val 0.043): breadth even
better preserved (chain 0.90, occupancy 1.00, chainset 0.90, general 0.37,
edge 0.23) — breadth prediction (70%) YES. But witness 0.25, same 56-token
stereotyped template — witness≥0.7 (60%) NO. Same cert volume/epochs as the
task-pure v1 (0.96 solo), so the collapse is INTERFERENCE from co-trained
long-CoT data, not insufficient volume. Skill-through-verbalization degrades
monotonically with data diversity: ablated-pure 0.96 → narrated-pure 0.58 →
narrated-mixed 0.25.

Round 3 (armC_sft_replay3, running): ABLATED certs (answer-only targets, the
0.96-solo variant) + same replay x4. If witness stays high AND breadth holds,
the C3 statement is sharp: this procedure is best installed weight-level,
bypassing the CoT channel; the verbalized version is fragile to mixing.
Predictions: witness ≥ 0.7 @55%; breadth within 0.05 @75%.

## 2026-08-06 ~16:00 — the SFT 2x2: verbalization and mixing each tax the skill ~40%

armC_sft_replay3 (ablated answer-only certs + replay x4, val 0.048):
witness 0.57 / think 36 tok; breadth intact (chain 0.90, occupancy 1.00,
chainset 0.73, general 0.33, edge 0.23, judge 0.53). Completed 2x2 (witness
spot-check accuracy):

                   pure     + replay
  ablated certs    0.96       0.57
  narrated certs   0.58       0.25

Two ~independent multiplicative taxes: teaching through the CoT channel
(~0.6x) and co-training on diverse data (~0.6x). Best joint cell = ablated
certs + replay: cert skill at 0.57 with zero collateral damage, and per-task
routing emerges (instant answers on witness, full-length thinks elsewhere).
0.57 is plausibly winner-ID-capped (judge ~0.5; winner-gated scoring) — the
decision half that gold certificates structurally can't teach.

Predictions for the consolidation RL leg (armC_sftrl3 from replay3):
- witness ≥ 0.75 by step 50 (RL supplies winner discrimination) @65%
- chain ≥ 0.80 at step 50 (breadth maintained under RL) @70%
- witness think stays < 100 tok median @70%

## 2026-08-06 ~17:00 — sftrl3 interim: think inflation + entropy collapse degrade chain

Consolidation leg through ~step 15: witness 0.57→0.60 (t84, branch_sd 0.65 —
the RB-branching sweet spot); but chain 0.88→0.38. Diagnosis (eyes on data):
thinks are sane but INFLATED (replay's verbose cell-by-cell board re-parsing
style), now overrunning the 1088-tok budget → forced close mid-reasoning →
confident wrong answer. Two compounding causes: (1) self-distillation
sharpened the policy to entropy 0.05, so exploration back to concise
conclusions is nearly dead; (2) correctness-gated length price has zero
gradient among wrong samples (known failure mode from arm A). Note armC pure
RL concluded within the same budget; the replay style is more verbose per
unit content. Lesson forming for the recipe: replay data should be
length-filtered or entropy-preserved (sample at temp 1.0, or mix multiple
temps) — self-distilled SFT at temp 0.6 bakes in both verbosity and
overconfidence.

## 2026-08-06 ~18:30 — sftrl3 final: same accuracy as pure RL at 30x cheaper witness; discrimination is the shared bottleneck

Final window (steps ~40-50): witness 0.61/t84, chain 0.78 (recovered from
0.38 dip), judge 0.61, general 0.44, occupancy 0.99, chainset 0.57,
mate1_v2 0.38, mate2 0.29, edge 0.23; val_edge kind_win 0.217 (ties armC
best), val_edge reward -0.62 (best of any leg).

Prediction grades:
- witness ≥0.75 by 50 (65%) → NO (0.61; +0.04 over the leg).
- chain ≥0.80 held (70%) → NO by a hair (0.78, after a transient 0.38 dip
  that RL corrected — the inflation-truncation was self-healing).
- witness think <100 tok (70%) → YES (84 chars ≈ 28 tok, stable).

The synthesis:
1. armC_sftrl3 ≈ pure-RL armC@298 across the curriculum, sometimes better
   (general 0.44 vs 0.42, mate1_v2 0.38, judge 0.61), with witness at the
   SAME 0.61 pure RL reached — but at ~28 think tokens vs ~900. The
   SFT-installed procedure delivers equal accuracy at ~30x less think
   compute. "Cheap skill installation" is real; acceleration beyond the
   RL asymptote is not (on this budget).
2. Witness ceilings at ~0.6 in EVERY leg, and judge sits at the same ~0.5-0.6
   — winner discrimination is the shared bottleneck no teaching channel
   cracked (ablated SFT, narrated SFT, replay mixes, RL, RL-after-SFT).
   The verification half is nearly free; the decision half is the hard
   kernel. This is the project's cleanest statement of the SFT/RL division.
3. Entropy 0.05 did NOT kill RL: chain recovered, judge/general improved.
   Post-self-distillation RL grinds slowly but is not dead at 1.7B scale.

Artifacts: checkpoints/armC_sftrl3/global_step_50 (best overall policy),
side channels for all four legs, wandb runs + controller mirrors.

## 2026-08-06 ~19:50 — Arm D launched: witness 2x2-5x5, LoRA SFT, no thinking

Motivation (user): RL teaches new skills too slowly. Arm D isolates the
witness task on tiny boards and asks (1) can Qwen3-1.7B learn it AT ALL,
(2) does CoT then buy anything. Test 1: teacher-forced SFT on a rank-32
LoRA from BASE instruct Qwen3-1.7B (not an RL ckpt), thinking disabled
(chat-template empty think block), pass bar mean score > 0.90 on held-out
boards via the real grader.

Design deltas vs arm C certificates:
- Boards 2x2-5x5, dedup by position, and only boards whose winner has a
  UNIQUE minimal winning path (BFS path counting) — unambiguous target.
  Pool sizes: 2x2 has only 9 such boards in existence (6B/3W), 3x3 841,
  4x4/5x5 capped at 1150. Split 2697 train / 151 val / 302 test.
- Token-importance loss weights: every completion token weight 2.0
  (structure/winner errors gate score to -1 = 2.0 achievable-score loss),
  except path-cell tokens which get 1 - grader_score(gold with that cell
  deleted) ~ 0.22-0.67. Flows through verl no_padding sft_loss as weighted
  CE (sum(w*ce)/sum(w)). Uniform-weight control trains alongside
  (ARMD_UNIFORM=1 binarizes the mask).
- Precomputed input_ids/loss_mask in parquet (hexenv/armd_sft_dataset.py);
  acceptance: decoded tokens+weights eyeballed, 20/20 gold rows score 1.0,
  prompt reconstruction byte-identical to curriculum witness rows.

Baseline (base model, no think, temp 0, 302 test rows): mean -0.877,
perfect 0.000, 114/302 unparsed; hallucinates off-board cells. Zero point.

Predictions (registered before training):
- P1: weighted run reaches mean > 0.90 on test at temp 0 within 3 epochs @60%
- P2: per-size monotone: 5x5 is the weakest size @70%
- P3: score decreases monotonically in gold path_len (3..6, n>=20 bins) @70%
- P4: weighted vs uniform final |delta mean| < 0.05 (weighting doesn't
  matter at convergence on this budget) @60%
- P5: >=1 of the two runs still emits an off-board cell on some test row
  after 3 epochs @55%

Hypers: lr 1e-4 cosine, bs 64, 3 epochs, lora r32 alpha 64, max_len 2048.
wandb: verl loss curves + per-epoch generation evals (train-sample/val/test
score, per-size + per-path_len) logged by scripts/eval_armD_witness.py.

## 2026-08-06 ~20:30 — Arm D interim: 3 epochs miss the 0.90 bar; undertrained, not capacity-capped

LoRA r32, 3 epochs, temp-0 test (302 rows): weighted 0.852, uniform 0.836
(baseline -0.877). Trajectory ep1->2->3: weighted 0.558->0.786->0.852, still
climbing steeply; TRAIN-split score is only 0.896 with NO train/val/test gap
(0.896/0.883/0.852) => optimization-limited, not data- or capacity-limited.
Failure anatomy (weighted ep3): 75/302 imperfect, only 14 wrong-winner,
0 unparsed — errors are near-miss paths with one broken link; monotone in
size (3x3 0.983 / 4x4 0.866 / 5x5 0.706) and path length (plen3 0.985 ->
plen6 0.643). Winner-ID + format are learned; chain-tracing at length is the
residual skill.

Weighting effect so far: weighted beats uniform on MEAN score (+0.016) but
loses on %PERFECT (0.752 vs 0.808) — the exact signature of the importance
weights (protect winner/structure, discount individual links).

Decision: minimal adjustment = same recipe, 8 epochs (fresh runs, both arms).
Fat FSDP checkpoints (8GB/epoch) exceeded disk; rolling export-to-peft-adapter
(67MB) + delete keeps it flat. Prediction: weighted_e8 test mean > 0.90 by
epoch 8 @70%; train-split mean > 0.97 by epoch 8 @70%.

## 2026-08-06 ~22:00 — Arm D test 1 PASSES: 0.980 no-think witness via rank-32 LoRA, 8 epochs

Weighted run, temp-0 test (302 held-out boards): ep4 0.946 -> ep8 0.980 mean,
94.0% perfect, 1 wrong winner, 0 unparsed, 0 off-board cells. Per-size at
ep8: 2x2 1.00, 3x3 1.00, 4x4 0.993, 5x5 0.947. Per-path-len: 3 1.00, 4 0.997,
5 0.980, 6 0.938, 8 -0.18 (n=2 — long paths are the surviving failure).
Uniform control: crosses 0.90 at ep5, plateaus 0.934-0.939; weighted is
above uniform at EVERY epoch >=4 (max delta +0.082 at ep4, final +0.046).
Residual errors: single broken link mid-path on 4x4/5x5; format and winner
fully solved.

Prediction grades:
- P1 mean>0.90 within 3 epochs (60%) -> NO (0.852 at ep3; crossed at ep4).
- Extension: >0.90 by ep8 (70%) -> YES (0.980); train>0.97 by ep8 (70%) ->
  YES (0.990).
- P2 5x5 weakest size (70%) -> YES.
- P3 monotone in path_len 3-6 (70%) -> YES (1.00/0.997/0.980/0.938).
- P4 |weighted-uniform|<0.05 final (60%) -> YES by the letter (0.046) but
  the spirit was wrong: weighting helped consistently, whole trajectory.
- P5 off-board cell survives 3 epochs (55%) -> ep3 not re-checked, ep8 NO
  (zero in 302x~5 cells) — hallucination fully trained out.

Headline for the arm D question ("can it learn the task at all?"): YES,
cheaply. ~40 min of teacher-forced LoRA SFT (2697 rows x 8 ep, r32, no CoT)
takes base Qwen3-1.7B from -0.877 (0% perfect, 38% unparsed) to 0.980.
Contrast: ~10h of curriculum RL (arm C) plateaued at ~0.61 on 5x5 witness.
The task was never hard — it was undemonstrated. RL's bottleneck (winner
discrimination gating path credit) simply doesn't exist under teacher
forcing. Token-importance weighting (grader-counterfactual weights) is a
real but modest accelerant: reaches any given score ~1 epoch sooner and
adds ~0.05 asymptotically over uniform CE.

Artifacts: checkpoints/armD_sft_{weighted,uniform}_e8/adapter_ep{1..8}
(67MB peft adapters; fat FSDP ckpts rolled up and deleted), curves in wandb
runs armD_sft_*_e8_scores (test/val/train x size x path_len), rollout jsonls
in results/armD/.

Next (arm D phase 2, per plan): does CoT buy anything? No-think saturates
2x2-5x5, so the phase-2 frontier must be where no-think fails: path_len>=7
and/or 6x6-7x7 boards. Design: same pipeline, long-path-biased sampling,
compare no-think LoRA vs think-target LoRA at matched compute.

## 2026-08-07 ~00:15 — Arm D v2: constructive boards 2x2-9x9, filter corrected to induced paths

User's cross-check exposed that v1's uniqueness filter counted unique
SHORTEST paths (BFS), but "minimal winning path" = unique INDUCED path
(chord-free vertex set; hex = triangular lattice). Enumerator reproduces the
user's count sequence exactly through n=8 (1,3,11,54,365,3848,68914,2195830;
n=9 = 126,004,636 pending, pure-python DFS). v1 leak quantified: 359/2697
train boards (13%) had a second longer induced path — 0% at 2x2, 22% at 5x5,
growing with size.

v2 pipeline (scripts/witness_constructive.py + build_armD_witness_v2.py):
PLANT a random induced path (self-avoiding walk, p_forward controls length;
25% long-path bucket), add winner distractors kept only if induced-path
count stays 1, add loser stones at game-consistent counts (half adjacent to
the path as blocking attempts), reject if loser connects. 100% acceptance at
~1ms/board at 9x9 (rejection sampling was already ~50% at 5x5). Every board
reachable by legal alternating play (unique path => winner connects exactly
on the final stone). Solver self-play considered and rejected as overkill
(cost + density fights the uniqueness filter).

Dataset: 6636 train / 351 val / 702 test, sizes 2-9 (2x2 pool is exactly
9 boards — matches the closed-form enumeration), winner-balanced, path_len
2-29, mean plen 11.4 at 9x9. Same no-think targets + token-importance
weights as v1.

Predictions for armD2_sft_weighted (r32 LoRA, lr 1e-4, 8 epochs):
- P-v2-1: overall v2-test mean > 0.90 by ep8 @55%
- P-v2-2: per-size monotone decreasing, 9x9 lowest @80%
- P-v2-3: 9x9 mean > 0.75 by ep8 @55%
- P-v2-4: transfer to v1 playout test (2-5) mean >= 0.93 @60%
- P-v2-5: path_len>=10 bin mean < 0.70 at ep8 (long paths stay hard =>
  the CoT-phase frontier) @60%

## 2026-08-07 ~00:20 — Arm D v2 PASSES at 9x9: 0.975 overall; the CoT frontier is exact-path perfection at plen>=14

armD2_sft_weighted (r32 LoRA, 8ep on 6636 constructive boards 2x2-9x9),
temp-0 ep8: v2 test 0.975 mean / 93.2% perfect (n=500), crossed 0.90 at ep2.
Per-size: 3x3 1.00, 4x4 0.991, 5x5 0.970, 6x6 0.968, 7x7 0.982, 8x8 0.941,
9x9 0.970. 34/500 imperfect: 3 wrong winner, 0 unparsed. n=9 minimal-path
count independently confirmed = 126,004,636 (user's closed-form vs my DFS).

Prediction grades:
- P-v2-1 overall >0.90 (55%) -> YES (0.975).
- P-v2-2 monotone in size, 9x9 lowest (80%) -> NO: 8x8 is the trough
  (0.941), 9x9 ties 5x5. Difficulty tracks PATH LENGTH, not board area;
  size ordering was a proxy that broke.
- P-v2-3 9x9 >0.75 (55%) -> YES (0.970, huge margin).
- P-v2-4 v1 playout-test transfer >=0.93 (60%) -> YES by a hair (0.932;
  by size 2/3/4 near-ceiling, 5x5 0.840). Constructive-trained skill
  mostly transfers to real-game boards; residual ~0.14 gap at 5x5 says
  playout boards (dense, tangled stone sets) are genuinely harder than
  equal-size constructive ones.
- P-v2-5 plen>=10 mean <0.70 (60%) -> NO (0.927) — but for an
  instructive reason: MEAN saturates on long paths because one broken
  link costs only ~2/(2L+1); %PERFECT is the honest metric and it
  collapses: plen>=10 71.9%, plen>=14 36.0%, plen>=17 ~15%. Failure
  reading: long-path errors include LOOPS (model re-enters visited
  cells — a plen-23 sample repeats a 6-cell cycle verbatim), the
  classic no-scratchpad signature.

Arm D question 1 is now fully answered: Qwen3-1.7B + r32 LoRA learns the
witness task to 0.94-1.00 mean at EVERY size 2x2-9x9 with no thinking,
~80 min SFT total. Question 2 (does CoT buy anything?) now has a sharp
target: exact-path perfection on plen>=14, where no-think sits at 36% and
shows visited-cell loops that a scratchpad should fix. Metric for phase 2:
frac_perfect on a long-path-enriched test set, not mean score.

Artifacts: data/armD2/, checkpoints/armD2_sft_weighted/adapter_ep{1..8},
results/armD/armD2_weighted_ep*_{test,val,train,v1test}.jsonl, wandb
armD2_sft_weighted(+_scores).

## 2026-08-07 ~00:10 — Long-path yardstick built; no-think baseline pinned

data/armD2/test_longpath.parquet: 500 boards, sizes 7-9, stratified 125 per
plen bin (14-17 / 18-21 / 22-25 / 26+), winner-balanced, disjoint from all
armD2 splits. Generated in 13s (low p_forward walks).

Baselines on it (temp 0, no think):
- base Qwen3-1.7B: -0.995 mean, 0% perfect (floored).
- armD2_sft_weighted ep8: 0.874 mean, 42.8% perfect, with a SMOOTH
  monotone decline in perfect rate: plen 14-15 ~78%, 18-19 ~63%, 22 ~43%,
  25-26 ~15%, 28-29 ~5%, 31+ 0%. (The v2-test thin-bin numbers at 14-17
  were pessimistic — 50% on n=18 there vs 77% on n=125 here.)

This is the phase-2 metric: frac_perfect on test_longpath, per bin. Wide
dynamic range (78 -> 0), zero saturation, failure mode (visited-cell loops)
mechanistically CoT-addressable. wandb: armD2_sft_weighted_scores
longpath/* (step 0 = base, step 8 = ep8 adapter).

## 2026-08-07 ~01:30 — Phase-2 pilot SURPRISE: the model's own CoT is anti-fuel; more thinking hurts more

Pilot (25 boards plen 8-32, k=8, temp 1.0, forced close only injection):
- base + think: 0 everywhere, natural close 0% (as predicted @70% YES).
- ep8 + think @1024: pass@8 0.60/0.40/0.20/0/0 across bins. Prediction
  "pass@8>0.3 at 14-17" @55% -> YES — but the control inverts the story:
- ep8 NO-think temp1 k=8 (control): pass@8 1.00/0.80/0.80/0.20/0.20 —
  STRICTLY dominates thinking at every bin.
- ep8 + think @2048: 0.60/0/0/0/0 — doubling budget makes it WORSE
  (natural close 1%; the model never finishes thinking on its own, and
  longer rambles drift further before the forced close).

Eyes on CoTs: think content is verbose cell-by-cell board RE-PARSING (with
misreads), never path-search; answers succeed in spite of it. Same failure
family as arm C's replay-verbosity lesson, now measured causally: at this
scale/skill state, the untrained think channel subtracts accuracy —
conditioning the answer on a long, error-laden re-read is worse than
answering from the trained direct mapping.

Also new: no-think temp-1 sampling has fuel DEEP into the frontier
(pass@8 0.20 at plen 22-32, where temp-0 is ~5%). Best-of-k no-think
self-distillation is viable at all bins.

Consequence for the phase-2 design: harvesting own-CoT successes (arm T as
drafted) would distill CoTs that measurably hurt. Options: (B) no-think
best-of-k self-distillation first — establishes the no-CoT ceiling any CoT
method must beat, ~45 min; (T) run the CoT harvest anyway for the honest
negative; (C) RL-with-think on long paths — can reward CREATE a useful
thinking procedure where distillation finds none (the C1 question at its
sharpest). Decision pending user.

Ops note: vllm 0.24 offline engines hang at shutdown ~50% of the time on
this box (orphan VLLM::EngineCore holds GPU; parent in do_wait). Guard all
one-shot vllm scripts with timeout -s KILL and pkill -x the orphans.

## 2026-08-07 ~02:00 — Arm B launched: no-think best-of-8 self-distillation

Plan B->T->C confirmed by user. B and T share one harvest pool (~2600
boards, plen 8-32, deep bins oversampled 700 each) so B doubles as T's
data-matched no-CoT control. B: harvest ep8 no-think k=8 temp-1.0 perfect
answers (1/board, canonicalized), SFT CONTINUING from adapter_ep8
(lora_adapter_path) on harvest + 1000-row armD2 replay, 3 epochs, per-epoch
eval.

Predictions (before harvest results):
- B raises yardstick overall frac_perfect 42.8% -> >=55% @60%
- gains concentrate at plen>=22 (the newly-fueled bins) @55%
- v2-test (short-path breadth) drop <= 0.02 mean @70%
- harvest yield at 26-32 bin: 15-30% of boards give a perfect sample @60%

## 2026-08-07 ~04:30 — Arm B result: one best-of-8 round buys +10 points, no forgetting

Harvest: 1469/2600 boards yielded a perfect no-think sample (yields by bin:
98/86/67/47/19%; 51 answers were valid non-gold walks — the grader admits
chorded walks; 0 repeated cells). Distill = continue ep8 adapter on
1469 own-answers + 1000 armD2 replay, 3 ep (val loss 0.0003 — data nearly
in-distribution already).

Yardstick frac_perfect: 42.8% -> 53.2% (ep2; ep3 53.0%), mean 0.874->0.915.
Per-bin (ep8 -> bok3): 14-17 72.8->82.4, 18-21 56.0->63.2, 22-25 32.0->40.0,
26-32 10.4->26.4 (2.5x — the newly-fueled deepest bin gained most).
v2-test breadth IMPROVED 0.975->0.981.

Prediction grades:
- >=55% overall @60% -> NO, narrowly (53.2%).
- gains concentrate plen>=22 @55% -> YES for 26-32 (largest gain by far);
  22-25 gained no more than shallower bins.
- v2-test drop <=0.02 @70% -> YES (it rose).
- 26-32 harvest yield 15-30% @60% -> YES (19.4%).

The no-CoT ceiling after ONE round: 53%. Rounds are cheap (~25 min);
iterating would likely keep climbing (fresh fuel: bok pass@8 now higher).
This is the bar arm T must beat.

Arm T predictions (registered before T harvest):
- T-1: T think-enabled eval beats 42.8% ep8-no-think baseline @40%
- T-2: T beats arm B's 53.2% @20%
- T-3: harvested think content is still board re-parsing, no path-search
  emergence (eyes-on-data judgment) @75%
- T-4: on T's own adapter, think-enabled eval <= its no-think eval
  (thinking still net-negative even after training on think successes) @70%

## 2026-08-07 ~06:00 — Arm T result: sharp negative — distilling forced-closed CoTs distills the truncation

Harvest (1600 boards, k=8 think temp-1.0): yields 282/152/62/11 per 400 by
bin — think-successes exist but thin at depth. 843 verbatim (think, answer)
rows + same 1000 replay, 3 ep continue from ep8.

Yardstick (temp 0): think-mode 5.2% perfect (mean 0.60-0.65) vs 42.8%
no-think baseline and 53.2% arm B. The T adapter's own NO-think mode fell to
38.2% (think-training mildly damaged the direct skill). v2test breadth
intact (0.98).

Mechanism (eyes on data): 498/500 eval CoTs hit the full 1024 budget — the
model never learned to CLOSE thinking. Root cause: the harvested CoTs were
themselves all forced-closed (natural close ~1%), so the targets teach
"re-parse until an arbitrary cutoff"; the </think> position carries no
learnable stopping signal. Rejection-sampling the CoT channel cannot work
when the sampler never terminates thinking — you distill the truncation
artifact. (This also retro-explains arm C's think-inflation episode.)

Prediction grades: T-1 (beats 42.8% @40%) NO. T-2 (beats B @20%) NO.
T-3 (content still board re-parsing @75%) YES. T-4 (think <= no-think on own
adapter @70%) YES — thinking still net-negative after training on it.

B vs T at matched boards: no-CoT distillation +10.4 pts; own-CoT
distillation -37.6 pts (think-mode). The CoT channel is not just unhelpful
here; SFT-based attempts to install it are actively destructive at this
scale. Remaining hope for the channel: RL (arm C), which uniquely can put
gradient on the CLOSE decision (correctness-gated length shaping) instead
of cloning truncated rambles.

## 2026-08-07 ~06:40 — Arm C (RL-with-think) launched

Question: can REWARD create a useful think channel where distillation
provably cannot (arm T)? RL uniquely gives gradient on the close decision:
correctness-gated length shaping (HEX_LEN_LAMBDA=0.3, char cap 3000) prices
think length among correct samples, and a rising close-bias schedule
(384:0,768:4,1048:8) supplies exploration over think lengths.

Setup: start policy = bok ep3 merged to full HF (best no-think, 53.2%
yardstick). Data: 4000 fresh witness boards plen 8-32 (~half plen>=18),
disjoint from all pools/evals; val 128. GRPO n=8, batch 32, lr 1e-6,
KL 0.001, RB answer branching 4, witness answer budget 200 tok, RESP_LEN
1256 (think ~1048). 100 steps, save every 25. Exp: armD_rl_think.
Note: the post-close "\n\nAnswer:" scaffold is injected mask-0 (standard
forced-close infra); the CoT itself receives no injected content, per the
phase-2 constraint.

Predictions:
- C-1: natural-close rate (train rollouts) > 50% by step 50 @55%
- C-2: think-mode yardstick at step 100 BEATS bok no-think 53.2% @35%
- C-3: RL prunes thinking toward empty (<50 tok median by step 100) and
  MATCHES 53.2% without exceeding it @40% (the "CoT adds nothing and RL
  discovers that" outcome)
- C-4: train reward mean rises >= 0.1 over the leg @70%

## 2026-08-07 ~10:20 — Arm C crash at step-100 save (disk full); salvaged at step 75

The run completed ~99 training steps, then died writing global_step_100
(torch save "unexpected pos" = ENOSPC; disk hit 100%, even the agent
harness wedged until a log truncation freed blocks). Steps 25/50/75 saved
intact; step-100 partial save deleted; optimizer states deleted per ops
rule. Steps 50/75 merged to hf; yardstick evals (think + no-think) running.

Training signal before the crash (side channel, temp-1.0 train mix):
emitted-answer perfect rate 9.4% -> 46.4% (window means), train score
0.36 -> ~0.64 plateau by step ~60. Think length shrank only mildly
(~2500 -> 2200 chars median): RL is NOT pruning the CoT to empty; it is
improving think-mode answers while the think text remains verbose
rules-rumination (a perfect plen-26 answer was observed after an
incoherent, mid-sentence-truncated think). C-4 (reward +0.1) already YES.

Analysis correction recorded: side-channel `score` is the RB-branching
MEAN over 4 answers, not the emitted answer's score — early read of "1.0s
vanished, 0.8 pile-up = reward hacking" was wrong; emitted answers pass
all checks. Re-grade emitted text for perfect-rate.

## 2026-08-07 ~11:00 — Arm C final + phase-2 synthesis: RL is the only channel that improves CoT, and it still loses to silence

Yardstick frac_perfect (temp 0):
                      think    no-think
  bok (pre-RL)         ~5%*     53.2%     (*arm-T family measurement)
  RL step 50           30.8%    54.4%     <- best overall number to date
  RL step 75           36.8%    49.0%
  (run died at the step-100 save; trajectory still rising)

Prediction grades:
- C-1 unbiased natural close >50% by s50 @55% -> NO. 500/500 eval thinks
  hit the 1024 budget; every training-time close was bias-assisted. The
  lambda=0.3 length price (max 0.3) is dominated by partial-credit stakes,
  so gradient went to better answers, not shorter thinks.
- C-2 think beats 53.2% @35% -> NO (36.8% at salvage; rising ~+6/25 steps —
  extrapolation says crossing was plausible near step ~150, unproven).
- C-3 prune-to-empty-and-match @40% -> NO. Thinks stayed ~2200 chars;
  instead RL-on-think ERODED the no-think mode 54.4 -> 49.0 (interference:
  the two modes converge toward each other rather than one winning).
- C-4 train reward +0.1 @70% -> YES (0.36 -> 0.64).

PHASE-2 ANSWER (the arm D question 2, this scale, this budget): the model
cannot yet be made to USE CoT profitably on witness.
  scripted-CoT SFT: format-without-computation (old arm C)
  own-CoT SFT (T):  distills truncation, actively harmful (5.2%)
  RL (C):           uniquely IMPROVES the think channel (5 -> 37%, still
                    climbing) but thinking stayed dominated by the same
                    policy's silence, and training it taxed the direct mode.
Across every channel, the verbalized stream never became the computation
carrier — answers improved while CoTs remained incoherent rules-rumination
(C3's sharpest form yet). The best policy of the whole program is
RL-step-50 NO-think: 54.4% yardstick / 95.3% v2test.

Open thread if ever resumed: (a) RL past step 100 (crossing extrapolation),
(b) stronger close pricing (lambda>=1 or hard budget curriculum), (c) B
round 2 (best-of-k iterate, no CoT needed). Shards deleted; hf checkpoints
kept at steps 50/75; disk back to 52G free.

## 2026-08-07 ~12:00 — Minimal-CoT probe launched: RL with EXACTLY 2 think tokens

User's design: same RL recipe as arm C but the think phase is hard-capped at
2 tokens (HEX_THINK_CAP_TOKENS=2 in hex_agent_loop, then forced </think>).
Splits "RL improves the policy" from "CoT content carries computation": if
2-token RL matches/beats the 1024-token leg, the ramble never mattered; the
2 tokens can at most become a learned steering register. Rollouts ~6x
cheaper (RESP_LEN 256 vs 1256).

Same start policy (bok ep3 merged), same data (verl_witness_long), same
hypers minus length shaping/close bias (both moot). 100 steps, save 25,
save_contents=[model,extra] (optimizer-state fix live).

Pre-RL baseline, 2-token eval mode: 53.2% / 0.913 longpath, 95.2% v2test —
IDENTICAL to bok no-think (junk 2 tokens are inert).

Predictions:
- M-1: 2tok-mode yardstick > 53.2% (its own start) by step 100 @70%
  (user suspicion "yes"; my odds high since arm C showed RL gains route
  through the answer head, which is fully present here)
- M-2: exceeds arm-C-RL's best no-think number (54.4% @ s50) @50%
- M-3: exceeds arm-C-RL think-mode s75 (36.8%) @85%
- M-4: the 2 think tokens collapse to a near-deterministic token pair by
  step 100 (a register, not content; entropy over the pair -> ~0) @60%
- M-5: train reward mean rises >= 0.15 over the leg @65%

## 2026-08-07 ~12:40 — 2-token probe v1 was a null: both tokens were the deterministic opener; relaunching with register exploration

User's attractor worry confirmed and then some. Measured on bok:
p('\n' | '<think>') = 1.0000, p('Okay' | '<think>\n') = 0.9996. The v1 probe's
"2 free tokens" were exactly '<think>'+'\n' — a constant prefix, zero CoT
capacity, zero exploration (pair entropy 0.00 bits over 1408 rollouts; KL
guard vacuous because the reference itself is deterministic there).

Fix (v2, relaunched): (1) '<think>\n' prefilled as context in cap mode so
both capped tokens are content positions; (2) think-phase sampling override
HEX_THINK_TEMP=10, HEX_THINK_TOPK=50 — exploration over ~50 candidate
registers guaranteed at sampling time regardless of policy collapse
(off-policy distortion on 2 tokens bounded by PPO clipping, same argument
as close-bias). Matching eval mode: --think-prefill.

Predictions unchanged (M-1..M-5) with one amendment:
- M-4 restated: the pair distribution SHARPENS toward a small register set
  under RL (measured as entropy decline in the side channel) @60%; and
- M-6 (new): some register pair yields a detectably higher mean reward than
  the policy-modal 'Okay\n'-style pair during training (|effect| >= 0.02
  RB-mean, any step-50 window) @40%.

## 2026-08-07 ~13:20 — 2-token probe v2 airborne with real register exploration

Exploration guard (answering the user's attractor concern): think-phase
sampling override temp 2.0 (vllm max) + top_k 50 + logit bias -8 on the
collapsed modal openers 'Okay'/'Alright' (model-derived candidates only, no
injected content). Result: 63 distinct register pairs, 2.89 bits over the
first 384 rollouts ('Okay,\n' still modal at 64% via the BPE comma-variant;
tail includes 'I need','I'm','Hmm,','Wait,','Let's','First,'). GRPO now has
genuine contrast across registers. Ops: killing the driver does NOT kill
stale Ray workers — a leftover AgentLoopWorker poisoned one relaunch;
full pkill of raylet/gcs/ray:: needed between runs.

## 2026-08-07 ~14:00 — How mode-collapsed is the thinking? The 8-bit continuation tree is a single boilerplate trunk (side-thread)

Enumerated the full within-8-bit (p>=1/256) continuation tree from
(prompt + '<think>\n') on the pre-RL bok policy, best-first over the token
tree, batched forward. Two tasks:
  task0: 402 kept prefixes, 39 leaves, 0.884 leaf prob mass
  task1: 383 kept prefixes, 37 leaves, 0.875 leaf prob mass
First token after '<think>\n' is 'Okay' at p=1.000 (0.001 elsewhere); tokens
2-3 (',' , ' let') are also ~0.00 bits. ALL 37-39 leaves are one semantic
template: "Okay, let's|let me try to figure out who wins|the winner of this
Hex game. The board is 8x8, and ...". The only within-budget degrees of
freedom are paraphrase swaps (let's/let me x try to/see/tackle x figure
out/determine x who wins/who won/the winner of). NOT ONE leaf in either tree
contains a move, a cell coordinate, or any position-specific reasoning — the
entire 8-bit horizon is ritual restatement of the prompt, zero computation.

Effective first-token entropy ~= 0 bits; the "diversity" is entirely
paraphrase noise at depth >=4. So the model's thinking is mode-collapsed
onto a content-free preamble, exactly as the user guessed.

Consequences (why this is load-bearing, not a curiosity):
- Explains the v1 2-token null directly: the 2 "free" think tokens are
  'Okay'+',' — forced boilerplate at ~0 bits, carrying zero task info. A
  2-token cap cannot help because there is nothing but preamble to cap.
- Sharpest form yet of C-3 (concept/verbalization decoupling): within any
  reasonable surprisal budget the verbalized stream is provably preamble,
  not the computation carrier. The answer head must be doing the work
  regardless of the think text.
- Predicts the v2 register probe's ceiling: forcing exploration over the
  2-token opener (temp2/topk50/suppress 'Okay') samples DIFFERENT preambles,
  not different computations — so per-register reward differences should be
  ~noise (consistent with the step-50 read: 0.912-0.948 spread, reshuffling).

Tool: scripts/think_tree.py (--budget N enumerates the p>=2^-N tree). Run on
GPU alongside RL (10GB free under vllm's fixed 0.55 fraction; no disruption).

## 2026-08-07 ~15:00 — Automated grammar induction over the thinking-continuation set; correction + cross-task answer

Built scripts/think_grammar.py: leaves -> word seqs -> trie -> minimal acyclic
DFA (DAWG, Revuz suffix-merge) -> cut-point BNF (name only branch/shared states,
collapse forced chains into multi-word terminals). Canonical + verified exact
(re-enumerate DAWG language, assert == leaf set). This is the honest version of
the hand-grammar: the unique minimal grammar whose language is exactly the set.

CORRECTION to 2026-08-07 ~14:00 entry: the "37-39 leaves" was a max_depth=24
TRUNCATION artifact (frontier was still in-budget at the cap -> false leaves).
Honest 8-bit tree is ~90-94 leaves per task and is STILL truncated at
max_depth=64 -- i.e. there exist >=64-word thinking prefixes with cumulative
p>=1/256. A 64-token "thought" at high probability still hasn't begun to reason
about the position; it is still reciting rules. That strengthens, not weakens,
the zero-computation conclusion.

Task-0 grammar: 94 leaves -> DAWG 752 states / 843 edges / 116 nonterminals.
Every production is boilerplate: "connect the top edge to the bottom edge",
"Black and White take turns placing stones", "the board is 8x8". Not one
production references a move, a cell coordinate, or a position evaluation.

Does the grammar change between training examples? Measured on tasks 0-3:
- exact leaf-string Jaccard: ~0.00-0.11 (LOW) -- so at the string level, yes,
  each task's leaf set is nearly disjoint from the others'.
- BUT prefix-Jaccard vs depth tells the real story (mean pairwise):
    first 1-3 words  J=1.00   (identical ritual opener)
    first ~8 words   J~0.90
    first 12 words   J~0.69
    first 16 words   J~0.25
    first 20 words   J~0.08  -> ~0.02 by 50 words
- unique-word (vocab) Jaccard across tasks: 0.80 (HIGH).
- structural sizes near-identical: 90-94 leaves, 752-846 states, 843-935 edges.

Reading: the grammar TRUNK is task-invariant -- identical first 3 words, ~70-90%
shared through ~12 words (the "Okay, let's try to figure out ... this Hex game.
The board is 8x8, and the ..." ritual). It fans into task-conditioned paraphrase
past ~15 words, but the divergence is lexically/semantically the SAME boilerplate
(vocab J=0.80, same production themes), never task-specific hex content. So the
prompt perturbs WHICH paraphrase of the rules gets recited, not WHAT is computed
-- consistent with C-3 (the verbalized stream is decoupled from the computation).

Tools: scripts/think_tree.py (now importable: load_model/build_context/
enumerate_budget_tree; memory-safe via logits_to_keep=1 + chunking) and
scripts/think_grammar.py. Ran on GPU alongside the live RL job (watch free mem;
OOMed once when the RL rollout phase spiked, fixed with chunk+last-token logits).

## 2026-08-08 ~00:15 — 2-token probe verdict: RL through a 2-token CoT = plain RL, and it works

Run history: OOM'd twice to GPU co-tenants (17.6G then 5.3G squatters);
salvaged at step 50 of 100 (train score flat 0.92-0.94 throughout, so the
lost half is unlikely to change the verdict). Optimizer-less checkpoints
worked for resume as designed.

Yardstick frac_perfect (temp 0, n=500, paired):
  bok floor (either mode)        53.2%
  RL-2tok, 2tok prefill mode     56.6%   (McNemar 38/21, z=2.2 vs floor)
  RL-2tok, no-think mode         56.0%   (2tok vs no-think on same model:
                                          9/6, z=0.77 — the 2 tokens are
                                          inert at eval too)
  [1024-leg s50 for reference: no-think 54.4%, think 30.8%]
Per-bin gains concentrate at plen>=18 (0.632->0.688, 0.400->0.456,
0.280->0.304); v2test breadth intact (94.3-94.6%).

Prediction grades:
- M-1 2tok-mode > 53.2% @70% -> YES (56.6%, z=2.2, at half the planned
  steps).
- M-2 > 1024-leg's 54.4% @50% -> YES (56.6/56.0), and WITHOUT the no-think
  erosion the 1024-leg suffered (54.4->49.0 by its s75) — at ~5x lower
  rollout cost.
- M-3 > 36.8% @85% -> YES trivially.
- M-4 register sharpening @60% -> NO: pair entropy rose 3.5->4.25 bits
  under the exploration sampler; no winner emerged.
- M-5 train reward +0.15 @65% -> NO (+~0.01; the 2tok floor was already
  0.93 — most of the 1024-leg's "RL gain" was just un-breaking the ramble).
- M-6 a register with >=0.02 reward edge @40% -> NO: per-register means
  compressed to 0.912-0.948 with rank instability across windows = noise;
  registers are inert.

USER'S SUSPICION CONFIRMED, with a sharpening: RL with (exactly) 2 CoT
tokens consistently improves the policy (+3.4 pts, z=2.2, gains at the long
-path frontier) — but the mechanism is pure answer-head improvement. The 2
tokens carry nothing (eval-mode-invariant, register-invariant), and this
minimal leg BEATS the 1024-token RL leg on every measure while being ~5x
cheaper and erosion-free. Verbose CoT was not merely useless for RL here;
it was an active tax on both compute and the direct-mode skill.

New program best: armD_rl_think2 step 50, 56.6%/56.0% yardstick.

## 2026-08-08 — Leaves persisted; leave-one-out grammar coverage of held-out training examples

Q from user: were the grammar completions stored, and does a grammar built from
all-but-one training example cover the held-out one? Note the grammars are built
from DETERMINISTIC enumeration of the p>=1/256 budget tree (not sampled rollouts);
those enumerated leaves are now dumped to results/think_leaves.jsonl (555 leaves,
6 tasks: task,rank,bits,prob,reason,text,words). [The RL run's *sampled* rollouts
live separately in results/rollouts/armD_rl_think2.jsonl.]

LOO (budget 8, max_depth 64, tasks 0-5): build minimal DAWG from the other 5
tasks' leaves (~354-404 leaves pooled), trace each held-out leaf.
  held  n_leaf  exact  full-trace  median_L  median_L/len
    0     94       2       2          15        0.50
    1     91      40      56          29        1.00
    2     94      46      68          31        1.00
    3     90       1       1          15        0.50
    4     94      50      71          31        1.00
    5     92      50      76          31        1.00
  TOTAL 555: exact-match 189 (34.1%), full-path-traced 274 (49.4%).
  (exact = held-out leaf is verbatim in the pooled language; full-trace = it is a
   prefix of / equal to some pooled string, i.e. every word follows a valid edge.)

Reading: coverage is BIMODAL. 4 of 6 held-out examples (1,2,4,5) are ~half
reconstructable verbatim from the others (median held-out leaf traces its ENTIRE
length), i.e. their high-prob thinking is a shared boilerplate dialect. 2 of 6
(0,3) are near-idiosyncratic (only ~1-2% exact; median leaf diverges at ~word 15,
halfway). So the pairwise ~0 Jaccard was misleading about pooled coverage: five
~90-leaf samples of the boilerplate distribution exactly cover ~34% of a sixth,
~49% as a path -- the thinking "grammar" substantially generalizes across
training examples, with an outlier subpopulation. The shared trunk (first ~15
words) generalizes to ALL held-out examples regardless (median_L>=15 everywhere).

## 2026-08-08 ~01:00 — Ritual-prefill probe launched (user's grammar -> design)

User mapped the near-deterministic grammar of the model's think openings
("Okay, " PREAMBLE (". " BOARD)? ...) — the mystery GPU squatter was their
decode runs. Implication accepted: the 2-token window sat inside a
zero-variance ceremonial prefix; gradient needs a window where content
varies by task.

Measured (bok, ritual 'Okay, let me try to figure out the winner of this
Hex game. ' prefilled): per-position sample entropy 0.9 -> ~1.1 bits over
the first 4 tokens (still the BOARD boilerplate clause), 2-3 bits at
positions 10-17, 4-4.8 bits by 19+. So: extend the gradient-free prefill
THROUGH the board clause (size-parameterized, task metadata only), then
16 free tokens landing in the 2-4.5-bit zone. Natural temp-1.0 exploration
suffices — no suppression hack this leg.

Setup: HEX_THINK_RITUAL=1, cap 16 free tokens, RESP_LEN 224, same start
(bok merged), same data, 50 steps target (heads-up vs the 2-tok leg's
56.6%). Exp: armD_rl_ritual.

Predictions:
- R-1: yardstick (ritual eval mode) > 53.2% floor @75%
- R-2: > 2-tok leg's 56.6% (the free window carries usable signal beyond
  answer-head-only RL) @40%
- R-3: free-window content becomes task-specific (names cells/edges vs
  generic boilerplate, eyes-on-data) by step 50 @50%
- R-4: think-mode vs no-think gap on the trained model stays < 2 pts
  (answer head still does the work) @60%

## 2026-08-08 ~02:30 — Ritual-prefill leg verdict: new best (58.0%), but the window still carries no content

armD_rl_ritual (gradient-free size-parameterized ritual prefill + 16 free
think tokens, 50 steps from bok):
  ritual mode  58.0% yardstick (vs floor: McNemar 55/31, z=2.6)
  no-think     57.0%
  v2test       95.4% (breadth intact)
Comparisons: vs 2-tok leg 56.6% -> +1.4 pts, McNemar 36/29, z=0.87 — NOT
significant. vs its own no-think: z=0.75 — mode gap ~1 pt, not significant.

Prediction grades:
- R-1 > floor @75% -> YES (z=2.6).
- R-2 > 2-tok leg @40% -> nominally yes, statistically NO (z=0.87);
  graded NO by the registered bar ("carries usable signal beyond
  answer-head RL" — unproven).
- R-3 task-specific window content @50% -> NO: RL shifted mass among
  boilerplate phrasings ('The rules say...' -> 'Let me start by looking at
  the current state of the board.'), and at temp-0 eval ZERO of 500
  windows name a cell.
- R-4 mode gap < 2 pts @60% -> YES (1.0 pt).

Synthesis of the two window probes: RL improves this policy by ~+3-5 pts
regardless of whether the free CoT window is 2 ceremonial tokens or 16
post-ritual tokens; the window's content remains inert boilerplate in both.
The user's grammar hypothesis was right about WHERE variance lives, but
even a well-placed window doesn't get task content selected at this
scale/budget — the answer head keeps absorbing the gradient. Program best
is now armD_rl_ritual s50 (58.0/57.0), a chain of: SFT (0.98 short paths)
-> best-of-8 distill (53.2) -> RL (58.0), with every CoT mechanism tested
and none load-bearing.

## 2026-08-08 ~04:00 — Interference study (arm E): 3 tasks, shared boards; pre-run checklist passed

Question (user): does SFT install destructively on top of OTHER SFT learning,
and does final per-task performance depend on training ORDER? Three no-think
tasks on ONE shared constructive-board pool (sizes 5-7, test boards disjoint):
  T_wit  (path)    witness — winner + unique path (reuses armD graders)
  T_cell (judge)   occupancy: "which player has a stone on cell X?" (existing
                   occupancy task's grader; queries balanced ~50% occupied)
  T_list (listing) reading-order full-board scan of a color class (existing
                   listing grader; empty-target instances excluded as the
                   grader scores [] as -1)
Uniform answer-token loss weight across all tasks (clean cross-task choice).
Data: 1600 train / 400 test boards, one row per task per board.

Pre-run checklist (user-specified):
[1] wandb: WANDB_API_KEY present; train+eval log to wandb. OK.
[2] each task solved by an independent Opus agent from the verbatim prompt:
    wit/cell/list all CORRECT, all reported SOLVABLE, AMBIGUITY: none.
[3] base Qwen3-1.7B (no-think, temp0) on one instance each: all 3 wrong, but
    the failure mode is COMPETENCE not ambiguity — Qwen systematically emits
    TRANSPOSED coordinates ('1b' for b1), violating the explicit 'column
    letter + row number' spec, plus board misreads. SFT teacher-forcing
    directly corrects this. Note: the coordinate convention is SHARED across
    all 3 tasks -> possible positive transfer (format learned once helps all).
[4] optimizer state NOT snapshotted (save_contents=[model,extra], live).

Two data bugs the checklist/asserts caught before training: (a) 'list empty'
on a full board -> [] target -> grader scores -1 (excluded); (b) occupancy
57% 'Neither' majority -> rebalanced to ~50% occupied queries.

Start model = BASE instruct Qwen3-1.7B (clean multi-task-from-scratch, not
bok). Predictions:
- E-1: sequential w/o replay -> earlier task drops >20 pts frac_perfect by
  end @65%
- E-2: mixed hits >=0.90 frac_perfect on all 3 @80%
- E-3: final per-task perf depends on ORDER (recency: last-trained highest)
  @70%
- E-4: positive transfer — a later task's stage-1 accuracy is higher than
  base zero-shot by more than format alone (shared coord convention) @45%

## 2026-08-08 ~05:30 — Interference matrix: sequential SFT is catastrophic, order = recency, mixing fixes it

frac_perfect on held-out test (rows=adapter, cols=task):
  adapter                 wit    cell   list
  wit_only               0.790  0.405  0.200   <- solo ceilings on diagonal
  cell_only              0.000  0.978  0.003
  list_only              0.138  0.357  0.988
  fwd2  wit>cell         0.090  1.000  0.018
  fwd3  wit>cell>list    0.223  0.895  0.990
  rev2  list>cell        0.000  1.000  0.212
  rev3  list>cell>wit    0.840  0.912  0.410
  mixed                  0.855  0.995  0.995

Answers to the user's questions:
1. Does SFT install destructively on top of other SFT? YES, catastrophically.
   One stage of cell training tanks the prior task: wit 0.790->0.090 (-0.70),
   list 0.988->0.212 (-0.78). Classic catastrophic forgetting.
2. Order dependence? YES, dominated by RECENCY. Final (stage-3) per task:
   - wit: trained-first 0.223 vs trained-last 0.840 (Delta +0.62)
   - list: trained-last 0.990 vs trained-first 0.410 (Delta +0.58)
   - cell (middle both) ~0.90 either way. Last task wins; earlier partially lost.
3. Mixing removes it: 0.855/0.995/0.995, all at/above solo ceilings at once.

Mechanism (eyes on data): forgetting is COMPUTATION loss with FORMAT
RETAINED. Post-cell witness still emits well-formed {"winner","path"} but
the path is a degenerate straight line (c1..c7); post-cell listing still
emits a JSON array but incomplete/wrong. The shared output schema (and the
coordinate convention) is reinforced across tasks and survives; the
task-specific board-reading->tracing computation is overwritten. So SFT
interference is not mode collapse -- it's selective erasure of the
per-task procedure.

Positive transfer (bonus): wit_only=0.790 but wit under mixed=0.855 and under
rev3 (wit last)=0.840 -- BOTH exceed the solo ceiling. Co-trained/last-trained
witness benefits from cell+list board-reading. (E-4 shared-convention
transfer: YES.)

Prediction grades:
- E-1 earlier task drops >20pts @65% -> YES (-70 to -78 pts).
- E-2 mixed >=0.90 all three @80% -> NO by letter (wit 0.855), but wit's solo
  ceiling is 0.79 so mixed is AT ceiling; the 0.90 bar was miscalibrated.
- E-3 order dependence, recency @70% -> YES, decisively (Delta ~0.6).
- E-4 positive transfer beyond format @45% -> YES.

Synthesis for the working model "RL selects, SFT installs": SFT installs,
but sequentially it OVERWRITES prior tasks' computation (format survives) --
the same catastrophic-forgetting / replay lesson as arm C's RL-after-SFT,
now shown SFT->SFT. Final perf is recency-dominated. The fix is identical:
co-train (mix), which additionally buys positive cross-task transfer.
"SFT on one task doesn't degrade others" is FALSE for sequential SFT and
TRUE only under mixing.

## 2026-08-08 ~06:30 — SFT-scheduler experiment launched (Joshua's Q: the multi-task RL scheduler, but SFT)

Faithful mechanical port of the arm-C Neyman-with-floors allocator driving
SFT instead of RL. 11 curriculum categories, canonical gold teacher-forcing
targets (all verified score 1.0, 0 dropped, 17244 rows). Chunked adaptive
loop from BASE Qwen3-1.7B: round -> sample M=1600 ~ weights -> train 1 epoch
(continue LoRA) -> per-category eval frac_perfect (temp0) -> reweight.
share_c = sqrt(p_c(1-p_c))/sqrt(k_c); p_c=eval frac_perfect (EMA0.5,
optimistic 0.5 prior); k_c=mean target length (SFT cost analog); floor 0.03;
w=1. Arms: uniform (static 1/11) vs port (adaptive). 8 rounds, matched budget.

Conceptual note (why this is interesting): sqrt(p(1-p)) is RL's stochastic-
reward gradient SD. SFT targets are deterministic -> that variance is not the
SFT gradient variance, so the allocation rule loses its derivation. The port
tests whether the RL-motivated scheduler still helps once teacher forcing
removes the gradient-scarcity problem it was built for.

Predictions:
- S-1: port does NOT beat uniform on final MEAN frac_perfect (Delta < +0.02)
  @70% (uniform already near-ceilings under SFT; interference study showed it)
- S-2: port DOES beat uniform on final WORST-task frac_perfect by >=0.05
  @45% (adaptive budget to the hardest lagging task is its one plausible edge)
- S-3: port reaches a given mean frac_perfect in fewer total examples
  (sample-efficiency) @40%
- S-4: port's weights concentrate on high-p(1-p)/low-cost cats (judge/occupancy
  early, move/witness late); the myopic starvation of a p~0 task is prevented
  only by the floor @60%
- S-5: mate2 (5-move, hardest) is the worst-task in both arms at the end @55%

## 2026-08-08 ~08:00 — SFT-scheduler verdict: the RL cost term is BACKWARDS for SFT; scheduler slightly hurts

Final frac_perfect (8 rounds, matched budget, temp0), uniform vs port:
  cat         uniform  port   delta
  chain         0.842  0.975  +0.133
  judge         0.659  0.659  +0.000
  occupancy     0.958  0.950  -0.008
  winset        0.006  0.011  +0.006
  chainset      0.422  0.200  -0.222
  witness       0.327  0.140  -0.187
  mate1_v2      0.644  0.739  +0.094
  mate2         0.556  0.544  -0.011
  edge_m1       0.595  0.619  +0.024
  gen_m1        0.699  0.748  +0.049
  general       0.407  0.311  -0.096
  MEAN          0.556  0.536  -0.020
  MEDIAN        0.595  0.619  +0.024
  WORST(winset) 0.006  0.011
MEAN trajectory uniform: .20 .26 .30 .43 .45 .52 .53 .556
MEAN trajectory port   : .21 .30 .40 .44 .48 .50 .52 .536
  (port leads r2-r3 by ~.10, uniform overtakes by r6, finishes higher.)

Port weights (x1000) converge by r2 and hold: cheap tasks (k~4-5:
chain/judge/occupancy/mate/edge/gen/general) 100-140; the 3 EXPENSIVE tasks
(chainset k21, winset k34, witness k38) SLAMMED to the floor (~28-42).

Mechanism / the finding: the cost-corrected Neyman term 1/sqrt(k) is
sensible in RL (k = rollout tokens; cheap-to-sample tasks yield more
gradient per GPU-sec) but INVERTED in SFT, where k = target length and long
targets = the HARD compositional tasks. So the port systematically defunded
witness (the project's whole target skill), chainset, winset -- the exact
tasks needing the most budget -- to overfund cheap lookups. Net: mean -0.02
(chainset -0.22 + witness -0.19 outweigh chain +0.13 etc.); it raised the
MEDIAN task (rich-get-richer on cheap-learnable) while lowering the MEAN and
gutting the hard tasks. The sqrt(p(1-p)) variance term, having no SFT
gradient meaning, just chased mid-p tasks that uniform would have learned
anyway.

Prediction grades:
- S-1 port !> uniform on mean (Delta<+0.02) @70% -> YES (-0.020).
- S-2 port beats worst by >=0.05 @45% -> NO (both ~0.01; winset unlearnable).
- S-3 sample-efficiency @40% -> PARTIAL: port faster r2-r3, overtaken by r6,
  finished lower. Early yes, net no.
- S-4 weights concentrate low-cost/high-p(1-p); floor prevents p~0 starvation
  @60% -> YES decisively (winset/witness held exactly at floor 0.029).
- S-5 mate2 worst-task @55% -> NO (winset is worst, 0.006/0.011).

Synthesis (answers Joshua's Q "how does the scheduler go with SFT"): it does
NOT help and slightly hurts. The scheduler solves an RL problem (gradient
scarcity on rare-success skills) that teacher forcing removes; worse, its
cost correction -- correct for RL -- is anti-correlated with task difficulty
in SFT and defunds the hardest tasks. Uniform mixing is the right default for
multi-task SFT; if anything, an SFT scheduler should weight cost the OTHER
way (spend MORE on long-target/hard tasks). The whole exercise re-confirms
the session's throughline: RL machinery ported to SFT tends to be
unnecessary-to-harmful, because the two regimes' notions of "where the
signal is" diverge.

## 2026-08-08 ~09:00 — Self-judge falsifier (Joshua's "should I retry" idea): verification learnable, but NOT on the model's own errors

Idea: augment witness with an is_correct self-judgment (label = grade of the
model's own sampled answer); a working signal enables retry/attempt-budget RL.
Cheapest falsifier: freeze bok answerer, build an is_correct verifier, measure
detection by error type on constructed probes AND bok's natural answers.

Data reality (caught pre-train, checklist eyeball): bok is 92% CORRECT on
won boards (median board p=1.0), and natural WINNER errors are ~0 (9/8000) --
on a finished board the winner is visually obvious; bok's mistakes are
long-path LINK errors. So on-policy self-judge signal is starved. Pivoted to
BALANCED CONSTRUCTED training (gold->Yes, broken-link->No, wrong-winner->No).

Results (r32 LoRA from base, temp0):
                       balanced_acc   link-detect   winner-detect
  base zero-shot         0.53 (chance, 27% unparsed -- can't even format)
  SFT, PROBE (constructed) 0.995         0.98          1.00
  SFT, NATURAL (bok's own) 0.545         0.088         0.00(n=9)

So: verifying a witness answer is TRIVIALLY SFT-learnable in-distribution
(0.995, both error types; winner-flip is easy -- the path connects the wrong
edges). But it does NOT transfer to the model's OWN mistakes (0.545 ~ base
0.53). Mechanism (eyes on data): constructed errors delete a cell -> blatant
ADJACENCY GAP; bok's natural link errors are FULLY-ADJACENT paths that step
onto a non-winner cell (subtle MEMBERSHIP/color errors). The gap-trained
judge never learned to check membership -> catches 9% of real link errors.

This empirically confirms the pre-registered design warning #1 (judge your
OWN samples, not gold/constructed -- distribution match is load-bearing).
The tension it exposes: on-policy errors are the right distribution but
sparse+skewed (bok too accurate, ~0 winner errors); constructed errors are
rich+balanced but the wrong distribution (near-zero transfer). To build the
"should I retry" signal you must train on real on-policy errors, which needs
an answerer weak enough (or boards hard enough) to supply a balanced
on-policy error set.

Theme callback: the skill is SFT-installable in-distribution, but
METACOGNITION about one's own outputs is distribution-sensitive -- the naive
version fails exactly where it needs to work.

Next (if pursued): (a) on-policy judge -- sample bok errors on hard/long
boards, train on THOSE, test natural transfer (predict link-detect jumps,
winner-detect stays ~0 for lack of data); (b) then the attempt-budget RL
(pick N tries, last graded, cost in N) on top of a working self-judge.
Artifacts: data/selfjudge/, checkpoints/selfjudge/adapter, results/selfjudge_*.

## 2026-08-09 ~05:00 — On-policy self-judge + AUC: the signal is weak, SFT-capped, and distribution-INVARIANT

Followed up the constructed-judge's natural failure by (a) mining bok's REAL
errors on hard boards (8-9, long paths; 52% error rate) and training an
on-policy judge, and (b) measuring discrimination by AUC of P(Yes) logprob
(the argmax Yes/No collapses to a constant and hides ranking signal).

On-policy mining confirmed winner-errors are intrinsically rare: 17/734
errors (~2%) even on hard boards -- bok gets the winner right ~98% while
botching the path. So on-policy can only teach LINK detection.

Argmax (misleading): on-policy judge collapses to always-No (natural
balanced_acc 0.500, "catches" 100% of errors + 0% of correct); constructed
judge was always-Yes (0.545). Both degenerate at the decision threshold.

AUC of self-correctness on the SAME natural test (the real measure):
  base (no training)   0.578
  constructed judge    0.645
  on-policy judge      0.646
Findings:
1. A weak self-correctness signal EXISTS and SFT improves it a little
   (0.578 -> 0.645, +0.07 AUC). Not a null.
2. It is WEAK and CAPPED at ~0.65, and DISTRIBUTION-INVARIANT: constructed
   (blatant gaps) and on-policy (subtle real errors) give the SAME AUC
   (0.645 vs 0.646). My prediction that on-policy would fix discrimination
   was WRONG -- distribution mismatch was not the limiter.
3. The real limiter is BOARD-READING: verifying a subtle self-error (a
   fully-adjacent path stepping on one non-winner cell) requires re-checking
   every cell's membership on an 8-9 board -- the same capacity that caps
   the answerer. Self-verification is about as hard as answering here.
4. The usable signal lives in the logprob, not the hard Yes/No (which
   collapses to the base-rate prior). A retry policy must threshold P(Yes).

For Joshua's "should I retry": a real but weak (AUC ~0.65) signal is
available and SFT nudges it up slightly; it could drive a retry policy
marginally better than blind retry, but it is not a strong detector at 1.7B
and does not improve with more/better error data -- it's board-reading-
bound. The attempt-budget RL (idea B) would be building on a ~0.65-AUC
primitive; low ceiling expected at this scale. Verification is cheap only
for GROSS errors (constructed probe 0.995); for the subtle errors a
competent answerer actually makes, verification ~ answering in difficulty.

Prediction grades (this leg):
- "on-policy link-detection jumps up" -> argmax YES (0->100% via always-No)
  but that's a threshold artifact; AUC NO (0.646 ~ constructed 0.645).
- "winner-detection stays ~0 for lack of data" -> YES (2% natural winner
  errors; untrained).
Artifacts: scripts/build_selfjudge_onpolicy.py, sj_auc.py,
checkpoints/selfjudge_op/adapter, results/selfjudge_op_*, /tmp/sj_auc2.log.

## 2026-08-09 ~06:30 — Error structure + decomposition: the info IS there per-cell (0.998); the holistic judge just doesn't use it

Joshua asked if bok's errors are "valid but non-minimal" and whether per-cell
membership is distinguishable >> 0.65.

Error structure (600 hard boards, bok temp1 k4; 2102 correct / 211 errors):
  of CORRECT answers: non-minimal (len>shortest) = 1.7%  (bok gives minimal paths)
  errors (one error can trip >1 check):
    wrong_winner 0.9% | membership 84.8% | adjacency 25.6% | edge 4.3%
    "removable" (winner-stones-in-answer still connect edges) = 12.8%
=> NOT non-minimal. The characteristic error is MEMBERSHIP: while tracing, bok
asserts a path cell that isn't actually the winner's stone (hallucinated
stone, often also breaking adjacency). Non-minimal-valid paths would score 1.0
anyway (grader accepts any valid chain), so they aren't errors.

Per-cell probe -- ask "which player has a stone on X?" on the EXACT cells bok
hallucinated (500 not-stone / 500 stone from error paths):
  base zero-shot (Yes/No logprob AUC)      0.499  (base can't read boards at all)
  occupancy LoRA trained on 5-7 (size-OOD) 0.625  (biased 'Neither' on 8-9)
  occupancy LoRA trained on 7-9 (on-size)  0.998  balanced (1.000 / 1.000)
vs the holistic answer-judge on the same errors: base 0.578 / SFT 0.645 AUC.

THE FINDING: the information needed to catch bok's errors is FULLY available
to the model per-cell -- trained occupancy nails the exact hallucinated cells
at 99.8%. The holistic self-judge caps at 0.65 NOT for lack of information or
capability, but for lack of DECOMPOSITION: judging the whole answer gestalt
instead of checking each cell's occupancy. bok's hallucinations are a TRACING
artifact (path pressure makes it assert a stone it would correctly call empty
in isolation), not a board-reading limit.

Implication for "should I retry": a DECOMPOSED verifier -- per-cell occupancy
(0.998 each) + deterministic adjacency/edge checks -- would catch ~all
membership errors, i.e. near-perfect self-verification. The holistic
0.65-AUC judge is the wrong architecture; the signal is there, decompose it.
This also refines the session's "verification is cheap" line: cheap even for
SUBTLE errors, but only when decomposed into the per-cell percepts the model
has; asked holistically it collapses to the base-rate prior.

Answer to Joshua: errors are membership (hallucinated stone), not
non-minimality; and the operative per-cell question (occupancy) is
distinguishable at ~1.0, far above 0.65 -- the model knows, it just isn't
asked cell-by-cell.

## 2026-08-09 ~07:30 — Decomposed verifier: 0.545 -> 0.965 by decomposition (100% error detection)

Built the verifier the per-cell finding implied: verdict(claimed winner C,
path P) = Yes iff geometry ok [code: adjacency + endpoints on C's edges] AND
every cell of P is C's stone [occupancy LoRA (5-9), per cell]. A valid path
of color C implies C won, so this also validates the winner claim; a flipped
winner fails per-cell occupancy vs the claimed color.

On 1974 bok answers (92.4% correct; errors 137 membership / 7 winner /
6 geometry):
  DECOMPOSED  balanced_acc 0.965  detect-error(No|bad) 1.000  confirm-correct 0.930
  (holistic answer-judge, same distribution: balanced 0.545 / AUC 0.645)
Per error type detection: membership 100%, geometry 100%, winner 100%.

So decomposition recovers whole-answer verification from 0.545 -> 0.965 and
catches EVERY error, confirming the thesis: the information was fully present
per-cell; the holistic judge failed only for lack of decomposition. The one
residual is 7% false-negatives on CORRECT answers (a long correct path
occasionally has one cell the occupancy oracle misreads -> flagged) -- for a
"should I retry" use this is the safe direction (100% error recall; ~7%
unnecessary retries). A stronger per-cell oracle (bigger model / majority
vote) would lift confirm-correct.

Ops: occupancy LoRA at lr 1e-4 DIVERGED on the 5-9 data (loss -> 0.03 then
blew up to 3.5; also a fully-failed run stuck at 4.0). lr 3e-5 + 0.1 warmup,
2 epochs = clean (loss ~2e-4). Note for utility-adapter training on this box.

## 2026-08-09 ~08:00 — Single-token 'any behavior' RL test launched (Joshua's design)

Can RL move a single, TASK-NEUTRAL token purely via a reward differential,
with competence held constant? Witness + a suffix marker " NOTE" after the
JSON (grader ignores trailing text -> correctness identical with/without).
SFT from bok on 50/50 marker targets -> P(marker)~=0.5, witness intact
(val 0.018). RL from the merged model: reward = witness_score - lambda *
marker_present (reward_verl HEX_MARKER/HEX_MARKER_PENALTY; kind_win
unaffected). The only reward-differentiating token is the marker decision
(the token after '}').

Run A (cost): lambda=+0.3. Run B (reward): lambda=-0.3. Measure P(marker)
over steps from the side channel.

Predictions:
- ST-1: cost run drives P(marker) 0.5 -> <0.1 by step 40 @85%
- ST-2: witness kind_win stays ~flat (marker task-neutral) @80%
- ST-3: reward run drives P(marker) -> >0.9 @80%
- ST-4: convergence is fast/monotone (<30 steps) @70%
If both directions move cleanly, RL demonstrably teaches an arbitrary
single-token behavior (the sharpest form of 'RL selects').

## 2026-08-09 ~08:40 — Single-token RL: YES, RL teaches an arbitrary task-neutral token (both directions)

Marker " NOTE" suffix, SFT'd to P(marker)~0.5 with witness intact, then RL
with reward = witness - lambda*marker (correctness/kind_win unaffected).
P(marker) by ~step (256 rollouts/step):
  COST   (lambda=+0.3): 0.32 0.36 0.26 0.25 0.20 0.20 0.12 0.08 0.07 0.03
                        0.02 0.01 ... 0.00 (pinned from step ~16)
  REWARD (lambda=-0.3): 0.27 0.34 0.32 0.40 0.45 0.58 0.60 0.65 0.82 0.91
                        0.96 0.97
Witness exact-win over the runs: cost 0.61->0.78, reward 0.62->0.47 (noisy,
no collapse; shaped score mean stayed 0.92 cost / 1.2 reward = ~0.9 witness +
marker bonus). So competence held while the single task-neutral token moved
to either extreme on a +/-0.3 differential, in ~12-16 steps, monotone.

Prediction grades: ST-1 (<0.1 by 40) YES; ST-2 (win flat) YES; ST-3 (>0.9)
YES (0.97); ST-4 (fast/monotone <30) YES.

Verdict: RL demonstrably teaches an ARBITRARY single-token behavior -- moves
probability on a semantically meaningless token to 0 or ~1 purely by a small
reward differential, competence pinned by SFT. This is the sharpest form of
the session's throughline: RL SELECTS (reliably shifts mass toward reward,
even for a meaningless token) but does not CREATE (all the earlier arms:
it can't install a skill/CoT that isn't already sampled). "Can RL teach
literally any behavior?" -- for a behavior already in the support as a
reachable token choice, yes, cleanly. The constraint is reachability/support,
not the reward's semantics.

## 2026-08-09 ~23:56 — Arm E launch: can RL SELECT an instrumentally-useful task? (Joshua's design)

Sharpest extension of the marker result (RL moves a task-NEUTRAL token): now
make the single token task-CONSEQUENTIAL. Model is shown a 5x5 hex board and
five perception tasks; it must pick ONE auxiliary task to solve first (not
evaluated), then answer the EVALUATED task. Question: can RL on the single
selection token learn to pick the auxiliary task whose solution best scaffolds
the evaluated one — under maximally favorable conditions?

Tasks (all exactly computable from the board via BFS; no benzene):
  A colors+positions of ALL stones (reading order)
  B winner + one winning path
  C ALL empty cells (reading order)         <- EVALUATED task
  D black stones connected to top / to bottom; white to left / to right
  E black connected to NEITHER top nor bottom; white to neither left nor right
Selection is over the 4 non-evaluated tasks {A,B,D,E}. DESIGNED-useful helper
for C is A: empties == complement of stones, so a correct A makes C mechanical
(copy-complement); B/D/E give only partial/irrelevant stone info. A-E are all
single tokens in the Qwen tokenizer (32-36). No-think throughout.

THE CRUX / GATE: an instrumental differential must EXIST — solving A first must
raise P(C correct) vs solving B/D/E. Measured directly in R2/R3 before any RL.
If null, report "no instrumental gradient" (still informative) and expect R4 to
drift like the marker.

Staged plan (each a gate for the next; matches Joshua's 4-run spec):
  R1 feasibility SFT: fixed-order "solve X then Y", confirm model can emit the
     tagged two-task format and score A-E at 5x5. Gate: A,C,D,E frac_perfect >=0.7.
  R2 teacher-forced-helper SFT: <selected-task>X</selected-task><helper>gold X
     </helper><evaluated-task>gold C</evaluated-task>, X~unif{A,B,D,E}, helper
     TF gold, GRADIENT ONLY ON C. Then measure Delta = acc(C|A) - mean acc(C|B,D,E).
  R3 own-helper SFT: helper span = model's OWN sampled answer (offline pre-gen
     with R2 model), loss still only on C. Re-measure Delta (helper may be wrong;
     this is the regime R4 lives in).
  R4 single-token selection RL from R3: model emits X (1 tok), then its own
     helper + its own C answer; reward = C score; POLICY LOSS MASKED TO THE X
     TOKEN ONLY. Measure P(select=X) trajectory.

Favorability levers: evaluated=C(empties) w/ helper A(stones)=exact complement;
boards moderately full so direct empty-enumeration is error-prone but
complement-after-A is reliable; escalate 5x5->7x7 if Delta too weak; no-think.

Pre-registered predictions (odds):
  P1 R1 feasibility: SFT reaches >=0.7 frac_perfect on A and C @5x5           80%
  P2 R2 differential (gold helper): acc(C|A) - mean acc(C|B,D,E) >= +0.15     55%
  P3 R3 differential (own helper):  Delta >= +0.10                            45%
  P4 R4 selection RL: IF Delta(R3)>=+0.10, P(select=A) rises >=+0.2 in <=40 steps 60%(cond)
  P5 R4 competence: C-accuracy of picked-A rollouts within noise of R3        75%
  Meta: at least one of {P2,P3} is a null (differential too weak)             40%

Checklist before R1/R4 launches: wandb (wired in run_armD_sft/run_pilot);
Opus-solve each of A-E verbatim; base-qwen no-think temp0 eyeball; NO optimizer
snapshot (save_contents=[model,extra] already in launchers). Artifacts under
hexenv/arme.py, scripts/*_arme_*, data/arme/, results/arme/.

## 2026-08-10 ~00:20 — Arm E Phase 0 checklist PASSED; tasks well-posed, base fails on competence

Built hexenv/arme.py (task defs + gold + graders + prompts). Gold scores
1.0/perfect on all 5 tasks x300 random 5x5 boards; invariants verified
(A.black|white == complement of C; E == stones - edge-connected). 5x5 fill
~12.8 stones / ~12.2 empties (good regime: direct empty-enum error-prone,
complement-after-A reliable).

Checklist:
1. wandb: WANDB_API_KEY in .env; launchers log wandb. OK.
2. Opus-solve (2 boards, all 5 tasks verbatim): BOTH scored 1.0/perfect on
   A-E. Flagged 2 STATEMENT ambiguities (not data): (i) "connected to an edge"
   defined only implicitly; (ii) a stone on a full winning chain double-lists
   in D (top&bottom). Both agents inferred the intended reading correctly ->
   fixed the preamble to state both explicitly. Cheap ambiguity removal.
3. base-Qwen no-think temp0 eyeball (R1 format, 3 boards): model FOLLOWS THE
   TAGGED FORMAT CLEANLY (valid <task-A>/<task-C> JSON every time) -> format
   unambiguous. Failure is pure BOARD-READING COMPETENCE: hallucinated stones;
   2/3 boards lazily dumped all 25 cells as "empty" (C score ~ -0.08). This is
   the SFT-fixable failure (cf occupancy LoRA 0.998), NOT ambiguity. Ideal.
4. No optimizer snapshot: save_contents=[model,extra] already in
   run_armD_sft.sh / run_pilot.sh.

Design locked: evaluated=C; selectable helpers {A,B,D,E}; A=designed-useful
(exact complement). R1 data = per board, 4 examples (X,C) for X in {A,B,D,E},
full-completion loss (feasibility). R2/R3 reuse the same SFT harness
(ArmDWitnessSFTDataset float masks) with loss masked to the evaluated C span
only. Adapters chain base->R1->R2->R3; R4 RL from R3-merged.

## 2026-08-10 ~01:00 — Arm E pipeline built; R1 SFT running

All staged machinery committed before any gate: pool (5000/600 disjoint),
R1/R2/R3 SFT builders (R2/R3 mask gradient to the evaluated-C span only),
eval_arme (r1/select_tf/select_own -> acc(C|helper=X) differential),
hex_select_loop (R4: response_mask=1 on ONLY the selection token; selection
constrained to {A,B,D,E} via vLLM allowed_token_ids), R4 launcher+dataset,
P(select=X) trend analysis. Derisked R4: verl splats sampling_params into
vLLM SamplingParams (vllm_async_server.py:587), so allowed_token_ids + string
stop pass through. Adapters chain base->R1->R2->R3; R4 RL from R3-merged.
R1 (feasibility) fitting fast (loss ~0.03 in epoch 1); will eval epoch-1
checkpoint for gate G1 rather than wait 3 epochs.

## 2026-08-10 ~00:45 — R1 gate: tasks feasible BUT evaluated C is at CEILING at 5x5 -> escalate board

R1 (1 epoch, loss 0.016) eval on 300 test boards, mode=r1 (fixed-order X then C):
  helper own-accuracy:  A 0.980   B 0.890   D 0.663   E 0.443
  C (evaluated) perfect: 0.997/0.983/0.997/0.980 across helpers; delta=+0.010
Reading: the evaluated task C (list ~12 empties on 5x5) is SATURATED at ~0.98
regardless of which helper preceded it. This is the pre-registered "no
instrumental gradient" risk realized by ceiling: a scaffold cannot lift a task
the model already solves. delta ~ 0 by construction at 5x5.

Decision (still "most favorable conditions" — engineering favorability): move
to a LARGER, SPARSER board so C-alone is hard (many empties, error-prone to
enumerate) but C-given-A is mechanical (empties = all cells minus the short
stone list A). Keeps the user's evaluated=C / useful-helper=A design and the
cleanest scaffold (exact complement). Target a regime where C-alone perfect
~0.3-0.7 (headroom). Try 7x7 sparse first; escalate to 9x9 if C stays >0.85.

Note for R4: A is both the useful helper AND the highest-competence helper
(0.98) while decoys D/E are weak (0.66/0.44). If RL selects A, "useful" and
"easy" are confounded; acceptable for a first demonstration, will flag.

## 2026-08-10 ~01:30 — KEY FINDING: helper CONTENT is ignored for board-derivable tasks; pivot evaluated C->E

7x7 sparse R1 (1 epoch). Measured C (empties) accuracy three ways:
  C solo (no helper, untrained format): 0.010
  C after ANY helper (A/B/D/E):         0.98-0.99  (delta A-vs-others = +0.003)
Read the samples: for helper=E (whose output is a SHORT partial list, NOT the
stone set), the model still emits a PERFECT 34-cell empties list -> it is
RE-READING THE BOARD for C, not using the helper's content. So:
  - the "helper helps" jump (0.01->0.99) is a FORMAT/position effect: C-as-2nd-
    task is trained (0.99); C-as-solo is an untrained format (0.01 OOD).
  - helper CONTENT provides ZERO instrumental value for C: C is board-derivable,
    so which helper was solved is irrelevant. No content differential exists.

Implication for arm E: "select the useful helper" has a gradient ONLY if the
evaluated task is (a) NOT directly board-derivable by the 1.7B, AND (b) made
solvable by a SPECIFIC helper's content. C fails (a). The connectivity task E
fits: E (stones connected to NEITHER edge) needs edge-connectivity, which the
model does only ~0.66 as a first task (NOT ceiling -> a real bottleneck), and
helper D (edge-connected sets) reduces E to a set-subtraction
  E_black = black_stones - D.black_top - D.black_bottom
while A/B/C supply no connectivity (warmup only). Predict acc(E|D) >> acc(E|A,B,C).

DECISION: pivot evaluated task C -> E, useful helper -> D. Parametrize EVALUATED
in arme.py. This departs from the user's C example but is REQUIRED for any
instrumental signal (and is itself the finding: RL-selection needs the helper to
supply a computation the model lacks, not merely restate readable board facts).

## 2026-08-10 ~02:30 — E landscape + structural hypothesis: difficulty is non-outsourceable

R1(E) 1 epoch, 7x7 sparse. E (connected-to-neither) accuracy:
  E solo (untrained format) 0.777
  E after OWN helper: A 0.803  B 0.827  C 0.807  D 0.850   (delta D-vs-ABC +0.038)
  helper own-accuracy: A 0.977  B 0.947  C 0.980  D 0.820
Notably B (winner+path) own-acc 0.947 -> on these SPARSE 7x7 boards the model
finds winning connections EASILY; nothing is hard enough to force helper use.

STRUCTURAL HYPOTHESIS (why no natural differential exists here): every task
A-E is a deterministic function of the FULLY-VISIBLE board, and SFT teaches the
model to compute the evaluated task DIRECTLY (A .98, B .95, C .99, D .82, E .78).
The only "hard" part is CONNECTIVITY (tracing chains). But any helper that
supplies connectivity (D) is ITSELF a connectivity task — as hard as the
connectivity-bound evaluated task (E). The perception helpers (A=stones,
C=empties) are easy but supply NO connectivity. So there is no "easy helper
unlocks a hard evaluated task" pair: the difficulty cannot be OUTSOURCED to an
easy auxiliary. => the natural instrumental differential is ~0, not because RL
fails, but because NO task is instrumentally useful.

This is the sharpening of the C finding: not merely "C is board-derivable" but
"the evaluated task's difficulty (connectivity) is non-outsourceable within this
family." Caveat before concluding: the clean GOLD-helper differential (R2
select_tf) is unmeasured; own-D was only 0.82 correct, so gold-D (perfect
edge-connectivity in context) could lift E more than +0.04. Waiting on R2.
If gold-D differential is real (>~+0.15), R4 selection RL is a clean test; if
tiny, R4 tests RL's sensitivity to a sub-0.05 instrumental edge (marker analog).

## 2026-08-10 ~03:15 — Gate G2 (gold-helper differential) NULL on sparse: +0.035; pivot to dense boards

R2(E) SFT (continue from R1, gradient on E only, 1 epoch). Clean GOLD-helper
differential (select_tf, gold helper X in context, generate E):
  E | gold A 0.930  B 0.912  C 0.910  D 0.953   ->  delta(D vs A,B,C) = +0.035
Even a PERFECT helper D lifts E only +0.035; all helpers put E at 0.91-0.95.
Confirms: sparse 7x7 -> E easy (0.91 after a useless helper) -> no headroom.
(Note gold-helper 0.91 > own-helper 0.80 > solo 0.78: format/warmup lifts E to
~0.91 regardless of helper; gold-D adds a marginal +0.035 on top.)

Mechanism recap: E|gold-D difficulty = the SUBTRACTION E=stones-D_edge; E|board
difficulty = CONNECTIVITY. On sparse both are easy -> tiny gap. To get a real
gap I need connectivity HARD but subtraction manageable -> DENSE tangled boards
(near-miss stones whose chains almost-but-don't reach an edge). Pivot: regenerate
pool ARME_FILL=dense (mean ~24 stones, tangled multi-chains, E-size ~7), retrain,
re-measure gold-D differential. If dense also gives a small gap, the
non-outsourceable-difficulty null is complete and R4 becomes a test of RL's
sensitivity to a sub-0.05 edge (expected: no selection).

## 2026-08-10 ~04:00 — DENSE boards create a REAL differential: E-solo 0.42, own-D helps +0.098

R1-dense (1 epoch). E (connected-to-neither) on DENSE tangled 7x7:
  E solo 0.423  (HARD now; was 0.777 sparse -> dense connectivity is genuinely hard)
  E after OWN helper: A 0.527  B 0.543  C 0.507  D 0.623   delta(D vs ABC)=+0.098
  helper own-acc: A 0.967 C 0.967 (easy perception) | B 0.767 | D 0.537 (D now HARD)
So on dense boards D helps E by +0.098 EVEN with the model's own D only 53.7%
correct. This validates the premise: when the evaluated task is beyond direct
competence (E-solo 0.42) and a helper supplies the missing computation
(connectivity), the helper is genuinely useful — and A/C (easy perception, no
connectivity) don't help beyond warmup. GOLD-D differential expected LARGER
(own-D is badly muddied at 0.54). This is the regime R4 needs.

Note for R4: the differential the model EXPERIENCES with its own helper is
+0.098 (D vs A,B,C) — ~3x the sparse gold gap and, crucially, present with
self-generated helpers, which is exactly what R4 rolls out.

Prediction: R2-dense gold-D differential >= +0.15 @70%.
Next: R2-dense (gold-D upper bound) -> R3-dense (own-helper) -> R4-dense selection RL.

## 2026-08-10 ~04:40 — R2-dense: the premise FAILS cleanly — no helper is instrumentally useful

R2-dense (select format, gradient on E only, gold helper in ctx). Clean measures:
  select_tf (GOLD helper, no generation cost):
    E | gold A 0.877  B 0.825  C 0.825  D 0.740   -> delta(D vs ABC) = -0.102
    i.e. a PERFECT connectivity helper D does NOT help E; it slightly HURTS.
  select_own (model generates own helper):
    E | own  A 0.792  B 0.203  C 0.825  D 0.310   -> delta = -0.297
    helper own-acc ~0 for ALL (A .000 C .007 B .025 D .003)

Two mechanisms, from reading samples:
1. CONTENT IGNORED: on board gi=0, gold-A-ctx and gold-D-ctx yield the IDENTICAL
   E answer -> the model re-derives E from the board and ignores helper content
   (same as the C finding). So gold-D can't help; its longer context slightly
   hurts (0.74 vs 0.88).
2. GENERATION-COST CONFOUND: R2 gradients on E only, with helper GOLD in ctx ->
   it never trains helper GENERATION in the select format. So own-helpers are
   garbage; generating a HARD helper (B/D connectivity) derails the subsequent E
   (0.20-0.31), while easy helpers (A/C perception) leave E intact (0.79-0.83).

=> The instrumental differential is <= 0. NO helper is useful: the eval is
board-re-derivable (content ignored), and the only useful content (connectivity)
is (a) ignored when given and (b) costly to generate. The user's gate "if R2/R3
work out" is NOT met. RL-select-useful-task has no positive gradient here; the
ONLY exploitable reward signal is "avoid hard-to-generate helpers" (a generation
artifact), which RL would ride toward A/C, not the 'useful' D.

Robustness: null confirmed across C-sparse (+.003), E-sparse-gold (+.035),
E-dense-gold (-.102), E-dense-own (-.297). The r1-format own-D +0.098 earlier
was a format artifact (r1 trains helper-gen+eval jointly; select format does not).

STRUCTURAL CONCLUSION: in a family where every task is a deterministic function
of a fully-visible board, SFT makes the eval directly solvable, so no auxiliary
task is instrumentally necessary; and the one hard skill (connectivity) cannot be
outsourced to an easy helper. "Select the useful sub-task" has no referent here.

## 2026-08-10 ~05:10 — User redirect: winner task + gradient-on-both-tasks (helper useful, not necessary)

Joshua's redirect after the premise-failure report:
 - Find a helper that is USEFUL even if not strictly necessary. His intuition:
   evaluated = WINNER should benefit from helper D (which stones connect to
   left/right, top/bottom) -> winner is an intersection check on D's lists
   (Black wins iff a stone is in black_top AND black_bottom; White iff in
   white_left AND white_right).
 - KEY FIX he flagged: the two-task SFT should put gradient on BOTH tasks (not
   just the evaluated one) so the model is trained to do the eval with NO helper
   too -> makes W-solo a fair in-distribution baseline. (My R2/R3 gradient-on-
   eval-only left helper-generation untrained and solo OOD -- the confound behind
   the garbage own-helpers.)
 - If everything is ~ceiling, move to a bigger board.
 - Note: witness (task B) already contains winner as a subtask.

New setup: added task W (winner) to arme.py; EVALUATED=W, ARME_HELPERS=A,C,D
(D useful; A=stones, C=empties decoys; B excluded as it trivially contains the
winner). SFT (build_arme_winner.py): select-format, gradient on helper+winner,
PLUS solo-W examples (gradient on winner). Training arme_win from base on dense
7x7. Will measure: W-solo vs W|gold-{A,C,D} (select_tf) vs W|own-{A,C,D}
(select_own). If gold-D >> gold-A/C and > solo, D is genuinely useful -> R4.
If winner ~ceiling on 7x7, escalate to 9x9.

## 2026-08-10 ~05:45 — WINNER experiment: D is genuinely USEFUL (user hypothesis CONFIRMED)

arme_win (dense 7x7, select-format SFT, gradient on helper+winner + solo-W).
Winner accuracy (n=400), after fixing a grader artifact (model emits bare
"White" not {"winner":"White"} after dict-valued helpers -> parse_answer now
lenient for W):
  W solo (trained baseline)          0.907
  W | gold A 0.945  C 0.915  D 0.993   -> delta(D vs A,C) = +0.063  (+0.086 vs solo)
  W | own  A 0.902  C 0.910  D 0.948   -> delta(D vs A,C) = +0.041
  own-helper acc: A 0.943  C 0.980  D 0.385 (D hard to generate)

FIRST POSITIVE DIFFERENTIAL. Gold-D nearly maxes the winner (0.993) -- knowing
which stones connect to each edge reduces the winner to an intersection check,
exactly Joshua's intuition. And own-D STILL wins (0.948 > solo 0.907 > A/C 0.90)
despite own-D only 38% correct: the CONTENT benefit outweighs the generation
cost. So D is both the USEFUL helper and the REWARD-MAX selection -> R4 has a
real (if modest) gradient to select D.

Why the earlier E/C nulls but W works: winner is a SINGLE global connectivity
predicate (does one spanning chain exist) that D's edge-sets answer directly via
intersection; E/C were either board-trivial or required re-deriving the same
work. The corrected SFT (gradient on BOTH tasks + solo) was also load-bearing --
it made helper-generation and solo-W in-distribution.

Modest magnitude (winner solo 0.907 is high). Per Joshua's "bigger board if
~perfect" -> a larger/harder board should widen the gap. Deciding escalate-vs-run
after checking the R4 selection prior P(X) (untrained: X was always teacher-forced
through SFT).

## 2026-08-10 ~06:40 — R4 machinery validated; 3 fixes; 7x7 winner too easy (warmup swamps D) -> escalate

Built + debugged the selection RL (hex_select loop) to WORKING (GROUP_N=8):
loop samples selection (allowed A/C/D), generates helper+winner, grades, logs.
Three fixes:
 - FIX 1 (exploration): SFT selection prior was degenerate P(D)=100% (accident of
   untrained prior). Trained the X token with X~uniform in win SFT -> prior ~uniform
   (A .31/C .35/D .34), like the marker 50/50 init.
 - FIX 2 (gen temp): at rollout temp 1.0 per-helper winner reward was A .74/C .74/
   D .71 (D NOT advantaged -- generating hard D at high temp derails the winner).
   Decoupled: selection at rollout temp (explore), helper+winner GREEDY
   (ARME_GEN_TEMP=0) so D's content survives.
 - FIX 3 (OOM): GROUP_N=16 (512 rollouts x3 generates) OOM-killed raylet
   (pids/cgroup). GROUP_N=8 (256, marker-proven) stable.

Greedy-gen per-helper winner reward (7x7, smoke): A .866 / C .930 / D .917.
C(decoy) ~= D(useful) > A: winner is EASY on 7x7 (solo .907) so ANY helper warms
it to ~.9 and D's connectivity edge (+.04) is swamped by C's warmup. No clean D
preference. Need winner HARD (solo ~.5-.6) so warmup != solution. Escalating to
9x9 dense. Tension noted: own-D generation degrades on bigger boards.

## 2026-08-11 ~00:50 — R4 PAYOFF: RL learns to SELECT the useful helper (P(D) 0.35 -> 0.99)

Gold-helper selection RL (evaluated=W winner, helpers A/C/D, useful=D; only the
1-token selection is trained; helper teacher-forced gold; winner greedy; reward =
winner score). From the uniform-X win model. BATCH=16 GROUP_N=8 (128 rollouts/step),
lr=4e-6, KL=0.001.

Per-helper winner reward (constant across steps): sel D:+1.00  C:+0.6..0.9  A:+0.4..0.8.
P(select) trajectory:
  step 0: A .24  C .41  D .35   (uniform-ish prior)
  step 5: A .00  C .03  D .97
  step10: A .00  C .01  D .99   (converged, holding)
So RL drives the single selection token from ~1/3 to ~0.99 on D in ~5-10 steps,
monotone -- MARKER-SCALE. This is the arm-E positive result:

  RL CAN be trained to select the most instrumentally-useful auxiliary task,
  when (a) a helper is genuinely useful (gold-D makes the winner a trivial
  intersection -> reward 1.0 vs 0.5-0.9 for decoys), and (b) usefulness is
  isolated from the model's ability to GENERATE the helper (teacher-forced gold).

What it took (each a real obstacle, logged above):
  - a useful helper to EXIST: winner<-D (connectivity), after C/E gave nulls
    (board-derivable, content ignored). Difficulty must be non-board-derivable
    AND suppliable by the helper.
  - SFT gradient on BOTH tasks + solo (Joshua's fix) so helper-gen and solo-eval
    are in-distribution.
  - uniform selection prior (was degenerate P(D)=1.0 by accident) via training the
    X token uniformly -- the marker 50/50 analog; without it, no exploration.
  - greedy helper/eval generation (decoupled from selection temp) OR gold helper,
    else high-temp derailing of the hard helper D erases its edge.
  - memory: RESP_LEN 256, BATCH 16 (worker-OOM hang at 512/560).

Caveat (the honest boundary): this is the GOLD-helper regime -- usefulness
isolated. In the OWN-helper regime (model generates its own D) the differential
is muddied because D is BOTH the useful AND the hardest-to-generate helper
(own-D ~0.38 acc), so RL would chase the net-best (easy decoy C ~= D). Running
that contrast next.

Prediction grades (original C-design, graded in spirit after the pivot):
  P2 gold-helper differential >=+0.15: NO numerically (winner gold-D +0.06-0.08),
     but D reaches 0.99->1.0 and is decisively best -> the SELECTION differential
     is huge even if the accuracy gap is modest.
  P3 own-helper >=+0.10: NO (muddied by generation cost).
  P4 RL moves P(useful) >=+0.2 in <=40 steps: YES emphatically (+0.64 to 0.99 by
     step 10) -- in the gold regime.
  Meta "at least one null": YES (C and E both null).

## 2026-08-11 ~01:05 — R4 correction: clean win is steps 4-7; then over-optimization collapses competence

Full 12-step gold trajectory (128/step) — correcting the "converged/holding" note:
  step  P(A)  P(C)  P(D)   Rperf  meanR_by_sel
   0    .22   .41   .37    .95    D:+1.00
   2    .06   .34   .61    .84    D:+1.00
   4    .00   .03   .97    .93    D:+0.87
   5    .00   .02   .98   1.00    D:+1.00   <- CLEAN: P(D)~.98, winner reward +1.0
   7    .00   .09   .91   1.00    D:+1.00
   8    .00   .79   .21    .94             (a wobble)
   9    .00   .01   .99    .62    D:+0.24
  10    .00   .00  1.00    .20    D:-0.61   <- competence COLLAPSING
  11    .00   .00  1.00    .06    D:-0.88

So the SELECTION result is real and clean in steps 4-7 (P(D) .35->.98, winner
reward still +1.0). AFTER that, competence collapses: once P(D)~1 the group loses
selection contrast (advantage->0), and because ONLY the selection token is
trained/KL-anchored (winner tokens are mask-0, no gradient AND no KL), the shared
LoRA weights drift and BREAK winner generation (Rperf .06 by step 11). This is an
inherent artifact of "gradient on ONE token only" on a shared adapter -- nothing
maintains the eval competence. Mitigations: fewer steps / lower lr / KL on ALL
response tokens / a tiny SFT-anchor on the eval. The demonstration stands (RL
selects D by step 5, competence intact); the collapse is over-optimization, not a
failure of the selection mechanism. lr=4e-6 was aggressive -- for a clean full
run use lr~1-2e-6 and stop by ~step 10.

## 2026-08-11 ~01:25 — Own-helper R4 CONTRAST: RL does NOT select D (drifts to easy decoy C)

Own-helper selection RL (ARME_GOLD_HELPER=0: model generates its OWN helper, greedy;
lr=2e-6, 14 steps, BATCH16). Same winner reward, same model.
P(select) trajectory:  step0 A.22 C.41 D.37 | step8 A.24 C.45 D.31 | step13 A.03 C.56 D.41
Net: P(A) .22->.03 (RL drops the WORST helper hard), P(C) .41->.56 (rising), P(D)
.37->.41 (flat). Per-helper reward NOISY, roughly C>=D>A, C and D trading the top --
no consistent D advantage. RL mildly prefers the EASY DECOY C, NOT the useful D.

Contrast with gold-helper (P(D)->.98): when the model GENERATES its own helper, the
useful helper D (edge-connectivity) is the HARDEST to generate (own-D ~.38), so its
content edge is erased -- own-D winner ~= own-C winner. C is easy AND a good warmup,
so it is the net-reward-max. RL rides the actual reward; usefulness isn't in it once
you must pay to generate D.

ARM-E CONCLUSION (both regimes):
 RL reliably SELECTS the reward-max task-token (P ~1/3 -> arg-max in a few steps).
 Whether that == the USEFUL helper depends on the setup:
  - usefulness ISOLATED (gold helper): reward-max == useful D -> RL selects D (.35->.98).
  - NOT isolated (own helper): useful D is costliest to produce -> reward-max == easy
    decoy C -> RL selects C, not D.
 "Can RL learn to select the most useful sub-task?" It selects on the REWARD; "useful"
 wins only when the environment makes the useful action the rewarded one. Same
 selection-not-creation throughline: RL redistributes probability toward reward; it
 neither creates usefulness nor sees human intent.

## 2026-08-12 ~02:05 — ARM F KICKOFF: containment of HexHex CNN in Qwen3-1.7B (universal-representation stress test)

New arm (user-initiated, "unhinged"): can Qwen3-1.7B + per-layer affine adapters be
fine-tuned so its residual stream CONTAINS the activations of the superhuman HexHex
CNN (18 skip-layers x 64ch, 11x11), depth-aligned: Qwen layers 5..23 <-> CNN capture
points 0..18 (post-initial-conv + after each skip layer)? No LM loss — we don't care
about preserving Qwen's outputs. Killer eval = STITCHING: run Qwen to layer 5+k, affine
A_k into CNN layer k, finish in CNN, measure move-match / play strength per cut.

Setup verified tonight:
- HexHex repo cloned; pretrained 11_2w4_2000.pt loads (18 layers, 64ch, reach 1,
  switch+rotation wrappers). 20/20 vs random. Empty-board move (3,1) = pie-rule-safe.
- armF/hexhex_wrap.py: dump_acts (19x (B,64,11,11)), stitched_logits verified exact
  vs inner forward at cut 0.
- armF/render11.py: text render == canonical CNN input (always to-move perspective,
  'X to move'); Qwen tokenizer gives exactly 121 distinct cell tokens (~175-token
  prompt), verified via offset mapping.
- CNN acts will be computed on the fly (CNN is tiny); dataset stores boards only.

REGISTERED PREDICTIONS (before any probe/training run):
 P1 frozen-Qwen affine probes, per-layer R2 on val: z0-z2 R2>0.5: 75%; z14-z18
    R2<0.2: 70%. (Early layers ~= local board features Qwen must represent; deep
    layers = superhuman win-relevant features Qwen cannot have.)
 P2 joint FT (LoRA or full) reaches R2>0.8 on ALL 19 layers: 60%; >0.9 all: 35%.
 P3 stitching argmax move-match vs pure CNN at deepest cut (k=18, Qwen does all the
    work) >= 60%: 40%.
 P4 stitched player (every cut) beats random 20/20: 50%.

## 2026-08-12 ~02:50 — ARM F frozen-probe result: P1 half wrong; random-init control kills the "deep alignment"

76210 unique positions generated (900 selfplay T-schedule+eps games, 300 random).
Pipeline validated HARD: patch-affine control (r=1, shared weights, conv geometry)
hits R2=1.000 on z0 exactly as it must (z0 IS conv(patch)) -- cell/target alignment
is provably correct.

Frozen pretrained Qwen affine probes (val R2, 6000 train / 1000 val positions):
 z0 .57  z2 .47  z5 .33  z9 .28 (min)  z14 .33  z18 .37  -- U-shape.
 patch3 control: .51 at z2 declining to .21 deep. Qwen > patch3 everywhere >= z2.
P1 grading: early-layer half CORRECT (z0-z1 > .5, z2 .47 borderline). Deep-layer
half WRONG (predicted <.2 at 70%; actual .33-.37).

BUT random-init Qwen control (same arch/prompt, untrained): z18 R2 .354 vs
pretrained .370. Deep-layer probe R2 is ~entirely random-feature kernel capacity
(2048-dim nonlinear reservoir of the board), NOT pretrained representation.
Pretrained-over-random advantage: z0 +.13, z5 +.08, z14+ <=.04, z18 +.016.
DECONFUSION NOTE: naive linear probing overstates "universal representation"
alignment; capacity-matched random-init control removes ~all of the deep signal.
Frozen Qwen has NO privileged access to superhuman deep hex features. The arm-F
question is now purely about the TRAINED version.

Design note (caught before any training): causal attention makes single-copy
cell-token readout impossible even for z0 (cell can't attend to the row below).
All probes/training use a TWO-COPY board render; readout at copy-2 cell tokens.

## 2026-08-12 ~02:40 — ARM F r1 launched + P5 registered

armF_containment_r1 launched: 9000 steps, batch 16, lr 1e-5 (backbone blocks 0-22,
embeddings frozen), 1e-3 adapters (warm-started from probe ridge solution -- step-0
val R2 reproduced the frozen-probe numbers exactly, pipeline verified end to end).
0.62 s/step, 26.5GB, ~1.6h. Loss = mean over 19 layers of MSE on normalized acts.
Truncated-backbone gotcha caught in smoke: HF applies final RMSNorm to the last
hidden_states entry -> truncate to 23 blocks and replace norm with Identity so all
read points (out4..out22) stay raw.

P5 registered (before r1 finishes): a from-scratch random-init backbone trained with
the same recipe reaches LOWER val R2 than pretrained-init at equal steps: 70%.
(If wrong -> Qwen's pretraining contributes ~nothing to learning containment, and
the arm-F result is about transformer capacity, not universal representations.)

## 2026-08-12 ~05:00 — ARM F stitching eval: play strength survives full-trunk replacement; P3/P4 graded

Stitched network = Qwen(trained r1, step 9000) -> A_k -> CNN skiplayers[k:] ->
policy head. First vs-CNN play eval was degenerate (both players deterministic +
color alternation = 2 distinct games x10); fixed with shared random 4-ply openings
per game pair (paired comparison, 40 games/cut).

Agreement vs pure CNN (500 val positions, top1/top3/spearman on legal moves):
 cut 0: .892/.998/.981   cut 4: .672/.896/.943   cut 9: .584/.844/.920
 cut 14: .558/.816/.902  cut 18: .452/.714/.878  (smooth monotone decline)

Play strength (the headline): vs random 19-20/20 at every cut; vs pure CNN with
4-ply openings: cut0 21/40, cut4 15/40, cut9 18/40, cut14 22/40, cut18 18/40 =
94/200 = 47% overall. The stitched player is at PARITY with the pure CNN at every
cut -- including cut 18, where Qwen+A_18 does the ENTIRE trunk and only the linear
policy head remains. Top1 agreement 0.45 but no measurable strength loss: the
disagreements are ~equivalued moves (top3 .71, spearman .88), not blunders.

FROZEN baseline contrast (pretrained backbone + probe adapters, no FT), same eval:
 top1 .20 (cut0) -> .08 (cut18); vs CNN: 1/40, 1/40, 0/40, 0/40, 2/40 = 4/200.
 vs random: 20,19,20/20 at cuts 0-9 but degrades to 16/20 (cut14), 7/20 (cut18).
So the probe-only stitch is categorically broken as a player at depth, while FT
containment plays at teacher level. The FT didn't just improve R2 from .37 to .77;
it crossed the threshold from "noise player" to "full-strength teacher clone".

P3 graded (stitch top1 at k=18 >= 60%, registered 40%): NO -- .452. My 40% lean
against was right, but for the wrong texture: I expected agreement failure to mean
strength failure. It doesn't.
P4 graded (every cut beats random 20/20, registered 50%): NO by one game (19/20 at
cut 14); all other cuts 20/20. Effectively at ceiling.

DECONFUSION NOTE (the real finding): move-match is the wrong metric for
containment quality -- play-strength parity at 47% top1-disagreement means the
containment preserved the *decision-relevant* structure of the CNN's activations
even where per-channel MSE/R2 (.70-.76 deep) looks mediocre. Affine containment of
an 18-layer superhuman CNN trunk inside a fine-tuned 1.7B LM's residual stream is
REAL by the causal (stitching) standard, not just the correlational (probe) one.

Artifacts: armF/results/stitch_eval.json, stitch_eval_frozen.json,
stitch_r1.log, stitch_frozen.log.

## 2026-08-12 ~05:10 — ARM F language check: hosting the CNN costs ~0.15 nats

Spliced FT'd blocks 0..22 back into original Qwen3-1.7B (blocks 23-27, norms,
embeddings, lm_head untouched). Token NLL on English sample: 2.045 -> 2.197
(ppl 7.7 -> 9.0). Greedy generations all coherent: Paris/correct fibonacci
def/100degC. So the containment fine-tune found weights that host the full CNN
trunk at cell tokens AND remain a functioning language model on text -- the two
computations coexist in the same 2048-dim residual stream with modest
interference. (No LM loss was used; this is purely incidental preservation at
lr 1e-5 / 9000 steps.)

## 2026-08-12 ~06:50 — ARM F randinit control done: P5 YES; + activation-map rank check

Random-init FT control (same recipe, warm-started from probe_frozen_randinit.pt --
first launch was confounded by pretrained-probe warm start, step-0 R2 ~ -800;
killed, added --probe flag, relaunched with step-0 reproducing the randinit probe
0.441 vs 0.445). FINAL mean val R2 0.7169 (z0 .948, trough .637 z11, z18 .698)
vs pretrained-init 0.7728 (z0 .991, z18 .763). Trajectory tracked ~2000 steps
behind pretrained throughout (randinit@7000 = 0.706 ~ pretrained@3000 = 0.701).
P5 graded (pretrained wins at equal steps, registered 70%): YES. Reading: real
but modest pretraining contribution, concentrated early/local (z0 gap .043);
most of containment is learnable from scratch -- arm F's result is substantially
about transformer capacity + FT, with pretraining as a head start, not a
prerequisite.

Rank check (armF/rank_check.py, SVD of full 7744-dim activation maps over 10k
positions), prompted by Joshua's density question: maps are strongly
low-dimensional with an HOURGLASS depth profile. rank90: 160 (z0) -> 1293 (z9
peak) -> 557 (z18); participation ratio 71 -> 90 -> 7.4 (deep layers collapse
onto a handful of global decision directions feeding the linear policy head).
Top-2048 dims capture >=94.7% variance at EVERY layer => a one/few-token
"register" affine readout of the full map is NOT capacity-doomed (contra my
earlier rank-bound argument); open question is whether the ~5% tail (rank99
~4000) carries play-relevant signal, and whether Qwen can learn to pack the
principal coords. Relevant to a possible r2 with moves-in-order prompts:
per-move activation-column supervision + register readout.

## 2026-08-12 ~08:10 — ARM F r2 (moves format) spec + P6-P9 registered + launch

Joshua asked for denser signal via moves-in-order input; rank check said full
maps compress (>=95% var in 2048 dims). r2 sequence = preamble + move list cut
at a random ply + SINGLE-copy render of the post-cut board (the move list makes
a single copy causally readable — every stone is in context before the render).
Three simultaneous supervision streams, per layer l in 0..18 at hidden 5+l:
 1. col: at each move token t, the CNN's 64-ch column at the just-played cell,
    board AFTER move t (new adapters C_l 2048->64) — every prefix supervised.
 2. pca: at each move token t, top-128 whitened PCA coords of the full 7744-dim
    map after move t (P_l 2048->128; basis captures 55-83% var by layer).
 3. render: full 121x64 map at render cell tokens (A_l, as r1).
Data: 2200 games (1800 selfplay + 400 random, median 73 plies, 168k boards),
5 cuts/game -> 11000 seqs, maxlen 480 tokens; val = game_id % 15 == 0.
Density: ~282k supervised scalars per ~300-token seq (~940/token, ~2x r1's 443,
and ~38 distinct boards per sequence vs 1). Verified: 20-game exact replay,
column indexing, PCA round-trip, 3 full samples eyeballed (121/121 cells each).
Smoke: loss backward ok, 10.6GB. NOTE: r1 probe warm-start does NOT transfer to
the moves-format prompt (render step-0 R2 -0.17 vs +0.37 in-format) — another
data point that these readouts are prompt-context-sensitive.

Predictions registered BEFORE launch (run armF_moves_r2, 9000 steps, batch 12):
 P6 render-stream final mean val R2 within 0.05 of r1's 0.773 (single-copy +
    moves context instead of two-copy): 60%.
 P7 col-stream final mean R2 < render-stream mean R2 (one token must carry
    incrementally-tracked state vs 121 tokens reading a rendered board): 70%.
 P8 pca-stream final mean R2 > 0.5: 55%.
 P9 stitched play from render cells at parity (>=40% pooled vs CNN, paired
    4-ply openings, cuts 0/9/18): 70%.

## 2026-08-12 ~10:35 — r2 FINAL + P6-P8 graded; r3 (render-free) planned; P10 registered

r2 (armF_moves_r2, 9000 steps) final val R2: col 0.6232, pca 0.4820, render
0.6097. Per-layer: col z0 .786 z9 .527 z18 .582; pca z0 .094 z9 .609 z18 .403;
render z0 .760 z9 .525 z18 .684.
 P6 (render within 0.05 of r1 0.773, 60%): NO — 0.610. Single-copy render +
    moves context is materially worse than r1's two-copy render.
 P7 (col < render, 70%): NO — col BEAT render. The interesting miss: move
    tokens are causally blind to the render (they precede it), so 0.623 there
    is pure internal simulation from the move list. Render-free readout is not
    the handicap I assumed.
 P8 (pca > 0.5, 55%): NO — 0.482, close. pca z0 .094 is a whitening artifact:
    equal weight on tiny-variance directions; per-dim-normalized targets don't
    have this pathology.
Stitch (moves format) partial: cut0 7/40 vs CNN, cut9 16/40 (cut18 pending) —
well below r1 parity; render cells in moves format are causally weaker, in line
with lower render R2.

r3 plan (user): render-free. Input = preamble + move list ONLY; at each move
token, per-layer Linear(2048->7744) reconstructs the FULL normalized CNN map
after that move. No cuts needed: every move token is a supervised prefix (one
seq per game, ~73 prefixes/seq, ~147k scalars/move token). ~301M adapter params
(19 x 2048x7744) — user approved. Hex favors this: append-only state, no
removals; hard parts are the per-ply canonical flip and whether DEEP features
(not just stones) are linearly exposed.

Mini falsifier first (user-requested): z0 only, 1000 steps, joint FT
(train_movesonly_z0.py). Prediction registered BEFORE launch:
 P10 render-free z0 full-map pooled val R2 at step 1000 >= 0.70: 65%.
    (col-z0 hit .786 for the played-cell column; full map includes all 121
    cells incl. empty; z0 is post-first-conv i.e. mostly local stone features.)

## 2026-08-12 ~10:50 — P9 graded NO; depth gradient in moves-format stitching

Stitch (moves format, r2 best.pt step 9000): vs pure CNN with paired 4-ply
openings: cut0 7/40, cut9 16/40, cut18 21/40 — pooled 44/120 = 36.7% < 40% →
P9 NO (70% was too confident). But the structure is the story: PARITY at
cut 18 (entire trunk replaced by Qwen) and near-zero at cut 0. Inverse of the
naive difficulty ordering (z0 ~ stones). Reading: z-hat errors at shallow cuts
get amplified through the remaining CNN layers, while at cut 18 errors feed
straight into the policy head, and the (lower than r1) render R2 hurts shallow
cuts most. vs random: 18-20/20 at all cuts; 0 illegal argmaxes anywhere.

## 2026-08-12 ~11:15 — P10 graded NO (0.379); one token != whole board at hs[5]

Render-free z0 mini run (train_movesonly_z0.py, 1000 steps): val R2 0.379
(curve: .33/.35/.37/.38/.38 — converged for this budget). P10 (>=0.70, 65%) NO.
Contrast trio on the SAME z0 target: played-cell column from move token .786;
full map from 121 render-cell tokens .760; full map from ONE move token .379.
Capacity is not the binder (2048 dims >= 94.7% var of z0): the full board is
not linearly present in a single move token's state at hs[5]. Reading: causal
attention distributes state across move tokens; aggregating ~70 moves into one
token needs attention hops the first 5 layers don't provide (CNN gets z0 in one
conv because its input IS the board; Qwen must first collect it).

Next falsifier (before r3 commits to z_l <-> hs[5+l]): DEPTH SWEEP — same
render-free z0 full-map target, 23 adapters reading hs[1..23] simultaneously,
joint FT, 1000 steps. Predictions registered BEFORE launch:
 P11 peak-depth R2 >= 0.55 (somewhere in the stack the board IS linearly
     assembled): 60%.
 P12 argmax depth >= hs[10] (aggregation needs depth; early layers can't): 70%.

## 2026-08-12 ~11:40 — Parity hypothesis (user); P13-P14 registered before format A/B

Depth sweep at step 500: profile FLAT (hs1 .30 -> peak ~.36 @ hs7-13 -> .33 @
hs22). Evidence against "aggregation needs depth", for a uniform information
deficit. User hypothesis: LLMs are bad at PARITY — the move list gives stone
color only via list-position parity, and the canonical (to-move) target frame
additionally flips with total-count parity; wrong parity = wrong channel for
every stone = uniform R2 cap at all depths. Also explains col z0 .786 (own
color is local) vs full-map .379 (needs parity of ~70 stones).
A/B: same hs[5] -> z0 setup, input becomes numbered+colored moves ("1. e9 X"
newline-separated; readout at last token of each record). Registered BEFORE
launch:
 P13 numbered+colored format lifts z0 R2 by >=0.10 (to >=0.48): 70%.
 P14 reaches >=0.65: 35%.

## 2026-08-12 ~12:00 — Depth sweep final: P11 NO, P12 vacuous (plateau, not peak)

FINAL per-depth z0 R2 (render-free, one adapter per hs[1..23], joint FT 1000
steps): 0.310 @ hs1 rising to 0.383 @ hs11, then FLAT (+-0.004) through hs23.
 P11 (peak >= 0.55, 60%): NO — peak 0.383.
 P12 (argmax >= hs10, 70%): letter-YES (argmax hs12) but the spirit is a
    plateau: NO depth assembles the board. Depth-of-aggregation is falsified
    as the binding constraint; a uniform ~0.38 ceiling at every depth is the
    signature of an information deficit in the input encoding — consistent
    with the user's parity hypothesis (stone color = list-position parity;
    canonical frame = total-count parity).
Launched numbered-format A/B (--fmt numbered: "\n1. g1 X\n2. d4 O ...",
readout at the color token; maxlen 311->~690). Grading P13 (>= 0.48, 70%) and
P14 (>= 0.65, 35%) against plain baseline 0.379.

## 2026-08-12 ~12:30 — Numbered format A/B at 1000 steps: P13 NO, P14 NO — but curves diverge

numbered ("\n1. g1 X ...") final z0 R2 0.4456 vs plain 0.379 (+0.067).
 P13 (lift >= 0.10, 70%): NO — 0.067.
 P14 (>= 0.65, 35%): NO.
BUT: plain plateaued from step 600 (.371/.377/.379) while numbered was still
climbing steeply when LR annealed out (.414/.438/.446, +0.008/200 at the end).
Format helps directionally (parity hypothesis alive); 1000 steps is the binder
for numbered, information is the binder for plain. Extending A/B to 3000 steps
each (both arms, fair cosine schedules). Predictions BEFORE launch:
 P15 numbered@3000 >= 0.55: 60%.
 P16 numbered@3000 - plain@3000 >= 0.10: 55%.

## 2026-08-12 ~14:20 — 3000-step A/B: PARITY HYPOTHESIS CONFIRMED. P15 YES, P16 YES

plain@3000: 0.436 (so plain's 1000-step "plateau" at .379 was cosine-annealing
artifact, not an information ceiling — correction to the ~12:30 reading).
numbered@3000: **0.7744** (curve .561 @1000, .736 @2000, still climbing).
 P15 (numbered@3000 >= 0.55, 60%): YES — by +0.22.
 P16 (gap >= 0.10, 55%): YES — gap +0.338, 3x the bar.
Numbered format at 1000 steps (.446) already beat plain at 3000 (.436): ~3x
compute equivalent. And 0.774 ~= own-column .786 ~= 121-render-token .760:
with explicit move numbers + colors, ONE move token's state linearly contains
the full z0 map as well as per-cell readout geometries do. Joshua's parity
hypothesis (stone color = list-position parity; LLMs bad at parity) is the
binding constraint on render-free containment, not attention aggregation
(flat depth sweep) and not capacity (rank check).
Decision: r3 uses numbered format ("\nN. <move> <X|O>", readout at color
token), z_l <-> hs[5+l] depth alignment as user specced.

## 2026-08-12 ~14:50 — r3 launch (armF_movesfull_r3): render-free full-stack containment

train_movesfull.py: numbered move list only; at each color token, 19x
Linear(2048->7744) predict all normalized CNN maps (z_l <-> hs[5+l]); ~301M
adapter params; no cuts (every move token supervises its prefix); joint FT
(1e-5/1e-3), batch 8, 9000 steps. Smoke: 25.4GB peak incl. Adam. This is the
densest configuration of the arm: ~147k supervised scalars per move token.
Predictions registered BEFORE launch:
 P17 final mean val R2 (19 layers) >= 0.60: 55%.
 P18 z0 final >= 0.85 (mini hit .774 @3000 single-target, still climbing): 60%.
 P19 z18 final >= 0.50: 50%.
 P20 render-free stitch (z_k-hat from LAST move token -> CNN tail) at parity,
     >=40% pooled vs CNN, paired openings, cuts 0/9/18: 40%.

## 2026-08-12 ~12:10 — r3 complete (armF_movesfull_r3, 9000 steps): P17-P19 graded

Final val R² mean **0.5970**. Per-layer U-shape: z0 .823, z3 .612, z6 .548,
z9 .515 (trough), z12 .552, z15 .603, z18 **.656**. ~2.2h at 0.89s/step.

- **P17 NO by 0.003** (mean ≥.60 @55%): 0.5970. Painfully calibrated.
- **P18 NO** (z0 ≥.85 @60%): 0.823. Joint 19-layer objective costs z0 ~.05 vs
  the z0-only mini run at equal steps (0.62 vs 0.774 @3k) — shared backbone
  capacity, though the gap narrows by 9k.
- **P19 YES** (z18 ≥.50 @50%): 0.656. Render-free DEEP features are the
  best-predicted after z0 — the U-shape now mirrors r1's render-based profile
  (trough mid-stack), not r2's.

Headroom check (extension rule agreed with Joshua at ~11:00): steps 7000→8500
mean +.004, z0 +.01, at LR ≤10% of peak — still climbing under a dying LR.
Cosine-artifact precedent (plain z0 0.379@1k → 0.436@3k) says this understates
true headroom. **Extension approved criteria met.**

Extension plan (launching after stitch eval): warm restart from best.pt
(weights only, fresh AdamW), +9000 steps, peak LRs HALVED (5e-6 / 5e-4) to
soften the restart transient, and DOUBLED data — 2200 fresh seed-2 selfplay
games (games2.pt, 168k plies) concatenated; original 60-seq val split
preserved (gi offset), so step-0 R² must reproduce 0.5970 (pipeline check).

Predictions (registered before extension launch):
- **P21**: extension (r3ext, 9k more steps, 2x data) final mean val R² ≥ 0.65.
  **60%**
- **P22**: r3ext z0 ≥ 0.85 (P18 revisited with 2x budget+data). **45%**
- **P23**: r3ext z9 (trough) ≥ 0.58. **50%**

## 2026-08-12 ~12:55 — r3 stitch eval (P20 graded): render-free stitch is a
## competent-but-not-CNN-grade player

`eval_stitch_full.py` on r3 best.pt (step 9000). Agreement (304 val cuts):
top1 FLAT ~0.28-0.38 across all 19 cuts, top3 ~0.50-0.63, spearman ~0.80 flat.
Play (paired 4-ply openings): vs random 19-20/20 at every cut, illegal argmax
0/everything; vs pure CNN cut0 3/40, cut9 1/40, cut18 4/40 = **8/120 = 6.7%**.

- **P20 NO** (pooled ≥40% @40%): 6.7%. Decisive.
- Depth profile is neither r1's decline nor r2's inversion: FLAT. Render-free
  ẑ carries uniform moderate error at every depth; vs a superhuman opponent,
  ~0.80-spearman ranking loses ~always. "Beats random easily, can't touch the
  CNN" — the reconstruction is real but not decision-grade.
- Contrast anchors: r1 render stitch 94/200 (parity); r2 moves+render stitch
  44/120; r3 render-free 8/120. Each render removed costs a regime.

## 2026-08-12 ~15:50 — r3ext complete: P21-P23 graded; no third cycle

Warm restart worked exactly as designed: step-0 val R² 0.5969 reproduced the
r3 checkpoint (pipeline check passed), no restart dip, and halved peak LRs +
2x data (4400 games) lifted the whole profile ~uniformly. Final mean val R²
**0.6388** (r3: 0.5970). Per-layer: z0 .913, z1 .760, z3 .662, z9 .554
(trough), z15 .634, z18 .683.

- **P21 NO** (mean ≥.65 @60%): 0.6388. Overconfident.
- **P22 YES** (z0 ≥.85 @45%): 0.913. Joshua's cascade read (z0 improving pulls
  z1, z2, ... up) matches the trajectory: z0 gained fastest (+.09) and shallow
  layers followed (+.06), deep layers +.03.
- **P23 NO** (z9 ≥.58 @50%): 0.554. The mid-stack trough is the stubborn part.

No third cycle: agreed headroom rule (significant per-layer improvement in the
7000→8500 window) NOT met — mean +.0018/1500 steps, ~4x flatter than r3's
window, every layer ≤ +.001/500. Two 9k cycles bought +.60 then +.04;
a third projects ~+.02. Render-free containment looks asymptotic around
~.65 mean at this data/adapter/model scale, vs r1's render-based .773.

Running stitch eval on r3ext (does z0 .82→.91 move play strength at all?).
Prediction **P24**: r3ext pooled vs CNN (cuts 0/9/18, paired openings)
improves over r3's 8/120 but stays <25% — **55%**.

## 2026-08-12 ~16:20 — r3ext stitch eval: P24 YES (16/120); r3 program closed

r3ext stitch (`eval_stitch_full.py --ckpt .../r3ext/best.pt`): agreement top1
.48/.30/.30 at cuts 0/9/18 (r3: .39/.32/.28), spearman .88 shallow → .82 deep.
Play vs random 54/60; vs pure CNN cut0 4/40, cut9 5/40, cut18 7/40 =
**16/120 = 13.3%** (r3: 8/120).

- **P24 YES** (improves over 8/120 but <25%, @55%): 13.3%.
- Read: the extension's R² gains were shallow-concentrated (z0 +.09) and the
  play gains are correspondingly modest. Render-free stitch is a real player
  (beats random ~90%) but ~1.5 OOM in win-odds below the render-based r1
  stitch (94/200). The render isn't just convenience — reconstructing the
  full board from the move list burns model capacity that the CNN-tail
  computation then can't use.

Arm F r3 program CLOSED. Ladder (pooled vs CNN, paired openings):
r1 render 47% ≈ parity | r2 moves+1 render 36.7% | r3 render-free 6.7% |
r3ext (2x steps+data) 13.3%. Mean val R²: .773 | .61 (streams) | .597 | .639.

## 2026-08-12 (morning) — attention anatomy of the r3ext model: P25–P27 registered

Joshua asks: how important is attention at each layer of the trained r3ext
transformer, and what does each layer's attention attend to? His prediction is
committed as sha256 cba53da7c3ff5801958dd331082bd2dcbc7c0042c6f39f035182f19dcc2cf926
(to be revealed after results).

Design (`armF/attn_analysis.py`, r3ext best.pt, training val set of 60 seqs):
- Per-layer ablation, two variants: **zero** (attn sublayer output → 0) and
  **selfonly** (output → o_proj(v_self): per-token write survives, cross-token
  mixing dies). Metric: mean val R² damage vs baseline (0.6388 expected).
- Patterns: eager attention on 8 val seqs; from color-token queries, mass by
  key class (sink/preamble/self/newline/number/cell/color), by record recency
  (own/prev/older bins), and by hex-adjacency of attended move's cell to the
  query move's cell.

My predictions (registered before running):
- **P25 (65%)**: importance is front-loaded — the single most damaging zero
  ablation is in layers 0–4 (pre-first-readout), and mean zero-damage over
  layers 0–4 > 3× mean over layers 12–22. Rationale: board assembly must be
  ~complete by hs[5] (z0 R²=.91 read there); deep layers then run the
  CNN-tail-like compute largely per-token.
- **P26 (55%)**: no single deep layer's mixing is critical — every selfonly
  ablation at layers 12–22 costs ≤ 0.02 mean R² (redundancy across deep
  layers even if incremental copying exists).
- **P27a (70%)**: recency structure — per-token mass on the *previous* move
  record exceeds the average older record in ≥16/23 layers (incremental
  state-passing: prefix t = prefix t−1 + one stone).
- **P27b (45%)**: spatial structure — records whose cell is hex-adjacent to
  the query move's cell get ≥1.5× the per-record mass of non-adjacent records
  in a majority of layers. (Genuinely unsure: spatial neighborhood compute
  may live in MLPs on the residual, not in attention.)

## 2026-08-12 ~16:45 — Arm F finger D registered: can ONE transformer MLP block emulate ONE conv skip layer?

Joshua's side question: z2 -> Linear(7744->2048) -> Qwen3-style MLP block
(RMSNorm + SwiGLU 2048->6144->2048 + residual, no attention — single token, so
attention is a no-op) -> Linear(2048->7744) -> target z3. All params trained
(random init), MSE in per-dim-normalized space, data = 76,210 dedup selfplay
positions (armF/data/positions.pt), z from 11_2w4_2000 CNN.

Key deconfusion hazard flagged before building: the target map
z3 = swish(z2 + BN(conv(z2))) is affine-plus-one-swish, so a LINEAR map may
already score high. Controls: (a) identity baseline (per-dim scalar OLS,
closed form), (b) lin2048 = D·U no MLP (rank-2048 linear, equal budget),
(c) linfull = unconstrained Linear 7744->7744 (linear ceiling, no rank
constraint). The interesting number is MLP minus lin2048, not MLP alone.
Functional eval: stitch emulator in place of skiplayers[2], top1/spearman
agreement + paired-opening play vs pure CNN (same harness conventions as r1-r3).

Predictions (before any training):
- PD1 identity baseline per-dim R2 >= 0.60: 55%
- PD2 lin2048 val R2 >= 0.95: 45% (>= 0.90: 70%)
- PD3a mlp val R2 >= 0.99 ("identical output" operationalized): 35%
- PD3b mlp beats lin2048 by >= 0.02 R2 at equal steps: 75%
- PD4a mlp-stitched CNN vs pure CNN >= 40% paired-opening wins: 70%
- PD4b mlp-stitched top1 agreement >= 0.90: 50%

## 2026-08-12 — attn anatomy interim: selfonly confounded; P28 registered

Zero + selfonly sweeps done (details in next entry). Realization: BOTH remove
the sink bias — deep layers park 60–97% of attention on token 0, so the attn
sublayer output there is ≈ o_proj(v_sink), a learned bias; zero deletes it,
selfonly replaces it with an OOD o_proj(v_self) injection. Neither cleanly
isolates *content mixing*. Adding a third variant: **sinkself** = attention
mask restricted to key 0 + self (bias-delivery preserved, cross-token content
mixing killed).

**P28 (60%)**: under sinkself, every single-layer ablation at layers 12–22
costs ≤ 0.05 mean val R² — i.e. once the sink bias is preserved, no deep
layer's content mixing is individually critical. (Registered before running.)

## 2026-08-12 — attn anatomy results: P25 YES, P26/P27b/P28 NO, P27a YES; Joshua's prediction graded

Joshua's committed prediction (hash verified): "Attention is influential up to
the hs[5] input (so layers 0-4 attn), and then attends to bos or other constant
tokens at further layers." **Half right — the pattern half. The influence half
is falsified.**

Patterns (8 val seqs, eager, color-token queries; `attn_patterns.json`):
- L0–L2 = gather layers: broad content-directed mass over the move list
  (cell .20–.36, num up to .24, sink ~.00). At L0, 54% of mass on the query's
  OWN record (local binding of "n. cell color").
- From L3 on the sink takes over (L3 .71, mid-stack .32–.66, L20–22 .95–.97).
  Joshua's pattern claim correct (transition at L3, not L5).
- BUT the non-sink remainder is structured, not constant: previous-record mass
  exceeds the average older record's per-record mass in **23/23 layers**
  (ratios 3–30x; dedicated heads L5H10 .68, L9H0 .44, L12H1/H9 ~.22) —
  incremental state-passing, prefix t = prefix t−1 + one stone. **P27a YES
  (70%)**. Hex-adjacent past moves get more per-record mass than non-adjacent
  in 23/23 layers but only ~1.01–1.5x, never ≥1.5x: **P27b NO (45%)** by the
  letter, direction universal.

Ablations (60-seq val, baseline reproduces ckpt at .6387; `attn_ablate.json`,
`attn_ablate_sinkself.json`):
- zero: L0 catastrophic (−52); L1–L4 −.31…−.70; L5–L16 −.04…−.13; second
  catastrophic band L17–L20 (z18 → −89 at L17, −34 at L19). **P25 YES (65%)**
  (most damaging in 0–4; mean damage 0–4 ≈ 10.8 > 3× mean 12–22 ≈ 0.78).
- selfonly: uniformly worse, deep catastrophic (L18 −15.4 mean). **P26 NO
  (55%)**. Confound diagnosed: deep attn output ≈ o_proj(v_sink) = a learned
  per-layer bias; zero deletes it, selfonly swaps in OOD o_proj(v_self).
- sinkself (mask → keys {0, self}; bias preserved, content mixing killed) is
  the clean test: L0–4 still −.54…−1.17 (assembly is real mixing); L5–L13
  −.10…−.27 each (recency channel does distributed real work; L12: z9
  .55→.22); L14–L17 severe for deep maps (L16 −10.9 mean, z18 −78; L15 −.94,
  L17 −2.5); L18–L22 ≈ free (−.0003…−.20, L20/21/22 ≤ .008). **P28 NO
  (60%)** — deep mixing is individually important through L17.
- The zero-catastrophes at L17/L19 mostly vanish under sinkself → they were
  bias-removal artifacts; the L15–L17 sinkself damage is genuine mixing.

Story: three-phase program. (1) L0–L2 assemble the board from the move list
by broad attention; (2) L3–L17 park most mass on the sink (attention as
per-layer bias) while a persistent prev-record channel incrementally passes
board state forward — functionally load-bearing, not vestigial; (3) L18–L22
mixing is dispensable. Attention "importance" and "where it looks" dissociate:
sink-dominated layers still carry critical mixing in their small non-sink
remainder. Method lesson: zero-ablation of attention is confounded wherever
sinks dominate — use sink-preserving masks.

## 2026-08-12 ~17:10 — finger D main run graded (PD1-PD4); bypass decomposition launched

Results (8000 steps each, 72k train / 4k val, val R2 per-dim mean):
identity 0.205 | lin2048 0.921 | mlp 0.931 | linfull 0.957.
Agreement top1 .784/.803/.832; play vs pure CNN 28/60, 26/60, 27/60 — ALL at
~parity (43-47%) regardless of R2; single-layer replacement is forgiving.

- PD1 NO (identity >=.60 @55%): 0.205. One skip layer transforms per-dim
  content far more than residual-stream intuition suggested.
- PD2 NO (lin2048 >=.95 @45%): 0.921 (the >=.90 sub-line at 70% was YES).
- PD3a NO (mlp >=.99 @35%): 0.931 — not "identical output".
- PD3b NO (mlp >= lin2048+.02 @75%): +0.010 only. AND linfull (pure linear,
  no bottleneck) BEATS the MLP: the rank-2048 bottleneck costs -.036, the
  SwiGLU nonlinearity recovers only +.010. The transformer MLP block does
  NOT reproduce the conv layer; a plain unconstrained linear map is closer.
- PD4a YES (mlp stitch >=40% @70%): 26/60 = 43.3%. PD4b NO (top1 .803 < .90).

Reframe (per rule): main run confounds "relearn linear part through
bottleneck" with "do the nonlinear part". Bypass run launched: full-rank
Linear(7744->7744) + parallel bottleneck MLP path, all trained.
- PD5: linfull_mlp closes >=50% of linfull's residual gap (R2 >= 0.9786): 40%

## 2026-08-12 ~17:25 — finger D bypass graded (PD5 NO by a hair); finger D closed

linfull_mlp (full-rank Linear bypass + parallel D->QwenMLP->U): val R2 0.9751,
top1 .864, vs pure CNN 32/60 (53%) — best emulator on every metric. Closed 42%
of linfull's residual gap (threshold 50%) -> PD5 NO @40%.

Finger D ladder: identity .205 | lin2048 .921 | mlp .931 | linfull .957 |
linfull+mlp .975. Reads:
1. One conv skip layer is NOT per-dim residual passthrough (identity .205)
   but IS ~96% linear (linfull .957).
2. A pure project->MLP-block->project emulation (Joshua's original spec) loses
   to plain linear: rank-2048 skip costs -.036, SwiGLU buys back +.010.
3. "MLP + skip" is the right frame ONLY if the skip is native-width: then the
   MLP's contribution is purely nonlinear (+.018, 42% of the nonlinear residue).
   The binding constraint was the bottleneck, not MLP capacity — rhymes with
   r3's render-reconstruction tax.
4. Play vs pure CNN is insensitive across R2 .92-.975 (43-53%): one-layer
   substitution at these fidelities is already ~parity.
Artifacts: armF/fingerD_convmlp.py, armF/results/fingerD{,_bypass}.{json,log},
checkpoints/armF_fingerD/. Finger D CLOSED.

## 2026-08-12 ~17:15 — finger E opened: rank-1024 HexHex (compressed-state distillation); PE1-PE5 registered

Joshua's question: can HexHex be distilled to a rank-1024 state (fits in half
of Qwen's 2048 residual) "by doing awful things with superposition — 121
almost-orthogonal rank-64 subspaces of a rank-1024 state"? Conceptual answer
given before any run:
- The hand-designed near-orthogonal-subspace version FAILS on dense features:
  reading one cell through its projection pulls crosstalk from 120
  simultaneous interferers, noise/signal ~ 120*64/1024 ~ 7.5 (SNR 0.13).
  Superposition buys capacity under sparsity; a board state is maximally dense.
- What saves you is cross-cell CORRELATION (global low-rank), not sparsity.
  New measurement (rank_check_1024.json, 10k positions): a single global
  rank-1024 basis captures 87-94% of variance at every layer (trough z9
  .872, z1 .959, z18 .940). Near-orthogonality is the wrong desideratum —
  it fights the correlations that make 1024 enough. Frame: learned global
  compression, not 121 private slots.
- Design consequences: (a) conv weight tying is lost in a compressed basis
  (fine, fixed 11x11 board, ~2M params/layer); (b) the real risk is
  COMPOUNDING across 19 compression points, not per-layer fidelity (finger D:
  one rank-2048 transition bottleneck cost -.036; r2/r3: shallow errors
  amplify through the trunk).

Cheapest falsifier (fingerE_pca.py): chain frozen per-layer PCA
project+reconstruct through the whole trunk (compress at every z_l, real conv
layers in between), agreement + paired-opening play vs pure CNN, rank sweep
{512, 1024, 2048} + single-layer-z9-only control (isolates compounding).
Decision rule: chained-1024 >=40% play parity -> frozen compressed state
suffices, distillation phase is about transitions; <20% -> joint training of
encoders/decoders/transitions must recover compounding.

Predictions (registered before running):
- PE1 (40%): chained k=1024 top1 agreement vs pure CNN >= 0.60.
- PE2 (40%): chained k=1024 play vs pure CNN >= 40% (>=24/60 paired games).
- PE3 (60%): chained k=2048 play >= 45% (>=27/60).
- PE4 (50%): compounding is dominant: single-z9-only k=1024 top1 exceeds
  chained k=1024 top1 by >= 0.15.
- PE5 (85%): chained k=1024 still beats random >= 18/20.

## 2026-08-12 ~17:40 — finger E falsifier graded: PE1-PE3 NO, PE4-PE5 YES; frozen PCA insufficient, compounding dominant

Chained frozen per-layer PCA (fingerE_pca.py, basis 10k train positions, 60
paired games/variant):
  chained k=512:  z18 R2 .474 | top1 .318 | vs CNN 11/60 | vs rand 19/20
  chained k=1024: z18 R2 .599 | top1 .424 | vs CNN  4/60 | vs rand 20/20
  chained k=2048: z18 R2 .721 | top1 .525 | vs CNN  4/60 | vs rand 19/20
  single z9 k=1024: z18 R2 .900 | top1 .729 | vs CNN 27/60 | vs rand 20/20
- PE1 NO (.424 < .60 @40%). PE2 NO (6.7% << 40% @40%). PE3 NO (@60%) — the
  genuine surprise: even rank-2048 chained is a broken player at 4/60, so
  finger D's "one-layer substitution is forgiving" does NOT extend to 19
  simultaneous mild perturbations; play sensitivity is a compounding
  phenomenon, not a fidelity threshold. (k=512's 11/60 vs k=1024/2048's 4/60
  is nonmonotone — treat 4-11/60 as one "broken" band, paired-opening noise.)
- PE4 YES (.729 - .424 = .305 >= .15 @50%): single-point rank-1024
  compression at the variance trough is nearly free (45% parity, finger-D
  band); chaining is what kills. PE5 YES (@85%).
- Immediate-vs-chained: per-layer basis keeps 87-94% var, but chained R2
  decays monotonically to ~.60 (k=1024) by z18 — each conv layer amplifies
  the previous truncation error (same mechanism as r2/r3 shallow-error
  amplification, now measured cleanly per-rank).
- Decision rule (pre-registered): <20% parity -> frozen compressed state does
  NOT suffice; finger E phase 2 = jointly trained compression (encoders/
  decoders/transitions), where the student can route around its own
  compression errors rather than eat them layer-by-layer.
Artifacts: armF/fingerE_pca.py, armF/results/fingerE_pca.{json,log},
rank_check_1024.json (var@{512,1024,1536,2048} per layer).

## 2026-08-12 ~17:50 — finger E phase 2 registered: jointly trained rank-1024 student (anchored + policy-only)

Design (fingerE_distill.py): student = E_in Linear(338->1024) from flat board
(z0 is linear in the board and rank<=226, so no information bottleneck at
entry) -> 18 transition blocks T_l(h) = Linear(1024->1024) + QwenMLP-delta
(1024->3072->1024, native-width-skip per finger D) -> policy.
Two variants, equal budget (8000 steps, batch 512, on-the-fly CNN teacher):
- anchored: 19 decoders D_l Linear(1024->7744); loss = mean_l per-dim-norm
  MSE(D_l(h_l), z_l) + KL(teacher || policyhead(D_18(h_18))), frozen CNN
  policy head. Stays affinely mapped to HexHex's activations = embeddable
  in Qwen the arm-F way.
- policy_only: trained Linear(1024->121) head, KL only. Existence proof for
  "a rank-1024 agent", correspondence severed.
The comparison prices the representational rent of anchoring at rank 1024.

Predictions (registered before running):
- PE6 (60%): anchored student mean decoded val R2 (over 19 layers) >= 0.80
  (frozen chained PCA managed 0.70 at k=1024; joint training routes around).
- PE7 (55%): anchored student play vs pure CNN >= 40% (>=24/60). The crux —
  chained k=2048 PCA had mean R2 .80 and still played 4/60, but it had no
  policy loss and adversarially compounded truncation errors.
- PE8 (75%): policy-only student play vs pure CNN >= 40%.
- PE9 (55%): representational rent is small: (policy_only wins - anchored
  wins) <= 10 games out of 60.
