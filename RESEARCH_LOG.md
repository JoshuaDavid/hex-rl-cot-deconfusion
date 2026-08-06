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
