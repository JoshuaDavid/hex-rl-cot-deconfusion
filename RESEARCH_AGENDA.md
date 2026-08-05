# RL Theory-Invention in Hex: A Deconfusion Agenda

**Status**: agenda drafted, nothing run yet. **Budget**: 100 A100-hours. **Purpose**: personal deconfusion, not publication. The deliverable is "I am less confused about what RL does to a model," with evidence I'd bet on.

## What I actually want to know

Four confusions, in priority order:

- **C1 — Creation vs. selection.** When RL improves a model at a task, does it bake in *new* cognitive paths, or does it make the smallest possible policy update — sharpening/reweighting behaviors the base model already had in its sampling distribution?
- **C2 — Vocabulary emergence.** Does new vocabulary/notation emerge in the CoT under RL? If it does: can *earlier checkpoints* (or the base model) understand that vocab — functionally (does it help them play?) and declaratively (can they define it, and does the definition predict usage)?
- **C3 — Concept/verbalization coupling.** When an internal concept forms (probe-detectable in the residual stream), does the CoT verbalize it? Is the CoT load-bearing, decorative, or actively decoupled from the computation?
- **C4 — Cheap preservation.** Does a halfhearted fixed-β KL-to-reference term preserve off-domain behavior? Does CoT *style* drift leak off-domain even when capabilities are preserved?

Secondary trophy, not the point: provably-near-optimal play on solved boards, and zero-shot board-size transfer.

## Why hex

- **Uncontaminated.** Qwen3 has ~no hex theory from pretraining (verify: Phase 1 quiz). Whatever theory appears was invented here, and whatever vocabulary appears is not parroting. Chess cannot give this.
- **Perfect ground truth.** 7×7, 8×8, 9×9 hex are fully solved. A DFPN solver (benzene) is a theory-neutral verifier: it rewards the game's mathematics, not another agent's concepts. Exact move-optimality metrics, no Elo proxies.
- **Theory is size-invariant and local** (bridges, ladders, edge templates). Memorized solutions are not. So: train on 5×5–7×7, evaluate zero-shot on 8×8/9×9 *with solver-exact scoring*. Transfer ⇒ invented theory; collapse ⇒ memorization. This is the discriminator.
- **Cheap.** ≤49-cell boards, short games, dense solver reward. TinyZero-scale discovery (~tens of GPU-hours), not DeepScaleR-scale (~thousands).
- **Probe targets exist.** Hex concept probes were built for AlphaZero ([arXiv:2211.14673](https://arxiv.org/abs/2211.14673)); adapt to a transformer residual stream.

## Setup (fixed unless a gate falsifies it)

- **Model**: Qwen3-1.7B, thinking mode. (Pilot plumbing on 0.6B. Escalate to 4B only if 1.7B fails board comprehension.)
- **Harness**: verl, GRPO, full board state in the prompt each turn as ASCII (never move-history-only — don't spend model capacity on being a world model).
- **Reward**: solver-based per-move (win-preserving = +, win-throwing = −) on ≤7×7. Win/loss terminal on anything the solver can't do in-loop.
- **Openings**: swap rule or solver-balanced starts (first player provably wins; naive self-play signal degenerates as play improves).
- **Preservation arm**: fixed small β, KL(π_θ‖π_ref) on ~30k self-distilled replay pairs (Tülu-3 prompts, base-Qwen3-1.7B responses). Deliberately *not* RECAP — the halfheartedness is the experimental condition. β=0 control run.
- **Checkpoints every ~100 steps.** The checkpoint sequence is the dataset for C1–C4. Do not skimp on this.

## The four confusions, operationalized

Each: boldest compact hypothesis stated *before* data, then falsifiers with costs.

### C1: creation vs. selection

**Boldest hypothesis**: RL is pure selection. Every move the RL'd model plays at temperature ~0 lies inside the base model's pass@k envelope for feasible k (≤1024). No new paths, only sharpened sampling.

Falsifiers:
- **pass@k envelope test** (method: [Yue et al., arXiv:2504.13837](https://arxiv.org/abs/2504.13837)): on a fixed set of solver-scored positions, compare base-model pass@k (k ∈ {8, 64, 256, 1024}) against RL'd pass@1. If the RL'd model reliably finds solver-approved moves the base model essentially never samples → selection hypothesis dies. *Cost: inference only, ~1–2 A100-hr given checkpoints.*
- **Probe delta**: concepts probe-detectable in late checkpoints but absent in base activations under any prompting. Creation evidence, weaker (probe sensitivity confound). *Cost: cheap given probe infra.*
- Caveat to log now: temperature/sharpening partially confounds pass@k; report the whole k-curve, not one number.

### C2: vocabulary emergence + backward intelligibility

**Boldest hypothesis**: no new vocabulary emerges — the KL anchor plus small budget means CoT reweights existing phrases only.

Detection: n-gram frequency ratios (late CoT vs. base CoT on identical positions); recurring spans with ref-perplexity spikes. *Cost: pure analysis on stored rollouts.*

If vocab is found, backward-intelligibility falsifiers:
- **Functional**: give the base model a late checkpoint's CoT as advice-in-context; measure move-quality delta vs. (a) paraphrased-into-plain-language control, (b) shuffled-CoT control. If raw ≫ shuffled but ≈ paraphrase, the vocab is compressive but translatable; if raw ≫ paraphrase, the vocab carries content that doesn't survive translation — the interesting case. *Cost: minutes.*
- **Declarative**: ask base model to define the term; check whether its definition predicts the term's usage contexts in late CoTs. *Cost: minutes.*

### C3: concept vs. verbalization (legibility)

Per checkpoint, build the 2×2: {probe detects concept} × {CoT verbalizes it}. Concepts: bridge/two-bridge safety, ladders, edge templates (adapted from 2211.14673). Verbalization: grader LLM + spot-check by hand.

**Registered prediction**: probe-yes/verbalize-no is the dominant cell — concepts form silently and CoT narrates post-hoc. If verbalization *precedes* probe detection at earlier layers/checkpoints, that's the surprising outcome; stop and look hard.

Supporting metrics per checkpoint: CoT ref-perplexity (drift), CoT-swap test (does the CoT predict the move before the move is emitted), think/no_think gap (is CoT load-bearing *at all*).

### C4: cheap preservation

Per checkpoint, both arms (β>0, β=0): trimmed MMLU/GSM8K/IFEval, KL on held-out non-hex prompts, CoT style metrics on non-hex tasks.

**Registered prediction**: at ~1k steps on this narrow a domain, forgetting is small even at β=0 — in which case the honest conclusion is "this regime doesn't stress preservation," not "KL works." Log this now so future-me doesn't overclaim. Style leakage (CoT drift on non-hex tasks) may appear even without capability loss — per [obfuscation-transfer results](https://arxiv.org/html/2601.23086v2), style generalizes.

## Protocol (the loop)

1. Look at the data / observations so far.
2. State the boldest, most compact hypothesis the observations do not exclude.
3. Design falsifiers. If a test costs <5 min GPU, just run it; otherwise find a cheaper test with the same discriminating power. **No sweeps-to-fill-time.**
4. Check, note what's confusing, iterate.

Hard rules:
- **Never launch a multi-hour run without having read ≥3 concrete training examples and ≥1 full trajectory with its evaluation verdict.** Eyes on samples, every phase.
- When an abstract question stalls, go concrete: one position, one rollout, one probe, one checkpoint pair.
- Register predictions (with rough odds) in `log.md` before each phase; grade them after.
- Subagents run in parallel with self-contained briefs: APIs, file paths, the hypothesis, the registered prediction, the report format. A subagent that doesn't know the prediction can't tell you whether you were surprised.

## Ordering: maximal information first

### Phase 0 — CPU only, today
- Benchmark benzene/DFPN latency on 7×7 midgame positions (in-loop feasibility; >1s ⇒ cache/precompute design).
- Token arithmetic: board prompt × game length × CoT × group size vs. context and throughput.
- Generate ~20 solver games; **read them**. Write the ASCII board renderer; **look at it** and confirm a human can play from it.

### Phase 1 — ~30 min A100, one batched script, untuned model
1. Legal-move rate from ASCII board, few-shot. (<90% ⇒ representation problem; fix before anything else.)
2. Board comprehension: stone lookups, adjacency, connectivity queries. **Biggest plan-killer; ~50/50 prior.**
3. Contamination quiz: bridges, templates, first-player win, ladders. Sets the C2 baseline — what vocab already exists.
4. think vs. no_think move quality vs. random-mover baseline. (CoT must be load-bearing at baseline or C3 starts confounded.)
5. GRPO group-variance check: 8 rollouts/position — is there within-group reward variance? (Zero variance ⇒ zero advantage ⇒ reward/opponent shaping needed before RL can start.)

**Read every sample this phase generates.** Gate: 1, 2, 5 pass → Phase 2. 2 fails → try richer board encodings (coordinates-per-row, cell lists, hybrid); still fails → Qwen3-4B and halve the main run.

### Phase 2 — pilot, ~15h (0.6B permitted for plumbing, 1.7B for the real gate)
- Reward slope over first 50–100 GRPO steps; throughput reality-check; degeneracy watch (format hacking, first-move-advantage collapse, mode collapse).
- Read trajectories at step 0, 50, 100. Note the first observable CoT change — that observation seeds the Phase-3 hypothesis via the loop.
- Gate: nonzero reward slope + no unpatched degenerate hack.

### Phase 3 — main runs
- ~50h: multi-size GRPO (5×5–7×7), KL arm, checkpoints every ~100 steps.
- ~15h: β=0 control, same seeds/data order.
- Standing monitor: eval mini-suite + rollout sampling per checkpoint; read a trajectory per checkpoint. If something confusing appears mid-run, the loop says: stop, hypothesize, cheap-falsify — don't wait for the run to finish.

### Phase 4 — analysis, ~15h + slack
- C1 pass@k envelope; C2 vocab mining + backward-intelligibility; C3 probes × verbalization 2×2 across checkpoints; C4 both arms.
- 8×8/9×9 zero-shot transfer with precomputed solver labels.

Budget ledger: 0.5 (Phase 1) + 15 + 50 + 15 + 15 = 95.5, ~4.5h slack. Standing rule: any test <5 min, just run it and log it; the ledger only tracks the big items.

## Registered predictions (drafted by Claude at agenda time — re-anchor to your own odds before Phase 1)

| Claim | Odds |
|---|---|
| Legal-move rate >90% few-shot (P1.1) | 75% |
| Useful board comprehension at 1.7B (P1.2) | 50% |
| Qwen3 already verbalizes bridge-like hex concepts (P1.3) | 30% |
| Reward slope positive in pilot (P2) | 60% |
| Near-optimal 7×7 by end of main run | 35% |
| Zero-shot 8×8/9×9 transfer clearly above size-matched-memorization baseline | 25% |
| C1: selection hypothesis survives pass@k test | 55% |
| C2: genuinely new vocab emerges | 20% |
| C2b: if vocab emerges, base model functionally understands it | 60% |
| C3: probe-yes/verbalize-no dominates | 65% |
| C4: measurable forgetting at β=0 within 1k steps | 40% |

## Logging

`log.md`, append-only, one entry per loop iteration: date / what I looked at / boldest surviving hypothesis / falsifier + cost / result / **what is still confusing**. The last field is the point of the document.

## Prior art (anchors, not a lit review)

- [Does RLVR incentivize reasoning beyond the base model? (pass@k method)](https://arxiv.org/abs/2504.13837) — the C1 measurement instrument.
- [Analyzing Concepts in AlphaZero in Hex](https://arxiv.org/abs/2211.14673) — hex concept probes to adapt for C3.
- [Baker et al., CoT obfuscation under monitor pressure](https://arxiv.org/abs/2503.11926) + [obfuscation transfers to unseen tasks](https://arxiv.org/html/2601.23086v2) — C3/C4 background; note our reward never touches the CoT, so drift here is *indirect* pressure only.
- [Karvonen, Chess-GPT world models](https://adamkarvonen.github.io/machine_learning/2024/01/03/chess-world-models.html) — probe methodology template; validate probes on stone-position features first (they definitely exist) before trusting them on concepts.
- [Grandmaster-Level Chess Without Search](https://arxiv.org/abs/2402.04494) — what distillation buys; deliberately *not* used here (invention purity).
- TinyZero — budget calibration: emergent behavior from GRPO on a small model in tens of GPU-hours is a thing that happens.
- [RECAP](https://arxiv.org/abs/2510.21978) — deliberately not used; fixed-β "halfhearted" KL is the condition under test.

## Failure modes and honest concerns

- **Board comprehension at 1.7B** is the single most likely killer, and the cheapest to test. It is Phase 1 item 2 for that reason.
- **pass@k conflates sharpening with creation** at the margins. Report full k-curves; treat "RL'd model inside the k=1024 envelope" as compatible-with-selection, not proof.
- **Solver reward purity**: solver labels leak "the game's mathematics" but no agent's concepts. Defensible, but per-move optimality reward does shape *which* theory is learnable (it rewards not-losing, not elegance). Log if this seems to matter.
- **Self-play nonstationarity**: opponent is the policy itself; reward curves can oscillate without meaning anything. Anchor progress to solver-optimality on a fixed position set, never to self-play winrate.
- **Single seed everywhere.** Conclusions are "what happened in this run," suggestive-not-established. Fine for deconfusion; say so when writing anything up.
- **CoT decorative from the start**: if Phase 1 item 4 shows no think/no_think gap, C3 needs reframing before the main run (e.g., "does RL *create* load-bearing CoT where none existed").
- **Nothing forgets at this scale** (C4 moot): 40% likely. That is itself an answer; don't torture the metrics to manufacture an effect.

## Review lens

- **Olah**: is the null explicit for each confusion? (Selection, no-vocab, silent-concepts, no-forgetting — all stated with odds.) Are probes validated on features that definitely exist before being trusted on concepts?
- **Radford**: what's the smallest run that tells us anything? (Phase 1 is 30 minutes and can kill the plan; the main run doesn't start until two gates pass.) Are we building tooling before the effect is real? (Probe infra waits for Phase 3 checkpoints; only the renderer and reward plumbing are built up front.)

