# C3 probe design (executed in Phase 4, designed early)

## Activation collection
- Fixed probe-position set: ~500 positions (mix of sizes, held out from training).
- Per checkpoint: forward pass on the *prompt only* (no generation), collect residual
  stream at layers {25%, 50%, 75%, 100%} at the last prompt token (and optionally
  mean-pooled over board-render tokens).
- 1.7B = 28 layers, d=2048. 500 pos x 4 layers x 2048 floats = tiny. Store fp16 npz.

## Labels (programmatic, from board state — theory-neutral)
Validation tier (features that DEFINITELY exist, per Karvonen chess-GPT method):
- stone color at queried cells; side to move; stone count.
Concept tier:
- bridge present for mover (pattern: two same-color stones at hex-distance-2 sharing
  exactly two common neighbors, both empty); count of bridges.
- mover has winning position (solver label — the ultimate "evaluation" concept).
- connectivity: largest same-color group span (rows for B / cols for W).
- edge-template-2 present (2nd-row stone with both downward carrier cells empty).
- ladder-ish: TBD; hard to label statically; may drop (log if dropped).
Probe = logistic regression per (layer, concept) on frozen activations; train/test split
over positions; AUC reported. Baseline control: probe on shuffled labels.

## Verbalization side of the 2x2
- Use each checkpoint's *sampled CoTs* on the same positions (from eval_checkpoint runs).
- Grader prompt: "does this CoT explicitly discuss <concept>?" (definition given),
  grader = biggest available model; hand-spot-check 20 per cell.
- 2x2 per (checkpoint, concept): probe-detectable x CoT-verbalized.

## Registered prediction (from agenda): probe-yes/verbalize-no dominates (65%).
Surprise condition: verbalization precedes probe detectability at earlier checkpoints.

## CoT-swap / load-bearing tests (C3 support)
- think/no_think gap per checkpoint (already in eval flow).
- CoT-swap: feed position A's prompt + position B's CoT prefix (or shuffled own CoT),
  force-continue after </think>, measure move-quality delta.
- Truncated-CoT: cut CoT at 25/50/75%, close think tag, measure degradation curve.
