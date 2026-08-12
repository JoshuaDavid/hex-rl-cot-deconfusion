# Arm F — affine containment of a superhuman CNN inside Qwen's residual stream

Question (Joshua's design, "extremely unhinged"): can Qwen3-1.7B + 19 per-layer
affine adapters be jointly fine-tuned so its residual stream *contains* the
activations of the HexHex CNN (18-layer 64ch superhuman 11x11 Hex player),
depth-aligned Qwen layers 5–23 ↔ HexHex layers 0–18? LM preservation NOT
required. A stress test of the universal representation hypothesis with a
causal (stitching) standard, not just probes.

## Answer (one line)

Yes by the causal standard: after joint FT, stitched Qwen→A_k→CNN-tail plays at
**parity with the pure CNN at every cut, including k=18** (Qwen does the entire
trunk), while the frozen-probe stitch is a broken player — and Qwen's language
survives almost intact despite no LM loss.

## Setup (everything in `armF/`)

- Teacher: HexHex `11_2w4_2000.pt` (github.com/harbecke/HexHex), loaded via
  `hexhex_wrap.py`. 19 capture points: z_0 = post-initial-conv, z_k = after
  skiplayer k−1, each (B,64,11,11). Activations from `internal_model` (no
  rotation averaging inside the trunk). `stitched_logits(cnn, z, k)` resumes at
  `skiplayers[k:]` + policy head — verified exact at k=0.
- Board→text: one Qwen token per cell (" X"/" O"/" ."), canonical to-move
  perspective so text == CNN input exactly (`render11.py`).
- **Two-copy render** (the key design catch): causal attention means a copy-1
  cell token cannot see rows below it, so even z0 is uncomputable there.
  Readout at copy-2 cell tokens, which see the full board in copy 1.
- Alignment: z_l ↔ `hidden_states[5+l]` (HF: entry j = input to block j).
  Backbone truncated to 23 blocks with `norm→Identity` so the deepest read is
  raw out22 (HF otherwise final-norms the last hidden_states entry).
- Data: 76210 dedup'd positions (selfplay temp-schedule + random),
  `gen_positions.py`; pretokenized cache `pretokenize.py` (armF/data/ is
  gitignored — tokens.pt is 114MB).
- Training (`train_containment.py`): loss = mean over 19 layers of MSE(A_l h,
  z_l normalized per-layer-per-channel); backbone lr 1e-5 (embeddings frozen),
  adapters lr 1e-3 warm-started from the ridge probe solution (step-0 val R²
  must reproduce the probe — built-in pipeline check); batch 16, 9000 steps,
  bf16 autocast + fp32 weights, grad ckpt, ~0.62 s/step, ~1.6h on A100-40GB.

## Validation anchors (why we trust the numbers)

- Patch-affine probe (shared-weight, F.unfold r=1 = conv geometry) hits
  **R²=1.000 on z0** — cell/target alignment provably correct.
- Warm-started adapters reproduce frozen-probe R² at step 0 exactly.
- Play evals vs deterministic opponents need **shared random 4-ply openings**
  (paired per game pair) — otherwise alternating colors gives only 2 distinct
  games repeated (0/10/20 artifacts).

## Findings

1. **Naive probing overstates alignment.** Frozen pretrained-Qwen probe R² is
   U-shaped (.57 z0, .28 z9, .37 z18) — but a random-init Qwen control gets
   ~the same deep (.354 vs .370 at z18). Deep probe R² ≈ random-feature
   reservoir capacity. Pretrained advantage only early/local (z0 +.13).
2. **Joint FT: mean val R² 0.773** (z0 .991, trough .70 at z9–z11, z18 .763).
   Didn't cross the predicted 0.8-everywhere bar.
3. **Stitching (headline):** top1 move-match declines smoothly .89 (cut 0) →
   .45 (cut 18); top3 .998→.71; spearman .98→.88. But **play strength doesn't
   decline**: vs pure CNN with paired 4-ply openings, 94/200 = 47% overall
   (cut 18: 18/40); vs random 19-20/20 everywhere. Frozen-probe stitch: top1
   .20→.08, 4/200 vs CNN, and at deep cuts even loses to random (7/20, k=18).
   **Move-match is the wrong metric** — disagreements are among equivalued
   moves; the decision-relevant structure is preserved even where per-channel
   R² looks mediocre.
4. **Language survives incidentally:** splicing FT blocks 0..22 back into full
   Qwen (`eval_language.py`): NLL 2.045→2.197 (ppl 7.7→9.0), coherent greedy
   generations. Both computations coexist in the 2048-dim stream.
5. Predictions: P1 deep-half wrong, P2 missed, P3 NO (.452 < .60), P4 NO by
   exactly one game. P5 (randinit-init FT control loses at equal steps, 70%)
   — see RESEARCH_LOG for grading.

## Gotchas for reruns

- Randinit control MUST warm-start adapters from `probe_frozen_randinit.pt`
  (`--probe`), else step 0 is R² ≈ −800 and the comparison is unfair. Seeds
  match across `qwen_embed.load_qwen(random_init=True)` and
  `train_containment.load_backbone(random_init=True)` (both manual_seed(0);
  verified via step-0 R² reproduction).
- Eval OOM: hidden_states are fp32 under autocast; use_cache=False and
  batch ≤32.
- Checkpoints: `checkpoints/armF_containment_r1/best.pt` (bf16 backbone +
  adapters + step, no optimizer). Load with `load_backbone(random_init=True)`
  as arch shell then overwrite (see `eval_stitch.load_trained`).

## Repro

```
python armF/gen_positions.py && python armF/pretokenize.py
python armF/probe_frozen.py            # + --random-init variant
python armF/train_containment.py --run-name armF_containment_r1
python armF/eval_stitch.py             # + --ckpt FROZEN baseline
python armF/eval_language.py
```
