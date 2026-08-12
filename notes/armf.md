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

## r2: moves format (denser supervision) — findings

Sequence = preamble + move list (cut at random ply) + ONE render; 3 streams
(`train_moves.py`): CNN column of the played cell at each move token, whitened
PCA-128 of the full map at each move token, full per-cell map at render cells.
9000 steps: col .623 / pca .482 / render .610 (all below r1's .773; P6-P8 NO).

6. **Move tokens beat render cells** (col > render) despite being causally
   BLIND to the render (they precede it) — Qwen simulates board state
   internally from the move list. This de-risked render-free r3.
7. **Moves-format stitch is worse and depth-INVERTED** (`eval_stitch_moves.py`,
   P9 NO): vs CNN cut0 7/40, cut9 16/40, cut18 21/40 (parity!). Shallow z-hat
   errors amplify through the remaining CNN tail; at k=18 errors hit only the
   policy head.
8. **One token ≠ whole board** (`train_movesonly_z0.py`): render-free full z0
   map from each move token caps at R² .379 (own-cell column: .786; 121
   render tokens: .760). Depth sweep (`train_movesonly_sweep.py`, adapters at
   hs[1..23]): FLAT ~.38 plateau — no depth assembles the board; NOT an
   aggregation-depth problem.
9. **Parity hypothesis (Joshua's)**: stone color = list-position parity, and
   the canonical frame flips with total-count parity; LLMs are bad at parity.
   Numbered+colored records ("\n1. g1 X") @1000 steps: .446 vs plain .379,
   still climbing when plain had plateaued (P13/P14 NO by the letter,
   direction confirmed). 3000-step A/B (P15/P16) pending → decides r3 format.

## r3: render-free full stack (numbered format) — findings

Render-free, per-layer Linear(2048→7744) full-map adapters at move tokens
(~301M params), numbered format, no cuts — every move token supervises its
own prefix (`train_movesfull.py`). r3 = 9000 steps; r3ext = warm restart from
best.pt (weights only, fresh AdamW, halved peak LRs 5e-6/5e-4) + 2200 fresh
seed-2 games (`gen_games.py --seed 2`, gi offset preserves val split — step-0
R² must reproduce the ckpt).

10. **R²: r3 0.597 → r3ext 0.639** (z0 .82→.91, trough z9 .52→.55, z18
    .66→.68). U-shape like r1. P17 NO by .003, P18 NO/P22 YES (z0 .91),
    P19 YES (z18 .66), P21 NO, P23 NO. Cascade dynamics (Joshua's read):
    z0 gains fastest, shallow follows, deep lags. Asymptotic ~.65 at this
    scale — headroom rule (7000→8500 window) failed for a third cycle.
11. **Render-free stitch plays, but far below CNN** (`eval_stitch_full.py`,
    P20 NO, P24 YES): agreement FLAT over depth (spearman ~.80-.88); beats
    random ~90%; vs pure CNN r3 8/120 = 6.7%, r3ext 16/120 = 13.3%.
    Ladder: r1 render 47% (parity) | r2 moves+render 36.7% | r3 render-free
    6.7% | r3ext 13.3%. Each render removed costs a regime — reconstructing
    the board from moves burns capacity the CNN-tail computation needs.

12. **Attention anatomy of r3ext** (`attn_analysis.py`): three-phase program.
    L0–L2 gather broadly over the move list (sink ~0); L3+ park 32–97% of
    mass on the token-0 sink, BUT the non-sink remainder is structured: the
    previous move record out-attends an average older record in 23/23 layers
    (3–30x per-record; heads L5H10 .68 mass) — incremental state-passing —
    and hex-adjacent past moves get ~1.2x mass in 23/23 layers. Sink-clean
    ablation (mask → keys {0,self}) shows mixing is critical L0–4
    (−.5…−1.2), real L5–L13 (−.10…−.27 each), severe L14–L17 for deep maps
    (L16 −10.9), free L18–L22. **Zero-ablating attention is confounded
    wherever sinks dominate** — deep attn output ≈ o_proj(v_sink) is a
    learned bias; zeroing it (or selfonly) wrecks layers for the wrong
    reason. Use sink-preserving masks.

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
