#!/usr/bin/env bash
# Janitor + B2 backup loops for a training run. Usage: support_loops.sh <exp>
# (Re)started after the 2026-08-06 disk crisis killed the ad-hoc versions.
set -uo pipefail
EXP=${1:?usage: support_loops.sh <exp_name>}
ROOT=/workspace/hex-rl-cot-deconfusion
B2=b2hex:claude-code-backups/hex-rl-cot-deconfusion
cd "$ROOT"

janitor() {
  while true; do
    bash scripts/ckpt_janitor.sh "$EXP" 2 >> results/janitor_${EXP}.log 2>&1
    sleep 600
  done
}

backup() {
  while true; do
    rclone copy results "$B2/results" -q
    rclone copy data "$B2/data" --exclude ".hf_home/**" -q
    rclone copy notes "$B2/notes" -q
    rclone copy checkpoints "$B2/checkpoints" \
      --include "*/hf/**" --include "*/huggingface/**" \
      --include "armF*/final.pt" --include "armF*/*.json" \
      --include "armF_fingerE/*.pt" --transfers 4 -q
    rclone copy armF/data "$B2/armF/data" --include "*.pt" -q
    sleep 7200
  done
}

janitor & backup & wait
