#!/usr/bin/env bash
# Merge a verl FSDP checkpoint to HF format and run the hex checkpoint eval.
# Usage: bash scripts/merge_and_eval.sh <exp_name> <step> [<k> <limit>]
set -euo pipefail
EXP=$1
STEP=$2
K=${3:-4}
LIMIT=${4:-150}
ROOT=/workspace/hex-rl-cot-deconfusion
CKPT=$ROOT/checkpoints/$EXP/global_step_${STEP}
HF_OUT=$CKPT/hf
if [ ! -d "$HF_OUT" ]; then
  /venv/verl/bin/python -m verl.model_merger merge --backend fsdp \
    --local_dir "$CKPT/actor" --target_dir "$HF_OUT"
fi
mkdir -p $ROOT/results/checkpoints
/venv/verl/bin/python $ROOT/scripts/eval_checkpoint.py \
  --model "$HF_OUT" \
  --corpus $ROOT/data/verl_hex/val_positions.jsonl \
  --out $ROOT/results/checkpoints/${EXP}_step${STEP}.jsonl \
  --k "$K" --limit "$LIMIT" --gpu-mem 0.75
