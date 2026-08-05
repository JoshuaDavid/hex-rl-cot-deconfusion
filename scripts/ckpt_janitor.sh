#!/usr/bin/env bash
# Merge each finished verl checkpoint to HF format, then prune shard/optimizer
# bulk, keeping the newest KEEP_FULL full checkpoints for resume.
# Usage: bash scripts/ckpt_janitor.sh <exp_name> [keep_full]
set -uo pipefail
EXP=$1
KEEP_FULL=${2:-2}
ROOT=/workspace/hex-rl-cot-deconfusion
DIR=$ROOT/checkpoints/$EXP

steps=$(ls -d "$DIR"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n)
[ -z "$steps" ] && exit 0
newest=$(echo "$steps" | tail -n "$KEEP_FULL")

for s in $steps; do
  ck=$DIR/global_step_$s
  # merge if not yet merged and actor shards exist and checkpoint is complete
  if [ -d "$ck/actor" ] && [ ! -d "$ck/hf" ]; then
    # completeness heuristic: data.pt written last by verl
    [ -f "$ck/data.pt" ] || continue
    echo "merging step $s"
    /venv/verl/bin/python -m verl.model_merger merge --backend fsdp \
      --local_dir "$ck/actor" --target_dir "$ck/hf" || continue
  fi
  # prune actor bulk unless among newest full ckpts
  if [ -d "$ck/actor" ] && [ -d "$ck/hf" ] && ! echo "$newest" | grep -qx "$s"; then
    echo "pruning actor shards for step $s"
    rm -rf "$ck/actor"
  fi
done
