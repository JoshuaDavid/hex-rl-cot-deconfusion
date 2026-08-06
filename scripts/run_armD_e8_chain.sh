#!/usr/bin/env bash
# Arm D 8-epoch extension: train weighted then uniform (fresh runs), with a
# rolling exporter that converts each epoch checkpoint to a 67MB peft adapter
# and deletes the 8GB FSDP dir (disk stays flat), then epoch-curve evals.
set -uo pipefail
cd /workspace/hex-rl-cot-deconfusion
export HF_HOME=/workspace/.hf_home

run_one() {
  local exp=$1 uniform=$2
  ARMD_UNIFORM=$uniform ARMD_EXP=$exp bash scripts/run_armD_sft.sh \
    trainer.total_epochs=8 > results/armD/train_${exp}.log 2>&1 &
  local pid=$!
  local done_training=0
  while :; do
    for d in checkpoints/$exp/global_step_*; do
      [ -e "$d/data_0.pt" ] || continue
      local n=${d##*_} ep
      ep=$((n / 42))
      /venv/verl/bin/python scripts/export_armD_adapter.py "$d" \
        checkpoints/$exp/adapter_ep$ep >> results/armD/export_${exp}.log 2>&1 \
        && rm -rf "$d"
    done
    [ "$done_training" = 1 ] && break
    if ! kill -0 $pid 2>/dev/null; then done_training=1; sleep 5; fi
    sleep 15
  done
  wait $pid || echo "WARN: $exp training exited nonzero"
}

run_one armD_sft_weighted_e8 0
run_one armD_sft_uniform_e8 1

set +x; source .env; set -x
export WANDB_API_KEY
for arm in weighted uniform; do
  loras=""
  for ep in 1 2 3 4 5 6 7 8; do
    a=checkpoints/armD_sft_${arm}_e8/adapter_ep$ep
    [ -d "$a" ] && loras="$loras $ep=$a"
  done
  /venv/verl/bin/python scripts/eval_armD_witness.py --model Qwen/Qwen3-1.7B \
    --lora $loras \
    --data test=data/armD/test.parquet val=data/armD/val.parquet train=data/armD/train.parquet \
    --limit 302 --out results/armD/${arm}_e8 --wandb-run armD_sft_${arm}_e8 \
    > results/armD/eval_${arm}_e8.log 2>&1
done
echo CHAIN_DONE
