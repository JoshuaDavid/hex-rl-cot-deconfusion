#!/usr/bin/env bash
# Arm D v2 ladder: weighted-loss r32 LoRA on constructive 2x2-9x9 witness,
# 8 epochs, rolling adapter export, per-epoch evals incl. v1 playout-test
# transfer.
set -uo pipefail
cd /workspace/hex-rl-cot-deconfusion
export HF_HOME=/workspace/.hf_home
EXP=armD2_sft_weighted

SPE=$(/venv/main/bin/python -c "
import pandas as pd; print(len(pd.read_parquet('data/armD2/train.parquet'))//64)")
echo "steps per epoch: $SPE"

ARMD_UNIFORM=0 ARMD_EXP=$EXP bash scripts/run_armD_sft.sh \
    data.train_files=data/armD2/train.parquet \
    data.val_files=data/armD2/val.parquet \
    trainer.total_epochs=8 > results/armD/train_${EXP}.log 2>&1 &
pid=$!
done_training=0
while :; do
  for d in checkpoints/$EXP/global_step_*; do
    [ -e "$d/data_0.pt" ] || continue
    n=${d##*_}
    ep=$((n / SPE))
    /venv/verl/bin/python scripts/export_armD_adapter.py "$d" \
      checkpoints/$EXP/adapter_ep$ep >> results/armD/export_${EXP}.log 2>&1 \
      && rm -rf "$d"
  done
  [ "$done_training" = 1 ] && break
  if ! kill -0 $pid 2>/dev/null; then done_training=1; sleep 5; fi
  sleep 15
done
wait $pid || echo "WARN: $EXP training exited nonzero"

set +x; source .env; set -x
export WANDB_API_KEY
loras=""
for ep in 1 2 3 4 5 6 7 8; do
  a=checkpoints/$EXP/adapter_ep$ep
  [ -d "$a" ] && loras="$loras $ep=$a"
done
/venv/verl/bin/python scripts/eval_armD_witness.py --model Qwen/Qwen3-1.7B \
  --lora $loras \
  --data test=data/armD2/test.parquet val=data/armD2/val.parquet \
         train=data/armD2/train.parquet v1test=data/armD/test.parquet \
  --limit 500 --out results/armD/armD2_weighted --wandb-run $EXP \
  --max-tokens 200 \
  > results/armD/eval_${EXP}.log 2>&1
echo CHAIN_DONE
