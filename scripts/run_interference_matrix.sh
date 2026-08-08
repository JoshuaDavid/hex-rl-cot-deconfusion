#!/usr/bin/env bash
# Arm E interference matrix. 8 LoRA-r32 SFT runs from base Qwen3-1.7B:
#   baselines: wit, cell, list (single-task ceilings)
#   forward  : wit -> cell -> list  (sequential, adapter continues)
#   reverse  : list -> cell -> wit
#   mixed    : all three at once
# Then eval every resulting adapter on all 3 held-out test sets (one engine
# load). Adapters exported to peft, FSDP shards deleted (disk discipline).
set -uo pipefail
cd /workspace/hex-rl-cot-deconfusion
ID=data/interference
CK=checkpoints/interference
mkdir -p "$CK" results/interference

# train_stage <exp> <trainfile> <valfile> <epochs> <adapter-in|NONE>
train_stage() {
  local exp=$1 tf=$2 vf=$3 ep=$4 adin=$5
  local spe steps out="$CK/$exp/adapter"
  if [ -d "$out" ]; then echo "[skip] $exp exists"; return; fi
  local extra=""
  [ "$adin" != "NONE" ] && extra="model.lora_adapter_path=$adin"
  ARMD_EXP=interference/$exp bash scripts/run_armD_sft.sh \
    data.train_files=$tf data.val_files=$vf trainer.total_epochs=$ep \
    $extra > results/interference/train_${exp//\//_}.log 2>&1
  # last checkpoint = ep * (rows/64)
  local last
  last=$(ls -d "$CK/$exp"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
  /venv/verl/bin/python scripts/export_armD_adapter.py "$last" "$out" \
    >> results/interference/train_${exp//\//_}.log 2>&1
  rm -rf "$CK/$exp"/global_step_*
  echo "[done] $exp -> $out"
}

echo "=== TRAINING ==="
# baselines
train_stage wit_only   $ID/wit_train.parquet  $ID/wit_test.parquet  2 NONE
train_stage cell_only  $ID/cell_train.parquet $ID/cell_test.parquet 2 NONE
train_stage list_only  $ID/list_train.parquet $ID/list_test.parquet 2 NONE
# forward wit->cell->list (stage1 == wit_only)
train_stage fwd2_witcell      $ID/cell_train.parquet $ID/cell_test.parquet 2 $CK/wit_only/adapter
train_stage fwd3_witcelllist  $ID/list_train.parquet $ID/list_test.parquet 2 $CK/fwd2_witcell/adapter
# reverse list->cell->wit (stage1 == list_only)
train_stage rev2_listcell     $ID/cell_train.parquet $ID/cell_test.parquet 2 $CK/list_only/adapter
train_stage rev3_listcellwit  $ID/wit_train.parquet  $ID/wit_test.parquet  2 $CK/rev2_listcell/adapter
# mixed
train_stage mixed $ID/mixed_train.parquet $ID/wit_test.parquet 2 NONE

echo "=== EVAL ==="
set +u; source .env; set -u; export WANDB_API_KEY HF_HOME=/workspace/.hf_home
# adapter index legend (step field encodes which adapter)
LORAS="\
 1=$CK/wit_only/adapter 2=$CK/cell_only/adapter 3=$CK/list_only/adapter \
 4=$CK/fwd2_witcell/adapter 5=$CK/fwd3_witcelllist/adapter \
 6=$CK/rev2_listcell/adapter 7=$CK/rev3_listcellwit/adapter \
 8=$CK/mixed/adapter"
for try in 1 2 3; do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1500 ]; do sleep 5; done
  timeout -s KILL 3000 /venv/verl/bin/python scripts/eval_armD_witness.py \
    --model Qwen/Qwen3-1.7B --lora $LORAS \
    --data wit=$ID/wit_test.parquet cell=$ID/cell_test.parquet list=$ID/list_test.parquet \
    --max-tokens 512 --out results/interference/mtx --wandb-run interference_matrix \
    > results/interference/eval.log 2>&1
  grep -q "OVERALL" results/interference/eval.log && break
  pkill -9 -x "VLLM::EngineCore" 2>/dev/null; sleep 20
done
pkill -9 -x "VLLM::EngineCore" 2>/dev/null
echo "MATRIX_DONE"
