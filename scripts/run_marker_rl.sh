#!/usr/bin/env bash
# Launch a marker single-token RL run. Args: EXP_NAME  PENALTY
set +e
cd /workspace/hex-rl-cot-deconfusion
pkill -9 -f raylet 2>/dev/null
pkill -9 -f gcs_server 2>/dev/null
sleep 3
EXP=$1
PEN=$2
MODEL_PATH=checkpoints/marker_sft/hf_merged \
DATA_DIR=data/verl_witness_long \
RESP_LEN=256 LEN_LAMBDA=0 ANSWER_BRANCH=1 STEPS=50 SAVE_FREQ=999 \
EXP_NAME="$EXP" HEX_WITNESS_ANSWER_BUDGET=200 \
HEX_MARKER=" NOTE" HEX_MARKER_PENALTY="$PEN" \
bash scripts/run_pilot.sh > "results/${EXP}.log" 2>&1
