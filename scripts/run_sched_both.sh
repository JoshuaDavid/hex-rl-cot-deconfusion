#!/usr/bin/env bash
# Launch both SFT-scheduler arms sequentially (uniform then port).
cd /workspace/hex-rl-cot-deconfusion
set +e
pkill -9 -f raylet 2>/dev/null
pkill -9 -f gcs_server 2>/dev/null
sleep 2
rm -rf checkpoints/sched_uniform checkpoints/sched_port \
       results/multitask/sched_* data/multitask/_chunk_* \
       results/multitask/both_done.flag 2>/dev/null
set -a; source /workspace/hex-rl-cot-deconfusion/.env; set +a
export HF_HOME=/workspace/.hf_home
mkdir -p results/multitask
ARM=uniform /venv/verl/bin/python scripts/scheduler_sft.py \
    > results/multitask/uniform.log 2>&1
pkill -9 -f raylet 2>/dev/null; sleep 3
ARM=port /venv/verl/bin/python scripts/scheduler_sft.py \
    > results/multitask/port.log 2>&1
echo DONE > results/multitask/both_done.flag
