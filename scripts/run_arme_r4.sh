#!/usr/bin/env bash
# Arm E R4: single-token task-selection RL. Only the selection token is trained
# (hex_select agent loop sets response_mask=1 on that token alone). Reward = the
# model's own Task-C score; GRPO's group baseline contrasts C-reward across the
# X's sampled in each group.
# Args: EXP_NAME  MODEL_PATH(merged R3)  [STEPS]
set +e
cd /workspace/hex-rl-cot-deconfusion
pkill -9 -f raylet 2>/dev/null; pkill -9 -f gcs_server 2>/dev/null
pkill -9 -x 'VLLM::EngineCore' 2>/dev/null
sleep 3
EXP=${1:-arme_r4_cost}
MODEL_PATH_IN=${2:-checkpoints/arme_win/hf_merged}
STEPS_IN=${3:-50}

export ARME_EVAL=${ARME_EVAL:-W}
export ARME_HELPERS=${ARME_HELPERS:-A,C,D}
MODEL_PATH="$MODEL_PATH_IN" \
DATA_DIR=data/arme/rl \
RESP_LEN=560 BATCH=32 GROUP_N=8 STEPS="$STEPS_IN" SAVE_FREQ=999 \
KL_COEF=0.001 USE_KL=True \
EXP_NAME="$EXP" \
ARME_HELPER_BUDGET=200 ARME_ANSWER_BUDGET=16 \
bash scripts/run_pilot.sh \
  data.max_prompt_length=896 \
  actor_rollout_ref.rollout.agent.default_agent_loop=hex_select \
  trainer.val_before_train=False \
  trainer.test_freq=999 \
  > "results/arme/${EXP}.log" 2>&1
