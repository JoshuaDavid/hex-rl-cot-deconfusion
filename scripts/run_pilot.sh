#!/usr/bin/env bash
# GRPO pilot on hex positions. Modeled on /workspace/verl-smoke/run_smoke.sh
# (thread/pid workarounds from notes/verl_setup.md).
set -xuo pipefail

source /venv/verl/bin/activate
export HF_HOME=/workspace/.hf_home
export VLLM_USE_V1=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export RAY_prestart_worker_first_driver=0
export RAY_num_server_call_thread=2
export TOKENIZERS_PARALLELISM=false
export GRPC_ENABLE_FORK_SUPPORT=0
set +x; source /workspace/hex-rl-cot-deconfusion/.env; set -x
export WANDB_API_KEY

export PYTHONPATH=/workspace/hex-rl-cot-deconfusion:${PYTHONPATH:-}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
DATA_DIR=${DATA_DIR:-data/verl_hex}
RESP_LEN=${RESP_LEN:-2176}
ROLLOUT_TEMP=${ROLLOUT_TEMP:-1.0}
STEPS=${STEPS:-100}
BATCH=${BATCH:-32}
GROUP_N=${GROUP_N:-8}
KL_COEF=${KL_COEF:-0.001}
USE_KL=${USE_KL:-True}
EXP_NAME=${EXP_NAME:-pilot_1p7b}
SAVE_FREQ=${SAVE_FREQ:-25}
CKPT_DIR=${CKPT_DIR:-/workspace/hex-rl-cot-deconfusion/checkpoints/$EXP_NAME}

export HEX_ROLLOUT_LOG=/workspace/hex-rl-cot-deconfusion/results/rollouts/${EXP_NAME}.jsonl
export HEX_LEN_LAMBDA=${LEN_LAMBDA:-0}
export HEX_CLOSE_BIAS=${CLOSE_BIAS:-}
export HEX_THINK_CHAR_CAP=${THINK_CHAR_CAP:-3300}
mkdir -p "$(dirname "$HEX_ROLLOUT_LOG")" "$CKPT_DIR"

cd /workspace/hex-rl-cot-deconfusion

python3 -m verl.trainer.main_ppo \
    +ray_kwargs.ray_init.include_dashboard=False \
    ray_kwargs.ray_init.num_cpus=6 \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=/workspace/hex-rl-cot-deconfusion/hexenv/agent_loops.yaml \
    actor_rollout_ref.rollout.agent.default_agent_loop=hex_forced_close \
    reward.num_workers=1 \
    transfer_queue.backend.SimpleStorage.num_data_storage_units=1 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${DATA_DIR:-data/verl_hex}/train.parquet \
    data.val_files=${DATA_DIR:-data/verl_hex}/val.parquet \
    data.train_batch_size=$BATCH \
    data.max_prompt_length=768 \
    data.max_response_length=$RESP_LEN \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMP \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.n=$GROUP_N \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    reward.custom_reward_function.path=/workspace/hex-rl-cot-deconfusion/hexenv/reward_verl.py \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=hex-rl-cot-deconfusion \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=25 \
    trainer.val_before_train=True \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.total_epochs=100 \
    trainer.total_training_steps=$STEPS \
    "$@"
