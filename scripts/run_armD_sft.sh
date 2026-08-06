#!/usr/bin/env bash
# Arm D test 1: witness 2x2-5x5, rank-32 LoRA SFT from base Qwen3-1.7B,
# no-think targets, token-importance loss weights.
# Uniform-weight control: ARMD_UNIFORM=1 ARMD_EXP=armD_sft_uniform bash scripts/run_armD_sft.sh
set -xuo pipefail
source /venv/verl/bin/activate
set +x; source /workspace/hex-rl-cot-deconfusion/.env; set -x
export WANDB_API_KEY
export HF_HOME=/workspace/.hf_home
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/hex-rl-cot-deconfusion:${PYTHONPATH:-}
cd /workspace/hex-rl-cot-deconfusion
EXP=${ARMD_EXP:-armD_sft_weighted}
torchrun --standalone --nproc_per_node=1 -m verl.trainer.sft_trainer \
    data.train_files=data/armD/train.parquet \
    data.val_files=data/armD/val.parquet \
    data.custom_cls.path=/workspace/hex-rl-cot-deconfusion/hexenv/armd_sft_dataset.py \
    data.custom_cls.name=ArmDWitnessSFTDataset \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=2048 \
    model.path=Qwen/Qwen3-1.7B \
    model.lora_rank=32 \
    model.lora_alpha=64 \
    optim.lr=1e-4 \
    trainer.total_epochs=3 \
    trainer.save_freq=after_each_epoch \
    trainer.project_name=hex-rl-cot-deconfusion \
    trainer.experiment_name=${EXP} \
    trainer.default_local_dir=checkpoints/${EXP} \
    trainer.logger='["console","wandb"]' \
    "$@"
