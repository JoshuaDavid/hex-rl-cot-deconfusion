#!/usr/bin/env bash
# Certificate-SFT branch: 2 epochs of gold witness certificates on ckpt-250.
set -xuo pipefail
source /venv/verl/bin/activate
export HF_HOME=/workspace/.hf_home
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
cd /workspace/hex-rl-cot-deconfusion
torchrun --standalone --nproc_per_node=1 -m verl.trainer.sft_trainer \
    data.train_files=data/sft_cert/train.parquet \
    data.val_files=data/sft_cert/val.parquet \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=2048 \
    data.truncation=right \
    model.path=checkpoints/armC/global_step_250/hf \
    optim.lr=1e-5 \
    trainer.total_epochs=2 \
    trainer.project_name=hex-rl-cot-deconfusion \
    trainer.experiment_name=armC_sft_cert \
    trainer.default_local_dir=checkpoints/armC_sft_cert \
    trainer.logger='["console"]' \
    "$@"
