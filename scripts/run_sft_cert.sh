#!/usr/bin/env bash
# Certificate-SFT branch: 2 epochs of gold witness certificates on ckpt-250.
# v2: custom dataset keeps <think> in the loss (v1's per-turn templating
# stripped it — the think-ablation accident, RESEARCH_LOG 2026-08-06).
set -xuo pipefail
source /venv/verl/bin/activate
export HF_HOME=/workspace/.hf_home
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/workspace/hex-rl-cot-deconfusion:${PYTHONPATH:-}
cd /workspace/hex-rl-cot-deconfusion
torchrun --standalone --nproc_per_node=1 -m verl.trainer.sft_trainer \
    data.train_files=data/sft_cert/train.parquet \
    data.val_files=data/sft_cert/val.parquet \
    data.custom_cls.path=/workspace/hex-rl-cot-deconfusion/hexenv/sft_cert_dataset.py \
    data.custom_cls.name=CertSFTDataset \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=2048 \
    data.truncation=right \
    model.path=checkpoints/armC/global_step_250/hf \
    optim.lr=1e-5 \
    trainer.total_epochs=2 \
    trainer.project_name=hex-rl-cot-deconfusion \
    trainer.experiment_name=${SFT_EXP:-armC_sft_cert_v2} \
    trainer.default_local_dir=checkpoints/${SFT_EXP:-armC_sft_cert_v2} \
    trainer.logger='["console"]' \
    "$@"
