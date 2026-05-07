#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TASK=${TASK:-gsm8k}
METHOD=${METHOD:-ll}
K=${K:-2}
GPUS=${GPUS:-8}
OUT=${OUT:-outputs/grpo-${TASK}-${METHOD}-k${K}}

accelerate launch --num_processes "${GPUS}" train_grpo.py \
  --model_name_or_path "${MODEL}" \
  --task "${TASK}" \
  --method "${METHOD}" \
  --num_blocks "${K}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_generations 4 \
  --steps_per_generation 1 \
  --num_iterations 1 \
  --loss_type grpo \
  --learning_rate 1e-6 \
  --lr_k1 1e-6 \
  --lambda_aux 10 \
  --gradient_checkpointing \
  --output_dir "${OUT}"
