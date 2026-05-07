#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
DATASET=${DATASET:-metamathqa}
METHOD=${METHOD:-ll}
K=${K:-2}
GPUS=${GPUS:-8}
OUT=${OUT:-outputs/sft-${DATASET}-${METHOD}-k${K}}

torchrun --nproc_per_node "${GPUS}" train_sft.py \
  --model_name_or_path "${MODEL}" \
  --dataset_preset "${DATASET}" \
  --method "${METHOD}" \
  --num_blocks "${K}" \
  --max_seq_length 1024 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-5 \
  --lr_k1 2e-5 \
  --lambda_aux 10 \
  --gradient_checkpointing \
  --output_dir "${OUT}"
