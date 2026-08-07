#!/usr/bin/env bash
# Finetune GR00T N1.5 on NERO b2 (train split only, 111 episodes).
# Mirrors the official SO-101 new-embodiment recipe from
# getting_started/3_0_new_embodiment_finetuning.md: batch 64 / 10k steps /
# lr 1e-4 / tune projector + diffusion head, backbone LLM+ViT frozen.
set -euo pipefail
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"

REPO="$NERO_ROOT/third_party/Isaac-GR00T-n1d5"
BENCH="$NERO_ROOT/bench"
OUT=${OUT:-${NERO_CKPT:-$NERO_ROOT/checkpoints}/gr00t_n15_nero_b2}
STEPS=${STEPS:-10000}
BS=${BS:-64}

cd "$REPO"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export PYTHONPATH="$BENCH/configs:${PYTHONPATH:-}"
export HF_ENDPOINT=https://hf-mirror.com   # proxy CDN stalls on HF LFS
export no_proxy='*' NO_PROXY='*'
export TOKENIZERS_PARALLELISM=false

exec .venv/bin/python scripts/gr00t_finetune.py \
  --dataset-path "$BENCH/data/b2_gr00t_train" \
  --data-config nero_data_config:NeroDualCamDataConfig \
  --embodiment-tag new_embodiment \
  --video-backend decord \
  --num-gpus 1 \
  --batch-size "$BS" \
  --max-steps "$STEPS" \
  --save-steps 2500 \
  --learning-rate 1e-4 \
  --dataloader-num-workers 8 \
  --report-to tensorboard \
  --output-dir "$OUT" \
  "$@"
