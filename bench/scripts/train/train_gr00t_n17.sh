#!/usr/bin/env bash
# Finetune GR00T N1.7 on NERO b2 (train split only, 111 episodes).
#
# Hyper-parameters are N1.7's own FinetuneConfig defaults -- global_batch_size 64,
# max_steps 10000, lr 1e-4, tune projector + diffusion head with the LLM and ViT
# frozen -- which happen to match the N1.5 run (batch 64 / 10k / 1e-4) exactly,
# so the two GR00T rows differ by backbone rather than by training budget. The
# 32/2000 numbers in getting_started/finetune_new_embodiment.md are tuned for the
# 5-episode `cube_to_bowl_5` demo, not for a 111-episode set.
#
# GPU is NOT defaulted: pass CUDA_VISIBLE_DEVICES explicitly so this can never
# land on someone else's card.
set -euo pipefail
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"

REPO="$NERO_ROOT/third_party/Isaac-GR00T"
BENCH="$NERO_ROOT/bench"
OUT=${OUT:-${NERO_CKPT:-$NERO_ROOT/checkpoints}/gr00t_n17_nero_b2}
STEPS=${STEPS:-10000}
BS=${BS:-64}
SAVE_STEPS=${SAVE_STEPS:-2500}
EXP_NAME=${EXP_NAME:-gr00t_n17}
# 底模本地快照,与 oft_train.sh 的 ${NERO_CKPT}/openvla-7b 同一约定。
BASE_MODEL=${BASE_MODEL:-$BENCH/models/GR00T-N1.7-3B}
# Smoke runs set WANDB=0 so they never land in the real gr00t_n17 run.
if [ "${WANDB:-1}" = "1" ]; then WANDB_FLAG=--use-wandb; else WANDB_FLAG=--no-use-wandb; fi

if [ "${NERO_DRYRUN:-0}" = "1" ]; then
  # CPU-only rehearsal (model build + processor + dataset, stops before Trainer).
  # Force the GPUs away so it can never squat on a card that isn't ours.
  export CUDA_VISIBLE_DEVICES=""
elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "ERROR: set CUDA_VISIBLE_DEVICES explicitly (only use cards allocated to you)" >&2
  exit 1
fi

cd "$REPO"
export HF_ENDPOINT=https://hf-mirror.com   # proxy CDN stalls on HF LFS
export no_proxy='*' NO_PROXY='*'
export TOKENIZERS_PARALLELISM=false
# Unset when nvidia/Cosmos-Reason2-2B access is granted; see n17_launch_finetune.py.
# The dir is nested under Qwen/ on purpose: gr00t_n1d7.get_backbone_cls() picks the
# backbone class by substring-matching model_name against "nvidia/Cosmos-Reason2" or
# "Qwen/Qwen3-VL", and raises "Unsupported model name" otherwise -- so a local path
# has to contain one of those literals.
export NERO_BACKBONE_MODEL=${NERO_BACKBONE_MODEL:-$BENCH/models/Qwen/Qwen3-VL-2B-Instruct}

exec .venv/bin/python "$BENCH/scripts/train/n17_launch_finetune.py" \
  --base-model-path "$BASE_MODEL" \
  --dataset-path "$BENCH/data/b2_n17_train" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$BENCH/configs/nero_n17_config.py" \
  --num-gpus 1 \
  --global-batch-size "$BS" \
  --max-steps "$STEPS" \
  --learning-rate 1e-4 \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit 1 \
  --save-only-model \
  --dataloader-num-workers 8 \
  --output-dir "$OUT" \
  --experiment-name "$EXP_NAME" \
  $WANDB_FLAG \
  --wandb-project nero_backbone_bench \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  "$@"
