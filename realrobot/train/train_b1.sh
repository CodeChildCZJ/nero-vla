#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# 全 B 训练 pi05_nero_b1 (openpi-agilex 纯官方栈). 数据落地后跑这个.
# 用法: bash train_b1.sh [GPU] [STEP_FOR_NORM_CPU]
#   GPU 默认 2 (等 v8 训练结束腾出); serve 在 GPU3 别占.
# 前提: LeRobot 数据集已 scp 到 ~/.cache/huggingface/lerobot/local/pick_pink_sponge_b1
set -e
GPU=${1:-2}
CFG=pi05_nero_b1
REPO=local/pick_pink_sponge_b1
AGX=${NERO_ROOT}/third_party/openpi-agilex
PY=$AGX/.venv/bin/python3
DS=~/.cache/huggingface/lerobot/$REPO

echo "=== 0. 检查数据集存在 ==="
ls -d $DS/ || { echo "❌ 数据集不在 $DS, 先让 windows scp"; exit 1; }
ls $DS/meta/info.json $DS/data/ 2>/dev/null && echo "✅ 数据集结构 OK"

echo "=== 1. compute_norm_stats (写 assets/pi05_nero_b1/$REPO/norm_stats.json) ==="
cd $AGX
NS=$AGX/assets/$CFG/$REPO/norm_stats.json
if [ -f "$NS" ]; then
  echo "✅ norm_stats 已存在, 跳过 ($NS)"
else
  HF_HUB_OFFLINE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" $PY scripts/compute_norm_stats.py --config-name $CFG
fi

echo "=== 2. train (超参冻 v8: 30k/h10/ema0.99/cosine) ==="
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train.py $CFG \
  --exp-name b1 \
  --checkpoint-base-dir ${NERO_CKPT} \
  --no-wandb-enabled --overwrite
