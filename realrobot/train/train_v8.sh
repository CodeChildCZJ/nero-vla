#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# Pi0.5 NERO v8: 62ep conditioned multimodal (海绵位置→grasp构型 r~0.93-0.95 双数据确认).
# 治 v7 闭环漂移 = 位置+approach全覆盖(左中右均衡)+SOP抓取保持. 配方基于v7, czj 调 3 处:
#   ①action_horizon 16→10 (=官方pi05_libero) ②ema_decay None→0.99 (官方默认, GPU1空间够)
#   ③跑 GPU1. batch16/mask off/pi05_base/cosine 不变. no-crop 干净隔离(crop留v9).
# 30k step, save每3k(keep全留), 真机挑最优ckpt. norm_stats 已重算(local/pick_pink_sponge_v8).
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}   # czj 指定 GPU1 (现全空97G); 占了再覆盖去GPU2挤
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.9}   # GPU1空, 给0.9(~87G)容 EMA(+~12.5G)
export HF_LEROBOT_HOME=${NERO_DATA_ROOT}
unset OPENPI_TRACE
cd ${NERO_ROOT}/third_party/openpi-agilex
exec uv run scripts/train.py \
  pi05_nero_pick_pink_sponge_v8 \
  --exp-name v8 \
  --checkpoint-base-dir ${NERO_CKPT} \
  --no-wandb-enabled \
  --overwrite
