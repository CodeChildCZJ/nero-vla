#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# Pi0.5 NERO v7: 官方/agilex recipe — 去 mask (mask_gripper_state=False) + 数据从源头修 copycat
# (recorder qpos[7]=follower readback ≠ action[7]=leader, lead-lag验lag+3 action领先3帧). ema_decay=0.99 回标准.
# 23ep diverse(任意位置铺开, 闭位起步, 抓取闭到0). 10000 step, save每2000. 重算 norm_stats(夹爪0~100双峰).
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}   # 起训前按GPU占用覆盖
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION:-0.8}
export HF_LEROBOT_HOME=${NERO_DATA_ROOT}
unset OPENPI_TRACE
cd ${NERO_ROOT}/third_party/openpi-agilex
exec uv run scripts/train.py \
  pi05_nero_pick_pink_sponge_v7 \
  --exp-name v7 \
  --checkpoint-base-dir ${NERO_CKPT} \
  --no-wandb-enabled \
  --overwrite
