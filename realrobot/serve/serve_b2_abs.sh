#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# B-abs (pi05_nero_b2_abs) 推理 server — 全官方 openpi-agilex 栈, b2 recovery 数据 + 绝对动作空间
# 参数化: $1=step (默认3000), $2=port (默认9096)
# 用法: SERVE_GPU=3 bash serve_b2_abs.sh 3000 9096
# 契约跟 b1 完全一致 (client_b 不用改): use_delta_joint_actions=False, 模型直接输出绝对值
#   observation/image=cam_high(H,W,3 uint8 RGB) / observation/wrist_image=cam_wrist
#   observation/state=8维[follower j1..j7 deg原生 + follower夹爪raw mm] / prompt
#   server 返回【绝对值】chunk[10x8]: 关节绝对deg + 夹爪绝对mm; client 直接下发
# ⚠️ ckpt 在 147 训, 起前先 rsync 到 145 本地 ${NERO_CKPT}/pi05_nero_b2_abs/b2_abs/$STEP
STEP=${1:-3000}
PORT=${2:-9096}
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=${SERVE_GPU:-3}      # 起前先看 nvidia-smi 挑空卡
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${SERVE_MEM:-0.2}   # Pi0.5 推理 ~15-19G
export HF_HUB_OFFLINE=1
export OPENPI_TRACE=${NERO_ROOT}/realrobot/server_trace_b2_abs_${STEP}.jsonl
cd ${NERO_ROOT}/third_party/openpi-agilex
exec .venv/bin/python3 scripts/serve_policy.py \
  --env ALOHA --port $PORT \
  policy:checkpoint \
  --policy.config pi05_nero_b2_abs \
  --policy.dir ${NERO_CKPT}/pi05_nero_b2_abs/b2_abs/$STEP
