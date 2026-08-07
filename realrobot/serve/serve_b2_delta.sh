#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# B-delta(131recovery) (pi05_nero_b2) 推理 server — 全官方 openpi-agilex 栈, b2 recovery 数据 + delta 动作空间
# 参数化: $1=step (默认29999), $2=port (默认9429)
# 用法: SERVE_GPU=1 SERVE_MEM=0.25 bash serve_b2_delta.sh 21000 9421
# ⚠️契约跟 b2_abs/b1 一致 (client_b 不用改, 无 shim): use_delta_joint_actions=True 但
#   推理时 DeltaActions(input) + AbsoluteActions(output) → server 输出仍是【绝对 deg+mm】
#   (AbsoluteActions 在 output 端把 delta 加回 state, 所以 client 收到绝对值直接下发)
#   observation/state=8维[follower j1..j7 deg原生 + follower夹爪raw mm] / prompt
# ckpt 在 145 本地 tree ${NERO_CKPT}/pi05_nero_b2/b2_delta/$STEP (留 21k/27k/29999)
STEP=${1:-29999}
PORT=${2:-9429}
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=${SERVE_GPU:-1}      # 起前先看 nvidia-smi 挑空卡
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${SERVE_MEM:-0.25}   # 冷启编译留余量, 防自杀OOM
export HF_HUB_OFFLINE=1
export OPENPI_TRACE=${NERO_ROOT}/realrobot/server_trace_b2_delta_${STEP}.jsonl
cd ${NERO_ROOT}/third_party/openpi-agilex
exec .venv/bin/python3 scripts/serve_policy.py \
  --env ALOHA --port $PORT \
  policy:checkpoint \
  --policy.config pi05_nero_b2 \
  --policy.dir ${NERO_CKPT}/pi05_nero_b2/b2_delta/$STEP
