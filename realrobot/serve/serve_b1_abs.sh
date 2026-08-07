#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# b1_abs(32clean) (pi05_nero_b1_abs) 推理 server — 全官方 openpi-agilex 栈, b1 clean 数据 + 绝对动作空间
# 参数化: $1=step (默认29999), $2=port (默认9529)
# 用法: SERVE_GPU=2 SERVE_MEM=0.25 bash serve_b1_abs.sh 21000 9521
# 契约跟 b2_abs 一致 (client_b 不用改, 无 shim): use_delta_joint_actions=False, 直接输出绝对值
#   observation/state=8维[follower j1..j7 deg原生 + follower夹爪raw mm] / prompt
#   server 返回【绝对 deg+mm】chunk[10x8]; client 直接下发
# ckpt 在 145 本地 tree ${NERO_CKPT}/pi05_nero_b1_abs/b1_abs/$STEP (留 21k/27k/29999)
STEP=${1:-29999}
PORT=${2:-9529}
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=${SERVE_GPU:-2}      # 起前先看 nvidia-smi 挑空卡
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${SERVE_MEM:-0.25}   # 冷启编译留余量, 防自杀OOM
export HF_HUB_OFFLINE=1
export OPENPI_TRACE=${NERO_ROOT}/realrobot/server_trace_b1_abs_${STEP}.jsonl
cd ${NERO_ROOT}/third_party/openpi-agilex
exec .venv/bin/python3 scripts/serve_policy.py \
  --env ALOHA --port $PORT \
  policy:checkpoint \
  --policy.config pi05_nero_b1_abs \
  --policy.dir ${NERO_CKPT}/pi05_nero_b1_abs/b1_abs/$STEP
