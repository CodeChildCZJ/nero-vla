#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# B-ref (pi05_nero_b2_ref) 推理 server — b2 recovery 数据 + follower[t+1]绝对 + 关节rad + 夹爪[0,1]
# 参数化: $1=step, $2=port
# 用法: SERVE_GPU=1 bash serve_b2_ref.sh 21000 9221
# ⚠️ 契约跟 b1/B-abs 不同 (ref 模型在 rad+夹爪[0,1] 空间训练, 转换烤进数据集):
#   client 发: observation/state=8维[follower j1..j7 用 *RAD* + 夹爪 *[0,1]* (0=闭/0mm, 1=张/76mm)]
#              observation/image=cam_high / observation/wrist_image=cam_wrist / prompt
#   server 返回【绝对值】chunk[10x8]: 关节绝对 *RAD* + 夹爪绝对 *[0,1]*
#   → windows ref-client 负责: 下发前 rad->deg(*180/pi) + [0,1]->mm(*76); 上传 state 前 deg->rad + mm->[0,1]
STEP=${1:?need step}
PORT=${2:?need port}
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=${SERVE_GPU:?need SERVE_GPU}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${SERVE_MEM:-0.2}   # Pi0.5 推理 ~15-19G
export HF_HUB_OFFLINE=1
export OPENPI_TRACE=${NERO_ROOT}/realrobot/server_trace_b2_ref_${STEP}.jsonl
# ckpt 目录: 默认原训练路径; 从硬备份起则 REF_CKPT_DIR=${NERO_CKPT}/b2_ref_KEEP/<step>
CKPT_DIR=${REF_CKPT_DIR:-${NERO_CKPT}/pi05_nero_b2_ref/b2_ref/$STEP}
cd ${NERO_ROOT}/third_party/openpi-agilex
exec .venv/bin/python3 scripts/serve_policy.py \
  --env ALOHA --port $PORT \
  policy:checkpoint \
  --policy.config pi05_nero_b2_ref_serve \
  --policy.dir "$CKPT_DIR"
