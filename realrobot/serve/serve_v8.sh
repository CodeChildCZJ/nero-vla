#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# v8 推理 server — 参数化: $1=step (默认3000), $2=port (默认9095)
# 用法: bash serve_v8.sh 3000 9095   (训练在GPU2, server放GPU3 42G free)
# 真机 smoke: windows client 连 <linux_ip>:$PORT, horizon=10/no-crop 已对齐
STEP=${1:-3000}
PORT=${2:-9095}
export PATH=$HOME/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=${SERVE_GPU:-3}     # 训练占GPU2, server用GPU3; 占了再换
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=${SERVE_MEM:-0.3}   # 推理~15G, 0.3(29G)够且容GPU3的42G free
export HF_LEROBOT_HOME=${NERO_DATA_ROOT}
export OPENPI_TRACE=${NERO_ROOT}/realrobot/server_trace_v8_${STEP}.jsonl
cd ${NERO_ROOT}/third_party/openpi-agilex
exec uv run scripts/serve_policy.py \
  --env ALOHA --port $PORT \
  policy:checkpoint \
  --policy.config pi05_nero_pick_pink_sponge_v8 \
  --policy.dir ${NERO_CKPT}/pi05_nero_pick_pink_sponge_v8/v8/$STEP
