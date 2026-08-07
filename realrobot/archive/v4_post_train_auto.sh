#!/bin/bash
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
# Smart auto pipeline:
# 1. Wait v4/4999 + train exit
# 2. Check loss plateau (last 1k step reduction < 10%)
# 3a. If plateau: start serve_v4_4999 + notify
# 3b. If not plateau: start v4_extend continue training to 15k + notify, plateau check repeats

set -u

WAIT_FOR_4999() {
  until [ -d ${NERO_CKPT}/pi05_nero_pick_pink_sponge_v4/v4/4999 ]; do sleep 60; done
  # Wait train process exit
  while pgrep -f "scripts/train.py.*pi05_nero_pick_pink_sponge_v4 " > /dev/null; do sleep 30; done
}

WAIT_FOR_4999
echo "v4 reached step 4999, checking plateau..."

# Plateau check
PLATEAU_OUTPUT=$(cd ${NERO_ROOT}/third_party/openpi-agilex && PATH=$HOME/.local/bin:$PATH uv run python ${NERO_ROOT}/realrobot/check_v4_plateau.py 2>&1)
echo "$PLATEAU_OUTPUT"

if echo "$PLATEAU_OUTPUT" | grep -q "PLATEAU=YES"; then
  echo "PLATEAU REACHED, starting server"
  setsid nohup ${NERO_ROOT}/realrobot/serve_v4_4999.sh > ${NERO_ROOT}/realrobot/serve_v4_4999.log 2>&1 < /dev/null &
  disown
  until grep -q "server listening on 0.0.0.0:9095" ${NERO_ROOT}/realrobot/serve_v4_4999.log 2>/dev/null; do sleep 5; done
  # (公开版移除:此处原有一行内部通知 hook,与训练流水线逻辑无关)
else
  echo "NOT PLATEAU, continuing to 15k step"
  # (公开版移除:此处原有一行内部通知 hook,与训练流水线逻辑无关)

  # v4_extend = new TrainConfig 15k step, resume from v4/4999 via weight_loader
  # NOTE: openpi doesn't support arbitrary resume across configs. We use --resume which requires same config name.
  # Simpler approach: just rerun v4 with --num-train-steps 15000 --resume (resumes from existing ckpt)
  cd ${NERO_ROOT}/third_party/openpi-agilex
  export PATH=$HOME/.local/bin:$PATH
  export CUDA_VISIBLE_DEVICES=3
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
  export HF_LEROBOT_HOME=${NERO_DATA_ROOT}
  setsid nohup uv run scripts/train.py \
    pi05_nero_pick_pink_sponge_v4_extend \
    --exp-name v4_extend \
    --checkpoint-base-dir ${NERO_CKPT} \
    --no-wandb-enabled \
    --overwrite > ${NERO_ROOT}/realrobot/train_v4_extend.log 2>&1 < /dev/null &
  disown
  echo "v4_extend training launched"
fi
