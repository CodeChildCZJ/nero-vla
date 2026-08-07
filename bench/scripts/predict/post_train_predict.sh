#!/bin/bash
# Wait for an openpi bench training process to exit, then immediately produce its preds npz.
#   usage: post_train_predict.sh <config_name> <bb_tag> <gpu_id>
# e.g.    post_train_predict.sh pi05_nero_b2_train pi05 5
# Deliberately does NOT chain the next training run -- GPU assignment is a human decision.
set -uo pipefail
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
CFG="${1:?config}"; BB="${2:?bb tag}"; GPU="${3:?gpu}"
BENCH="$NERO_ROOT/bench"
OPENPI="$NERO_ROOT/third_party/openpi-agilex"
CKPT_ROOT="${NERO_CKPT:-$NERO_ROOT/checkpoints}/$CFG/bench"
FINAL=29999

echo "[$(date '+%F %T')] waiting for training '$CFG' to exit..."
while pgrep -f "train.py $CFG" > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] training process gone"

if [ ! -d "$CKPT_ROOT/$FINAL/params" ]; then
  echo "[$(date '+%F %T')] ABORT: $CKPT_ROOT/$FINAL/params missing; steps present: $(ls "$CKPT_ROOT" 2>/dev/null | tr '\n' ' ')"
  exit 1
fi
# orbax writes asynchronously; wait until the params dir stops changing.
while [ -n "$(find "$CKPT_ROOT/$FINAL/params" -newermt '-90 seconds' -print -quit 2>/dev/null)" ]; do
  echo "[$(date '+%F %T')] params still settling..."; sleep 60
done

echo "[$(date '+%F %T')] running predict on GPU $GPU"
cd "$OPENPI"
CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv/bin/python "$BENCH/scripts/predict/predict_openpi.py" \
  --config-name "$CFG" \
  --ckpt-dir "$CKPT_ROOT/$FINAL" \
  --out "$BENCH/preds/preds_${BB}.npz"
rc=$?
echo "[$(date '+%F %T')] predict rc=$rc"
[ $rc -eq 0 ] && "$OPENPI/.venv/bin/python" "$BENCH/scripts/score/score.py"
echo "[$(date '+%F %T')] done"
