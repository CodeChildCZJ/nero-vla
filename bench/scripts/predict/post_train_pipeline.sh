#!/bin/bash
# Wait for an openpi bench training run to exit, then run the whole post-training pipeline:
#   predict -> score -> sampling-variance footnote -> strip train_state -> (optionally) next training
#
#   usage: post_train_pipeline.sh <config_name> <bb_tag> <gpu_id> <exp_name> [next_config] [next_tag] [next_exp]
#   e.g.   post_train_pipeline.sh pi05_nero_b2_train pi05 5 pi05_b2train pi0_nero_b2_train pi0 pi0_b2train
#
# Written as a standalone file on purpose: bash reads a running script by byte offset, so
# rewriting the file an armed waiter is executing corrupts its control flow. To change this
# script, kill the waiter first, then edit, then re-arm.
set -uo pipefail
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"
CFG="${1:?config}"; BB="${2:?bb tag}"; GPU="${3:?gpu}"; EXP="${4:?exp name}"
NEXT_CFG="${5:-}"; NEXT_BB="${6:-}"; NEXT_EXP="${7:-}"
BENCH="$NERO_ROOT/bench"
CKPT_ROOT="${NERO_CKPT:-$NERO_ROOT/checkpoints}/$CFG/$EXP"
FINAL=29999
OPENPI="$NERO_ROOT/third_party/openpi-agilex"
PY=$OPENPI/.venv/bin/python
MIN_FREE_GB_FOR_NEXT=100

log() { echo "[$(date '+%F %T')] $*"; }

log "waiting for training '$CFG' to exit..."
while pgrep -f "train.py $CFG" > /dev/null; do sleep 60; done
log "training process gone"

if [ ! -d "$CKPT_ROOT/$FINAL/params" ]; then
  log "ABORT: $CKPT_ROOT/$FINAL/params missing; steps present: $(ls "$CKPT_ROOT" 2>/dev/null | tr '\n' ' ')"
  exit 1
fi
# orbax writes asynchronously; wait until the params dir stops changing.
while [ -n "$(find "$CKPT_ROOT/$FINAL/params" -newermt '-90 seconds' -print -quit 2>/dev/null)" ]; do
  log "params still settling..."; sleep 60
done

cd "$OPENPI" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU"          # JAX would otherwise build a mesh over all 6 cards
export XLA_PYTHON_CLIENT_PREALLOCATE=false

log "predict -> preds_${BB}.npz (noise seed 0)"
$PY "$BENCH/scripts/predict/predict_openpi.py" \
  --config-name "$CFG" --ckpt-dir "$CKPT_ROOT/$FINAL" \
  --out "$BENCH/preds/preds_${BB}.npz"
rc=$?
log "predict rc=$rc"

if [ $rc -eq 0 ]; then
  $PY "$BENCH/scripts/score/score.py"
  log "sampling-variance footnote (200 anchors x 5 seeds)"
  $PY "$BENCH/archive/variance_openpi.py" \
    --config-name "$CFG" --ckpt-dir "$CKPT_ROOT/$FINAL" --tag "$BB" \
    --ref-preds "$BENCH/preds/preds_${BB}.npz"
  log "variance rc=$?"
  # preds are in hand -> the optimizer state has no further use; contract says keep params only.
  if [ -d "$CKPT_ROOT/$FINAL/train_state" ]; then
    sz=$(du -sh "$CKPT_ROOT/$FINAL/train_state" 2>/dev/null | cut -f1)
    rm -rf "$CKPT_ROOT/$FINAL/train_state" && log "stripped final train_state ($sz freed)"
  fi
else
  log "predict FAILED -- keeping train_state so the run can be resumed/debugged"
fi

df -h /home2 | tail -1

if [ -n "$NEXT_CFG" ]; then
  # The next run may already have been started by hand elsewhere -- possibly on ANOTHER HOST
  # (this bench was run across two training machines), which pgrep here cannot see. Hence the
  # explicit marker file:
  # whoever launches the run off-box drops $BENCH/.skip_next_<tag> to disarm this fallback.
  # Starting a second copy would fight over its checkpoint dir and burn a GPU, so guard hard.
  if [ -f "$BENCH/.skip_next_${NEXT_BB}" ]; then
    log "NOT starting $NEXT_CFG: marker $BENCH/.skip_next_${NEXT_BB} present ($(cat "$BENCH/.skip_next_${NEXT_BB}"))"
  elif pgrep -f "train.py $NEXT_CFG" > /dev/null; then
    log "NOT starting $NEXT_CFG: already running elsewhere (pid $(pgrep -f "train.py $NEXT_CFG" | tr '\n' ' '))"
  elif [ -f "$BENCH/preds/preds_${NEXT_BB}.npz" ]; then
    log "NOT starting $NEXT_CFG: preds_${NEXT_BB}.npz already exists"
  else
    free_gb=$(df -BG --output=avail /home2 | tail -1 | tr -dc '0-9')
    if [ "$free_gb" -lt "$MIN_FREE_GB_FOR_NEXT" ]; then
      log "NOT starting $NEXT_CFG: only ${free_gb}G free (< ${MIN_FREE_GB_FOR_NEXT}G)"
    else
      log "starting next training: $NEXT_CFG on GPU $GPU (${free_gb}G free)"
      cd "$BENCH" || exit 1
      setsid nohup ./scripts/train/train_openpi_bench.sh "$NEXT_CFG" "$GPU" "$NEXT_EXP" \
        > "$BENCH/logs/train_${NEXT_BB}_b2train.log" 2>&1 < /dev/null &
      disown
      sleep 120   # let train.py register in the process table before the next waiter polls
      setsid nohup bash "$BENCH/scripts/predict/post_train_pipeline.sh" "$NEXT_CFG" "$NEXT_BB" "$GPU" "$NEXT_EXP" \
        >> "$BENCH/logs/post_predict_${NEXT_BB}.log" 2>&1 < /dev/null &
      disown
      log "armed post_train_pipeline for $NEXT_CFG"
    fi
  fi
fi
log "done"
