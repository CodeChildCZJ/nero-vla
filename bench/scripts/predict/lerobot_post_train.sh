#!/usr/bin/env bash
# Post-training pipeline for a LeRobot backbone (smolvla | act) on NERO b2.
#
#   ./lerobot_post_train.sh smolvla 5            # wait for the 20k ckpt, then run everything
#   NERO_DRYRUN=1 ./lerobot_post_train.sh smolvla 5   # print the plan, touch no GPU
#
# DELIVERY-FIRST ORDERING (gr00t-n17's rule, adopted board-wide): predict and score come
# FIRST and nothing else can block them. Every stage after them is a footnote or an audit
# that the leaderboard row does not depend on, so a hanging or crashing validation costs
# its own artifact and never the row. The reverse order -- audits first -- is what turns a
# broken leak-check into a missing backbone.
#
# NOTHING here claims the GPU is free (gr00t-n15's rule): this pid still holds a CUDA
# context until it exits, so a downstream leg reading a "DONE" line would be racing it.
set -uo pipefail
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"

BB=${1:?usage: $0 <smolvla|act> <gpu>}
GPU=${2:?usage: $0 <smolvla|act> <gpu>}
BENCH="$NERO_ROOT/bench"
PY="$NERO_ROOT/third_party/lerobot/.venv/bin/python"
OUT=$BENCH/runs/$BB
case "$BB" in
  smolvla) STEP=020000 ;;
  act)     STEP=100000 ;;
  *) echo "unknown backbone: $BB" >&2; exit 2 ;;
esac
CKPT=$OUT/checkpoints/$STEP/pretrained_model
PREDS=$BENCH/preds/preds_$BB.npz
VARJSON=$BENCH/logs/variance/variance_$BB.json
DRY=${NERO_DRYRUN:-0}
WARN=0

log() { echo "[$(date '+%F %T')] $*"; }
step() {  # step <name> <cmd...>
  local name=$1; shift
  log "--- $name ---"
  if [ "$DRY" = 1 ]; then echo "    DRYRUN: $*"; return 0; fi
  "$@"; local rc=$?
  if [ $rc -ne 0 ]; then log "!!! $name FAILED rc=$rc"; WARN=$((WARN+1)); fi
  return 0
}

log "backbone=$BB gpu=$GPU ckpt=$CKPT dry=$DRY"

# ---------------------------------------------------------------- wait for ckpt
# Key the wait on the ARTIFACT, not on the process or the directory: `checkpoints/`
# already exists from step 5000 onward, and `last` is a symlink that points at whatever
# was saved most recently -- neither tells you the FINAL checkpoint is complete. The
# delivered row is pre-committed to $STEP (no-val-selection), so wait for exactly that
# path and for its bytes to stop moving.
if [ "$DRY" != 1 ]; then
  while [ ! -f "$CKPT/config.json" ]; do log "waiting for $CKPT ..."; sleep 60; done
  while [ -n "$(find "$CKPT" -newermt '-90 seconds' -print -quit 2>/dev/null)" ]; do
    log "ckpt still settling..."; sleep 30
  done
  log "ckpt present and settled: $(du -sh "$CKPT" | cut -f1)"
fi

# ============================ STAGE 1: THE ROW (nothing may block this) =========
step "predict (batch 1, full 1397, global-index noise pinning)" \
  "$PY" "$BENCH/scripts/predict/lerobot_predict.py" --backbone "$BB" --ckpt "$CKPT" \
        --device "cuda:$GPU" --seed 0

if [ "$DRY" != 1 ] && [ ! -f "$PREDS" ]; then
  log "!!! $PREDS was not written -- the write-side finiteness gate rejects non-finite"
  log "!!! predictions on purpose, so check the predict output above before re-running."
  WARN=$((WARN+1))
fi
step "score (shared leaderboard)" "$PY" "$BENCH/scripts/score/score.py"

# ============================ STAGE 2: the sampling footnote ====================
# Its own wait/failure surface, deliberately after the row. Emits the per-draw provenance
# npz and self-verifies subset-vs-full bit-exactness against the row written above.
step "sampling variance (200 x 5 draws + provenance)" \
  "$PY" "$BENCH/archive/lerobot_sampling_variance.py" --backbone "$BB" \
        --ckpt "$CKPT" --device "cuda:$GPU"
if [ "$DRY" != 1 ] && [ ! -f "$VARJSON" ]; then
  log "!!! $VARJSON missing -- footnote did not land (the row above is unaffected)"
  WARN=$((WARN+1))
fi

# ============================ STAGE 3: audits (row-independent) =================
step "action parameterization at the TRAINING TARGET" \
  "$PY" "$BENCH/archive/lerobot_action_parameterization.py" --backbone "$BB" --ckpt "$CKPT"
step "leak check (ckpt-baked stats vs train-111 recompute, float32 cast)" \
  "$PY" "$BENCH/archive/lerobot_leak_check.py" --ckpt "$CKPT"
step "live-stats provenance (poison injection, not comparison)" \
  "$PY" "$BENCH/archive/lerobot_live_stats_provenance.py" --ckpt "$CKPT"

# Overfit diagnostic. FULL population on BOTH splits (--n 0): a 400-anchor draw carries
# ~+-0.13 on the ratio, which is the same order as real gaps between backbones. `--eps val`
# additionally asserts the rebuilt anchors bit-match the frozen val_anchors.npz before any
# train number is believed -- a rebuild that cannot reproduce the frozen harness makes
# every number downstream of it worthless.
step "train-split eval: VAL (also the tool self-check)" \
  "$PY" "$BENCH/archive/lerobot_train_split_eval.py" --ckpt "$CKPT" --backbone "$BB" \
        --eps val --n 0 --device "cuda:$GPU"
step "train-split eval: TRAIN (7475 anchors)" \
  "$PY" "$BENCH/archive/lerobot_train_split_eval.py" --ckpt "$CKPT" --backbone "$BB" \
        --eps train --n 0 --device "cuda:$GPU"
# --backbone is NOT optional here even though it has a default. The default is `act`, so
# omitting it on a SmolVLA run would compute and write the ACT row -- right filename, right
# schema, plausible numbers, wrong model. Same silent-wrong-artifact class as openpi's
# pre-flight writing another leg's MAEs into their own json.
step "split-difficulty control (CPU, image-blind constants)" \
  "$PY" "$BENCH/archive/lerobot_split_difficulty.py" --backbone "$BB"
step "overfit-ratio significance (CPU, paired episode jackknife)" \
  "$PY" "$BENCH/archive/lerobot_ratio_signif.py" --backbone "$BB"
# Board rule (d): the ratio must be reported in a STATED weighting, because frame-weighted
# and episode-equal move a model and its baseline in opposite directions. For ACT that
# flipped the sign of the val arm-vs-hold margin while leaving the 2.3x ratio alone, so the
# caliber is load-bearing on margins and reassuring on ratios -- either way it has to be measured.
step "weighting calibers (CPU, frame-weighted vs episode-equal)" \
  "$PY" "$BENCH/archive/lerobot_weighting_calibers.py" --backbone "$BB"

# The board row's training budget is DERIVED from this checkpoint's own train_config.json,
# never copied from the neighbouring row's meta. ACT ran 100000 x 8 = 21.04 epochs and
# SmolVLA 20000 x 64 = 33.66, so a hand-copied block would label one row with the other's
# budget: plausible, self-consistent, and invisible to every structural check on the board.
# The script's --verify-against mode re-derives ACT's shipped block as a must-pass.
step "training-budget block (derived from ckpt, not typed)" \
  "$PY" "$BENCH/archive/lerobot_budget_block.py" --ckpt "$CKPT"

step "paired significance over the whole board" \
  "$PY" "$BENCH/scripts/score/paired_signif.py" --all

# ============================ close out =========================================
log "warnings: $WARN"
log "this pid ($$) still holds a CUDA context on GPU $GPU until it exits -- do NOT read"
log "this line as 'GPU $GPU is free'. Verify with nvidia-smi after the process is gone."
exit 0
