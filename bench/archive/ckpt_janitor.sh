#!/bin/bash
# Session-scoped disk janitor for the openpi bench runs (NOT a cron, NOT persistent).
#
# Why: each openpi checkpoint is params(12G) + train_state(31G) = 43G, and orbax (max_to_keep=1)
# only deletes the previous step AFTER the new one is fully written -> a run peaks at ~86G on a
# disk with <90G free that three subagents share. Stripping train_state from an *intermediate*
# checkpoint once its write has quiesced drops that peak to ~55G. Cost: no crash-resume from that
# step; params (all the bench needs to produce preds) is kept.
#
# Safety rules -- it only ever removes <BASE>/<cfg>/bench/<step>/train_state when ALL hold:
#   * cfg is one of the two bench configs (nothing else on the box is touched)
#   * step != FINAL_STEP (the deliverable checkpoint keeps everything)
#   * step/params and step/_CHECKPOINT_METADATA exist (the save committed)
#   * nothing under step/train_state has been modified for QUIESCE_MIN minutes (write finished)
# Time-based rather than log-based on purpose: train.py's "Step N: loss=..." lines are
# block-buffered here and can lag the real step by thousands of iterations.
set -uo pipefail
BASE="${NERO_CKPT:?set NERO_CKPT}"
CFGS="pi05_nero_b2_train pi0_nero_b2_train"
FINAL_STEP=29999
QUIESCE_MIN=20
PRESSURE_GB=60   # above this much free space, do nothing (keep train_state for crash-resume)

log() { echo "[$(date '+%F %T')] $*"; }
log "janitor start (base=$BASE final=$FINAL_STEP quiesce=${QUIESCE_MIN}min)"

while true; do
  # Only act under disk pressure: with room to spare, keeping train_state means a crash between
  # step 20000 and 29999 can resume instead of restarting a 13h run.
  free_gb=$(df -BG /home2 | tail -1 | awk '{gsub("G","",$4); print $4}')
  if [ "${free_gb:-0}" -ge "$PRESSURE_GB" ]; then sleep 120; continue; fi
  log "disk pressure: free=${free_gb}G < ${PRESSURE_GB}G -- scanning for strippable train_state"
  for cfg in $CFGS; do
    # exp_name is chosen at launch (it is also the wandb run name), so scan every exp dir
    # rather than hardcoding one -- hardcoding "bench" silently disarmed this whole janitor
    # the moment the runs were renamed to pi05_b2train / pi0_b2train.
    for d in "$BASE/$cfg"/*/; do
    [ -d "$d" ] || continue
    for s in $(ls "$d" 2>/dev/null | grep -E '^[0-9]+$' | sort -n); do
      [ "$s" = "$FINAL_STEP" ] && continue
      ts="$d/$s/train_state"
      [ -d "$ts" ] || continue
      [ -d "$d/$s/params" ] || continue
      [ -e "$d/$s/_CHECKPOINT_METADATA" ] || continue
      # any file under train_state touched within QUIESCE_MIN => still being written
      if [ -n "$(find "$ts" -newermt "-${QUIESCE_MIN} minutes" -print -quit 2>/dev/null)" ]; then
        continue
      fi
      sz=$(du -sh "$ts" 2>/dev/null | cut -f1)
      log "stripping $ts ($sz) -- committed + quiesced; params kept"
      rm -rf "$ts"
      log "free now: $(df -h /home2 | tail -1 | awk '{print $4}')"
    done
    done
  done
  free_gb=$(df -BG /home2 | tail -1 | awk '{gsub("G","",$4); print $4}')
  [ "$free_gb" -lt 25 ] && log "WARNING /home2 free=${free_gb}G (<25G)"
  sleep 120
done
