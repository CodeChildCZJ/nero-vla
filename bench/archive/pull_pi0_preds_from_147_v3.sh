#!/bin/bash
# pi0 trains on 147, but 145 owns the leaderboard. Pull preds_pi0.npz + the sampling-variance
# footnote back to 145 and re-score here. Delivery is PULL-only by design: 147's
# post_train_pipeline.sh contains no rsync/ssh back to 145 at all, so there is no remote-side
# port/credential to get wrong (the failure mode that bit gr00t-n17's `ssh -p 22`).
#
# v3 vs v2 -- v2 had the SAME silent-loss shape it was written to fix, just one stage later.
# On 147 the order is: predict -> preds_pi0.npz, THEN score, THEN variance_openpi.py (200 anchors
# x 5 seeds, ~15 min of GPU). v2 broke out of its wait 2 minutes after preds_pi0.npz settled and
# rsynced logs/variance/ immediately -- i.e. always BEFORE variance_pi0.json existed. It would have
# pulled an empty directory, logged "variance files now here: ", scored, and exited "done" with a
# zero exit code. The CONTRACT-mandatory pi0 footnote would have been produced on 147 and never
# arrived, with no error anywhere on either host.
#
# v3 therefore has TWO waits with independent deadlines, and pulls preds as soon as they exist so a
# slow/failed footnote can never delay the leaderboard row.
#
# Logs into logs/post_predict_pi0.log, which the bench monitor already tails.
set -uo pipefail
: "${NERO_ROOT:?please 'source env.sh' first}"
# The second training machine (host/port/user are site-specific -- this leg ran two boxes).
#   NERO_REMOTE_HOST      ssh destination, e.g. "user@host"
#   NERO_REMOTE_SSH_OPTS  extra ssh flags, e.g. "-p 2222" (may be empty)
: "${NERO_REMOTE_HOST:?set NERO_REMOTE_HOST to the second training machine (user@host)}"
: "${NERO_REMOTE_SSH_OPTS:=}"
BENCH="$NERO_ROOT/bench"
R="ssh -o BatchMode=yes -o ConnectTimeout=20 $NERO_REMOTE_SSH_OPTS"
HOST="$NERO_REMOTE_HOST"
# same repo path on both boxes -- that is what made a bare rsync of REMOTE work
REMOTE="$NERO_ROOT/bench"
VJSON="$REMOTE/logs/variance/variance_pi0.json"
log() { echo "[$(date '+%F %T')] [pull147-v3] $*"; }

# settled(remote_path) = exists AND untouched for 120s (np.savez / write finished)
settled() {
  timeout 90 $R $HOST "[ -f '$1' ] && [ -z \"\$(find '$1' -newermt '-120 seconds' -print -quit)\" ]" 2>/dev/null
}

# ---- stage 1: preds ------------------------------------------------------------------
log "stage 1: waiting for 147 to produce preds_pi0.npz ..."
while ! settled "$REMOTE/preds/preds_pi0.npz"; do sleep 300; done
log "147 has preds_pi0.npz, pulling"

timeout 600 rsync -a -e "$R" "$HOST:$REMOTE/preds/preds_pi0.npz" "$BENCH/preds/preds_pi0.npz"
rc=$?
log "rsync preds rc=$rc"

if [ $rc -eq 0 ]; then
  # transfer integrity: compare md5 on both hosts, don't trust the exit code alone
  m_local=$(md5sum "$BENCH/preds/preds_pi0.npz" 2>/dev/null | awk '{print $1}')
  m_remote=$(timeout 120 $R $HOST "md5sum $REMOTE/preds/preds_pi0.npz" 2>/dev/null | awk '{print $1}')
  if [ -n "$m_local" ] && [ "$m_local" = "$m_remote" ]; then
    log "preds md5 MATCH on both hosts: $m_local"
  else
    log "preds md5 MISMATCH local='$m_local' remote='$m_remote' -- DO NOT TRUST THIS ROW"
  fi
  log "re-scoring the full leaderboard on 145 (pi0 row is in, footnote still pending)"
  "$NERO_ROOT/third_party/openpi-agilex/.venv/bin/python" "$BENCH/scripts/score/score.py"
else
  log "preds rsync FAILED -- leaderboard has no pi0 row; retry by hand"
fi

# ---- stage 2: the sampling-variance footnote (separate deadline) ----------------------
mkdir -p "$BENCH/logs/variance"
log "stage 2: waiting for $VJSON (deadline 90 min) ..."
deadline=$(( $(date +%s) + 5400 ))
got=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if settled "$VJSON"; then got=1; break; fi
  sleep 120
done

if [ $got -eq 1 ]; then
  timeout 600 rsync -a -e "$R" "$HOST:$REMOTE/logs/variance/" "$BENCH/logs/variance/"
  log "rsync logs/variance rc=$?"
  # legacy path, harmless no-op if absent (only reachable if 147 ran a stale variance_openpi.py)
  timeout 600 rsync -a -e "$R" "$HOST:$REMOTE/variance/" "$BENCH/logs/variance/" 2>/dev/null
  log "rsync legacy variance rc=$?"
  log "variance files now here: $(ls "$BENCH/logs/variance" 2>/dev/null | tr '\n' ' ')"
  if [ -s "$BENCH/logs/variance/variance_pi0.json" ]; then
    log "FOOTNOTE OK -- variance_pi0.json:"
    cat "$BENCH/logs/variance/variance_pi0.json"
  else
    log "FOOTNOTE MISSING after a successful rsync -- investigate"
  fi
else
  log "!!! FOOTNOTE TIMEOUT: variance_pi0.json never settled on 147 within 90 min."
  log "!!! The pi0 row is on the board but the CONTRACT-mandatory sampling-variance footnote is"
  log "!!! ABSENT. Re-run by hand on 147 (GPU4 is free once training exits):"
  log "!!!   variance_openpi.py --config-name pi0_nero_b2_train --ckpt-dir <final> --tag pi0 \\"
  log "!!!       --ref-preds preds/preds_pi0.npz"
  timeout 600 rsync -a -e "$R" "$HOST:$REMOTE/logs/variance/" "$BENCH/logs/variance/" 2>/dev/null
  log "pulled whatever exists anyway: $(ls "$BENCH/logs/variance" 2>/dev/null | tr '\n' ' ')"
fi

# the remote pipeline's own predict/score/variance output, pulled LAST so it includes the
# variance rc line
timeout 120 $R $HOST "cat $REMOTE/logs/post_predict_pi0.log" 2>/dev/null | sed 's/^/[147] /'
log "done (preds rc=$rc, footnote got=$got)"
