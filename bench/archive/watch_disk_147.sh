#!/usr/bin/env bash
# 147-specific disk watch, tighter than watch_disk_both.sh.
# Why a second watcher instead of editing the first: watch_disk_both.sh is currently being
# executed by an armed monitor, and bash re-reads a running script by byte offset -- editing
# it in place corrupts the live process. New file = new byte stream = safe.
#
# 147 budget: pi0 saves at step 20000 (~08:40) and 29999 (~12:20). orbax max_to_keep=1 writes
# the new ckpt BEFORE dropping the old, so the run transiently needs 43G + 43G = 86G.
# ckpt_janitor.sh (PID 3005487 on 147) strips 20000/train_state under pressure (+31G), which
# cuts the true requirement to ~55G. Ladder below is set so I hear about it with hours of slack,
# not at the cliff.
# The second training machine (host/port/user are site-specific -- this leg ran two boxes).
#   NERO_REMOTE_HOST      ssh destination, e.g. "user@host"
#   NERO_REMOTE_SSH_OPTS  extra ssh flags, e.g. "-p 2222" (may be empty)
: "${NERO_REMOTE_HOST:?set NERO_REMOTE_HOST to the second training machine (user@host)}"
: "${NERO_REMOTE_SSH_OPTS:=}"
SSH="ssh $NERO_REMOTE_SSH_OPTS -o ConnectTimeout=20 $NERO_REMOTE_HOST"
prev=""; prev_t=0
while true; do
  line=$($SSH 'df -BG /home2 | tail -1' 2>/dev/null)
  if [ -z "$line" ]; then
    echo "[147 disk] ssh/df unreachable at $(date +%H:%M)"
    sleep 300; continue
  fi
  g=$(echo "$line" | awk '{gsub(/G/,"",$4); print $4+0}')
  now=$(date +%s)
  rate=""
  if [ -n "$prev" ] && [ "$prev_t" -gt 0 ]; then
    dt=$(( now - prev_t ))
    [ "$dt" -gt 0 ] && rate=$(awk -v d=$(( prev - g )) -v t="$dt" 'BEGIN{printf "%+.0fG/h", -d*3600.0/t}')
  fi
  # ladder: 130 = gr00t setup is eating into my margin; 100 = janitor-assisted peak only;
  # 85 = below my unassisted 86G peak; 60 = janitor trigger point, act now.
  for thr in 130 100 85 60; do
    if [ "$g" -lt "$thr" ] && { [ -z "$prev" ] || [ "$prev" -ge "$thr" ]; }; then
      echo "[147 disk] crossed below ${thr}G: ${g}G free ${rate:+(trend $rate)} -- pi0 needs 86G transient / 55G with janitor"
      break
    fi
  done
  # sustained fast drain is worth hearing about even without a crossing
  if [ -n "$rate" ]; then
    r=$(echo "$rate" | tr -d 'G/h+')
    case "$r" in -*) v=${r#-}; [ "${v%%.*}" -ge 25 ] && echo "[147 disk] draining $rate, now ${g}G free" ;; esac
  fi
  prev=$g; prev_t=$now
  sleep 300
done
