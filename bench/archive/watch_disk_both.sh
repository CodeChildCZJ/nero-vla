#!/usr/bin/env bash
# Disk headroom watch for the two openpi trainings.
# Peak need per model: params 12G + train_state 31G = ~43G for the in-flight ckpt,
# transiently ~86G while orbax writes 29999 before dropping 20000 (max_to_keep=1).
# WARN below 100G, ALARM below 60G -> at 60G the final ckpt may not fit.
# The second training machine (host/port/user are site-specific -- this leg ran two boxes).
#   NERO_REMOTE_HOST      ssh destination, e.g. "user@host"
#   NERO_REMOTE_SSH_OPTS  extra ssh flags, e.g. "-p 2222" (may be empty)
: "${NERO_REMOTE_HOST:?set NERO_REMOTE_HOST to the second training machine (user@host)}"
: "${NERO_REMOTE_SSH_OPTS:=}"
SSH="ssh $NERO_REMOTE_SSH_OPTS -o ConnectTimeout=20 $NERO_REMOTE_HOST"
last145=999; last147=999
while true; do
  l145=$(df -BG /home2 | tail -1); g145=$(echo "$l145" | awk '{gsub(/G/,"",$4); print $4+0}')
  l147=$($SSH 'df -BG /home2 | tail -1' 2>/dev/null) || l147=""
  g147=$(echo "$l147" | awk '{gsub(/G/,"",$4); print $4+0}')
  for h in 145 147; do
    [ "$h" = 145 ] && { g=$g145; prev=$last145; } || { g=$g147; prev=$last147; }
    [ -z "$g" ] && continue
    if [ "$g" -lt 60 ]; then
      echo "[disk $h] ALARM ${g}G free — final ckpt (43G) at risk, act now"
    elif [ "$g" -lt 100 ] && [ "$prev" -ge 100 ]; then
      echo "[disk $h] WARN crossed below 100G: ${g}G free"
    elif [ "$g" -lt 140 ] && [ "$prev" -ge 140 ]; then
      echo "[disk $h] note: crossed below 140G: ${g}G free"
    fi
    [ "$h" = 145 ] && last145=$g || last147=$g
  done
  if [ -z "$l147" ]; then echo "[disk 147] df unreachable over ssh at $(date +%H:%M)"; fi
  sleep 600
done
