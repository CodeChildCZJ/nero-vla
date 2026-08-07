#!/usr/bin/env bash
# openpi resolves predict-time norm_stats from checkpoint_dir/"assets", NOT the config
# assets dir (explicit upstream comment in policy_config.create_trained_policy). The
# lattice leak-proof was run on the SOURCE assets file; it only transfers to inference if
# the ckpt-baked copy is byte-identical. Verify that on the first saved step (20000).
#
# Emits one line per host when its ckpt copy appears, then exits once both are settled.
set -u
# The second training machine (host/port/user are site-specific -- this leg ran two boxes).
#   NERO_REMOTE_HOST      ssh destination, e.g. "user@host"
#   NERO_REMOTE_SSH_OPTS  extra ssh flags, e.g. "-p 2222" (may be empty)
: "${NERO_REMOTE_HOST:?set NERO_REMOTE_HOST to the second training machine (user@host)}"
: "${NERO_REMOTE_SSH_OPTS:=}"
EXPECT=9467e876b49b33168d5c2654482f0d09
P05="${NERO_CKPT:?set NERO_CKPT}/pi05_nero_b2_train/pi05_b2train/20000/assets/local/pick_pink_sponge_b2_train/norm_stats.json"
P0="$NERO_CKPT/pi0_nero_b2_train/pi0_b2train/20000/assets/local/pick_pink_sponge_b2_train/norm_stats.json"
done05=0; done0=0

verdict () {  # $1=tag $2=actual-md5-or-empty
  if [ -z "$2" ]; then
    echo "CKPT_NORMSTATS $1 FAIL: file present but md5 unreadable"
  elif [ "$2" = "$EXPECT" ]; then
    echo "CKPT_NORMSTATS $1 PASS: ckpt-baked copy md5=$2 == proven-clean source (leak proof transfers to inference)"
  else
    echo "CKPT_NORMSTATS $1 FAIL: ckpt-baked md5=$2 != source $EXPECT -- leak proof does NOT transfer, re-run lattice test on the ckpt copy"
  fi
}

while [ $done05 -eq 0 ] || [ $done0 -eq 0 ]; do
  if [ $done05 -eq 0 ] && [ -s "$P05" ]; then
    sleep 20  # let orbax finish the assets callback
    verdict pi05 "$(md5sum "$P05" 2>/dev/null | awk '{print $1}')"
    done05=1
  fi
  if [ $done0 -eq 0 ]; then
    m=$(ssh -o BatchMode=yes -o ConnectTimeout=15 $NERO_REMOTE_SSH_OPTS $NERO_REMOTE_HOST \
          "[ -s $P0 ] && sleep 20 && md5sum $P0" 2>/dev/null | awk '{print $1}')
    if [ -n "$m" ]; then verdict pi0 "$m"; done0=1; fi
  fi
  [ $done05 -eq 1 ] && [ $done0 -eq 1 ] && break
  sleep 120
done
echo "CKPT_NORMSTATS both hosts verified; watcher exiting"
