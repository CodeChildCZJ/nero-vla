#!/usr/bin/env bash
# usage: ckpt_loadtest.sh <ckpt_step_dir> <tag>
# Waits for <ckpt_step_dir>/params to appear and quiesce, then CPU-load-tests it and exits.
#
# Why bother: a corrupt/truncated orbax save is indistinguishable from a good one by byte
# count or du -sh -- pi0_base burned us exactly that way (nested params/params/ with four
# truncated shards, surfacing only as OUT_OF_RANGE at load time). The step-20000 save lands
# ~3.5h before the final one; if it is unreadable, the 29999 written by the same process
# probably is too, and I want that at 09:00 with runway, not at 12:30 mid-delivery.
# It also catches divergence (NaN/Inf weights), which a block-buffered loss log can hide.
#
# CPU-only by construction: CUDA_VISIBLE_DEVICES="" + JAX_PLATFORMS=cpu, so it cannot touch
# the GPU running the very training it inspects. nice/ionice so the 12G read does not
# compete with the trainer's I/O.
# Verified 2026-08-05 against pi05_nero_b2/b2_delta/29999/params -> 51 arrays / 3.35B / PASS.
set -uo pipefail
: "${NERO_ROOT:?please 'source env.sh' first}"
CKPT="${1:?ckpt step dir}"; TAG="${2:?tag}"
OPENPI="$NERO_ROOT/third_party/openpi-agilex"
log() { echo "[$(date '+%F %T')] [$TAG] $*"; }

log "waiting for $CKPT/params ..."
while [ ! -d "$CKPT/params" ]; do sleep 120; done
log "params dir appeared; waiting for the write to quiesce"
while [ -n "$(find "$CKPT/params" -newermt '-120 seconds' -print -quit 2>/dev/null)" ]; do sleep 60; done
log "quiesced. size $(du -sh "$CKPT/params" | cut -f1) | step dir holds: $(ls "$CKPT" | tr '\n' ' ')"

cd "$OPENPI" || exit 1
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu nice -n 15 ionice -c3 \
  .venv/bin/python - "$CKPT/params" <<'PY'
import sys, numpy as np, jax, jax.numpy as jnp
import openpi.models.model as _model
p = sys.argv[1]
try:
    params = _model.restore_params(p, dtype=jnp.bfloat16)
except Exception as e:
    print(f"CKPT_LOADTEST FAIL restore raised {type(e).__name__}: {e}")
    raise SystemExit(1)
leaves = jax.tree.leaves(params)
n = sum(int(np.prod(x.shape)) for x in leaves)
bad = sum(1 for x in leaves if not bool(jnp.isfinite(x.astype(jnp.float32)).all()))
print(f"CKPT_LOADTEST arrays={len(leaves)} params={n/1e9:.2f}B nonfinite_arrays={bad}")
print("CKPT_LOADTEST " + ("PASS" if (bad == 0 and n > 2.5e9) else "FAIL"))
PY
log "loadtest exit rc=$?"
