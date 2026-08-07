#!/usr/bin/env bash
# 1-step CPU training run of the EXACT SmolVLA config lerobot_train_bb.sh will launch,
# plus the post-train chain against the checkpoint it produces.
#
# Why on CPU, before the GPU window: the SmolVLA argument set has three pieces that have
# never executed -- the top-level `--rename_map` JSON, `--policy.empty_cameras=1`, and
# `--policy.scheduler_decay_steps` on a policy that (unlike ACT) accepts it. Each fails
# loudly and within seconds, but it would fail while I am holding a card that a queue is
# waiting on. CPU time here is free; window time there is not.
#
# The 1-step checkpoint is the real prize. Its BAKED NORMALIZATION STATS are identical to
# what the 20k run will bake (same dataset, same train-111 aggregate, computed before step
# 1), so the parameterization measurement and the leak check are VALID from it -- they are
# statements about the stats artifact, not about the weights. That closes two board items
# early and de-risks the whole post-train chain. The PREDICTIONS are of course garbage;
# nothing here produces a leaderboard number.
#
# Kept OUT of runs/ and deleted at the end: a stale checkpoint that predict could silently
# load is a real failure mode this bench has already hit once (openvla-oft's probe dir).
set -uo pipefail

: "${NERO_ROOT:?please 'source env.sh' first}"
BENCH="$NERO_ROOT/bench"
VENV="$NERO_ROOT/third_party/lerobot/.venv"
ROOT="${NERO_DATA_V30:-$HOME/.cache/huggingface/lerobot/local/pick_pink_sponge_b2_bench}"
REPO_ID=local/pick_pink_sponge_b2_bench
SMOKE="${TMPDIR:-/tmp}/smolvla_cpu_smoke"
KEEP=${KEEP_SMOKE:-0}
WARN=0
log() { echo "[$(date '+%F %T')] $*"; }

EPS=$(BENCH="$BENCH" "$VENV/bin/python" - <<'PY'
import json, os
s = json.load(open(os.environ["BENCH"] + "/data/split.json"))
eps = s["train_episodes"]
assert len(eps) == 111 and not (set(eps) & set(s["val_episodes"]))
print("[" + ",".join(map(str, eps)) + "]")
PY
) || { echo "split.json read failed"; exit 1; }

# Remove but do NOT create: lerobot-train's `cfg.validate()` hard-raises FileExistsError on an
# output_dir that already exists (resume is False), so pre-creating it fails the run before step 1.
# This bit me on this script's FIRST REAL EXECUTION -- it had only ever been `bash -n` checked,
# which cannot see it. The real launcher is clear (it mkdirs runs/, never runs/<bb>), and that
# asymmetry is the point: a smoke that mirrors the launcher's *args* but not its *filesystem
# preconditions* validates less than it looks like it does.
rm -rf "$SMOKE"

# CUDA_VISIBLE_DEVICES="" is a hard guarantee, not a hint: this script must be unable to
# take a card even if a flag is wrong, because both cards belong to other legs right now.
log "=== 1-step CPU train (config validation) ==="
CUDA_VISIBLE_DEVICES="" HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
"$VENV/bin/lerobot-train" \
  --policy.path=lerobot/smolvla_base \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.scheduler_decay_steps=20000 \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.cam_high":"observation.images.camera1","observation.images.cam_wrist":"observation.images.camera2"}' \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$ROOT" \
  --dataset.episodes="$EPS" \
  --batch_size=1 \
  --steps=1 \
  --save_freq=1 \
  --output_dir="$SMOKE" \
  --job_name=smolvla_cpu_smoke \
  --policy.device=cpu \
  --policy.push_to_hub=false \
  --wandb.enable=false 2>&1 | tail -40
rc=${PIPESTATUS[0]}
[ $rc -ne 0 ] && { log "!!! CPU train FAILED rc=$rc -- fix the config BEFORE the GPU window"; exit 1; }

CKPT=$SMOKE/checkpoints/000001/pretrained_model
[ -d "$CKPT" ] || { log "!!! no checkpoint at $CKPT"; ls -R "$SMOKE/checkpoints" 2>/dev/null | head; exit 1; }
log "checkpoint written: $(du -sh "$CKPT" | cut -f1)"

# The rename map must survive into the checkpoint, because predict recovers it from there
# rather than hardcoding it -- if it does not round-trip, predict dies at make_policy with
# a feature mismatch and the failure would first appear after a 4-hour run.
"$VENV/bin/python" - "$CKPT" <<'PY' || WARN=$((WARN+1))
import json, pathlib, sys
tc = json.loads((pathlib.Path(sys.argv[1]) / "train_config.json").read_text())
rm = tc.get("rename_map")
want = {"observation.images.cam_high": "observation.images.camera1",
        "observation.images.cam_wrist": "observation.images.camera2"}
print(f"[rename_map in ckpt] {rm}")
assert rm == want, f"rename_map did not round-trip: {rm} != {want}"
print("[rename_map] round-trips into the checkpoint OK")
PY

step() { log "--- $1 ---"; shift; "$@" 2>&1 | tail -25; local rc=${PIPESTATUS[0]}
         [ $rc -ne 0 ] && { log "!!! rc=$rc"; WARN=$((WARN+1)); }; return 0; }

step "predict on 3 anchors (rename recovery, processors, write gate)" \
  env CUDA_VISIBLE_DEVICES="" "$VENV/bin/python" "$BENCH/scripts/predict/lerobot_predict.py" \
      --backbone smolvla --ckpt "$CKPT" --device cpu --limit 3 \
      --out "$SMOKE/preds_SMOKE_NOT_A_ROW.npz"
step "action parameterization at the training target (VALID from this ckpt)" \
  env CUDA_VISIBLE_DEVICES="" "$VENV/bin/python" "$BENCH/archive/lerobot_action_parameterization.py" \
      --backbone smolvla --ckpt "$CKPT" --n 16
step "leak check on the ckpt-baked stats (VALID from this ckpt)" \
  env CUDA_VISIBLE_DEVICES="" "$VENV/bin/python" "$BENCH/archive/lerobot_leak_check.py" --ckpt "$CKPT"

log "warnings: $WARN"
if [ "$KEEP" = 1 ]; then
  log "KEEPING $SMOKE (KEEP_SMOKE=1). It is NOT under runs/, so no predict path can glob it."
else
  rm -rf "$SMOKE"; log "removed $SMOKE (a stale smoke ckpt is a real failure mode)"
fi

# Explicit machine-readable verdict, and it is not decoration. This script ran to completion
# with `warnings: 0` and no such line, so the watcher tailing it reported "EXITED with no
# verdict" -- it could not distinguish "finished clean" from "died silently after the last
# step". A validation script whose own success is only inferable from the ABSENCE of bad
# lines cannot be consumed by anything automated; absence of evidence is the one thing a log
# tail is worst at. State the conclusion, and make the exit code agree with it.
if [ "$WARN" = 0 ]; then
  log "SMOKE PASS -- config validated end to end (train step, rename_map round-trip, predict, action parameterization, leak check)"
  exit 0
fi
log "SMOKE FAIL -- $WARN warning(s); fix the config BEFORE the GPU window"
exit 1
