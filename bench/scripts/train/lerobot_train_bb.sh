#!/usr/bin/env bash
# Train a LeRobot backbone (act | smolvla) on NERO b2 train_episodes only.
#
#   ./lerobot_train_bb.sh act     4
#   ./lerobot_train_bb.sh smolvla 5
#
# Arg1 = backbone, Arg2 = GPU index (CONTRACT.md: only GPU 4 and 5 are ours).
#
# TRAIN/VAL DISCIPLINE (CONTRACT.md): --dataset.episodes is pinned to the 111
# train ids from split.json, so the 20 val episodes never enter the sampler.
# The normalization stats in the working copy were re-aggregated over those same
# 111 episodes by prepare_b2_v30.py -- LeRobot does NOT filter meta/stats.json by
# --dataset.episodes, so without that step val stats would leak into the normalizer.
set -euo pipefail
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"

BB=${1:?usage: $0 <act|smolvla> <gpu>}
GPU=${2:?usage: $0 <act|smolvla> <gpu>}
BENCH="$NERO_ROOT/bench"
VENV="$NERO_ROOT/third_party/lerobot/.venv"
# b2 数据集本体(不公开,格式见 docs/DATA.md)+ prepare_b2_v30.py 产出的 v3.0 工作副本。
NERO_DATA="${NERO_DATA:-$HOME/.cache/huggingface/lerobot/local/pick_pink_sponge_b2}"
ROOT="${NERO_DATA_BENCH:-${NERO_DATA}_bench}"
REPO_ID=local/pick_pink_sponge_b2_bench

# 111 train episode ids, straight from split.json (single source of truth)
EPS=$(SPLIT="$BENCH/data/split.json" "$VENV/bin/python" - <<'PY'
import json, os
s = json.load(open(os.environ["SPLIT"]))
eps = s["train_episodes"]
assert len(eps) == 111 and not (set(eps) & set(s["val_episodes"]))
print("[" + ",".join(map(str, eps)) + "]")
PY
)

OUT=$BENCH/runs/${BB}
mkdir -p "$BENCH/logs" "$BENCH/runs"     # parent only -- see the guard below

# lerobot-train's cfg.validate() hard-raises FileExistsError if output_dir exists (resume=False).
# It raises EARLY, but "early" still means after the launcher has been backgrounded and after the
# GPU window has been claimed, and the traceback says nothing about what to do. Fail here instead,
# in the foreground, with the decision spelled out -- deleting a previous run's checkpoints is not
# something a launcher may do on its own. (Found by the CPU config smoke, which hit the same raise
# from the opposite cause: it pre-created the dir itself. bash -n cannot see either.)
if [ -e "$OUT" ]; then
  echo "[FATAL] $OUT already exists. lerobot-train refuses to write into it and will abort."
  echo "        Inspect it first: a previous $BB run's checkpoints may be in there."
  echo "        Then either move it aside or delete it DELIBERATELY -- this script will not."
  du -sh "$OUT" 2>/dev/null; find "$OUT" -maxdepth 2 -name pretrained_model 2>/dev/null | head
  exit 1
fi

# --- wandb (runs go online so the loss curve can be watched live) ------------
# Project/run names are fixed by the bench lead: project nero_backbone_bench,
# run name = the backbone id (LeRobot uses cfg.job_name as the wandb run name).
# The entity comes from $WANDB_ENTITY (see env.example.sh); LeRobot picks it up
# from the environment, so there is no --wandb.entity flag below.
# Known red herring: wandb.Api() in this venv (0.27.2) errors "relogin required",
# but wandb.init(mode=online) works fine -- don't let that push you to offline+sync.
# The key lives in ~/.netrc; export it explicitly because wandb-core does not
# always pick netrc up when the *_PROXY vars are set.
WANDB_PROJECT_NAME=nero_backbone_bench
if [[ -z "${WANDB_API_KEY:-}" && -r "$HOME/.netrc" ]]; then
  export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc")
fi

case "$BB" in
  act)
    # Official LeRobot/ACT recipe = the train-config defaults (batch 8, 100k steps,
    # chunk_size 100, lr 1e-5, resnet18). Trained from scratch: ACT has no base model.
    # NOTE: ACTConfig.get_scheduler_preset() returns None -- ACT runs a constant LR and
    # has no scheduler_decay_steps field, so passing that flag here is a hard error.
    POLICY_ARGS=(--policy.type=act)
    EXTRA_ARGS=()
    STEPS=100000
    BATCH=8
    SAVE_FREQ=100000   # ACT ran with a single end-of-training save
    ;;
  smolvla)
    # Official recipe from docs/source/smolvla.mdx: smolvla_base, batch 64, 20k steps.
    # hardware_guide.mdx recommends unfreezing both for real finetuning.
    # scheduler_decay_steps defaults to 30k; left at that while training only 20k the LR
    # would never finish decaying, so pin it to STEPS (docs/source/hardware_guide.mdx).
    # smolvla_base was pretrained with 3 cameras named camera1/2/3; b2 has 2 named
    # cam_high/cam_wrist, so without a rename + one empty camera the run dies at
    # startup with "Feature mismatch between dataset/environment and policy config".
    # --rename_map is a TOP-LEVEL arg (not --policy.rename_map) and gets baked into the
    # saved preprocessor, so lerobot_predict.py picks it up from the checkpoint.
    POLICY_ARGS=(--policy.path=lerobot/smolvla_base
                 --policy.freeze_vision_encoder=false
                 --policy.train_expert_only=false
                 --policy.scheduler_decay_steps=20000
                 --policy.empty_cameras=1)
    EXTRA_ARGS=(--rename_map='{"observation.images.cam_high":"observation.images.camera1","observation.images.cam_wrist":"observation.images.camera2"}')
    STEPS=20000
    BATCH=64
    # Intermediate checkpoints, unlike ACT (single save at 100k).
    # ACT's overfit, measured like-for-like (same ckpt, eval mode, raw units, K=8, on
    # the TRAIN split -- scripts/lerobot_train_split_eval.py): train MAE 1.2457 vs val
    # 2.6665 = 2.14x. (The earlier "1.7x" was wrong: it compared the TRAIN-mode
    # normalized running l1, dropout+aug live, against an eval-mode raw-unit val MAE,
    # so it mixed overfit with a units+mode discrepancy -- and it UNDERSTATED the gap.)
    # This leg is more exposed, not less: 20k x 64 = 33.7 epochs over the 38025 train
    # frames vs ACT's 100k x 8 = 21.0. So chart the curve.
    # DISCIPLINE: these 4 points are a DIAGNOSTIC, never a selection pool. The delivered
    # row is the FINAL checkpoint, pre-committed without looking at val -- picking
    # best-of-4 by val MAE is model selection on the leaderboard set (optimistic bias
    # growing with candidate count, and it breaks comparability with legs reporting
    # their final). If the curve's optimum is mid-run, that is a footnote, not the row.
    # 4 saves x (1.8G params + 5.4G optimizer) ~ 29G worst case. The reaper below runs
    # only AFTER the run, so treat 29G as held for the whole 3-4h. /home2 free was 184G
    # at 04:30 and 170G at 05:20 while OFT writes ~15G/save -- re-check `df -h /home2`
    # at launch rather than trusting either number, and drop SAVE_FREQ to 10000 (2 saves,
    # ~15G) if the margin has gone.
    SAVE_FREQ=5000
    ;;
  *) echo "unknown backbone: $BB" >&2; exit 1 ;;
esac

LOG=$BENCH/logs/train_${BB}_$(date +%Y%m%d_%H%M%S).log
echo "[train] $BB on GPU $GPU -> $OUT (steps=$STEPS batch=$BATCH), log=$LOG"

CUDA_VISIBLE_DEVICES=$GPU HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
"$VENV/bin/lerobot-train" \
  "${POLICY_ARGS[@]}" "${EXTRA_ARGS[@]}" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$ROOT" \
  --dataset.episodes="$EPS" \
  --batch_size=$BATCH \
  --steps=$STEPS \
  --save_freq=$SAVE_FREQ \
  --output_dir="$OUT" \
  --job_name=${BB} \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --wandb.project="$WANDB_PROJECT_NAME" \
  --wandb.mode=online \
  --wandb.disable_artifact=true \
  2>&1 | tee "$LOG"

# disable_artifact=true above: lerobot_train.py calls wandb_logger.log_policy() on
# every checkpoint save, which would upload the whole model (450 MB for SmolVLA) to
# the Hub-side artifact store. We only want the loss curves, and the weights stay local.
#
# The run URL is printed once at startup ("Track this run --> ..."), so when this is
# launched with nohup, grab it with:
#   grep -m1 -o 'https://wandb.ai/[^ ]*' <log> | sed 's/\x1b\[[0-9;]*m//g'
grep -m1 -o "https://wandb.ai/[^ ]*" "$LOG" | sed -e 's/\x1b\[[0-9;]*m//g' -e 's/^/[wandb] /' || true

# CONTRACT.md disk rule: keep params/, drop optimizer state. training_state/ is the
# optimizer+scheduler+rng blob (as large as the model, sometimes larger) and is only
# needed to RESUME -- inference reads pretrained_model/ alone.
find "$OUT/checkpoints" -maxdepth 2 -name training_state -type d -exec rm -rf {} + 2>/dev/null || true
echo "[ckpt] $(du -sh "$OUT" 2>/dev/null)"
echo "[ckpt] predict with: --ckpt $OUT/checkpoints/last/pretrained_model"
