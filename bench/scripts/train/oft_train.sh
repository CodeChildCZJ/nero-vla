#!/bin/bash
# Fine-tune OpenVLA-OFT on NERO b2 train_episodes (111 ep) for the backbone bench.
#   usage: oft_train.sh <gpu_id> [max_steps] [batch_size] [grad_accum] [save_freq] [draccus args...]
#   probe (no checkpoint, offline wandb):
#     WANDB_MODE=offline oft_train.sh 4 30 8 4 999999
#
# Recipe follows the official OFT fine-tuning recipe (README / LIBERO.md):
#   parallel decoding + action chunking + continuous actions + L1 regression, 2 camera images,
#   proprio input, LoRA r32, lr 5e-4. FiLM is off (that is "OFT+", for multi-instruction language
#   grounding; NERO b2 is a single fixed instruction).
# Deviation from official: 1 GPU instead of 8, so the effective batch is made up with
# grad_accumulation_steps and the step count is scaled to the bench budget (recorded in the log).
set -euo pipefail
: "${NERO_ROOT:?please 'source env.sh' first (see env.example.sh at the repo root)}"

BENCH="$NERO_ROOT/bench"
REPO=${NERO_ROOT}/third_party/openvla-oft
PY=${NERO_ROOT}/third_party/openvla-oft/.venv/bin/python

GPU=${1:?usage: oft_train.sh <gpu_id> [max_steps] [batch_size] [grad_accum] ...}
STEPS=${2:-15000}
BS=${3:-8}
# Official recipe is batch 4 x 8 GPUs = effective 32. We have 1 GPU, so per-device batch and
# accumulation trade off freely as long as their product stays 32; default keeps that invariant.
ACCUM=${4:-$((32 / BS))}
# Read save_freq BEFORE shifting: after `shift`, $5 refers to a different argument, so reading it
# later silently ignored the caller's save_freq (always fell back to STEPS/4) and leaked the
# leftover positional into the python command line.
SAVE=${5:-$((STEPS / 4))}
shift 5 2>/dev/null || shift $#

EFF=$((BS * ACCUM))
if [[ $EFF -ne 32 ]]; then
  echo "WARNING: effective batch $BS x $ACCUM = $EFF, official recipe is 32" >&2
fi

# Checkpointing: finetune_nero.py saves only on `log_step % save_freq == 0` and then breaks out of
# the loop at max_steps WITHOUT a final save. So if max_steps is not a multiple of save_freq, the
# whole run ends with nothing newer than the last multiple -- a 12h run could end with no usable
# checkpoint at all. Default to 4 saves per run so the last one lands exactly on max_steps.
if [[ $SAVE -gt $STEPS ]]; then
  echo "NOTE: save_freq $SAVE > max_steps $STEPS -> no checkpoint will be written (probe mode)" >&2
elif [[ $((STEPS % SAVE)) -ne 0 ]]; then
  echo "FATAL: max_steps $STEPS is not a multiple of save_freq $SAVE -> the final step would NOT be" >&2
  echo "       saved and the tail of the run would be lost. Pick divisible values." >&2
  exit 1
fi

# Official ALOHA recipe decays LR 10x at half the run (50000 of 100005) -- the docs call this out as
# what makes the L1 loss spike down. The knob defaults to 100_000, so on a short run the milestone
# would never fire and we would train flat at 5e-4 the whole way; keep the official 1/2 ratio.
DECAY=$((STEPS / 2))

export CUDA_VISIBLE_DEVICES=$GPU
export ROBOT_PLATFORM=NERO          # -> NERO_CONSTANTS: chunk 8, action/proprio dim 8, BOUNDS norm
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1             # weights are local; must not gate wandb (different lib)
export PYTHONPATH=$REPO

# wandb online so the loss curve can be watched live. Key lives in ~/.netrc; export explicitly
# because wandb-core does not always pick netrc up when the *_PROXY vars are set.
# The entity comes from $WANDB_ENTITY (see env.example.sh).
if [[ -z "${WANDB_API_KEY:-}" && -r "$HOME/.netrc" ]]; then
  export WANDB_API_KEY=$(awk '/api.wandb.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc")
fi

TS=$(date +%Y%m%d_%H%M%S)
LOG=$BENCH/logs/oft/train_${TS}.log
mkdir -p "$BENCH/logs/oft" "$BENCH/runs"

cd "$REPO"   # check_model_logic_mismatch() walks ./prismatic/
echo "=== OFT NERO b2 train | GPU $GPU | steps $STEPS | batch $BS x $ACCUM accum = $EFF | save@$SAVE | lr decay @ $DECAY | wandb ${WANDB_MODE:-online} | log $LOG ==="

exec $PY -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/finetune_nero.py \
  --vla_path "${NERO_CKPT}/openvla-7b" \
  --data_root_dir "$BENCH/data/b2_oft" \
  --dataset_name nero_b2 \
  --run_root_dir "$BENCH/runs" \
  --run_id_override openvla_oft \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size "$BS" \
  --grad_accumulation_steps "$ACCUM" \
  --learning_rate 5e-4 \
  --lr_warmup_steps 100 \
  --num_steps_before_decay "$DECAY" \
  --max_steps "$STEPS" \
  --save_freq "$SAVE" \
  --save_latest_checkpoint_only True \
  --image_aug True \
  --use_lora True \
  --lora_rank 32 \
  --merge_lora_during_training True \
  --num_workers 8 \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_project nero_backbone_bench \
  "$@" 2>&1 | tee "$LOG"
