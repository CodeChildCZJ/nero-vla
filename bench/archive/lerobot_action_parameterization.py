#!/usr/bin/env python3
"""What do ACT / SmolVLA actually regress on: absolute joint angles or arm-delta?

This is a leaderboard COLUMN (gr00t-n17 / team-lead): the pass line IS the delta=0
predictor, so a delta-arm stack clears it nearly for free (const-relative 3.064 vs
pass line 3.072) while an absolute stack must learn the state->action identity map
(const-absolute 21.189, 6.9x worse). Rows from the two families are not directly
comparable on total MAE.

n17 could only grep the predict path, which cannot see how the TRAINING target is
built -- LeRobot really does ship relative-action machinery
(`processor/relative_action_processor.py::to_relative_actions`, "relative = action
- state (for masked dims)"), wired into pi0 / pi05 / pi0_fast / groot behind
`config.use_relative_actions`. So "LeRobot is conventionally absolute" is not good
enough; this measures it.

The measurement reconstructs the TRAIN-time tensor: dataset -> preprocessor ->
batch[ACTION], which is literally the tensor ACT's `F.l1_loss(batch[ACTION],
actions_hat)` and SmolVLA's flow target `u_t = noise - actions` are computed
against. It is unnormalized back to raw units and compared against two
hypotheses:

    H_abs   : target == raw dataset action column
    H_delta : target == raw action - observation.state (broadcast over the chunk)

Both are scored, so the answer is a comparison and not an assertion -- and the
losing hypothesis' residual doubles as the must-differ control proving the test
could have distinguished them.

CPU only, no GPU, no model weights.
"""

import os
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

BENCH = pathlib.Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="act")
    ap.add_argument("--repo-id", default="local/pick_pink_sponge_b2_bench")
    ap.add_argument("--root", default=os.environ.get("NERO_DATA_V30", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2_bench")))
    ap.add_argument("--split", default=str(BENCH / "data" / "split.json"))
    ap.add_argument("--n", type=int, default=64, help="training samples to check")
    args = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import ACTION, OBS_STATE

    split = json.loads(pathlib.Path(args.split).read_text())
    train_eps = sorted(split["train_episodes"])

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.pretrained_path = args.ckpt
    cfg.device = "cpu"

    # --- static facts about the pipeline, printed so the numeric result is читаемый
    print("=== config-level facts ===")
    print(f"  policy type                : {getattr(cfg, 'type', type(cfg).__name__)}")
    print(f"  normalization_mapping      : {cfg.normalization_mapping}")
    for flag in ("use_relative_actions", "adapt_to_pi_aloha"):
        print(f"  {flag:<27}: {getattr(cfg, flag, '<absent from this config class>')}")

    # The training loop feeds the dataset through delta_timestamps so ACTION is a
    # chunk; rebuild that exactly rather than reading a single frame.
    delta_ts = None
    if getattr(cfg, "chunk_size", None):
        delta_ts = {ACTION: [i / 30.0 for i in range(cfg.chunk_size)]}
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=train_eps, delta_timestamps=delta_ts)
    print(f"\n[dataset] TRAIN split: {ds.num_episodes} eps / {ds.num_frames} frames")

    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    steps_pre = [type(s).__name__ for s in pre.steps]
    print(f"[pipeline] preprocessor steps : {steps_pre}")
    print(f"[pipeline] postprocessor steps: {[type(s).__name__ for s in post.steps]}")
    rel = [s for s in steps_pre if "elative" in s or "elta" in s]
    print(f"[pipeline] relative/delta steps present: {rel if rel else 'NONE'}")

    # --- pull the normalizer's action stats so we can invert by hand -------------
    norm_stats = None
    for s in pre.steps:
        st = getattr(s, "stats", None)
        if st and ACTION in st:
            norm_stats = st[ACTION]
            break
    if norm_stats is None:
        sys.exit("could not find ACTION stats on the normalizer -- cannot invert")
    a_mean = np.asarray(norm_stats["mean"], dtype=np.float64).ravel()
    a_std = np.asarray(norm_stats["std"], dtype=np.float64).ravel()
    print(f"\n[action stats baked in ckpt] mean = {np.round(a_mean, 3).tolist()}")
    print(f"[action stats baked in ckpt] std  = {np.round(a_std, 3).tolist()}")
    print("  (a delta-parameterized target would sit near mean~0 on every joint;")
    print("   absolute joint angles carry each joint's mechanical offset)")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)

    res_abs, res_delta, mags = [], [], []
    for i in idx:
        item = ds[int(i)]
        raw_action = item[ACTION].numpy().astype(np.float64)      # (chunk, 8) raw units
        raw_state = item[OBS_STATE].numpy().astype(np.float64)    # (8,)
        item = dict(item)
        item["task"] = item.get("task", "pick the pink sponge and place it in the blue bucket")
        batch = pre(item)                                          # what the loss sees
        tgt = batch[ACTION]
        tgt = tgt.detach().cpu().numpy().astype(np.float64)
        if tgt.ndim == 3:
            tgt = tgt[0]
        # invert MEAN_STD by hand (the postprocessor is built for policy OUTPUT,
        # feeding it the target tensor would be a different code path)
        tgt_raw = tgt[:, : a_mean.size] * a_std + a_mean

        h_abs = raw_action
        h_delta = raw_action - raw_state[None, :]
        k = min(tgt_raw.shape[0], h_abs.shape[0])
        res_abs.append(np.abs(tgt_raw[:k] - h_abs[:k]).max())
        res_delta.append(np.abs(tgt_raw[:k] - h_delta[:k]).max())
        mags.append(np.abs(h_abs[:k]).mean())

    res_abs, res_delta = np.array(res_abs), np.array(res_delta)
    print(f"\n=== training-target hypothesis test ({len(idx)} train samples) ===")
    print(f"  mean |raw action|                      : {np.mean(mags):10.4f}  (scale reference)")
    print(f"  max residual  vs H_abs  (action)       : {res_abs.max():10.3e}")
    print(f"  max residual  vs H_delta(action-state) : {res_delta.max():10.3e}")
    sep = res_delta.max() / max(res_abs.max(), 1e-12)
    print(f"  separation (H_delta / H_abs)           : {sep:10.3e}")

    ok = res_abs.max() < 1e-3 and res_delta.max() > 1.0
    print()
    if ok:
        print("VERDICT: ABSOLUTE. The tensor the loss regresses on is the raw dataset action")
        print("         column (joint positions), with NO proprio subtraction anywhere in the")
        print("         pipeline. The H_delta residual is the must-differ control: it is")
        print(f"         {sep:.1e}x larger, so this test could have told the two apart.")
    else:
        print("VERDICT: INCONCLUSIVE or DELTA -- inspect the residuals above before reporting.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
