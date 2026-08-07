#!/usr/bin/env python
"""Generic HF-LeRobot predict scaffold for the NERO b2 backbone bench (SmolVLA / ACT).

Produces the single deliverable required by CONTRACT.md:
    preds/preds_<bb>.npz   key 'pred'   shape (A, K', 8),  A=1397, K'>=8
row order strictly aligned to val_anchors.npz anchor order.

--- Why this reads observations through LeRobotDataset (do not "simplify" this) ---
The anchor observation must be byte-identical to what the policy saw at train time:
same AV1 video decode, same float32 [0,1] CHW image layout, same state dtype, same
task string. Re-decoding the mp4s by hand is how you silently get a distribution
shift that shows up as a bogus MAE. So we open the *same* LeRobotDataset class and
index the exact (episode, frame) anchors.

--- Unnormalization (the #1 trap in CONTRACT.md) ---
In LeRobot >=0.6 normalization lives OUTSIDE the policy, in processor pipelines:
    preprocessor : rename -> add_batch_dim -> [tokenize] -> to_device -> normalize
    postprocessor: unnormalize -> to_cpu
`policy.predict_action_chunk()` therefore returns NORMALIZED actions. Feeding those
straight to score.py would compare normalized units against raw-degree GT.
We always route the output through `postprocessor`, and we load the postprocessor
FROM THE CHECKPOINT so the unnormalize stats are exactly the ones training used.

--- Train/val discipline (CONTRACT.md: val frames must never be trained on) ---
This script only ever touches val episodes. Training must restrict episodes via
    lerobot-train --dataset.episodes='[<the 111 train ids from split.json>]'
which maps to LeRobotDataset(episodes=[...]). NOTE: LeRobot's normalization stats
(`dataset.meta.stats`) are loaded from meta/stats.json and are NOT filtered by that
`episodes=` argument -- they are aggregated over all 131 episodes. That leaks val
statistics into the normalizer, so we pre-bake train-only stats into the working
copy of the dataset; see prepare_b2_v30.py.
"""

import argparse
import hashlib
import io
import json
import os
import pathlib
import time

import numpy as np
import torch

# 仓库自包含:从本文件位置推导。bench/scripts/predict/*.py -> parents[2] == bench/
BENCH = pathlib.Path(__file__).resolve().parents[2]
# b2 数据集本体(不公开,格式见 docs/DATA.md);默认取 LeRobot 本地缓存,可用 NERO_DATA 覆盖。
NERO_DATA = pathlib.Path(os.environ.get(
    "NERO_DATA", pathlib.Path.home() / ".cache/huggingface/lerobot/local/pick_pink_sponge_b2"))
# v3.0-converted working copy (the v2.1 original is read-only per CONTRACT.md).
# Written by prepare_b2_v30.py next to the source dataset; NERO_DATA_BENCH overrides.
DEFAULT_ROOT = os.environ.get("NERO_DATA_BENCH", str(NERO_DATA) + "_bench")
DEFAULT_REPO_ID = "local/pick_pink_sponge_b2_bench"
PROMPT = "pick the pink sponge and place it in the blue bucket"


def build_anchor_index(ds, episodes, frames):
    """Map each (episode_index, frame_index) anchor -> relative row index in ds.

    `ds[i]` takes a *relative* index into the episode-filtered view, while the
    parquet 'index' column is absolute, so we resolve via the two id columns
    directly. This reads parquet columns only -- no video decode.
    """
    ep_col = ds.hf_dataset.data.column("episode_index").to_numpy()
    fr_col = ds.hf_dataset.data.column("frame_index").to_numpy()
    key2rel = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ep_col, fr_col, strict=True))}
    missing = [(int(e), int(f)) for e, f in zip(episodes, frames, strict=True) if (int(e), int(f)) not in key2rel]
    if missing:
        raise KeyError(f"{len(missing)} anchors not found in dataset, e.g. {missing[:5]}")
    return [key2rel[(int(e), int(f))] for e, f in zip(episodes, frames, strict=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["smolvla", "act"], help="writes preds_<bb>.npz")
    ap.add_argument("--ckpt", default=None,
                    help="path to .../pretrained_model dir. Omit + --random to shape-test the pipeline.")
    ap.add_argument("--random", action="store_true",
                    help="scaffold mode: randomly-initialised policy, validates shapes/plumbing only")
    ap.add_argument("--device", default="cuda:4")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N anchors")
    ap.add_argument("--subset", default=None,
                    help=".npy of anchor indices to run (for the sampling-variance footnote). "
                         "Rows are seeded by their GLOBAL anchor index, so a subset run reproduces "
                         "the corresponding rows of a full run bit-for-bit at the same --seed.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-anchor-change", action="store_true",
                    help="bypass the val_anchors.npz md5 pin (only after a board-agreed re-freeze)")
    args = ap.parse_args()

    if not args.random and not args.ckpt:
        ap.error("need --ckpt (a trained checkpoint) or --random (shape test)")

    # Resolve the output path HERE, before any model or dataset work, so the two
    # guards below can refuse *before* paying for a load (a must-pass test of a gate
    # whose pass branch loads a 7B model is expensive -- openvla-oft ate that cost).
    out = pathlib.Path(args.out) if args.out else BENCH / "preds" / f"preds_{args.backbone}.npz"

    # GUARD 1 -- scaffold must never be able to land in preds/.
    # `--random` runs the DEFAULT full anchor set through a RANDOMLY-INITIALISED
    # policy, so it emits a (1397, K', 8) npz with correct shape, correct dtype,
    # correct row order and finite values -- i.e. it passes every structural check
    # score.py applies, and would be RANKED as if it were a real backbone. With the
    # default --out it also OVERWRITES the real delivery. Shape and finiteness gates
    # cannot catch this: the file is structurally perfect and semantically noise.
    # (openvla-oft found the same hole on their leg; this is the LeRobot instance.)
    if args.random and out.resolve().parent == (BENCH / "preds").resolve():
        raise SystemExit(
            f"REFUSING scaffold write: --random (randomly-initialised weights) targets "
            f"{out}, which is inside the delivery directory. A scaffold npz is "
            f"shape-valid and model-independent, so score.py would rank noise and the "
            f"real row would be gone. Re-run with --out pointing outside preds/."
        )

    # GUARD 1b -- a real backbone must not land under ANOTHER backbone's delivery name.
    # This script serves two backbones, so `--backbone smolvla --ckpt <smolvla> --out
    # preds/preds_act.npz` (a stale --out copied from the ACT command line) silently
    # overwrites a delivered row with predictions from a different model: right shape,
    # right row order, finite, and scored without complaint. The ckpt<->backbone half of
    # this is caught by the normalization assert below (the two configs differ), but the
    # OUTPUT-NAME half has nothing else guarding it. Exact stem equality, not a prefix
    # test -- openpi-base measured that `'preds_pi05'.startswith('preds_pi0')` is True,
    # so a prefix judge passes exactly the swap it exists to block.
    if not args.random and out.resolve().parent == (BENCH / "preds").resolve():
        expect_stem = f"preds_{args.backbone}"
        if out.name != f"{expect_stem}.npz":
            raise SystemExit(
                f"REFUSING cross-backbone delivery write: --backbone {args.backbone} would "
                f"write {out.name} inside preds/, but this backbone's delivery name is "
                f"{expect_stem}.npz. Writing another leg's name here replaces a delivered "
                f"row with a structurally perfect file from the wrong model. Fix --out, or "
                f"aim outside preds/ for scratch runs."
            )

    # GUARD 2 -- val_anchors.npz is the row-alignment contract for EVERY leg.
    # Structural checks (1397 rows / K=8 / dtype / finite) all pass on a rebuilt or
    # reordered anchor file, so predict could align rows to version A while score.py
    # scores against version B: a plausible wrong MAE with no error anywhere.
    # Pin the content, not the mtime -- the mtime is 06:52 because openvla-oft
    # swapped the file out and back while testing a gate; content was byte-restored.
    # Verify-then-load is two reads of a file that other legs demonstrably rewrite on THIS
    # host (oft swapped it out and back at 06:52), so the bytes can change between the hash
    # and the load. (An earlier version of this comment blamed 145<->147 syncthing; probed
    # 2026-08-05 with tripwire files in both directions -- nothing propagated either way in
    # ~9 min, and 147 has no .stignore at all. Same-host concurrency is the real window.)
    # Read ONCE,
    # hash THOSE bytes, and load the anchors from the same in-memory buffer below, so the
    # thing verified is literally the thing used (no TOCTOU window at all).
    VAL_ANCHORS_MD5 = "cc34dda2a99ade1011c192e2b3e4e343"   # 414998 B, independently confirmed by oft
    _anchor_bytes = (BENCH / "data" / "val_anchors.npz").read_bytes()
    _am = hashlib.md5(_anchor_bytes).hexdigest()
    if _am != VAL_ANCHORS_MD5 and not args.allow_anchor_change:
        raise SystemExit(
            f"[FATAL] val_anchors.npz md5 {_am} != pinned {VAL_ANCHORS_MD5}. The frozen "
            f"anchor set is the cross-leg row contract; a silently rebuilt one yields a "
            f"wrong-but-plausible MAE. Pass --allow-anchor-change only if the board "
            f"agreed to re-freeze."
        )

    torch.manual_seed(args.seed)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors

    anchors = np.load(io.BytesIO(_anchor_bytes))   # the exact bytes GUARD 2 hashed
    episodes, frames = anchors["episodes"], anchors["frames"]
    K = int(anchors["K"])
    A = len(episodes)
    # glob_idx = position in the FULL anchor array; drives both the output row order
    # and the per-anchor RNG seed, so subsets stay comparable to full runs.
    glob_idx = np.arange(A)
    if args.subset:
        glob_idx = np.load(args.subset).astype(int)
        if glob_idx.min() < 0 or glob_idx.max() >= A:
            raise SystemExit(f"subset indices out of range [0,{A})")
    if args.limit:
        glob_idx = glob_idx[: args.limit]
    episodes, frames = episodes[glob_idx], frames[glob_idx]
    val_eps = sorted(set(int(e) for e in episodes))
    print(f"[anchors] A={A} K={K} over {len(val_eps)} val episodes; running {len(episodes)}")

    # ---- dataset restricted to val episodes (never used for training) ----
    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=val_eps)
    print(f"[dataset] {ds.num_episodes} eps / {ds.num_frames} frames, fps={ds.meta.fps}")
    rel_idx = build_anchor_index(ds, episodes, frames)

    # ---- policy + processors ----
    # Board-adopted (oft, 2026-08-05): absolute-space legs assert the normalization TYPE
    # explicitly. A silent switch to a quantile/min-max mapping does not raise anywhere --
    # it just unnormalizes with different constants, and oft measured that a q01/q99 mixup
    # lands at MAE 3.71, only 0.63 above the 3.072 pass line: indistinguishable from an
    # undertrained model, so the do-nothing flag is NOT a backstop for it. This leg is
    # MEAN_STD on all three keys (ckpt config.json), and preds are compared to RAW GT.
    # Per-backbone, NOT one table: measured from the shipped configs, SmolVLA is
    # VISUAL=IDENTITY (it normalizes images inside the VLM) while ACT is VISUAL=MEAN_STD.
    # A single shared expectation looks tidy and hard-fails the SmolVLA delivery run.
    # STATE/ACTION are the load-bearing keys here -- they set the units of the numbers
    # score.py compares against raw GT; VISUAL is pinned only to catch a recipe swap.
    EXPECTED_NORM_BY_BACKBONE = {
        "act":     {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        "smolvla": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
    }
    EXPECTED_NORM = EXPECTED_NORM_BY_BACKBONE.get(args.backbone)
    if EXPECTED_NORM is None:
        raise SystemExit(
            f"[FATAL] no normalization-type expectation registered for backbone "
            f"'{args.backbone}'. Add one (measured from its config, not assumed) before "
            f"delivering a row -- an unasserted leg is exactly the q99-mixup exposure."
        )

    def assert_norm_types(cfg_obj):
        got = {k: getattr(v, "value", str(v)) for k, v in cfg_obj.normalization_mapping.items()}
        if got != EXPECTED_NORM:
            raise SystemExit(
                f"[FATAL] {args.backbone} normalization_mapping {got} != expected {EXPECTED_NORM}. The "
                f"unnormalization constants decide the units of every predicted action; a "
                f"quantile/min-max mapping yields a wrong-but-plausible MAE that clears the "
                f"pass line. Re-freeze the expectation only if the board changed the recipe."
            )

    if args.random:
        # Shape-test only: config derived from dataset features, weights random.
        cfg = make_policy_config(args.backbone, device=args.device)
        assert_norm_types(cfg)
        policy = make_policy(cfg, ds_meta=ds.meta)
        pre, post = make_pre_post_processors(cfg, dataset_stats=ds.meta.stats)
    else:
        cfg = PreTrainedConfig.from_pretrained(args.ckpt)
        assert_norm_types(cfg)          # the CHECKPOINT's declared mapping, not a default
        cfg.pretrained_path = args.ckpt
        cfg.device = args.device
        # smolvla_base ships input_features named camera1/2/3, b2 has cam_high/cam_wrist,
        # so training is launched with --rename_map. make_policy re-derives features from
        # ds_meta and would raise a feature mismatch without the same map, so recover it
        # from the checkpoint's train_config.json rather than hardcoding it here.
        rename_map = None
        tc = pathlib.Path(args.ckpt) / "train_config.json"
        if tc.exists():
            rename_map = json.loads(tc.read_text()).get("rename_map") or None
            if rename_map:
                print(f"[rename_map] from checkpoint: {rename_map}")
        policy = make_policy(cfg, ds_meta=ds.meta, rename_map=rename_map)
        # Load processors FROM THE CHECKPOINT: carries the exact training-time
        # normalize/unnormalize stats, which is what makes preds comparable to raw GT.
        # The saved device step records a bare "cuda", which resolves to cuda:0 and
        # collides with a policy placed on any other card -- pin it to our device.
        pre, post = make_pre_post_processors(
            cfg, pretrained_path=args.ckpt, dataset_stats=ds.meta.stats,
            preprocessor_overrides={"device_processor": {"device": args.device}},
            postprocessor_overrides={"device_processor": {"device": args.device}},
        )
    policy.eval()

    preds = []
    t0 = time.time()
    for n, ri in enumerate(rel_idx):
        item = ds[ri]
        item["task"] = item.get("task", PROMPT)
        # Policies keep observation/action queues across calls; reset so every anchor
        # is an independent open-loop query rather than a continuation of the last one.
        policy.reset()
        # SmolVLA denoises from random noise, so its prediction depends on RNG state.
        # Reseed per anchor (not once per run) to make each anchor's result independent
        # of iteration order -- reruns and any future batching stay comparable.
        torch.manual_seed(args.seed * 1_000_003 + int(glob_idx[n]))
        batch = pre(item)                              # normalize (+tokenize for smolvla)
        chunk = policy.predict_action_chunk(batch)     # (1, K', 8) NORMALIZED
        chunk = post(chunk)                            # (1, K', 8) raw joint units
        preds.append(chunk.squeeze(0).float().cpu().numpy())
        if n % 200 == 0:
            print(f"  [{n}/{len(rel_idx)}] {time.time() - t0:.1f}s", flush=True)

    pred = np.stack(preds).astype(np.float32)          # (A, K', 8)
    print(f"[done] pred {pred.shape} in {time.time() - t0:.1f}s")
    if pred.shape[1] < K:
        raise SystemExit(f"chunk horizon {pred.shape[1]} < required K={K}")

    # `out` was resolved before the model load so the scaffold/anchor guards could
    # refuse early; do not recompute it here or those guards stop covering this write.

    # Refuse to WRITE a non-finite file into preds/. gr00t-n15 measured that a NaN
    # row does not just spoil its own row: score.py's `mae >= HOLD_MAE` guard is
    # False for NaN (silent), and a NaN sort key breaks Timsort's transitivity, so
    # the WHOLE board misorders -- observed pushing the hold-state pass line to #1.
    # The gate belongs here too, not only in the scorer: a file that never lands
    # cannot corrupt anyone. Scoped to the K prefix score.py actually reads, so it
    # will not fire on padding past the scoring window.
    bad = ~np.isfinite(pred[:, :K, :])
    if bad.any():
        raise SystemExit(
            f"REFUSING TO WRITE {out.name}: {int(bad.sum())}/{bad.size} non-finite cells in the "
            f"K={K} prefix (first at anchor {int(np.argwhere(bad.any(axis=(1, 2)))[0][0])}). "
            "A non-finite preds file misorders the entire leaderboard, not just this row."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    # anchor_idx lets a partial run be re-aligned later; score.py only reads 'pred'.
    np.savez(out, pred=pred, anchor_idx=glob_idx.astype(np.int32))
    print(f"[write] {out}")

    # gt is indexed by glob_idx, so this is valid for full, --limit and --subset runs alike.
    gt = anchors["gt"].astype(np.float32)[glob_idx]
    err = np.abs(pred[:, :K, :] - gt)
    scope = "FULL" if len(glob_idx) == A else f"subset n={len(glob_idx)}"
    print(f"[selfcheck {scope}] MAE={err.mean():.3f}  arm={err[..., :7].mean():.3f}  grip={err[..., 7].mean():.3f}")
    print("[selfcheck] per-joint " + " ".join(f"{v:.2f}" for v in err.mean(axis=(0, 1))))


if __name__ == "__main__":
    main()
