#!/usr/bin/env python3
"""Same-mode train-split MAE: separate genuine overfit from a train/val METRIC MISMATCH.

Why (openvla-oft's methodological correction, which I accept): I reported an ACT
"train/val gap 1.73x" using the training-time running l1 loss vs the val MAE.
Those are NOT the same measurement -- train l1 is logged in TRAIN mode (dropout
0.1 active, augmentation on) on NORMALIZED actions, while val MAE is an eval-mode
clean forward in RAW joint units. The gap therefore mixes overfit with a units +
mode discrepancy, and the two cannot be separated from those two numbers.

The cheap fix is to measure the SAME quantity on the train split: identical
eval-mode forward, identical raw-unit MAE, identical K=8 anchor construction,
same checkpoint. Then `train_MAE` vs `val_MAE` is a like-for-like comparison and
the ratio means what it says.

TOOL VALIDATION FIRST (--eps val): this script rebuilds anchors itself rather
than importing the frozen builder (which reads the val split only). A rebuild is
only trustworthy if it reproduces the frozen artifact, so `--eps val` asserts
bit-equality against `val_anchors.npz` before any train number is believed. The
frozen harness is READ, never written.

Anchor construction mirrors scripts/prepare/build_val_anchors.py exactly: K=8, STRIDE=5,
gt = action[t:t+K], state = observation.state[t], read from the ORIGINAL
per-episode parquets (the bench copy is v3.0-packed into a single file).

Note the pass line is split-dependent: hold-state on TRAIN anchors is NOT the
3.072 val pass line, so it is recomputed here and both are printed. Comparing a
train MAE against the val pass line would be the apples/oranges error this script
exists to avoid.
"""

import os
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

BENCH = pathlib.Path(__file__).resolve().parents[1]
B2_RAW = pathlib.Path(os.environ.get("NERO_DATA", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2")))
K = 8
STRIDE = 5
PROMPT = "pick the pink sponge and place it in the blue bucket"


def build_anchors(eps: list[int]):
    """Byte-for-byte the logic of scripts/prepare/build_val_anchors.py, over given episodes."""
    e_out, f_out, gts, states = [], [], [], []
    for e in eps:
        df = pd.read_parquet(B2_RAW / f"data/chunk-000/episode_{e:06d}.parquet")
        act = np.stack(df["action"].to_numpy())
        st = np.stack(df["observation.state"].to_numpy())
        for t in range(0, len(act) - K, STRIDE):
            e_out.append(e)
            f_out.append(t)
            gts.append(act[t : t + K])
            states.append(st[t])
    return (
        np.array(e_out, np.int32),
        np.array(f_out, np.int32),
        np.stack(gts).astype(np.float32),
        np.stack(states).astype(np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default="act")
    ap.add_argument("--eps", choices=["train", "val"], default="train")
    ap.add_argument("--device", default="cpu")
    # DEFAULT IS 0 = FULL POPULATION. It used to be 400, and openvla-oft caught what that
    # cost: a 400-anchor draw carries sd 0.063 on the ACT val MAE (95% window +-0.124), so
    # the seed-0 draw read 2.6665 against the full-1397 truth 2.7272, and the train/val
    # ratio inherited ~+-0.13 -- the same order as the real gaps between backbones. Worse,
    # BOTH hold lines were subsample artifacts (train 3.352 / val 3.1188 vs the full 3.0799
    # / 3.0720). A default that silently narrows the population is worse than a missing
    # flag: it returns a plausible number of the wrong thing, with nothing in the output to
    # say so. Third instance on this bench (gr00t-n17's --n-anchors 200 inflated the
    # gripper floor +33%), so the fix is theirs: default all, WARN on any subset.
    ap.add_argument("--n", type=int, default=0, help="anchors to evaluate (0 = ALL, the default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repo-id", default="local/pick_pink_sponge_b2_bench")
    ap.add_argument("--root", default=os.environ.get("NERO_DATA_V30", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2_bench")))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path[:0] = [str(BENCH / "scripts" / "predict"), str(BENCH / "scripts" / "score"),
                    str(BENCH / "archive")]   # was one flat scripts/ dir
    from lerobot_predict import build_anchor_index

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    split = json.loads((BENCH / "data" / "split.json").read_text())
    eps = sorted(split[f"{args.eps}_episodes"])
    episodes, frames, gt, state = build_anchors(eps)
    print(f"[anchors] {args.eps}: {len(episodes)} anchors over {len(eps)} episodes, K={K} stride={STRIDE}")

    # ---- tool validation: the val rebuild must bit-match the frozen artifact ----
    if args.eps == "val":
        A = np.load(BENCH / "data" / "val_anchors.npz")
        checks = {
            "episodes": np.array_equal(episodes, A["episodes"]),
            "frames": np.array_equal(frames, A["frames"]),
            "gt": np.array_equal(gt, A["gt"]),
            "state": np.array_equal(state, A["state"]),
        }
        print(f"[validate] vs frozen val_anchors.npz: {checks}")
        if not all(checks.values()):
            sys.exit("FAIL: rebuild does not reproduce the frozen anchors -- mirror logic is wrong")
        print("[validate] PASS -- rebuild reproduces the frozen harness bit-for-bit")

    # ---- subsample ----
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(episodes))
    subsampled = bool(args.n) and args.n < len(idx)
    if subsampled:
        idx = np.sort(rng.choice(idx, size=args.n, replace=False))
        print(f"[WARNING] SUBSAMPLED {args.n}/{len(episodes)} anchors (seed {args.seed}). Both the "
              f"model MAE and this split's hold line are then single draws: measured spread on "
              f"ACT/val is sd 0.063 at n=400, i.e. a train/val RATIO carries ~+-0.13 at 95%. "
              f"Do NOT compare a subsampled ratio against another leg's full-population ratio. "
              f"Use --n 0 (the default) for anything that goes in a report.", flush=True)
    episodes_s, frames_s, gt_s, state_s = episodes[idx], frames[idx], gt[idx], state[idx]
    print(f"[eval] running {len(idx)} anchors on {args.device}")

    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=eps)
    rel_idx = build_anchor_index(ds, episodes_s, frames_s)

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.pretrained_path = args.ckpt
    cfg.device = args.device
    rename_map = None
    tc = pathlib.Path(args.ckpt) / "train_config.json"
    if tc.exists():
        rename_map = json.loads(tc.read_text()).get("rename_map") or None
    policy = make_policy(cfg, ds_meta=ds.meta, rename_map=rename_map)
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=args.ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": args.device}},
    )
    policy.eval()  # THE point of this script: same mode as the val measurement

    preds, t0 = [], time.time()
    for n, ri in enumerate(rel_idx):
        item = ds[ri]
        item["task"] = item.get("task", PROMPT)
        policy.reset()
        torch.manual_seed(args.seed * 1_000_003 + int(idx[n]))
        chunk = post(policy.predict_action_chunk(pre(item)))
        preds.append(chunk.squeeze(0).float().cpu().numpy())
        if n % 50 == 0:
            print(f"  [{n}/{len(rel_idx)}] {time.time() - t0:.1f}s", flush=True)
    pred = np.stack(preds).astype(np.float32)[:, :K, :]
    print(f"[done] {pred.shape} in {time.time() - t0:.1f}s")

    # ---- score, with score.py's exact definitions -------------------------------
    err = np.abs(pred - gt_s)
    hold = np.abs(np.broadcast_to(state_s[:, None, :], gt_s.shape) - gt_s)
    res = {
        "split": args.eps,
        "n_anchors": int(len(idx)),
        # A reader cannot tell a subsample from a full run by looking at n_anchors alone
        # (they would need to know the population), so state both explicitly.
        "n_anchors_full_population": int(len(episodes)),
        "subsampled": subsampled,
        "subsample_seed": int(args.seed) if subsampled else None,
        "K": K,
        "mae": float(err.mean()),
        "arm": float(err[..., :7].mean()),
        "grip": float(err[..., 7].mean()),
        "hold_mae": float(hold.mean()),
        "hold_arm": float(hold[..., :7].mean()),
        "hold_grip": float(hold[..., 7].mean()),
    }
    print(f"\n=== {args.backbone} on {args.eps} split ({len(idx)} anchors, K={K}, eval mode) ===")
    print(f"  MAE            {res['mae']:.4f}   arm {res['arm']:.4f}   grip {res['grip']:.4f}")
    print(f"  hold-state     {res['hold_mae']:.4f}   arm {res['hold_arm']:.4f}   grip {res['hold_grip']:.4f}")
    print(f"  margin vs its OWN split's hold line: {res['hold_mae'] - res['mae']:+.4f}")
    print("  (this split's hold line is NOT the 3.072 val pass line -- do not cross-compare)")

    out = pathlib.Path(args.out) if args.out else BENCH / "logs" / f"trainsplit_eval_{args.backbone}_{args.eps}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"[write] {out}")

    # Keep the per-anchor arrays, not just the summary. A train/val RATIO of two point
    # estimates has no scale (gr00t-n15), and the fix is a cluster jackknife over EPISODES
    # of the ratio itself -- which needs per-episode errors on both splits. Recomputing
    # them means re-taking a GPU, so 2 MB of npz here buys that analysis for free later.
    # `episodes` is the cluster label; keep it alongside so the jackknife cannot mis-group.
    pnpz = out.with_name(f"trainsplit_preds_{args.backbone}_{args.eps}.npz")
    np.savez(pnpz, pred=pred, gt=gt_s, state=state_s, episodes=episodes_s, frames=frames_s,
             anchor_idx=idx.astype(np.int32))
    print(f"[write] {pnpz}  (per-anchor arrays for the ratio jackknife)")


if __name__ == "__main__":
    main()
