#!/usr/bin/env python3
"""Independent audit of val_anchors.npz against the raw LeRobot parquet.

val_anchors.npz is the only ground truth every one of the 7 backbone legs scores
against, so a silent extraction bug in it would void the whole leaderboard at
once. This script re-derives the anchors from the raw dataset WITHOUT importing
build_val_anchors.py or any lerobot code -- it reads the parquet with pandas and
compares row by row, so a shared helper cannot make both sides wrong the same
way.

Checked, over ALL 1397 anchors (not a sample):
  structural  shapes/dtypes, K, no NaN, episodes subset of split.val_episodes,
              every val episode represented, frames in range, rows sorted
  state       state[i] == observation.state at (episodes[i], frames[i])
  gt          gt[i,k] == action at frame frames[i]+k for k in 0..K-1
              (the action COLUMN, so an accidental state-shift build is caught)
  boundary    frames[i]+K-1 <= last frame of that episode (no cross-episode bleed
              and no clamped/repeated tail rows)
  stride      spacing between consecutive anchors of an episode, and whether the
              tail was truncated consistently across all 20 episodes
"""
import os
import argparse
import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd

BENCH = pathlib.Path(__file__).resolve().parents[1]
DS = pathlib.Path(os.environ.get("NERO_DATA", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2")))


def fail(msg):
    print(f"FAIL  {msg}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", default=str(BENCH / "data" / "val_anchors.npz"))
    ap.add_argument("--dataset", default=str(DS))
    ap.add_argument("--split", default=str(BENCH / "data" / "split.json"))
    args = ap.parse_args()

    bad = 0
    A = np.load(args.anchors)
    eps, frames, gt, state = A["episodes"], A["frames"], A["gt"], A["state"]
    K = int(A["K"])
    stride = int(A["stride"]) if "stride" in A.files else None
    n = len(eps)
    print(f"anchors: {args.anchors}")
    print(f"  n={n}  K={K}  stride={stride}  gt{gt.shape}{gt.dtype}  state{state.shape}{state.dtype}")

    split = json.load(open(args.split))
    val_eps = set(split["val_episodes"])
    train_eps = set(split["train_episodes"])

    # ---------------- structural ----------------
    if gt.shape != (n, K, 8):
        bad += fail(f"gt shape {gt.shape} != ({n},{K},8)")
    if state.shape != (n, 8):
        bad += fail(f"state shape {state.shape} != ({n},8)")
    if not np.isfinite(gt).all():
        bad += fail(f"gt has {(~np.isfinite(gt)).sum()} non-finite values")
    if not np.isfinite(state).all():
        bad += fail(f"state has {(~np.isfinite(state)).sum()} non-finite values")

    seen = set(int(e) for e in eps)
    if not seen <= val_eps:
        bad += fail(f"anchors reference NON-val episodes: {sorted(seen - val_eps)}")
    if seen & train_eps:
        bad += fail(f"anchors reference TRAIN episodes: {sorted(seen & train_eps)}")
    missing = val_eps - seen
    if missing:
        bad += fail(f"val episodes with zero anchors: {sorted(missing)}")
    print(f"  episodes: {len(seen)}/{len(val_eps)} val eps represented, "
          f"train-intersection {len(seen & train_eps)}")

    # rows must be grouped/sorted so that row i is addressable by (ep, frame)
    order = np.lexsort((frames, eps))
    if not np.array_equal(order, np.arange(n)):
        print("  NOTE: rows are not sorted by (episode, frame) -- not an error, "
              "but row order is load-bearing for every leg's preds alignment")
    if len(set(zip(eps.tolist(), frames.tolist()))) != n:
        bad += fail("duplicate (episode, frame) anchors")

    # ---------------- per-episode against raw parquet ----------------
    # map episode_index -> parquet path by reading the column, not by trusting the
    # filename (a renamed/reordered export would otherwise silently misalign)
    paths = sorted(glob.glob(str(pathlib.Path(args.dataset) / "data" / "chunk-*" / "*.parquet")))
    ep_path = {}
    for p in paths:
        head = pd.read_parquet(p, columns=["episode_index"])
        u = head["episode_index"].unique()
        if len(u) != 1:
            bad += fail(f"{p} contains {len(u)} episode ids, expected 1")
        ep_path[int(u[0])] = p
    print(f"  dataset: {len(paths)} parquet files, episode ids "
          f"{min(ep_path)}..{max(ep_path)}")

    n_state_bad = n_gt_bad = n_bound_bad = 0
    worst_state = worst_gt = 0.0
    ep_rows = {}
    for ep in sorted(seen):
        if ep not in ep_path:
            bad += fail(f"episode {ep} has anchors but no parquet")
            continue
        df = pd.read_parquet(ep_path[ep], columns=["frame_index", "observation.state", "action"])
        df = df.sort_values("frame_index")
        fidx = df["frame_index"].to_numpy()
        if not np.array_equal(fidx, np.arange(len(df))):
            bad += fail(f"episode {ep} frame_index is not 0..L-1 (gaps or restart)")
        st = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        ac = np.stack(df["action"].to_numpy()).astype(np.float64)
        L = len(df)

        rows = np.where(eps == ep)[0]
        ep_rows[ep] = (rows, L)
        for i in rows:
            f = int(frames[i])
            if f < 0 or f + K > L:
                n_bound_bad += 1
                bad += fail(f"anchor {i} ep{ep} f{f}: window [{f},{f+K-1}] "
                            f"escapes episode length {L}")
                continue
            ds = np.abs(st[f] - state[i].astype(np.float64)).max()
            dg = np.abs(ac[f:f + K] - gt[i].astype(np.float64)).max()
            worst_state = max(worst_state, ds)
            worst_gt = max(worst_gt, dg)
            # float32 storage of a float64 parquet value: 1e-4 is ~6 orders above
            # the float32 ulp at 100 deg and still far below any real misalignment
            if ds > 1e-4:
                n_state_bad += 1
                if n_state_bad <= 3:
                    bad += fail(f"anchor {i} ep{ep} f{f} state mismatch max|d|={ds:.6g}\n"
                                f"      parquet={st[f]}\n      anchors={state[i]}")
            if dg > 1e-4:
                n_gt_bad += 1
                if n_gt_bad <= 3:
                    bad += fail(f"anchor {i} ep{ep} f{f} gt mismatch max|d|={dg:.6g}\n"
                                f"      parquet[0]={ac[f]}\n      anchors[0]={gt[i][0]}")

    print(f"  state: {n - n_state_bad}/{n} exact (worst max|d| {worst_state:.3g})")
    print(f"  gt   : {n - n_gt_bad}/{n} exact (worst max|d| {worst_gt:.3g})")
    print(f"  bounds: {n - n_bound_bad}/{n} windows inside their episode")

    # ---------------- gt provenance: action column, not shifted state ----------
    # If the builder had grabbed observation.state instead of action, every check
    # above that uses `ac` would fail -- but the two are close, so also report the
    # margin, i.e. how far apart the two candidate sources actually are.
    ep0 = sorted(seen)[0]
    df = pd.read_parquet(ep_path[ep0], columns=["observation.state", "action"])
    st0 = np.stack(df["observation.state"].to_numpy())
    ac0 = np.stack(df["action"].to_numpy())
    print(f"  provenance margin on ep{ep0}: mean|action - state| = "
          f"{np.abs(ac0 - st0).mean():.4f}  (0 would make the gt-source check blind)")

    # ---------------- stride / tail policy ----------------
    strides, tails = set(), {}
    for ep, (rows, L) in sorted(ep_rows.items()):
        f = np.sort(frames[rows])
        strides.update(np.unique(np.diff(f)).tolist())
        tails[ep] = (len(rows), int(f[0]), int(f[-1]), L, L - (int(f[-1]) + K))
    print(f"  stride values across all episodes: {sorted(strides)}"
          + (f" (npz says {stride})" if stride is not None else ""))
    if stride is not None and strides != {stride}:
        bad += fail(f"stride in npz ({stride}) disagrees with the anchor spacing {sorted(strides)}")
    starts = {t[1] for t in tails.values()}
    if starts != {0}:
        print(f"  NOTE: not every episode starts at frame 0: {sorted(starts)}")
    slack = {ep: t[4] for ep, t in tails.items()}
    print(f"  tail slack (frames left unused after the last window): "
          f"min {min(slack.values())} max {max(slack.values())}")
    if min(slack.values()) < 0:
        bad += fail("negative tail slack -- a window ran off the end")
    if max(slack.values()) >= stride if stride else False:
        print(f"  NOTE: an episode had >= stride frames of slack "
              f"(eps {[e for e, s in slack.items() if s >= stride]}) -- an extra "
              f"anchor could have fit; consistent only if the builder needs K+ tail room")

    # expected count under "every stride-th frame whose K-window fits"
    exp = sum(len(range(0, L - K + 1, stride)) for _, (_, L) in ep_rows.items()) if stride else None
    if exp is not None:
        print(f"  count model: sum_ep |range(0, L-K+1, {stride})| = {exp} vs actual {n}"
              + ("  MATCH" if exp == n else "  MISMATCH"))
        if exp != n:
            for ep, (rows, L) in sorted(ep_rows.items()):
                e = len(range(0, L - K + 1, stride))
                if e != len(rows):
                    print(f"    ep{ep}: L={L} expected {e} anchors, got {len(rows)}")

    print()
    print("VERDICT:", "CLEAN -- val_anchors.npz reproduces from raw parquet"
          if bad == 0 else f"{bad} PROBLEM(S) -- see FAIL lines above")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
