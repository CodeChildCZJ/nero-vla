#!/usr/bin/env python3
"""Model-free rehearsal of lerobot_predict.py's observation half (ACT / SmolVLA).

Runs everything lerobot_predict.py does EXCEPT the policy forward pass: opens the
same LeRobotDataset with the same episode filter, indexes the same anchors through
build_anchor_index(), and inspects what comes back. CPU only, no checkpoint, so
the alignment + video-decode surface is cleared before a GPU is ever held.

Why it exists (gr00t-n17's finding, relayed by team-lead): predict has NO assert on
image *content*. A decoder that silently returns the same keyframe for every anchor,
a near-black frame, or a cam_high/cam_wrist mapping that collapsed to one stream all
produce a plausible MAE and never throw. Those failures are only visible here.

Checks:
  1. state alignment  -- ds observation.state == val_anchors.npz state, per anchor
  2. per-frame md5    -- every decoded frame distinct (anchors are 5 frames apart)
  3. near-black       -- frame mean < 5/255
  4. near-uniform     -- frame std  < 2/255
  5. camera aliasing  -- cam_high != cam_wrist on every anchor
"""
import os
import argparse
import hashlib
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BENCH / "scripts" / "predict"), str(BENCH / "scripts" / "score"),
                str(BENCH / "archive")]   # was one flat scripts/ dir

DEFAULT_ROOT = os.environ.get("NERO_DATA_V30", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2_bench"))
DEFAULT_REPO_ID = "local/pick_pink_sponge_b2_bench"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot_predict import build_anchor_index  # the exact mapping predict uses

    bad = []
    A = np.load(BENCH / "data" / "val_anchors.npz")
    n = args.limit if args.limit else len(A["episodes"])
    episodes = A["episodes"][:n]
    frames = A["frames"][:n]
    ref_state = A["state"][:n]
    val_eps = sorted(set(int(e) for e in episodes))
    print(f"[anchors] {n} anchors over {len(val_eps)} val episodes: {val_eps}")

    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=val_eps)
    print(f"[dataset] {ds.num_episodes} eps / {ds.num_frames} frames, fps={ds.meta.fps}")
    rel_idx = build_anchor_index(ds, episodes, frames)

    probe = ds[rel_idx[0]]
    img_keys = sorted(k for k in probe if k.startswith("observation.image"))
    print(f"[keys] images={img_keys}  state shape={tuple(probe['observation.state'].shape)}")
    if len(img_keys) != 2:
        bad.append(f"expected 2 camera keys, got {img_keys}")

    n_state_ok = 0
    img_stats = defaultdict(list)     # key -> [(anchor, mean, std)]
    frame_hash = defaultdict(dict)    # key -> {digest: first anchor}
    dup_frames, aliased = [], []
    t0 = time.time()
    for i, ri in enumerate(rel_idx):
        item = ds[ri]

        got = item["observation.state"].numpy().astype(np.float64)
        if not np.allclose(got, ref_state[i], atol=1e-3):
            bad.append(f"anchor {i} ep{episodes[i]} f{frames[i]} state mismatch: "
                       f"max|d|={np.abs(got - ref_state[i]).max():.4g}")
        else:
            n_state_ok += 1

        per_key = {}
        for k in img_keys:
            img = item[k].numpy()
            per_key[k] = img
            img_stats[k].append((i, float(img.mean()), float(img.std())))
            d = hashlib.md5(np.ascontiguousarray(img)).hexdigest()
            if d in frame_hash[k]:
                dup_frames.append((k, frame_hash[k][d], i))
            else:
                frame_hash[k][d] = i
        if len(img_keys) == 2:
            a, b = per_key[img_keys[0]], per_key[img_keys[1]]
            if a.shape == b.shape and np.array_equal(a, b):
                aliased.append(i)
        if i % 200 == 0:
            print(f"  [{i}/{n}] {time.time() - t0:.0f}s", flush=True)

    print(f"\nstate alignment: {n_state_ok}/{n} anchors match val_anchors (atol 1e-3)")
    # LeRobot returns images as float32 [0,1]; n17's thresholds are 0-255, rescale.
    for k in img_keys:
        rows = img_stats[k]
        m = np.array([r[1] for r in rows]) * 255.0
        s = np.array([r[2] for r in rows]) * 255.0
        print(f"image {k:28s} n={len(rows)}  mean {m.min():.1f}..{m.max():.1f}  "
              f"std {s.min():.1f}..{s.max():.1f}")
        dark = [rows[j][0] for j in np.where(m < 5)[0]]
        flat = [rows[j][0] for j in np.where(s < 2)[0]]
        if dark:
            bad.append(f"{k}: {len(dark)} near-black frames, first {dark[:5]}")
        if flat:
            bad.append(f"{k}: {len(flat)} near-uniform frames, first {flat[:5]}")
        uniq = len(frame_hash[k])
        print(f"   distinct frames {uniq}/{len(rows)}"
              + ("" if uniq == len(rows) else "  <-- repeats"))
    if dup_frames:
        print(f"duplicate frames: {len(dup_frames)} pairs, e.g. {dup_frames[:5]}")
        bad.append(f"{len(dup_frames)} byte-identical frame pairs across anchors")
    if aliased:
        bad.append(f"{img_keys[0]} == {img_keys[1]} on {len(aliased)} anchors, first {aliased[:5]}")

    print()
    if bad:
        for b in bad:
            print("FAIL ", b)
        print(f"\nVERDICT: {len(bad)} PROBLEM(S)")
        return 1
    print("VERDICT: CLEAN -- observation half of predict is ready, "
          "only the forward pass is untested")
    return 0


if __name__ == "__main__":
    sys.exit(main())
