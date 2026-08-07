#!/usr/bin/env python3
"""Model-free rehearsal of predict_gr00t_n17.py's observation half.

Everything predict does EXCEPT the forward pass: open the N1.7 val dataset,
decode each val episode's video once, rebuild the observation at every anchor and
check it against val_anchors.npz. Runs on CPU with no checkpoint loaded, so the
whole episode/frame-alignment and video-integrity surface can be cleared before a
GPU is ever held -- a misalignment found here costs nothing, the same bug found
at anchor 900 of a real run costs the card.

Also screens the images, which no assert in predict covers: an all-black frame,
a frozen frame repeated across anchors, or cam_high and cam_wrist swapped/aliased
would all produce a plausible-looking but meaningless MAE.
"""
import argparse
import hashlib
from collections import defaultdict
from copy import deepcopy
import pathlib
import sys
import time

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / "configs"))

import nero_n17_config  # noqa: E402,F401  (registers the modality config)
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS  # noqa: E402
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", default=str(BENCH / "data/b2_n17_val"))
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    bad = []
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    # register_modality_config() keys the registry by the tag's VALUE, not the enum
    modality = MODALITY_CONFIGS[embodiment_tag.value]
    print("modality keys:", {k: v.modality_keys for k, v in modality.items()})
    print("NOTE: this uses configs/nero_n17_config.py, while predict uses the copy "
          "baked into the checkpoint's processor -- the state/video keys are what "
          "matter here and a divergence in them would show up as a shape error.")

    loader = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=modality)
    obs_modality = deepcopy(modality)
    obs_modality.pop("action", None)
    pos_of_ep = {m["episode_index"]: i for i, m in enumerate(loader.episodes_metadata)}
    print(f"dataset {args.dataset_path}: {len(pos_of_ep)} episodes -> {sorted(pos_of_ep)}")

    A = np.load(BENCH / "data" / "val_anchors.npz")
    episodes, frames, ref_state = A["episodes"], A["frames"], A["state"]
    n = args.limit if args.limit else len(episodes)

    ds_eps, anchor_eps = set(pos_of_ep), set(int(e) for e in episodes[:n])
    if not anchor_eps <= ds_eps:
        bad.append(f"anchors need episodes {sorted(anchor_eps - ds_eps)} that the dataset lacks")
    extra = ds_eps - anchor_eps
    if extra and not args.limit:
        bad.append(f"val dataset carries {len(extra)} episodes with no anchors: {sorted(extra)}")

    by_ep = defaultdict(list)
    for i in range(n):
        by_ep[int(episodes[i])].append(i)
    state_keys = modality["state"].modality_keys

    n_state_ok = 0
    img_stats = defaultdict(list)      # key -> [(anchor, mean, std)]
    frame_hash = defaultdict(dict)     # key -> {digest: first anchor seen}
    dup_frames, aliased = [], []
    t0 = time.time()
    for ep in sorted(by_ep):
        traj = loader[pos_of_ep[ep]]
        for i in by_ep[ep]:
            step = extract_step_data(traj, int(frames[i]), obs_modality, embodiment_tag)

            got = np.concatenate([step.states[k][0] for k in state_keys])
            if not np.allclose(got, ref_state[i], atol=1e-3):
                bad.append(f"anchor {i} ep{ep} f{frames[i]} state mismatch: "
                           f"dataset={got} anchors={ref_state[i]}")
            else:
                n_state_ok += 1

            per_key = {}
            for k, v in step.images.items():
                img = np.asarray(v)
                per_key[k] = img
                img_stats[k].append((i, float(img.mean()), float(img.std())))
                d = hashlib.md5(np.ascontiguousarray(img)).hexdigest()
                if d in frame_hash[k]:
                    dup_frames.append((k, frame_hash[k][d], i))
                else:
                    frame_hash[k][d] = i
            ks = sorted(per_key)
            if len(ks) == 2 and per_key[ks[0]].shape == per_key[ks[1]].shape and \
                    np.array_equal(per_key[ks[0]], per_key[ks[1]]):
                aliased.append(i)
        del traj
        print(f"  ep{ep}: {len(by_ep[ep])} anchors  ({time.time() - t0:.0f}s)", flush=True)

    print(f"\nstate alignment: {n_state_ok}/{n} anchors match val_anchors")
    for k, rows in sorted(img_stats.items()):
        m = np.array([r[1] for r in rows])
        s = np.array([r[2] for r in rows])
        shape = "?"
        print(f"image {k:12s} n={len(rows)}  mean {m.min():.1f}..{m.max():.1f}  "
              f"std {s.min():.1f}..{s.max():.1f}")
        dark = [rows[j][0] for j in np.where(m < 5)[0]]
        flat = [rows[j][0] for j in np.where(s < 2)[0]]
        if dark:
            bad.append(f"{k}: {len(dark)} near-black frames, first anchors {dark[:5]}")
        if flat:
            bad.append(f"{k}: {len(flat)} near-uniform frames, first anchors {flat[:5]}")
        uniq = len(frame_hash[k])
        print(f"   distinct frames {uniq}/{len(rows)}"
              + ("" if uniq == len(rows) else "  <-- repeats, see below"))
    if dup_frames:
        # Anchors are 5 frames apart, so two byte-identical frames mean a stalled
        # camera or a decode that clamped to the same keyframe.
        print(f"duplicate frames: {len(dup_frames)} pairs, e.g. {dup_frames[:5]}")
        bad.append(f"{len(dup_frames)} byte-identical frame pairs across anchors")
    if aliased:
        bad.append(f"cam_high == cam_wrist on {len(aliased)} anchors "
                   f"(camera mapping collapsed), first {aliased[:5]}")

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
