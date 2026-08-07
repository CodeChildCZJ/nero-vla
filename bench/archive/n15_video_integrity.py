#!/usr/bin/env python3
"""Model-free integrity sweep of the val video decode path, run BEFORE predict.

predict_gr00t.py asserts nothing about the images it feeds the model, so a decoder
that silently returns a stale keyframe (or an all-black frame) produces a perfectly
plausible MAE and never raises. This checks the frames the anchors actually touch.

Per (episode, camera) it decodes exactly the anchor frame indices and flags:
  DARK       frame mean < 5            (near-black / decode returned nothing)
  FLAT       frame std  < 2            (frozen or uniform frame)
  DUP        identical md5 at two DIFFERENT frame indices  <- the stale-keyframe mode
  SAMECAM    cam_high md5 == cam_wrist md5 at the same anchor (wired to one stream)
Run with the same backend predict uses (decord).
"""
import argparse, hashlib, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
CAMS = ["observation.images.cam_high", "observation.images.cam_wrist"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-root", default=str(BENCH / "data/b2_gr00t_val"))
    ap.add_argument("--anchors", default=str(BENCH / "data" / "val_anchors.npz"))
    ap.add_argument("--json-out", default=str(BENCH / "logs/n15_video_integrity.json"))
    ap.add_argument("--dark-mean", type=float, default=5.0)
    ap.add_argument("--flat-std", type=float, default=2.0)
    args = ap.parse_args()

    import decord

    root = Path(args.val_root)
    a = np.load(args.anchors)
    eps, frames = a["episodes"], a["frames"]
    val_eps = sorted(set(eps.tolist()))
    print(f"anchors={len(eps)} over {len(val_eps)} val episodes; backend=decord")

    problems = []
    n_frames_checked = 0
    per_ep = {}

    for e in val_eps:
        idxs = sorted(set(frames[eps == e].tolist()))
        md5_by_cam = {}
        for cam in CAMS:
            mp4 = root / f"videos/chunk-000/{cam}/episode_{e:06d}.mp4"
            if not mp4.exists():
                problems.append({"type": "MISSING", "episode": int(e), "cam": cam, "path": str(mp4)})
                continue
            vr = decord.VideoReader(str(mp4))
            if max(idxs) >= len(vr):
                problems.append({"type": "SHORT", "episode": int(e), "cam": cam,
                                 "n_frames": len(vr), "max_anchor_idx": int(max(idxs))})
            keep = [i for i in idxs if i < len(vr)]
            batch = vr.get_batch(keep).asnumpy()  # (n,H,W,3) uint8
            n_frames_checked += len(keep)

            md5s = {}
            seen = defaultdict(list)
            for i, fi in enumerate(keep):
                fr = batch[i]
                m, s = float(fr.mean()), float(fr.std())
                if m < args.dark_mean:
                    problems.append({"type": "DARK", "episode": int(e), "cam": cam,
                                     "frame": int(fi), "mean": m})
                if s < args.flat_std:
                    problems.append({"type": "FLAT", "episode": int(e), "cam": cam,
                                     "frame": int(fi), "std": s})
                h = hashlib.md5(fr.tobytes()).hexdigest()
                md5s[fi] = h
                seen[h].append(int(fi))
            for h, fis in seen.items():
                if len(fis) > 1:
                    problems.append({"type": "DUP", "episode": int(e), "cam": cam,
                                     "frames": fis[:8], "n": len(fis)})
            md5_by_cam[cam] = md5s

        if len(md5_by_cam) == 2:
            hi, wr = md5_by_cam[CAMS[0]], md5_by_cam[CAMS[1]]
            same = [fi for fi in hi if fi in wr and hi[fi] == wr[fi]]
            if same:
                problems.append({"type": "SAMECAM", "episode": int(e),
                                 "frames": [int(x) for x in same[:8]], "n": len(same)})
        per_ep[int(e)] = {"n_anchor_frames": len(idxs)}

    by_type = defaultdict(int)
    for p in problems:
        by_type[p["type"]] += 1

    print(f"frames decoded: {n_frames_checked}")
    print(f"problems: {dict(by_type) if by_type else 'none'}")
    for p in problems[:20]:
        print("  ", json.dumps(p))
    if len(problems) > 20:
        print(f"   ... +{len(problems)-20} more")

    ok = not problems
    print("\nVERDICT:", "PASS - val decode path is clean" if ok else "FAIL - see problems above")
    print("  Blind spot: proves the decoded frames are distinct/non-degenerate; it does NOT"
          "\n  prove they are the RIGHT frames (a consistent off-by-N offset would pass).")
    Path(args.json_out).write_text(json.dumps({
        "n_anchors": int(len(eps)), "n_val_episodes": len(val_eps),
        "n_frames_decoded": n_frames_checked,
        "thresholds": {"dark_mean": args.dark_mean, "flat_std": args.flat_std},
        "problems": problems, "by_type": dict(by_type),
        "verdict": "PASS" if ok else "FAIL",
    }, indent=2))
    print(f"wrote {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
