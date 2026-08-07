"""Video-integrity gate for the openpi predict path (gr00t-n17's check, on my code path).

predict_openpi.py has NO assert on image CONTENT. A decoder that silently returns the same
keyframe for every anchor, a stuck/near-black frame, or a cam_high/cam_wrist mix-up all yield
a *plausible* MAE and never throw -- a wrong leaderboard row that looks fine.

Runs the real observation rebuild (same LeRobotDataset open + same to_hwc_uint8 helper as
predict_openpi.py), model-free, 0 GPU:
  - per-frame md5, both cameras: every anchor must be distinct
  - reject near-black (mean < 5) and near-uniform (std < 2)
  - assert cam_high != cam_wrist on every anchor

SCALE (lerobot-setup's catch): LeRobot's ds[i] yields torch float32 CHW in [0,1], while
n17's thresholds are 0-255. Applying 0-255 thresholds to [0,1] data flags 100% of frames as
"near-black" -- a total false alarm. This script thresholds AFTER to_hwc_uint8(), i.e. on the
exact uint8 array the policy actually receives, so the thresholds are the right ruler.

--selftest injects a duplicated frame, a black frame and a cam-swap into the collected
arrays and asserts each check fires. A check that cannot fail is not a check.
"""

import argparse
import hashlib
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from predict_openpi import SRC_REPO, to_hwc_uint8  # noqa: E402

BENCH = pathlib.Path(__file__).resolve().parents[1]
NEAR_BLACK_MEAN = 5.0     # on 0-255
NEAR_UNIFORM_STD = 2.0


def md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def check(tag, hi_md5, wr_md5, hi_mean, hi_std, wr_mean, wr_std, same_cam):
    n = len(hi_md5)
    res = {
        "anchors": n,
        "cam_high_distinct": len(set(hi_md5)),
        "cam_wrist_distinct": len(set(wr_md5)),
        "near_black": int(((hi_mean < NEAR_BLACK_MEAN) | (wr_mean < NEAR_BLACK_MEAN)).sum()),
        "near_uniform": int(((hi_std < NEAR_UNIFORM_STD) | (wr_std < NEAR_UNIFORM_STD)).sum()),
        "cams_identical": int(same_cam.sum()),
    }
    ok = (res["cam_high_distinct"] == n and res["cam_wrist_distinct"] == n
          and res["near_black"] == 0 and res["near_uniform"] == 0 and res["cams_identical"] == 0)
    res["verdict"] = "PASS" if ok else "FAIL"
    print(f"[{tag}] {res}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    A = np.load(BENCH / "data" / "val_anchors.npz")
    eps, frames = A["episodes"], A["frames"]
    total = len(eps) if args.limit is None else min(args.limit, len(eps))
    sel = np.arange(total)

    hi_md5, wr_md5 = [], []
    hi_mean, hi_std, wr_mean, wr_std, same = [], [], [], [], []

    for e in sorted(set(int(x) for x in eps[sel])):
        rows = np.flatnonzero(eps[sel] == e)
        ds = LeRobotDataset(SRC_REPO, episodes=[e])
        for r in rows:
            g = int(sel[r])
            t = int(frames[g])
            item = ds[t]
            assert int(item["frame_index"]) == t, f"frame mismatch ep{e} t={t}"
            hi = to_hwc_uint8(item["observation.images.cam_high"])
            wr = to_hwc_uint8(item["observation.images.cam_wrist"])
            hi_md5.append(md5(hi)); wr_md5.append(md5(wr))
            hi_mean.append(hi.mean()); hi_std.append(hi.std())
            wr_mean.append(wr.mean()); wr_std.append(wr.std())
            same.append(hi_md5[-1] == wr_md5[-1])
        print(f"  ep{e}: {len(rows)} anchors done", flush=True)

    hi_mean = np.array(hi_mean); hi_std = np.array(hi_std)
    wr_mean = np.array(wr_mean); wr_std = np.array(wr_std)
    same = np.array(same)
    print(f"\nbrightness range (0-255): cam_high mean {hi_mean.min():.1f}-{hi_mean.max():.1f} "
          f"std {hi_std.min():.1f}-{hi_std.max():.1f} | cam_wrist mean {wr_mean.min():.1f}-"
          f"{wr_mean.max():.1f} std {wr_std.min():.1f}-{wr_std.max():.1f}")
    real = check("REAL", hi_md5, wr_md5, hi_mean, hi_std, wr_mean, wr_std, same)

    if args.selftest:
        print("\n--- selftest: each corruption must flip the verdict to FAIL ---")
        d = list(hi_md5); d[1] = d[0]
        check("inject duplicate frame", d, wr_md5, hi_mean, hi_std, wr_mean, wr_std, same)
        m = hi_mean.copy(); m[3] = 0.4
        check("inject black frame", hi_md5, wr_md5, m, hi_std, wr_mean, wr_std, same)
        s = same.copy(); s[5] = True
        check("inject cam swap", hi_md5, wr_md5, hi_mean, hi_std, wr_mean, wr_std, s)
        u = hi_std.copy(); u[7] = 0.5
        check("inject flat frame", hi_md5, wr_md5, hi_mean, u, wr_mean, wr_std, same)

    return 0 if real["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
