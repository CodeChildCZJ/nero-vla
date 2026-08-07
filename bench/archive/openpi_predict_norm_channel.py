#!/usr/bin/env python3
"""Falsify the THIRD leak channel (gr00t-n15, relayed by team-lead 2026-08-05):

    "predict-time dataset can self-normalize with the VAL split's own stats if constructed
     with transforms=data_config.transform(); norm-stats AND sampler checks both pass through it."

For openpi the claim decomposes into three checkable facts, each measured here -- no model,
no GPU, no JAX:

  (A) STRUCTURAL: the LeRobotDataset in predict_openpi/variance_openpi/openpi_obs_dryrun is
      built as LeRobotDataset(repo, episodes=[e]) and this lerobot build has no `transforms=`
      kwarg at all (only `image_transforms`), so the channel cannot be opened by accident.
  (B) THE CHANNEL IS REAL BUT UNUSED: ds.meta.stats for episodes=[val_ep] is that val
      episode's OWN statistics. Print it and show it differs from the ckpt/config norm_stats.
      If they were numerically equal the channel would be undetectable; they are not.
  (C) RUNTIME: the obs actually handed to policy.infer is RAW -- state in dataset units
      (matches val_anchors bit-for-bit, which is the live assert in predict_openpi.py), and
      images uint8 0..255 after to_hwc_uint8. Then show, counterfactually, what the state
      WOULD look like had it been self-normalized with the val episode's stats, and confirm
      that predict_openpi.py's existing `allclose(atol=1e-4)` assert fires on every anchor.

Exit code 0 = all three PASS. Usage:
    python openpi_predict_norm_channel.py --config pi05_nero_b2_train [--n 40]
"""
import argparse, json, pathlib, sys

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
SRC_REPO = "local/pick_pink_sponge_b2"
ASSETS = pathlib.Path(__file__).resolve().parents[2] / "third_party" / "openpi-agilex" / "assets"

sys.path[:0] = [str(BENCH / "scripts" / "predict"), str(BENCH / "scripts" / "score"),
                str(BENCH / "archive")]   # was one flat scripts/ dir
from predict_openpi import to_hwc_uint8  # the exact converter the predict path uses  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pi05_nero_b2_train")
    ap.add_argument("--n", type=int, default=40, help="anchors to spot-check (spread over all val eps)")
    args = ap.parse_args()

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import inspect

    fails = []

    # ---- (A) structural -----------------------------------------------------------------
    sig = inspect.signature(LeRobotDataset.__init__)
    has_transforms = "transforms" in sig.parameters
    print(f"[A] LeRobotDataset kwargs: {list(sig.parameters)[1:]}")
    print(f"[A] `transforms=` kwarg present in this lerobot build: {has_transforms}")
    srcs = ["scripts/predict/predict_openpi.py", "archive/variance_openpi.py",
            "archive/openpi_obs_dryrun.py"]
    bad = []
    for s in srcs:
        txt = (BENCH / s).read_text()
        for ln, line in enumerate(txt.splitlines(), 1):
            if "LeRobotDataset(" in line:
                print(f"[A] {s}:{ln}: {line.strip()}")
                if "transform" in line:
                    bad.append(f"{s}:{ln}")
    if bad:
        fails.append(f"(A) a predict-path dataset is built with a transform: {bad}")
    print(f"[A] {'PASS' if not bad else 'FAIL'}: no predict-path dataset passes any transform\n")

    # ---- ckpt/config norm_stats (the ONLY artifact policy.infer normalizes with) ---------
    ns_path = ASSETS / args.config / "local/pick_pink_sponge_b2_train/norm_stats.json"
    ns = json.loads(ns_path.read_text())["norm_stats"]
    st_mean = np.asarray(ns["state"]["mean"], dtype=np.float64)
    st_std = np.asarray(ns["state"]["std"], dtype=np.float64)
    print(f"[ns] {ns_path}")
    print(f"[ns] state.mean = {np.round(st_mean, 4)}")
    print(f"[ns] state.std  = {np.round(st_std, 4)}\n")

    A = np.load(BENCH / "data" / "val_anchors.npz")
    eps, frames, states = A["episodes"], A["frames"], A["state"]
    sel = np.unique(np.linspace(0, len(eps) - 1, args.n).round().astype(int))

    # ---- (B)+(C) per-episode ------------------------------------------------------------
    max_state_dev = 0.0
    img_lo, img_hi = 255, 0
    raw_dtypes = set()
    cf_rows = []          # counterfactual: val-self-normalized state
    n_assert_would_fire = 0
    b_checked = 0
    b_maxreldiff = 0.0

    for e in sorted({int(eps[i]) for i in sel}):
        ds = LeRobotDataset(SRC_REPO, episodes=[e])

        # (B) does the val-episode-scoped dataset expose its own stats? how far from train stats?
        mstats = getattr(ds.meta, "stats", None)
        if mstats is not None and b_checked < 3 and "observation.state" in mstats:
            vm = np.asarray(mstats["observation.state"]["mean"], dtype=np.float64)
            vs = np.asarray(mstats["observation.state"]["std"], dtype=np.float64)
            rel = np.abs(vm - st_mean) / np.maximum(np.abs(st_mean), 1e-6)
            b_maxreldiff = max(b_maxreldiff, float(rel.max()))
            print(f"[B] ep{e:3d} ds.meta.stats['observation.state'].mean = {np.round(vm, 4)}")
            print(f"[B] ep{e:3d}   vs train norm_stats.mean            = {np.round(st_mean, 4)}")
            print(f"[B] ep{e:3d}   max relative gap = {rel.max():.3f}  (0 would mean undetectable)")
            print(f"[B] ep{e:3d}   val std {np.round(vs, 3)}")
            b_checked += 1

        for i in sel[np.asarray([int(eps[j]) == e for j in sel])]:
            t = int(frames[i])
            item = ds[t]
            ds_state = np.asarray(item["observation.state"], dtype=np.float32)
            max_state_dev = max(max_state_dev, float(np.abs(ds_state - states[i]).max()))
            raw = np.asarray(item["observation.images.cam_high"])
            raw_dtypes.add((str(raw.dtype), raw.shape[0] if raw.ndim == 3 else None))
            u8 = to_hwc_uint8(raw)
            img_lo, img_hi = min(img_lo, int(u8.min())), max(img_hi, int(u8.max()))
            # counterfactual self-normalization with THIS (val) episode's own stats
            if mstats is not None and "observation.state" in mstats:
                vm = np.asarray(mstats["observation.state"]["mean"], dtype=np.float64)
                vs = np.asarray(mstats["observation.state"]["std"], dtype=np.float64)
                z = (ds_state - vm) / np.maximum(vs, 1e-6)
                cf_rows.append(z)
                if not np.allclose(states[i], z, atol=1e-4):
                    n_assert_would_fire += 1
        del ds

    print()
    print(f"[C] state: max |ds.observation.state - val_anchors.state| over {len(sel)} anchors "
          f"= {max_state_dev:.2e}   (predict_openpi.py asserts < 1e-4 on ALL 1397)")
    if max_state_dev >= 1e-4:
        fails.append(f"(C) raw state mismatch {max_state_dev:.2e}")
    print(f"[C] raw image from ds: dtype/channels {sorted(raw_dtypes)}  -> after to_hwc_uint8: "
          f"range [{img_lo}, {img_hi}]  (uint8 0..255, no mean/std image norm applied)")
    if cf_rows:
        cf = np.stack(cf_rows)
        print(f"[C] counterfactual self-normalized state: mean|z| = {np.abs(cf).mean():.3f}, "
              f"max|z| = {np.abs(cf).max():.3f}   vs raw mean|state| = {np.abs(states[sel]).mean():.3f}")
        print(f"[C] anchors on which the existing atol=1e-4 assert WOULD fire if this channel "
              f"were open: {n_assert_would_fire}/{len(cf_rows)}")
        if n_assert_would_fire != len(cf_rows):
            fails.append("(C) the live assert would NOT catch a self-normalized state on every anchor")
    print(f"[B] max relative gap between a val episode's own mean and the train norm_stats mean: "
          f"{b_maxreldiff:.3f}  ({'detectable' if b_maxreldiff > 1e-3 else 'UNDETECTABLE'})")
    if b_maxreldiff <= 1e-3:
        fails.append("(B) val stats indistinguishable from train stats -> channel undetectable")

    print()
    if fails:
        print("RESULT: FAIL")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print("RESULT: PASS -- predict feeds RAW obs; the only normalization is Policy-internal "
          "(Normalize/Unnormalize built from checkpoint_dir/assets norm_stats).")


if __name__ == "__main__":
    main()
