#!/usr/bin/env python3
"""DIRECT proof that the N1.5 ckpt's baked norm-stats are TRAIN-111-only.

Reverse-falsification (`val_max > stat_max`) only fails to contradict "train-only".
This is the positive form team-lead ratified: replicate GR00T's own statistic
computation (gr00t/data/dataset.py:calculate_dataset_statistics) bit-for-bit over
the 111 train episodes and compare to the numbers actually baked into the ckpt,
then do the same over all 131 (the leaked counterfactual) and show it differs.

  train-111 recompute == ckpt baked   (dev 0.00e+00)   -> positive attribution
  full-131 recompute  != ckpt baked   (dev >> 0)       -> the test has power

Replication notes (must match upstream or "bit-for-bit" is meaningless):
  - paths iterated with sorted(), pd.concat(axis=0)  -> row order fixed
  - np.vstack([np.asarray(x, dtype=np.float32) ...]) -> float32 accumulation
  - mean/std/min/max/quantile along axis=0, np.quantile default interpolation
CPU-only; does not touch the GPU.
"""
import os
import argparse, glob, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
TRAIN_DIR = BENCH / "data/b2_gr00t_train"
SOURCE_DIR = Path(os.environ.get("NERO_DATA", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2")))

# meta/modality.json: state/action are both [0:7] single_arm + [7:8] gripper
SLICES = {"single_arm": slice(0, 7), "gripper": slice(7, 8)}
COLMAP = {"state": "observation.state", "action": "action"}
STAT_KEYS = ["max", "min", "mean", "std", "q01", "q99"]


def calculate_dataset_statistics(parquet_paths):
    """Verbatim port of gr00t/data/dataset.py (N1.5), trimmed to the low-dim cols."""
    all_low_dim_data_list = []
    for parquet_path in sorted(list(parquet_paths)):
        all_low_dim_data_list.append(pd.read_parquet(parquet_path))
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)

    out = {}
    for le_modality in all_low_dim_data.columns:
        if isinstance(all_low_dim_data[le_modality].iloc[0], str):
            continue
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        out[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return out, len(all_low_dim_data)


def baked(ckpt: Path):
    md = json.loads((ckpt / "experiment_cfg/metadata.json").read_text())
    assert list(md.keys()) == ["new_embodiment"], list(md.keys())
    return md["new_embodiment"]["statistics"]


def as_vec(recomputed, grp, sub, stat):
    """Project the raw-column stat onto GR00T's single_arm/gripper split."""
    return np.asarray(recomputed[COLMAP[grp]][stat], dtype=np.float64)[SLICES[sub]]


def compare(tag, recomputed, ck):
    """Return per-(grp,sub,stat) max abs deviation between a recompute and the ckpt."""
    rows = []
    for grp in ["state", "action"]:
        for sub in ["single_arm", "gripper"]:
            for stat in STAT_KEYS:
                a = as_vec(recomputed, grp, sub, stat)
                b = np.asarray(ck[grp][sub][stat], dtype=np.float64)
                assert a.shape == b.shape, (grp, sub, stat, a.shape, b.shape)
                dev = np.abs(a - b)
                rows.append((f"{grp}.{sub}.{stat}", float(dev.max()), int((dev == 0).sum()), len(a)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint dir containing experiment_cfg/metadata.json")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    ck = baked(ckpt)
    split = json.loads((BENCH / "data" / "split.json").read_text())
    train_eps, val_eps = split["train_episodes"], split["val_episodes"]
    assert not (set(train_eps) & set(val_eps))

    train_paths = sorted(glob.glob(str(TRAIN_DIR / "data/*/*.parquet")))
    src_paths = sorted(glob.glob(str(SOURCE_DIR / "data/*/*.parquet")))
    print(f"ckpt          : {ckpt}")
    print(f"train parquet : {len(train_paths)} (split says {len(train_eps)})")
    print(f"source parquet: {len(src_paths)} (split says {len(train_eps) + len(val_eps)} total)")
    assert len(train_paths) == len(train_eps), "train dir episode count != split"
    assert len(src_paths) == split["total_episodes"], "source dir != 131"

    print("\n[A] recompute over TRAIN-111 (what the trainer read) ...")
    rc_train, n_train_rows = calculate_dataset_statistics(train_paths)
    print(f"    rows={n_train_rows}")
    print("[B] recompute over FULL-131 (leaked counterfactual) ...")
    rc_full, n_full_rows = calculate_dataset_statistics(src_paths)
    print(f"    rows={n_full_rows}")

    rows_train = compare("train111", rc_train, ck)
    rows_full = compare("full131", rc_full, ck)

    print(f"\n{'stat':28s} {'dev(train111 vs ckpt)':>22s} {'bitexact':>10s} | {'dev(full131 vs ckpt)':>21s}")
    print("-" * 92)
    worst_train = 0.0
    min_sep = float("inf")
    n_discriminating = 0
    for (name, dt, ne, n), (_, df_, _, _) in zip(rows_train, rows_full):
        worst_train = max(worst_train, dt)
        flag = ""
        if df_ > 0:
            n_discriminating += 1
            min_sep = min(min_sep, df_)
        else:
            flag = "  <- full131 identical too (no power on this stat)"
        print(f"{name:28s} {dt:22.3e} {ne:>6d}/{n:<3d} | {df_:21.3e}{flag}")

    n_stats = len(rows_train)
    print("-" * 92)
    print(f"worst deviation train-111 vs ckpt : {worst_train:.3e}")
    print(f"stats where full-131 differs      : {n_discriminating}/{n_stats}"
          f"  (min non-zero separation {min_sep:.3e})")

    clean = worst_train == 0.0
    print("\nVERDICT:", "PASS - ckpt stats are bit-for-bit the TRAIN-111 statistics"
          if clean else "FAIL - ckpt stats do NOT reproduce from train-111")
    if clean and n_discriminating:
        print(f"  Power: had val leaked, {n_discriminating}/{n_stats} stat vectors would have moved"
              f" (by >= {min_sep:.3e}); they did not.")
    print("  Blind spot: proves WHICH episode set the baked stats came from; it does NOT"
          "\n  prove the sampler fed only those episodes (separate channel -- see"
          " n15_leak_two_channel.py).")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "ckpt": str(ckpt),
            "n_train_parquet": len(train_paths), "n_train_rows": int(n_train_rows),
            "n_full_parquet": len(src_paths), "n_full_rows": int(n_full_rows),
            "worst_dev_train111_vs_ckpt": worst_train,
            "n_stats": n_stats, "n_discriminating_full131": n_discriminating,
            "min_separation_full131": None if min_sep == float("inf") else min_sep,
            "per_stat": [{"stat": r[0], "dev_train111": r[1], "n_bitexact": r[2], "n_dims": r[3],
                          "dev_full131": f[1]} for r, f in zip(rows_train, rows_full)],
            "verdict": "PASS" if clean else "FAIL",
        }, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
