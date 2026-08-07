#!/usr/bin/env python3
"""Per-DIM discriminating power of each statistic, split by stat type.

lerobot-setup found (ACT/LeRobot, MEAN_STD) that mean/std move on 8/8 dims under a
leak while min/max move on only 1-3/8 -- i.e. the whole `val_max > stat_max` route
uses the two LEAST sensitive statistics. This tests that on a second, independent
stack with a different normalization (N1.5 = min_max) and adds the quantile class
(q01/q99) that LeRobot's stats.json doesn't carry.

Counts, per stat vector, how many DIMS move between the train-111 recompute and the
full-131 (leaked) recompute. A dim with zero movement is a slot where a leak is
structurally INVISIBLE -- that enumerates the saturation blind spot for free.
"""
import pathlib
import os
import json
from pathlib import Path

import numpy as np

import importlib.util
spec = importlib.util.spec_from_file_location(
    "direct", str(pathlib.Path(__file__).resolve().parent / "n15_leak_direct_recompute.py"))
direct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(direct)

BENCH = direct.BENCH
CKPT = Path(os.environ.get("NERO_CKPT", str(pathlib.Path(__file__).resolve().parents[2] / "checkpoints"))) / "gr00t_n15_nero_b2" / "checkpoint-5000"

import glob
train_paths = sorted(glob.glob(str(direct.TRAIN_DIR / "data/*/*.parquet")))
src_paths = sorted(glob.glob(str(direct.SOURCE_DIR / "data/*/*.parquet")))

print(f"recomputing train-111 ({len(train_paths)} eps) and full-131 ({len(src_paths)} eps) ...")
rc_train, n_tr = direct.calculate_dataset_statistics(train_paths)
rc_full, n_fu = direct.calculate_dataset_statistics(src_paths)
print(f"rows: train={n_tr} full={n_fu}\n")

rows = []
dead_slots = []
for grp in ["state", "action"]:
    for sub in ["single_arm", "gripper"]:
        for stat in direct.STAT_KEYS:
            t = direct.as_vec(rc_train, grp, sub, stat)
            f = direct.as_vec(rc_full, grp, sub, stat)
            d = np.abs(t - f)
            moved = int((d > 0).sum())
            rows.append((f"{grp}.{sub}", stat, moved, len(d), float(d.max())))
            for j, dv in enumerate(d):
                if dv == 0:
                    jj = j if sub == "single_arm" else 7
                    dead_slots.append(f"{grp}.{stat}.j{jj}")

print(f"{'column':22s}{'stat':>6s}{'dims moved':>13s}{'max|dev|':>13s}")
print("-" * 56)
for col, stat, moved, n, mx in rows:
    mark = "   <- BLIND" if moved == 0 else ("" if moved == n else "   <- partial")
    print(f"{col:22s}{stat:>6s}{f'{moved}/{n}':>13s}{mx:13.4f}{mark}")

# aggregate by stat type over all 16 dims (state+action, arm+gripper)
print("\n=== aggregated by statistic type (out of 8 dims per column) ===")
print(f"{'column.stat':28s}{'dims moved':>12s}{'max|dev|':>12s}")
agg = {}
for col, stat, moved, n, mx in rows:
    base = col.split(".")[0]
    key = f"{base}.{stat}"
    m, tot, mm = agg.get(key, (0, 0, 0.0))
    agg[key] = (m + moved, tot + n, max(mm, mx))
for stat in direct.STAT_KEYS:
    for base in ["state", "action"]:
        k = f"{base}.{stat}"
        m, tot, mm = agg[k]
        print(f"{k:28s}{f'{m}/{tot}':>12s}{mm:12.4f}")

print(f"\nzero-power slots (leak invisible here): {len(dead_slots)}")
for s in dead_slots:
    print("   ", s)

Path(BENCH / "logs/n15_leak_power_by_stat.json").write_text(json.dumps({
    "per_vector": [{"column": c, "stat": s, "dims_moved": m, "n_dims": n, "max_dev": x}
                   for c, s, m, n, x in rows],
    "aggregated": {k: {"dims_moved": v[0], "n_dims": v[1], "max_dev": v[2]} for k, v in agg.items()},
    "zero_power_slots": dead_slots,
}, indent=2))
print("\nwrote logs/n15_leak_power_by_stat.json")
