#!/usr/bin/env python3
"""The MUST-NOT-MATCH direction of the N1.5 leak proof.

n15_leak_direct_recompute.py proves the positive: ckpt stats == TRAIN-111 recompute,
bit-for-bit. This is its complement, the form gr00t-n17 ran on N1.7:

  (a) VAL-20 recompute        vs ckpt  -> must DIFFER
  (b) val dir's own meta/stats.json    -> must DIFFER  (this is the artifact
      gr00t/data/dataset.py:360 reads when a transform IS attached to the eval
      loader; predict_gr00t.py passes transforms=None so it never reaches the
      policy, but if it ever DID, this is the table that would be used)

⚠ THIS SCRIPT CANNOT EXCLUDE A LEAK, and the reason is the control group, not the
direction (gr00t-n17's point, 2026-08-05). A leaked table is aggregated over
train∪val = 131 episodes, and a 131-table differs from the 111-table AND from the
20-table. So "ckpt differs from VAL-20" is satisfied by a leaked ckpt too — the
counterfactual is simply pointing the wrong way. The watershed for any leak test is
whether its control group is train∪val, NOT whether it is positive/reverse, and NOT
how many artifacts it compared. The exclusion lives entirely in
n15_leak_direct_recompute.py, whose control IS the full 131 (20/24 stat vectors move).
This script's only jobs are (a) a sanity floor and (b) measuring how often the
compare-to-val style false-alarms on a stack whose arm is in ABSOLUTE space.

It cannot convict on its own either: coincidences are expected wherever
the arm parks against a mechanical stop, since the same stop is attained in both
splits. Every coincidence is therefore adjudicated automatically, not by assertion --
a slot is EXPLAINED iff the TRAIN-111 recompute alone already yields that exact
value (i.e. the ckpt could have got it with zero knowledge of val), and the number
of train/val frames attaining it is printed as the mechanism. Only a coincidence
the train split cannot produce would be a finding.

CPU-only, seconds, no GPU, no model.
"""
import argparse, glob, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n15_leak_direct_recompute import (  # noqa: E402
    BENCH, COLMAP, SOURCE_DIR, STAT_KEYS, TRAIN_DIR, baked,
    calculate_dataset_statistics, as_vec,
)

VAL_DIR = BENCH / "data/b2_gr00t_val"
# GR00T's single_arm/gripper keys are slices of the raw 8-wide column
SLICE_OFFSET = {"single_arm": 0, "gripper": 7}


def dims_equal(recomputed, ck):
    """Per-(grp,sub,stat): how many dims coincide with the ckpt, and which."""
    rows = []
    for grp in ["state", "action"]:
        for sub in ["single_arm", "gripper"]:
            for stat in STAT_KEYS:
                a = as_vec(recomputed, grp, sub, stat)
                b = np.asarray(ck[grp][sub][stat], dtype=np.float64)
                eq = np.flatnonzero(a == b)
                rows.append((f"{grp}.{sub}.{stat}", len(a), eq.tolist(),
                             float(np.abs(a - b).max()),
                             [float(a[i]) for i in eq]))
    return rows


def report(tag, rows, rc_train, attained, out):
    """Print the coincidence table and adjudicate every coinciding slot.

    rc_train: the TRAIN-111 recompute (the ckpt's proven provenance).
    attained: fn(grp, dim, value) -> (n_train_frames, n_val_frames) attaining it exactly.
    """
    n_dims = sum(r[1] for r in rows)
    n_eq = sum(len(r[2]) for r in rows)
    print(f"\n=== {tag} vs ckpt ===")
    print(f"{'stat':28s} {'dims':>5s} {'coincide':>9s} {'max|diff|':>13s}")
    print("-" * 62)
    coincidences, unexplained = [], []
    for name, n, eq, mx, vals in rows:
        print(f"{name:28s} {n:5d} {len(eq):9d} {mx:13.4f}")
        grp, sub, stat = name.split(".")
        for i, v in zip(eq, vals):
            # Can the TRAIN split alone produce this exact number? (ckpt == train
            # recompute bit-exact, so this is asking whether val was ever needed.)
            from_train = float(as_vec(rc_train, grp, sub, stat)[i])
            explained = from_train == v
            ntr, nval = attained(grp, SLICE_OFFSET[sub] + i, v)
            rec = {"stat": name, "dim": i, "value": v, "train_recompute": from_train,
                   "explained_by_train": explained,
                   "frames_attaining_train": ntr, "frames_attaining_val": nval}
            coincidences.append(rec)
            if not explained:
                unexplained.append(rec)
    print("-" * 62)
    print(f"{tag}: {n_eq}/{n_dims} slots coincide with the ckpt")
    if coincidences:
        print(f"  {'slot':28s} {'value':>16s} {'train-only?':>12s} {'attained tr/val':>18s}")
        for r in coincidences:
            print(f"  {r['stat'] + '[' + str(r['dim']) + ']':28s} {r['value']:16.6f}"
                  f" {'YES' if r['explained_by_train'] else 'NO':>12s}"
                  f" {str(r['frames_attaining_train']) + '/' + str(r['frames_attaining_val']):>18s}")
    for r in unexplained:
        print(f"  FINDING - train-111 cannot produce {r['stat']}[{r['dim']}] = {r['value']}")
    out[tag] = coincidences
    return n_eq, n_dims, unexplained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ck = baked(Path(args.ckpt))
    split = json.loads((BENCH / "data" / "split.json").read_text())
    val_eps = set(split["val_episodes"])

    # (a) VAL-20 recompute, straight from the source repo's parquet
    src = sorted(glob.glob(str(SOURCE_DIR / "data/*/*.parquet")))
    val_paths = [p for p in src if int(Path(p).stem.split("_")[-1]) in val_eps]
    assert len(val_paths) == len(val_eps), (len(val_paths), len(val_eps))
    train_paths = sorted(glob.glob(str(TRAIN_DIR / "data/*/*.parquet")))
    print(f"ckpt      : {args.ckpt}")
    print(f"val parquet: {len(val_paths)} episodes (split says {len(val_eps)})")
    rc_val, n_rows = calculate_dataset_statistics(val_paths)
    rc_train, n_train_rows = calculate_dataset_statistics(train_paths)
    print(f"val rows  : {n_rows}   train rows: {n_train_rows}")

    # raw columns, for the "how many frames actually attain this value" mechanism
    def raw(paths, col):
        import pandas as pd
        d = pd.concat([pd.read_parquet(p) for p in sorted(paths)], axis=0)
        return np.vstack([np.asarray(x, dtype=np.float32) for x in d[col]])

    cols = {g: {"tr": raw(train_paths, c), "val": raw(val_paths, c)} for g, c in COLMAP.items()}

    def attained(grp, dim, value):
        v = np.float32(value)
        return (int((cols[grp]["tr"][:, dim] == v).sum()),
                int((cols[grp]["val"][:, dim] == v).sum()))

    out = {"ckpt": args.ckpt, "n_val_parquet": len(val_paths), "n_val_rows": int(n_rows),
           "n_train_rows": int(n_train_rows)}
    res = {}
    a_eq, a_n, a_sus = report("VAL-20 recompute", dims_equal(rc_val, ck), rc_train, attained, res)

    # (b) the val dataset directory's OWN stats.json -- the third-channel artifact
    val_stats_f = VAL_DIR / "meta/stats.json"
    b = None
    if val_stats_f.exists():
        vs = json.loads(val_stats_f.read_text())
        print(f"\nval meta/stats.json present ({val_stats_f}), keys={sorted(vs.keys())[:4]}...")
        b = report("val meta/stats.json", dims_equal(vs, ck), rc_train, attained, res)
    else:
        print(f"\nval meta/stats.json ABSENT at {val_stats_f}"
              " -- dataset.py:360 would RECOMPUTE and write it on first transformed load")

    out["val_recompute"] = {"n_coincide": a_eq, "n_dims": a_n, "unexplained": a_sus}
    if b:
        out["val_meta_stats_json"] = {"n_coincide": b[0], "n_dims": b[1], "unexplained": b[2]}
    out["coincidences"] = res
    ok = not a_sus and (b is None or not b[2])
    out["false_alarm_rate"] = {"n_coincide": a_eq, "n_slots": a_n, "frac": a_eq / a_n}
    out["verdict"] = "PASS" if ok else "FAIL"
    print(f"\nFalse-alarm rate of this (weaker) direction: {a_eq}/{a_n} slots"
          f" = {100 * a_eq / a_n:.2f}% coincide with val by physics alone.")
    print("VERDICT:", "PASS - every coincidence is reproducible from TRAIN-111 alone,"
          " so val contributed nothing" if ok else
          "FAIL - a coinciding value that train-111 cannot produce")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
