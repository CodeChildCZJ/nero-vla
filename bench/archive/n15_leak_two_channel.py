#!/usr/bin/env python3
"""N1.5 two-channel leakage audit.

Channel A (sampler): prove the frames that entered gradients came from exactly
the 111 split.json train episodes -- by set-equality on episode ids AND by
frame arithmetic that ties the on-disk set to the number the trainer printed.

Channel B (norm-stats): the val_max > train_stat_max falsification test, run
per joint on the CHECKPOINT's baked statistics (what unnormalize reads).

Neither channel vouches for the other; both are reported separately.
"""
import os
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_DS = ROOT / "data" / "b2_gr00t_train"
VAL_DS = ROOT / "data" / "b2_gr00t_val"
CKPT = pathlib.Path(os.environ.get("NERO_CKPT", str(pathlib.Path(__file__).resolve().parents[2] / "checkpoints"))) / "gr00t_n15_nero_b2"
TRAIN_LOG = ROOT / "logs" / "train_gr00t_n15.log"

split = json.loads((ROOT / "data" / "split.json").read_text())
train_ids = sorted(split["train_episodes"])
val_ids = sorted(split["val_episodes"])

out = {}
print("=" * 78)
print("CHANNEL A -- TRAINING SAMPLER  (did val frames enter gradients?)")
print("=" * 78)

# A1: episode ids present in the dataset the trainer was pointed at
meta_ids = sorted(
    json.loads(l)["episode_index"]
    for l in (TRAIN_DS / "meta" / "episodes.jsonl").read_text().splitlines()
    if l.strip()
)
parq = sorted(TRAIN_DS.glob("data/**/episode_*.parquet"))
file_ids = sorted(int(p.stem.split("_")[-1]) for p in parq)
vid_ids = sorted({int(p.stem.split("_")[-1]) for p in TRAIN_DS.glob("videos/**/episode_*.mp4")})

print(f"A1 split.json train_episodes      : n={len(train_ids)}")
print(f"A1 meta/episodes.jsonl ids        : n={len(meta_ids)}  == split? {meta_ids == train_ids}")
print(f"A1 parquet files on disk          : n={len(file_ids)}  == split? {file_ids == train_ids}")
print(f"A1 video episode ids (both cams)  : n={len(vid_ids)}  == split? {vid_ids == train_ids}")
print(f"A1 train_ids INTERSECT val_ids    : {sorted(set(train_ids) & set(val_ids))}  (must be [])")
out["A1_meta_eq_split"] = meta_ids == train_ids
out["A1_files_eq_split"] = file_ids == train_ids
out["A1_videos_eq_split"] = vid_ids == train_ids
out["A1_intersection_empty"] = not (set(train_ids) & set(val_ids))

# A2: the episode_index column INSIDE each parquet (content, not filename)
inner_ids, inner_rows = [], 0
for p in parq:
    df = pd.read_parquet(p, columns=["episode_index"])
    u = np.unique(df["episode_index"].to_numpy())
    assert len(u) == 1, f"{p} mixes episodes {u}"
    inner_ids.append(int(u[0]))
    inner_rows += len(df)
inner_ids = sorted(inner_ids)
print(f"A2 episode_index column inside parquets: n={len(inner_ids)} == split? {inner_ids == train_ids}")
print(f"A2 any inner id in val set?            : {sorted(set(inner_ids) & set(val_ids))}  (must be [])")
out["A2_inner_eq_split"] = inner_ids == train_ids

# A3: frame arithmetic -- on-disk rows vs metadata vs what the trainer printed
meta_len = sum(
    json.loads(l)["length"]
    for l in (TRAIN_DS / "meta" / "episodes.jsonl").read_text().splitlines()
    if l.strip()
)
logged = None
if TRAIN_LOG.exists():
    for line in TRAIN_LOG.read_text(errors="ignore").replace("\r", "\n").splitlines():
        if "train dataset length" in line:
            logged = int(line.strip().split(":")[-1])
val_meta_len = sum(
    json.loads(l)["length"]
    for l in (VAL_DS / "meta" / "episodes.jsonl").read_text().splitlines()
    if l.strip()
)
print(f"A3 actual parquet rows (read)     : {inner_rows}")
print(f"A3 meta episodes.jsonl sum(length): {meta_len}")
print(f"A3 trainer printed dataset length : {logged}")
print(f"A3 val dataset frames (excluded)  : {val_meta_len}   total {meta_len + val_meta_len}")
print(f"A3 all three agree?               : {inner_rows == meta_len == logged}")
print(f"A3 headroom: one extra val ep would add ~{val_meta_len // len(val_ids)} frames -> detectable")
out["A3_frames_agree"] = inner_rows == meta_len == logged
out["A3_train_frames"] = inner_rows
out["A3_val_frames"] = val_meta_len

# A4: physical impossibility -- no val episode file exists under the train dir
stray = [i for i in val_ids if list(TRAIN_DS.glob(f"data/**/episode_{i:06d}.parquet"))]
print(f"A4 val episode parquets found under train dir: {stray}  (must be [])")
out["A4_no_val_files_in_train_dir"] = not stray

print()
print("=" * 78)
print("CHANNEL B -- NORM STATS  (val_max > ckpt_stat_max falsification)")
print("=" * 78)

stats = json.loads((CKPT / "experiment_cfg" / "metadata.json").read_text())
emb = next(iter(stats))
st = stats[emb]["statistics"]


def load_col(ds, col):
    eps = sorted(
        json.loads(l)["episode_index"]
        for l in (ds / "meta" / "episodes.jsonl").read_text().splitlines()
        if l.strip()
    )
    arrs = []
    for p in sorted(ds.glob("data/**/episode_*.parquet")):
        if int(p.stem.split("_")[-1]) not in eps:
            continue
        arrs.append(np.stack(pd.read_parquet(p, columns=[col])[col].to_numpy()))
    return np.concatenate(arrs, 0).astype(np.float64)


for col, key_prefix in (("action", "action"), ("observation.state", "state")):
    val = load_col(VAL_DS, col)
    tr = load_col(TRAIN_DS, col)
    # gr00t splits the 8 dims into single_arm(0:7) + gripper(7:8)
    grp = st[key_prefix]
    smax = np.concatenate([np.asarray(grp["single_arm"]["max"]), np.asarray(grp["gripper"]["max"])])
    smin = np.concatenate([np.asarray(grp["single_arm"]["min"]), np.asarray(grp["gripper"]["min"])])
    print(f"\n--- {col} ---   (ckpt stats vs val, per joint)")
    print(f"{'j':>2} {'ckpt_max':>10} {'val_max':>10} {'exceed':>9} | {'ckpt_min':>10} {'val_min':>10} {'undercut':>9}  verdict")
    hi = lo = sat = 0
    for j in range(8):
        e = val[:, j].max() - smax[j]
        u = smin[j] - val[:, j].min()
        # is this joint saturating (train hits the same hardware limit)?
        sat_hi = abs(tr[:, j].max() - val[:, j].max()) < 1e-9
        sat_lo = abs(tr[:, j].min() - val[:, j].min()) < 1e-9
        v = []
        if e > 1e-9:
            v.append("EXCEEDS-HI")
            hi += 1
        if u > 1e-9:
            v.append("EXCEEDS-LO")
            lo += 1
        if not v:
            v.append("saturated(hw-limit)" if (sat_hi or sat_lo) else "in-range")
            if sat_hi or sat_lo:
                sat += 1
        print(
            f"{j:>2} {smax[j]:>10.3f} {val[:,j].max():>10.3f} {e:>+9.3f} | "
            f"{smin[j]:>10.3f} {val[:,j].min():>10.3f} {u:>+9.3f}  {'/'.join(v)}"
        )
    nonsat = 8 - sat
    print(f"    -> {hi}/8 joints val exceeds ckpt max, {lo}/8 undercuts ckpt min")
    print(f"    -> saturating (blind-spot) joints: {sat}/8; test power denominator = {nonsat} non-saturating dims")
    out[f"B_{key_prefix}_exceed_hi"] = hi
    out[f"B_{key_prefix}_exceed_lo"] = lo
    out[f"B_{key_prefix}_saturating"] = sat

    # control group: what WOULD leakage look like? full-131 stats == val max bit-for-bit
    full = np.concatenate([tr, val], 0)
    leak_hits = int(sum(1 for j in range(8) if full[:, j].max() == val[:, j].max()))
    print(f"    -> control: if stats had been aggregated over all 131 eps, "
          f"full_max == val_max bit-for-bit on {leak_hits}/8 joints (test would go silent there)")
    out[f"B_{key_prefix}_leak_control_hits"] = leak_hits

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print(json.dumps(out, indent=2, default=str))
ok_A = all(
    out[k] for k in ("A1_meta_eq_split", "A1_files_eq_split", "A1_videos_eq_split",
                     "A1_intersection_empty", "A2_inner_eq_split", "A3_frames_agree",
                     "A4_no_val_files_in_train_dir")
)
ok_B = (out["B_action_exceed_hi"] + out["B_action_exceed_lo"]) > 0
print(f"\nCHANNEL A (sampler)   : {'PASS' if ok_A else 'FAIL'}")
print(f"CHANNEL B (norm-stats): {'PASS' if ok_B else 'INCONCLUSIVE'}")
sys.exit(0 if (ok_A and ok_B) else 1)
