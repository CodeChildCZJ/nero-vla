#!/usr/bin/env python3
"""POSITIVE (direct-recompute) proof that N1.7's baked stats are TRAIN-111-only.

Reverse falsification (`val_max > train_stat_max`) only fails to CONTRADICT
"train-only". team-lead re-prioritised the positive form: replicate GR00T's own
statistic computation over exactly the 111 train episodes, read from the ORIGINAL
131-episode source (never from the train dir, whose own provenance is what is in
question), and compare bit-for-bit to the numbers actually baked into the ckpt.

Why this is not a duplicate of archive/n17_stats_provenance.py: that one proved
live == {ckpt statistics.json, experiment_cfg copy, train dir meta}. All four are
the SAME table, so it proves consistency, not scope -- if the train dir's meta had
itself been generated over all 131 episodes, every one of those comparisons would
still pass. The 131-episode counterfactual below is what closes that hole, and
VAL-20 (the old must-differ control) never could: a 131-table differs from both.

N1.7 needs TWO recomputes, because it loads TWO normalization artifacts:
  A  meta/stats.json          absolute, 8 dims (operative for the GRIPPER only)
  B  meta/relative_stats.json the arm's 16x7 per-horizon-step table (7 of 8 dims)
B has never been recomputed forward by anyone on this bench; N1.5 has no such file.

Replication notes (port upstream or "bit-for-bit" is meaningless):
  - gr00t/data/stats.py:137 sorted(paths) + pd.concat(axis=0)     -> row order fixed
  - np.vstack([np.asarray(x, dtype=np.float32) ...])              -> float32 accum
  - mean/std are float32-accumulation-order dependent; min/max/quantile are not
  - the relative table is built with the REAL JointPose/JointActionChunk classes
    (gr00t/data/stats.py:323 load_relative_actions), then cross-checked against a
    vectorised form -- so "I ported it correctly" is asserted, not assumed
CPU-only. Does not touch the GPU.
"""
import pathlib
import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("NERO_DATA", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2")))
TRAIN_DIR = BENCH / "data/b2_n17_train"

LOWDIM = ["observation.state", "action", "timestamp"]
SLICES = {"single_arm": slice(0, 7), "gripper": slice(7, 8)}
COLMAP = {"state": "observation.state", "action": "action"}
STAT_KEYS = ["max", "min", "mean", "std", "q01", "q99"]


def ep_parquet(ep: int) -> Path:
    return SOURCE / f"data/chunk-000/episode_{ep:06d}.parquet"


# ---------------------------------------------------------------- artifact A
def calculate_dataset_statistics(parquet_paths, features=LOWDIM):
    """Verbatim port of gr00t/data/stats.py:137 (tqdm/prints stripped)."""
    all_low_dim_data_list = []
    for parquet_path in sorted(list(parquet_paths)):
        all_low_dim_data_list.append(pd.read_parquet(parquet_path))
    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)

    dataset_statistics = {}
    for le_modality in features:
        np_data = np.vstack(
            [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
        )
        dataset_statistics[le_modality] = dict(
            mean=np.mean(np_data, axis=0).tolist(),
            std=np.std(np_data, axis=0).tolist(),
            min=np.min(np_data, axis=0).tolist(),
            max=np.max(np_data, axis=0).tolist(),
            q01=np.quantile(np_data, 0.01, axis=0).tolist(),
            q99=np.quantile(np_data, 0.99, axis=0).tolist(),
        )
    return dataset_statistics, len(all_low_dim_data)


# ---------------------------------------------------------------- artifact B
def relative_trajectories(eps, horizon, use_upstream_classes=True):
    """Port of RelativeActionLoader.load_relative_actions (gr00t/data/stats.py:323).

    Upstream sources its rows from LeRobotEpisodeLoader over a dataset DIR; we read
    the source parquet instead so the episode set is ours to choose. Everything
    downstream of that (the reference frame, the subtraction, the float32 cast) is
    upstream's own code.
    """
    if use_upstream_classes:
        from gr00t.data.state_action.action_chunking import JointActionChunk
        from gr00t.data.state_action.pose import JointPose

    trajectories = []
    for ep in eps:  # ascending == LeRobotEpisodeLoader's positional order
        df = pd.read_parquet(ep_parquet(ep), columns=["observation.state", "action"])
        state_data = np.vstack([np.asarray(x, dtype=np.float32) for x in df["observation.state"]])
        action_data = np.vstack([np.asarray(x, dtype=np.float32) for x in df["action"]])
        arm = SLICES["single_arm"]
        state_data, action_data = state_data[:, arm], action_data[:, arm]

        usable_length = len(df) - (horizon - 1)  # delta_indices[-1] == horizon-1
        for i in range(usable_length):
            last_state = state_data[i]  # state delta_indices[-1] == 0
            actions = action_data[i : i + horizon]
            if use_upstream_classes:
                ref = JointPose(last_state)
                traj = JointActionChunk([JointPose(m) for m in actions]).relative_chunking(
                    reference_frame=ref
                )
                trajectories.append(np.stack([p.joints for p in traj.poses], dtype=np.float32))
            else:
                trajectories.append((actions - last_state).astype(np.float32))
    return trajectories


def calculate_stats_for_key(trajectories):
    """Verbatim port of gr00t/data/stats.py:366 (the reduction half)."""
    return {
        "max": np.max(trajectories, axis=0),
        "min": np.min(trajectories, axis=0),
        "q01": np.quantile(trajectories, 0.01, axis=0),
        "q99": np.quantile(trajectories, 0.99, axis=0),
        "mean": np.mean(trajectories, axis=0),
        "std": np.std(trajectories, axis=0),
    }


# ---------------------------------------------------------------- comparison
def baked(ckpt: Path):
    d = json.loads((ckpt / "statistics.json").read_text())
    assert "new_embodiment" in d, list(d)
    return d["new_embodiment"]


def cmp_absolute(recomputed, ref):
    """(label -> max abs deviation) for the sliced absolute table."""
    out = {}
    for grp in ("state", "action"):
        for sub in ("single_arm", "gripper"):
            for st in STAT_KEYS:
                a = np.asarray(recomputed[COLMAP[grp]][st], dtype=np.float64)[SLICES[sub]]
                b = np.asarray(ref[grp][sub][st], dtype=np.float64)
                assert a.shape == b.shape, (grp, sub, st, a.shape, b.shape)
                out[f"{grp}.{sub}.{st}"] = float(np.abs(a - b).max())
    return out


def cmp_relative(recomputed, ref):
    out = {}
    for st in STAT_KEYS:
        a = np.asarray(recomputed[st], dtype=np.float64)
        b = np.asarray(ref["relative_action"]["single_arm"][st], dtype=np.float64)
        assert a.shape == b.shape, (st, a.shape, b.shape)
        out[f"relative_action.single_arm.{st}"] = float(np.abs(a - b).max())
    return out


def verdict(name, devs):
    bad = {k: v for k, v in devs.items() if v != 0.0}
    print(f"  {name}: {len(devs) - len(bad)}/{len(devs)} cells dev = 0.00e+00")
    for k, v in sorted(bad.items(), key=lambda kv: -kv[1])[:8]:
        print(f"      {k:<38} dev {v:.6e}")
    return len(bad) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(pathlib.Path(os.environ.get("NERO_CKPT", str(pathlib.Path(__file__).resolve().parents[2] / "checkpoints"))) / "n17_smoke" / "smoke" / "checkpoint-10"))
    ap.add_argument("--json-out", default=str(BENCH / "logs/n17_leak_direct_recompute.json"))
    ap.add_argument("--horizon", type=int, default=0, help="0 = read from the baked table")
    args = ap.parse_args()

    split = json.loads((BENCH / "data" / "split.json").read_text())
    train_eps, val_eps = split["train_episodes"], split["val_episodes"]
    all_eps = sorted(train_eps + val_eps)
    assert not set(train_eps) & set(val_eps)
    print(f"train {len(train_eps)} | val {len(val_eps)} | source total {len(all_eps)}")

    ck = baked(Path(args.ckpt))
    H = args.horizon or np.asarray(ck["relative_action"]["single_arm"]["max"]).shape[0]
    print(f"horizon (from baked relative table) = {H}")

    # The train dir must carry the SAME episode files, or "sorted order" differs and
    # the float32 mean/std accumulation would not be reproducible even when clean.
    tr_names = sorted(p.name for p in (TRAIN_DIR / "data/chunk-000").glob("*.parquet"))
    src_names = sorted(ep_parquet(e).name for e in train_eps)
    assert tr_names == src_names, (
        f"train dir episode numbering differs from source: "
        f"{set(tr_names) ^ set(src_names)}"
    )
    print(f"train dir filenames == source filenames for the 111 train eps ({len(tr_names)})")

    results = {"ckpt": args.ckpt, "horizon": H}

    # ---- A: absolute table -------------------------------------------------
    print("\n=== A: meta/stats.json (absolute; gripper-operative) ===")
    rec_train, n_train = calculate_dataset_statistics([ep_parquet(e) for e in train_eps])
    print(f"  TRAIN-111 recompute over {n_train} rows")
    a_ok = verdict("TRAIN-111 vs ckpt baked", cmp_absolute(rec_train, ck))

    rec_all, n_all = calculate_dataset_statistics([ep_parquet(e) for e in all_eps])
    print(f"  leaked counterfactual: ALL-131 recompute over {n_all} rows")
    dev_all = cmp_absolute(rec_all, ck)
    moved = {k: v for k, v in dev_all.items() if v != 0.0}
    print(f"  ALL-131 vs ckpt baked: {len(moved)}/{len(dev_all)} cells MOVE "
          f"(min nonzero shift {min(moved.values()):.3e})" if moved else "  ALL-131: NOTHING MOVED")
    dead_abs = sorted(k for k, v in dev_all.items() if v == 0.0)
    print(f"  zero-resolution cells (leak undetectable here): {len(dead_abs)}/{len(dev_all)}")
    for k in dead_abs:
        print(f"      {k}")

    # ---- B: relative table -------------------------------------------------
    print("\n=== B: meta/relative_stats.json (RELATIVE arm, 16x7 = 7 of 8 dims) ===")
    tr_up = relative_trajectories(train_eps, H, use_upstream_classes=True)
    tr_vec = relative_trajectories(train_eps, H, use_upstream_classes=False)
    same = all(np.array_equal(a, b) for a, b in zip(tr_up, tr_vec))
    print(f"  upstream-classes vs vectorised port: bit-identical={same} "
          f"({len(tr_up)} chunks of {tr_up[0].shape})")
    assert same, "my vectorised port is NOT upstream's arithmetic -- do not trust the rest"

    rec_rel = calculate_stats_for_key(tr_up)
    b_ok = verdict("TRAIN-111 vs ckpt baked", cmp_relative(rec_rel, ck))

    rec_rel_all = calculate_stats_for_key(relative_trajectories(all_eps, H, False))
    dev_rel_all = cmp_relative(rec_rel_all, ck)
    moved_rel = {k: v for k, v in dev_rel_all.items() if v != 0.0}
    print(f"  leaked counterfactual ALL-131: {len(moved_rel)}/{len(dev_rel_all)} cells MOVE"
          + (f" (min nonzero shift {min(moved_rel.values()):.3e})" if moved_rel else ""))
    dead_rel = sorted(k for k, v in dev_rel_all.items() if v == 0.0)
    for k in dead_rel:
        print(f"      zero-resolution: {k}")

    results.update(
        train_rows=n_train, all_rows=n_all,
        A_train111_vs_baked=cmp_absolute(rec_train, ck),
        A_all131_vs_baked=dev_all, A_dead_cells=dead_abs,
        B_train111_vs_baked=cmp_relative(rec_rel, ck),
        B_all131_vs_baked=dev_rel_all, B_dead_cells=dead_rel,
        B_port_bit_identical=bool(same),
        PASS=bool(a_ok and b_ok and moved and moved_rel),
    )
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 66)
    if results["PASS"]:
        print("PASS: baked stats reproduce EXACTLY from the 111 train episodes,")
        print("      and the 131-episode leak counterfactual is detectable.")
    else:
        print("!!! NOT PASS -- read the per-cell deviations above")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
