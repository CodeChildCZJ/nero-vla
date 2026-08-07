#!/usr/bin/env python
"""Prepare a HF-LeRobot-0.6-readable working copy of the NERO b2 dataset.

Two things stand between b2 and `lerobot-train`:

1. FORMAT. b2 is codebase_version v2.1; LeRobot >=0.6 only reads v3.0 and raises
   BackwardCompatibilityError otherwise. The shipped converter
   (lerobot.scripts.convert_dataset_v21_to_v30) works IN PLACE and deletes the old
   meta/stats.json -- and the v2.1 original is shared read-only with the GR00T and
   openpi backbones (CONTRACT.md). So we convert a COPY and never touch the original.

2. NORMALIZATION LEAKAGE. LeRobotDataset(episodes=[...111 train ids...]) filters the
   frames you train on, but `dataset.meta.stats` is just meta/stats.json loaded whole
   -- aggregated over all 131 episodes, val included. Those stats become the
   normalize/unnormalize constants, so val statistics would leak into every backbone
   trained here. We therefore re-aggregate stats from the per-episode stats of the
   111 train episodes ONLY and write that as the working copy's stats.json.

Run once before training. Idempotent (re-copies from the pristine source each time).
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

# 仓库自包含:从本文件位置推导。bench/scripts/prepare/*.py -> parents[2] == bench/
BENCH = pathlib.Path(__file__).resolve().parents[2]
# b2 数据集本体(不公开,格式见 docs/DATA.md);默认取 LeRobot 本地缓存,可用 NERO_DATA 覆盖。
SRC = pathlib.Path(os.environ.get(
    "NERO_DATA", pathlib.Path.home() / ".cache/huggingface/lerobot/local/pick_pink_sponge_b2"))
# v3.0 工作副本,默认写在源数据集旁边;可用 NERO_DATA_BENCH 覆盖。
DST = pathlib.Path(os.environ.get("NERO_DATA_BENCH", str(SRC) + "_bench"))
REPO_ID = "local/pick_pink_sponge_b2_bench"
# LeRobot 第三方栈的 venv(见 docs/ 的环境说明);可用 NERO_LEROBOT_PY 覆盖。
PY = os.environ.get(
    "NERO_LEROBOT_PY", str(BENCH.parent / "third_party" / "lerobot" / ".venv" / "bin" / "python"))


def main():
    split = json.loads((BENCH / "data" / "split.json").read_text())
    train_eps = set(split["train_episodes"])
    val_eps = set(split["val_episodes"])
    assert len(train_eps) == 111 and len(val_eps) == 20 and not (train_eps & val_eps)

    # ---- 1. pristine copy (original stays v2.1, untouched) ----
    # The converter shuffles <root> -> <root>_old and <root>_v30 -> <root>; stale
    # siblings from an earlier run would send it down its recovery path, so clear them.
    for p in (DST, DST.with_name(DST.name + "_old"), DST.with_name(DST.name + "_v30")):
        if p.exists():
            shutil.rmtree(p)
    shutil.copytree(SRC, DST)
    print(f"[copy] {SRC} -> {DST}")

    # per-episode stats must be read BEFORE conversion (it rewrites meta/)
    per_ep = {}
    with open(SRC / "meta" / "episodes_stats.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            per_ep[int(rec["episode_index"])] = rec["stats"]
    print(f"[stats] read per-episode stats for {len(per_ep)} episodes")

    # ---- 2. v2.1 -> v3.0 (in place, on the copy) ----
    cmd = [PY, "-m", "lerobot.scripts.convert_dataset_v21_to_v30",
           f"--repo-id={REPO_ID}", f"--root={DST}", "--push-to-hub=false"]
    print("[convert]", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-5000:], file=sys.stderr)
        raise SystemExit(f"conversion failed rc={r.returncode}")
    ver = json.loads((DST / "meta" / "info.json").read_text())["codebase_version"]
    print(f"[convert] ok, codebase_version={ver}")
    # the converter parks the pre-conversion v2.1 copy here; 460MB we don't need
    stale = DST.with_name(DST.name + "_old")
    if stale.exists():
        shutil.rmtree(stale)
        print(f"[cleanup] removed {stale}")

    # ---- 3. train-only normalization stats ----
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_stats

    def to_np(s):
        return {k: {kk: np.asarray(vv) for kk, vv in v.items()} for k, v in s.items()}

    train_stats_list = [to_np(per_ep[e]) for e in sorted(train_eps)]
    train_stats = aggregate_stats(train_stats_list)
    write_stats(train_stats, DST)
    print(f"[stats] wrote train-only stats.json from {len(train_stats_list)} episodes")

    # ---- 4. verify the fix actually moved the numbers (i.e. it was really leaking) ----
    all_stats = aggregate_stats([to_np(per_ep[e]) for e in sorted(per_ep)])
    a, t = all_stats["action"], train_stats["action"]
    print("[verify] action mean  all131:", np.round(a["mean"], 3))
    print("[verify] action mean train111:", np.round(t["mean"], 3))
    print("[verify] max |mean| delta:", float(np.abs(a["mean"] - t["mean"]).max()))
    print("[verify] max |std|  delta:", float(np.abs(a["std"] - t["std"]).max()))
    print("[verify] keys:", sorted(train_stats.keys()))


if __name__ == "__main__":
    main()
