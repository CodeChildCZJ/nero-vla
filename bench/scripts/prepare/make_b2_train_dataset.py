#!/usr/bin/env python3
"""Materialize local/pick_pink_sponge_b2_train = b2 restricted to split.json train_episodes (111).

LeRobot v2.1 layout: parquet rewritten with contiguous episode_index + global index,
videos symlinked (no copy), meta/{info,episodes,episodes_stats,tasks}.jsonl regenerated.
Read-only w.r.t. the source dataset.
"""
import json, os, pathlib, shutil, sys
import numpy as np
import pandas as pd

# b2 数据集本体(不公开,格式见 docs/DATA.md);默认取 LeRobot 本地缓存,可用 NERO_DATA 覆盖。
SRC = pathlib.Path(os.environ.get(
    "NERO_DATA", pathlib.Path.home() / ".cache/huggingface/lerobot/local/pick_pink_sponge_b2"))
# 111-episode 训练子集,默认写在源数据集旁边;可用 NERO_DATA_TRAIN 覆盖。
DST = pathlib.Path(os.environ.get("NERO_DATA_TRAIN", str(SRC) + "_train"))
# 仓库自包含:从本文件位置推导。bench/scripts/prepare/*.py -> parents[2] == bench/
BENCH = pathlib.Path(__file__).resolve().parents[2]

train_eps = json.loads((BENCH / "data" / "split.json").read_text())["train_episodes"]
print(f"train episodes: {len(train_eps)}")

if DST.exists():
    print(f"removing existing {DST}")
    shutil.rmtree(DST)
(DST / "data" / "chunk-000").mkdir(parents=True)
(DST / "meta").mkdir(parents=True)

info = json.loads((SRC / "meta" / "info.json").read_text())
video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
for vk in video_keys:
    (DST / "videos" / "chunk-000" / vk).mkdir(parents=True)

src_stats = {}
for line in (SRC / "meta" / "episodes_stats.jsonl").read_text().splitlines():
    d = json.loads(line)
    src_stats[d["episode_index"]] = d["stats"]
src_eplen = {}
for line in (SRC / "meta" / "episodes.jsonl").read_text().splitlines():
    d = json.loads(line)
    src_eplen[d["episode_index"]] = d

episodes_out, stats_out = [], []
global_idx = 0
for new_e, old_e in enumerate(train_eps):
    df = pd.read_parquet(SRC / f"data/chunk-000/episode_{old_e:06d}.parquet")
    L = len(df)
    assert L == src_eplen[old_e]["length"], f"length mismatch ep{old_e}"
    df["episode_index"] = np.full(L, new_e, dtype=np.int64)
    df["index"] = np.arange(global_idx, global_idx + L, dtype=np.int64)
    df.to_parquet(DST / f"data/chunk-000/episode_{new_e:06d}.parquet", index=False)

    for vk in video_keys:
        src_v = (SRC / f"videos/chunk-000/{vk}/episode_{old_e:06d}.mp4").resolve()
        assert src_v.exists(), src_v
        (DST / f"videos/chunk-000/{vk}/episode_{new_e:06d}.mp4").symlink_to(src_v)

    episodes_out.append({"episode_index": new_e, "tasks": src_eplen[old_e]["tasks"], "length": L})

    st = json.loads(json.dumps(src_stats[old_e]))  # deep copy
    st["episode_index"] = {"min": [new_e], "max": [new_e], "mean": [float(new_e)],
                           "std": [0.0], "count": [L]}
    lo, hi = global_idx, global_idx + L - 1
    arr = np.arange(lo, hi + 1, dtype=np.float64)
    st["index"] = {"min": [int(lo)], "max": [int(hi)], "mean": [float(arr.mean())],
                   "std": [float(arr.std())], "count": [L]}
    stats_out.append({"episode_index": new_e, "stats": st})
    global_idx += L

total_frames = global_idx
info_out = json.loads(json.dumps(info))
info_out["total_episodes"] = len(train_eps)
info_out["total_frames"] = total_frames
info_out["total_videos"] = len(train_eps) * len(video_keys)
info_out["splits"] = {"train": f"0:{len(train_eps)}"}
(DST / "meta" / "info.json").write_text(json.dumps(info_out, indent=4))
(DST / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes_out))
(DST / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in stats_out))
shutil.copy(SRC / "meta" / "tasks.jsonl", DST / "meta" / "tasks.jsonl")

print(f"wrote {DST}: {len(train_eps)} eps / {total_frames} frames "
      f"(src 131/{info['total_frames']}), video keys {video_keys}")
