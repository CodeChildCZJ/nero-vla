#!/usr/bin/env python3
"""Extract NERO b2 frames to 256x256 JPEGs + per-episode state/action arrays for OpenVLA-OFT.

Mirrors the official OFT ALOHA preprocessing (`preprocess_split_aloha_data.py`):
480x640 -> 256x256 BICUBIC. The model's own image transform does the final 224x224.

Output layout:
  <out>/frames/ep{E:03d}/{cam_high,cam_wrist}/{frame:05d}.jpg
  <out>/traj.npz  -> per-episode 'ep{E:03d}_state' (T,8) and 'ep{E:03d}_action' (T,8)
"""
import argparse
import json
import os
import pathlib

import av
import numpy as np
import pandas as pd
from PIL import Image

# 仓库自包含:从本文件位置推导。bench/scripts/prepare/*.py -> parents[2] == bench/
BENCH = pathlib.Path(__file__).resolve().parents[2]
# b2 数据集本体(不公开,格式见 docs/DATA.md);默认取 LeRobot 本地缓存,可用 NERO_DATA 覆盖。
DS = pathlib.Path(os.environ.get(
    "NERO_DATA", pathlib.Path.home() / ".cache/huggingface/lerobot/local/pick_pink_sponge_b2"))
# h264 转码副本,由 build_gr00t_dataset.py 生成在 bench/data/ 下。
H264 = BENCH / "data" / "b2_h264"
CAMS = ("cam_high", "cam_wrist")


def decode_video(path, size):
    """Decode all frames of an mp4 to a list of PIL Images resized to (size,size)."""
    out = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            img = frame.to_image()
            out.append(img.resize((size, size), resample=Image.BICUBIC))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BENCH / "data" / "b2_oft"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    n_eps = json.loads((DS / "meta" / "info.json").read_text())["total_episodes"]

    traj = {}
    for ep in range(n_eps):
        df = pd.read_parquet(DS / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(df["action"].to_numpy()).astype(np.float32)
        assert len(state) == len(action)
        traj[f"ep{ep:03d}_state"] = state
        traj[f"ep{ep:03d}_action"] = action

        for cam in CAMS:
            dst = out / "frames" / f"ep{ep:03d}" / cam
            dst.mkdir(parents=True, exist_ok=True)
            done = len(list(dst.glob("*.jpg")))
            if done == len(state):
                continue
            vid = H264 / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{ep:06d}.mp4"
            imgs = decode_video(vid, args.size)
            if len(imgs) != len(state):
                print(f"[warn] ep{ep} {cam}: {len(imgs)} frames vs {len(state)} rows -> truncating to min")
            for i in range(min(len(imgs), len(state))):
                imgs[i].save(dst / f"{i:05d}.jpg", quality=args.quality)
        print(f"ep{ep:03d} T={len(state)} done", flush=True)

    np.savez_compressed(out / "traj.npz", **traj)
    print("WROTE", out / "traj.npz")


if __name__ == "__main__":
    main()
