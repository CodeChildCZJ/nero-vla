#!/usr/bin/env python3
"""Materialize GR00T-compatible LeRobot datasets from NERO b2.

Two outputs, both under BENCH/data:
  b2_gr00t_train : the 111 split.json train episodes  -> what we finetune on
  b2_gr00t_val   : the 20 split.json val episodes     -> only used to build obs at predict time

Both point at a shared h264 transcode of the b2 videos (b2_h264/), because the
original b2 mp4s are AV1 and neither decord nor torchcodec 0.1 can decode AV1.
Parquets are symlinked from the read-only source. Episode indices are kept
unchanged so LeRobot's episode_%06d path pattern still resolves.

Crucially, GR00T computes meta/stats.json by globbing data/*/*.parquet of the
dataset dir it is given -> the train dir only sees the 111 train episodes, so
the min/max normalization stats carry no val leakage.
"""
import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# 仓库自包含:从本文件位置推导。bench/scripts/prepare/*.py -> parents[2] == bench/
BENCH = pathlib.Path(__file__).resolve().parents[2]
# b2 数据集本体(不公开,格式见 docs/DATA.md);默认取 LeRobot 本地缓存,可用 NERO_DATA 覆盖。
SRC = pathlib.Path(os.environ.get(
    "NERO_DATA", pathlib.Path.home() / ".cache/huggingface/lerobot/local/pick_pink_sponge_b2"))
# 派生数据集写在 bench/data/ 下,与冻结的 split.json / val_anchors.npz 同级(不重名)。
OUT = BENCH / "data"
H264 = OUT / "b2_h264"
CAMS = ["observation.images.cam_high", "observation.images.cam_wrist"]

MODALITY = {
    "state": {"single_arm": {"start": 0, "end": 7}, "gripper": {"start": 7, "end": 8}},
    "action": {"single_arm": {"start": 0, "end": 7}, "gripper": {"start": 7, "end": 8}},
    "video": {
        "cam_high": {"original_key": "observation.images.cam_high"},
        "cam_wrist": {"original_key": "observation.images.cam_wrist"},
    },
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}


def transcode(args):
    src, dst = args
    if dst.exists() and dst.stat().st_size > 0:
        return dst, "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-g", "15", "-an", str(tmp),
    ]
    subprocess.run(cmd, check=True)
    tmp.rename(dst)
    return dst, "done"


def nframes(p):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(p)],
        capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def main():
    split = json.loads((BENCH / "data" / "split.json").read_text())
    eps_meta = [json.loads(l) for l in (SRC / "meta/episodes.jsonl").read_text().splitlines()]
    len_by_ep = {e["episode_index"]: e["length"] for e in eps_meta}

    # ---- 1. transcode all videos av1 -> h264 (shared by train and val dirs)
    jobs = []
    for cam in CAMS:
        for e in range(len(eps_meta)):
            jobs.append((SRC / f"videos/chunk-000/{cam}/episode_{e:06d}.mp4",
                         H264 / f"videos/chunk-000/{cam}/episode_{e:06d}.mp4"))
    print(f"transcoding {len(jobs)} videos av1->h264 ...", flush=True)
    with ThreadPoolExecutor(max_workers=12) as ex:
        done = 0
        for _dst, status in ex.map(transcode, jobs):
            done += 1
            if done % 40 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    # ---- 2. verify frame counts survived the transcode
    bad = []
    for cam in CAMS:
        for e in (split["val_episodes"] + split["train_episodes"][:5]):
            n = nframes(H264 / f"videos/chunk-000/{cam}/episode_{e:06d}.mp4")
            if n != len_by_ep[e]:
                bad.append((cam, e, n, len_by_ep[e]))
    if bad:
        print("FRAME COUNT MISMATCH:", bad[:10])
        sys.exit(1)
    print("frame counts verified on all val eps + 5 train eps", flush=True)

    # ---- 3. build the two dataset dirs
    info = json.loads((SRC / "meta/info.json").read_text())
    for name, eps in [("b2_gr00t_train", split["train_episodes"]),
                      ("b2_gr00t_val", split["val_episodes"])]:
        root = OUT / name
        (root / "meta").mkdir(parents=True, exist_ok=True)
        (root / "data/chunk-000").mkdir(parents=True, exist_ok=True)

        for e in eps:
            link = root / f"data/chunk-000/episode_{e:06d}.parquet"
            if not link.is_symlink():
                link.symlink_to(SRC / f"data/chunk-000/episode_{e:06d}.parquet")
            for cam in CAMS:
                d = root / f"videos/chunk-000/{cam}"
                d.mkdir(parents=True, exist_ok=True)
                vlink = d / f"episode_{e:06d}.mp4"
                if not vlink.is_symlink():
                    vlink.symlink_to(H264 / f"videos/chunk-000/{cam}/episode_{e:06d}.mp4")

        ninfo = json.loads(json.dumps(info))
        ninfo["total_episodes"] = len(eps)
        ninfo["total_frames"] = sum(len_by_ep[e] for e in eps)
        ninfo["total_videos"] = len(eps) * len(CAMS)
        ninfo["splits"] = {"train": f"0:{len(eps)}"}
        for cam in CAMS:
            ninfo["features"][cam]["info"]["video.codec"] = "h264"
        (root / "meta/info.json").write_text(json.dumps(ninfo, indent=4))
        with open(root / "meta/episodes.jsonl", "w") as f:
            for e in eps:
                f.write(json.dumps({"episode_index": e,
                                    "tasks": ["pick the pink sponge and place it in the blue bucket"],
                                    "length": len_by_ep[e]}) + "\n")
        (root / "meta/tasks.jsonl").write_text((SRC / "meta/tasks.jsonl").read_text())
        (root / "meta/modality.json").write_text(json.dumps(MODALITY, indent=4))
        # force stats recompute from this dir's parquets only
        stats = root / "meta/stats.json"
        if stats.exists():
            stats.unlink()
        print(f"{name}: {len(eps)} eps, {ninfo['total_frames']} frames -> {root}", flush=True)


if __name__ == "__main__":
    main()
