#!/usr/bin/env python3
"""开环 teacher-forcing audit: 喂训练原帧 obs, 看模型预测的 grip 闭不闭.
判 v5 是否在分布内学会了闭爪信号 (一刀切开 训练问题 vs 部署/OOD).

用法: CUDA_VISIBLE_DEVICES=3 uv run python openloop_audit.py <ckpt_dir> [ep_list]
默认 ckpt = v5/4000, ep = 25,50
输出: stdout 打印 grasp 区曲线 + 落 JSON 到 ${NERO_ROOT}/realrobot/openloop_audit_<step>.json
"""
import os
os.environ.setdefault("HF_LEROBOT_HOME", os.environ.get('NERO_DATA_ROOT', 'data'))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
import sys, json
import numpy as np
import pandas as pd

from openpi.training import config as _config
from openpi.policies import policy_config
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

CKPT = sys.argv[1] if len(sys.argv) > 1 else f"{os.environ.get('NERO_CKPT', 'checkpoints')}/pi05_nero_pick_pink_sponge_v5/v5/4000"
EPS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [25, 50]
REPO = "local/pick_pink_sponge_v3"
DATA = f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v3/data/chunk-000"
PROMPT = "pick the pink sponge and place it in the blue bucket"
GRIP, J4 = 7, 3
step = CKPT.rstrip("/").split("/")[-1]

print(f"[audit] ckpt={CKPT} step={step} eps={EPS}", flush=True)
cfg = _config.get_config("pi05_nero_pick_pink_sponge_v5")
policy = policy_config.create_trained_policy(cfg, CKPT)
ds = lerobot_dataset.LeRobotDataset(REPO)
print(f"[audit] policy + dataset loaded ({ds.num_frames} frames total)", flush=True)


def to_img(x):
    a = np.asarray(x.numpy() if hasattr(x, "numpy") else x)
    return a  # _NeroInputs 内部处理 CHW/float


out_all = {}
for ep in EPS:
    df = pd.read_parquet(f"{DATA}/episode_{ep:06d}.parquet")
    gidx = df["index"].values
    act_gt = np.stack(df["action"].values).astype(np.float32)   # (T,16) 绝对
    T = len(df)
    rows = []
    for f in range(T):
        frame = ds[int(gidx[f])]
        obs = {
            "observation/image": to_img(frame["observation.images.cam_high"]),
            "observation/wrist_image": to_img(frame["observation.images.cam_wrist"]),
            "observation/state": np.asarray(frame["observation.state"], dtype=np.float32),
            "prompt": PROMPT,
        }
        pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)  # (16,8) 绝对真实单位
        rows.append({
            "f": f,
            "rec_grip": round(float(act_gt[f, GRIP]), 1),
            "pred_grip0": round(float(pred[0, GRIP]), 1),         # chunk 第0步预测 grip
            "pred_grip_min": round(float(pred[:, GRIP].min()), 1), # chunk 16步内最小 grip (有没有想闭)
            "rec_j4": round(float(act_gt[f, J4]), 1),
            "pred_j4_0": round(float(pred[0, J4]), 1),
        })
        if f % 40 == 0:
            print(f"  ep{ep} {f}/{T}", flush=True)

    # 定位 grasp 区: rec_grip 从 >40 跌到 <25 的第一段
    rg = np.array([r["rec_grip"] for r in rows])
    opened = np.where(rg > 40)[0]
    grasp_f = None
    if len(opened):
        after = np.where((rg < 25) & (np.arange(T) > opened[0]))[0]
        if len(after):
            grasp_f = int(after[0])
    out_all[f"ep{ep}"] = {"T": T, "grasp_frame": grasp_f, "rows": rows}

    # 打印 grasp 区 ±12 帧
    print(f"\n===== ep{ep}  (T={T}, grasp_frame={grasp_f}) =====", flush=True)
    print(f"{'f':>4} {'rec_grip':>9} {'pred_g0':>8} {'pred_gmin':>10} | {'rec_j4':>7} {'pred_j4':>8}", flush=True)
    if grasp_f is not None:
        lo, hi = max(0, grasp_f - 12), min(T, grasp_f + 13)
    else:
        lo, hi = 0, min(T, 25)
    for r in rows[lo:hi]:
        mark = " <<GRASP" if r["f"] == grasp_f else ""
        print(f"{r['f']:>4} {r['rec_grip']:>9} {r['pred_grip0']:>8} {r['pred_grip_min']:>10} | {r['rec_j4']:>7} {r['pred_j4_0']:>8}{mark}", flush=True)
    # 关键判据数值
    if grasp_f is not None:
        win = rows[max(0, grasp_f - 3):grasp_f + 6]
        rec_min = min(r["rec_grip"] for r in win)
        pred0_min = min(r["pred_grip0"] for r in win)
        predmin_min = min(r["pred_grip_min"] for r in win)
        print(f"  [判据] grasp区 录制grip最小={rec_min}  预测chunk0最小={pred0_min}  预测chunk内最小={predmin_min}", flush=True)
        verdict = "闭(<=20)→学会了" if pred0_min <= 20 else ("半闭" if pred0_min <= 35 else "没闭(>35)→分布内也不闭")
        print(f"  [判据] 喂训练图模型预测 chunk0 grip 是否闭: {verdict}", flush=True)

outp = f"{os.environ.get('NERO_ROOT', '.')}/realrobot/openloop_audit_{step}.json"
with open(outp, "w") as f:
    json.dump(out_all, f)
print(f"\n[audit] 全量曲线落: {outp}", flush=True)
