#!/usr/bin/env python3
"""离线 teacher-forcing (直接加载ckpt, 不连server) — 量化任意 v7 ckpt 的:
  A. grasp帧: j4贴合 + 夹爪是否领先state(非copycat)
  B. 下降段: 模型命令下降幅度 / 训练下降幅度 = 下降比 (判欠训核心)
喂训练真obs逐帧, 无误差累积。一个进程加载一个ckpt跑完释放。
用法: CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \\
      uv run python tf_offline.py <ckpt_dir> <tag>
"""
import os
import sys, glob
import numpy as np, pandas as pd
import av
import jax
from openpi.policies import policy_config as pc
from openpi.training import config as _config

CKPT = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else "v7"
CFG = "pi05_nero_pick_pink_sponge_v7"
DS = f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v7"
PROMPT = "pick the pink sponge and place it in the blue bucket"
J4, GRIP = 3, 7

def decode_frames(mp4, idxs):
    want = set(int(i) for i in idxs); out = {}; i = 0
    c = av.open(mp4)
    for fr in c.decode(c.streams.video[0]):
        if i in want:
            out[i] = fr.to_ndarray(format="rgb24"); want.discard(i)
            if not want: break
        i += 1
    c.close(); return out

print(f"[load] {CKPT}  (GPU {jax.devices()})")
policy = pc.create_trained_policy(_config.get_config(CFG), CKPT, default_prompt=PROMPT)
print(f"[ready] tag={TAG}\n")

eps = sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))
grasp_rows, desc_rows = [], []
for ep_path in eps:
    epi = int(ep_path.split("episode_")[-1].split(".")[0])
    df = pd.read_parquet(ep_path)
    S = np.stack(df["observation.state"].values).astype(np.float32)
    A = np.stack(df["action"].values).astype(np.float32)
    T = len(df)
    closed = np.where(A[:, GRIP] > 50)[0]
    if len(closed) == 0:
        continue
    t_g = int(closed[0])
    # 下降最猛5帧 (未来8帧 j4 减最多)
    H = 8
    fdj = np.array([S[min(t + H, T - 1), J4] - S[t, J4] for t in range(T)])
    desc_frames = sorted(int(t) for t in np.argsort(fdj)[:5] if t < T - H)
    need = {t_g, max(0, t_g - 30), max(0, t_g - 60)} | set(desc_frames)
    high_mp4 = f"{DS}/videos/chunk-000/observation.images.cam_high/episode_{epi:06d}.mp4"
    wrist_mp4 = f"{DS}/videos/chunk-000/observation.images.cam_wrist/episode_{epi:06d}.mp4"
    high = decode_frames(high_mp4, need)
    wrist = decode_frames(wrist_mp4, need)

    def infer(t):
        return np.asarray(policy.infer({
            "observation/image": high[t], "observation/wrist_image": wrist[t],
            "observation/state": S[t], "prompt": PROMPT})["actions"])

    # A. grasp + approach
    for label, t in [("approach-60", max(0, t_g - 60)), ("approach-30", max(0, t_g - 30)), ("grasp", t_g)]:
        if t not in high or t not in wrist: continue
        ch = infer(t)
        grasp_rows.append({"ep": epi, "label": label, "t": t,
            "train_j4": float(A[t, J4]), "pred_j4_0": float(ch[0, J4]),
            "state_grip": float(S[t, GRIP]), "train_grip": float(A[t, GRIP]),
            "pred_grip_0": float(ch[0, GRIP]), "pred_grip_hmax": float(ch[:, GRIP].max())})
    # B. descent
    for t in desc_frames:
        if t not in high or t not in wrist: continue
        ch = infer(t)
        desc_rows.append({"ep": epi, "t": t,
            "train_descent": float(A[min(t + 15, T - 1), J4] - A[t, J4]),
            "pred_descent": float(ch[-1, J4] - ch[0, J4]),
            "train_j4": float(A[t, J4]), "pred_j4_0": float(ch[0, J4])})
    print(f"ep{epi}: t_g={t_g} done")

G = pd.DataFrame(grasp_rows); D = pd.DataFrame(desc_rows)
G.to_json(f"{os.environ.get('NERO_ROOT', '.')}/realrobot/tf_offline_grasp_{TAG}.json", orient="records")
D.to_json(f"{os.environ.get('NERO_ROOT', '.')}/realrobot/tf_offline_desc_{TAG}.json", orient="records")
gg = G[G.label == "grasp"]; ap = G[G.label.str.startswith("approach")]
ratio = D.pred_descent.mean() / D.train_descent.mean() if len(D) and D.train_descent.mean() else float("nan")
print("\n" + "=" * 70)
print(f"  TF-OFFLINE  tag={TAG}  ckpt={CKPT.split('/')[-1]}")
print("=" * 70)
print(f"  [下降比] 训练={D.train_descent.mean():.1f} 模型={D.pred_descent.mean():.1f} → {ratio:.0%}  (n={len(D)})")
print(f"  [grasp j4] 训练={gg.train_j4.mean():.1f} 模型={gg.pred_j4_0.mean():.1f} Δ={gg.pred_j4_0.mean()-gg.train_j4.mean():+.1f}")
print(f"  [grasp 夹爪] state={gg.state_grip.mean():.0f} 模型chunk[0]={gg.pred_grip_0.mean():.0f} chunk-max={gg.pred_grip_hmax.mean():.0f} (模型>>state=非copycat)")
print(f"  [approach 夹爪] state={ap.state_grip.mean():.0f} 模型chunk[0]={ap.pred_grip_0.mean():.0f} (应~0保持开)")
print("=" * 70)
