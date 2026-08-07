#!/usr/bin/env python3
"""离线 teacher-forcing — 主动下降段 j4 跟踪 (判别 model 学没学到"下扎")。
grasp 帧 j4 近静止, 贴合不够判别力; 真机"下扎浅"的核心是 approach 的主动下降段。
做法: 每 ep 找 j4 下降最快的若干帧, 喂真 obs, 比 model 命令的 j4-delta(pred-state) vs 训练 delta(action-state)。
  delta 匹配 → model 学到下降 → 真机浅=部署/闭环问题(非欠训)
  delta 偏小/反 → model 没学到下降 → 欠训, 等 9k
用法: python tf_eval_descent.py [host] [port] [tag]
"""
import os
import sys, glob
import numpy as np, pandas as pd
import av
from openpi_client import websocket_client_policy as wcp

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9095
TAG  = sys.argv[3] if len(sys.argv) > 3 else "v7_3k"
DS = f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v7"
PROMPT = "pick the pink sponge and place it in the blue bucket"
J4, GRIP = 3, 7

def decode_frames(mp4, idxs):
    want = set(int(i) for i in idxs); out = {}; i = 0
    c = av.open(mp4)
    for frame in c.decode(c.streams.video[0]):
        if i in want:
            out[i] = frame.to_ndarray(format="rgb24"); want.discard(i)
            if not want: break
        i += 1
    c.close(); return out

client = wcp.WebsocketClientPolicy(host=HOST, port=PORT)
print(f"[server] {HOST}:{PORT}  tag={TAG}  (主动下降段 j4-delta 跟踪)\n")

eps = sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))
rows = []
for ep_path in eps:
    epi = int(ep_path.split("episode_")[-1].split(".")[0])
    df = pd.read_parquet(ep_path)
    S = np.stack(df["observation.state"].values).astype(np.float32)
    A = np.stack(df["action"].values).astype(np.float32)
    T = len(df)
    j4_state = S[:, J4]
    # 平滑后找下降最快的帧 (j4 减小 = 下扎). 用未来8帧的 j4 变化作为"该往下走多少"
    H = 8
    future_dj4 = np.array([j4_state[min(t + H, T - 1)] - j4_state[t] for t in range(T)])
    # 下降最猛的 5 个帧 (future_dj4 最负)
    desc_frames = np.argsort(future_dj4)[:5]
    desc_frames = sorted(int(t) for t in desc_frames if t < T - H)
    if not desc_frames: continue
    high_mp4 = f"{DS}/videos/chunk-000/observation.images.cam_high/episode_{epi:06d}.mp4"
    wrist_mp4 = f"{DS}/videos/chunk-000/observation.images.cam_wrist/episode_{epi:06d}.mp4"
    high = decode_frames(high_mp4, desc_frames)
    wrist = decode_frames(wrist_mp4, desc_frames)
    for t in desc_frames:
        if t not in high or t not in wrist: continue
        obs = {"observation/image": high[t], "observation/wrist_image": wrist[t],
               "observation/state": S[t], "prompt": PROMPT}
        chunk = np.asarray(client.infer(obs)["actions"])  # (16,8) absolute
        # 训练: 未来 H 帧 j4 真实走到哪 (绝对) ; 训练 action chunk 头部 = A[t..t+H]
        train_j4_future = A[min(t + H, T - 1), J4]
        # 模型 chunk 第 H 步的 j4 (绝对)
        pred_j4_H = float(chunk[min(H, len(chunk) - 1), J4])
        pred_j4_last = float(chunk[-1, J4])
        rows.append({
            "ep": epi, "t": t, "state_j4": float(S[t, J4]),
            "train_j4_now": float(A[t, J4]), "train_j4_+H": float(train_j4_future),
            "pred_j4_0": float(chunk[0, J4]), "pred_j4_+H": pred_j4_H, "pred_j4_last": pred_j4_last,
            # delta = 该帧起 model 想让 j4 走多少 (chunk末 - chunk头), 训练同理
            "train_descent": float(A[min(t + 15, T - 1), J4] - A[t, J4]),
            "pred_descent": float(chunk[-1, J4] - chunk[0, J4]),
        })
    print(f"ep{epi}: desc_frames={desc_frames} done")

R = pd.DataFrame(rows)
R.to_json(f"{os.environ.get('NERO_ROOT', '.')}/realrobot/tf_descent_{TAG}.json", orient="records")
print(f"\n[saved] tf_descent_{TAG}.json  ({len(R)} rows)\n")
print("=" * 76)
print(f"  主动下降段 j4 跟踪  tag={TAG}  (n={len(R)} 帧, 取每ep下降最猛5帧)")
print("=" * 76)
print(f"  state_j4 均={R.state_j4.mean():.1f}  (下降段起点, 高=臂在上)")
print(f"  绝对 j4 (chunk 末步 vs 训练同步):")
print(f"    训练 j4(t)     ={R.train_j4_now.mean():7.1f}")
print(f"    模型 chunk[0]  ={R.pred_j4_0.mean():7.1f}   Δ vs 训练={R.pred_j4_0.mean()-R.train_j4_now.mean():+.1f}")
print(f"    模型 chunk末步 ={R.pred_j4_last.mean():7.1f}")
print(f"  ★下降幅度 (chunk末 - chunk头, 负=往下扎):")
print(f"    训练 descent(15步)={R.train_descent.mean():7.1f}")
print(f"    模型 descent(chunk)={R.pred_descent.mean():7.1f}")
ratio = R.pred_descent.mean() / R.train_descent.mean() if R.train_descent.mean() != 0 else float("nan")
print(f"    模型/训练 下降比 = {ratio:.0%}")
print("  判读: 比~100%=学到下降(真机浅是闭环/部署问题); 比<<100%=没学到=欠训等9k")
print("=" * 76)
