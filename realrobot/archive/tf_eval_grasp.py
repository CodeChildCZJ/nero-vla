#!/usr/bin/env python3
"""离线开环 teacher-forcing: 喂训练集帧 obs 给 v7 server, 对比 model commanded vs 训练 action.
诊断 2 件事 (windows/czj 要的):
  1. commanded j4@grasp 是否随 step 从~37 向训练~93 收敛 (下扎深度学没学到)
  2. 闭爪信号: model 在 approach(开爪期) 是否乱闭, 在 grasp 是否真闭; 是否在复读 state.grip (copycat)
逐帧喂训练真 obs (cam_high+cam_wrist 原生480x640 + 16维state), 无误差累积。
用法: python tf_eval_grasp.py [host] [port] [tag]
"""
import os
import sys, glob, json
import numpy as np, pandas as pd
import av
from openpi_client import websocket_client_policy as wcp

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9095
TAG  = sys.argv[3] if len(sys.argv) > 3 else "v7_3k"
DS = f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v7"
PROMPT = "pick the pink sponge and place it in the blue bucket"
J4, GRIP = 3, 7
GRIP_CLOSE_THR = 50.0  # action.grip>50 = 闭爪(closed); <50 = 开爪

def decode_frames(mp4, idxs):
    want = set(int(i) for i in idxs)
    out = {}
    container = av.open(mp4)
    stream = container.streams.video[0]
    i = 0
    for frame in container.decode(stream):
        if i in want:
            out[i] = frame.to_ndarray(format="rgb24")  # HWC uint8 RGB
            want.discard(i)
            if not want:
                break
        i += 1
    container.close()
    return out

client = wcp.WebsocketClientPolicy(host=HOST, port=PORT)
print("[meta]", client.get_server_metadata())
print(f"[server] {HOST}:{PORT}  tag={TAG}\n")

eps = sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))
rows = []
n_grasp = 0
for ep_path in eps:
    epi = int(ep_path.split("episode_")[-1].split(".")[0])
    df = pd.read_parquet(ep_path)
    S = np.stack(df["observation.state"].values).astype(np.float32)
    A = np.stack(df["action"].values).astype(np.float32)
    T = len(df)
    ga = A[:, GRIP]
    closed = np.where(ga > GRIP_CLOSE_THR)[0]
    if len(closed) == 0:
        print(f"ep{epi}: no grasp frame (action.grip never >{GRIP_CLOSE_THR}), skip")
        continue
    n_grasp += 1
    t_g = int(closed[0])  # 第一次闭爪 = 抓海绵
    samples = {
        "approach-60": max(0, t_g - 60),
        "approach-30": max(0, t_g - 30),
        "grasp": t_g,
        "grasp+10": min(T - 1, t_g + 10),
    }
    high_mp4 = f"{DS}/videos/chunk-000/observation.images.cam_high/episode_{epi:06d}.mp4"
    wrist_mp4 = f"{DS}/videos/chunk-000/observation.images.cam_wrist/episode_{epi:06d}.mp4"
    idxs = list(samples.values())
    high = decode_frames(high_mp4, idxs)
    wrist = decode_frames(wrist_mp4, idxs)
    for label, t in samples.items():
        if t not in high or t not in wrist:
            continue
        obs = {
            "observation/image": high[t],
            "observation/wrist_image": wrist[t],
            "observation/state": S[t],
            "prompt": PROMPT,
        }
        chunk = np.asarray(client.infer(obs)["actions"])  # (16,8)
        c0 = chunk[0]
        rows.append({
            "ep": epi, "label": label, "t": t,
            "train_j4": float(A[t, J4]), "pred_j4_0": float(c0[J4]), "pred_j4_hmean": float(chunk[:, J4].mean()),
            "train_grip": float(A[t, GRIP]), "state_grip": float(S[t, GRIP]),
            "pred_grip_0": float(c0[GRIP]), "pred_grip_hmax": float(chunk[:, GRIP].max()), "pred_grip_hmin": float(chunk[:, GRIP].min()),
        })
    print(f"ep{epi}: t_g={t_g} (T={T}) done")

R = pd.DataFrame(rows)
out_json = f"{os.environ.get('NERO_ROOT', '.')}/realrobot/tf_eval_{TAG}.json"
R.to_json(out_json, orient="records")
print(f"\n[saved] {out_json}  ({len(R)} rows, {n_grasp} episodes w/ grasp)\n")

print("=" * 78)
print(f"  TEACHER-FORCING 报告  tag={TAG}  (n={n_grasp} grasp episodes)")
print("=" * 78)
for label in ["approach-60", "approach-30", "grasp", "grasp+10"]:
    g = R[R.label == label]
    if len(g) == 0:
        continue
    tj, pj, pjh = g.train_j4.mean(), g.pred_j4_0.mean(), g.pred_j4_hmean.mean()
    tg, sg, pg0, pgmax = g.train_grip.mean(), g.state_grip.mean(), g.pred_grip_0.mean(), g.pred_grip_hmax.mean()
    # 闭爪判定: model chunk 任一步 grip>50 算"命令闭爪"
    frac_pred_close = (g.pred_grip_hmax > GRIP_CLOSE_THR).mean()
    frac_train_close = (g.train_grip > GRIP_CLOSE_THR).mean()
    print(f"\n[{label}] (n={len(g)})")
    print(f"  j4:   训练={tj:7.1f} | 模型chunk[0]={pj:7.1f}  chunk均={pjh:7.1f}  | Δ(模型-训练)={pj-tj:+.1f}")
    print(f"  grip: 训练={tg:6.1f} state={sg:6.1f} | 模型chunk[0]={pg0:6.1f} chunk内max={pgmax:6.1f}")
    print(f"  闭爪率: 训练={frac_train_close:.0%}  模型(chunk有>50)={frac_pred_close:.0%}")

# copycat 检验: 模型 grip 是否在复读 state.grip
print("\n" + "=" * 78)
print("  COPYCAT 检验: 模型chunk[0].grip vs state.grip 的偏离 (越大越说明从视觉决策非复读)")
gall = R.copy()
gall["dev_from_state"] = (gall.pred_grip_0 - gall.state_grip).abs()
print(f"  全样本 |pred_grip[0] - state_grip| mean={gall.dev_from_state.mean():.1f}  max={gall.dev_from_state.max():.1f}")
# 关键: grasp 帧, state 刚变闭, 模型是否也闭 (若领先=好); approach 帧 state 开, 模型是否保持开
gr = R[R.label == "grasp"]
ap = R[R.label.isin(["approach-60", "approach-30"])]
if len(gr):
    print(f"  grasp帧: state_grip均={gr.state_grip.mean():.0f} 模型chunk[0].grip均={gr.pred_grip_0.mean():.0f} chunk-max均={gr.pred_grip_hmax.mean():.0f}")
if len(ap):
    print(f"  approach帧: state_grip均={ap.state_grip.mean():.0f} 模型chunk[0].grip均={ap.pred_grip_0.mean():.0f} (应保持开~低)")
print("=" * 78)
