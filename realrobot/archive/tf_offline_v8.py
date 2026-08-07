#!/usr/bin/env python3
"""离线 teacher-forcing v8 — 加载 v8 ckpt, 喂 v8 训练真 obs 逐帧 (无误差累积).
核心新增 = windows 对齐的【复合 grasp 位姿合成】指标:
  模型输出 chunk 里有没有 同帧 j4<71 AND j7<45 (=真下扎+转腕到抓取). v7 此处=0.
判读(配 windows 在线 _analyze_smoke_traj):
  离线能合成 / 在线不能 = 闭环 OOD 漂 (v8 覆盖不够)
  离线在线都能 = 治好;  都不能 = 数据/模型容量.
用法: CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.3 \\
      uv run python tf_offline_v8.py <ckpt_dir> <tag>
"""
import os
import sys, glob
import numpy as np, pandas as pd
import av
import jax
from openpi.policies import policy_config as pc
from openpi.training import config as _config

CKPT = sys.argv[1]
TAG  = sys.argv[2] if len(sys.argv) > 2 else "v8_3k"
CFG  = "pi05_nero_pick_pink_sponge_v8"
DS   = f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v8"
PROMPT = "pick the pink sponge and place it in the blue bucket"
J4, J7, GRIP = 3, 6, 7
# 复合 grasp 位姿阈 (与 windows 在线检验一致)
J4_THR, J7_THR = 71.0, 45.0

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
grasp_rows, desc_rows, comp_rows = [], [], []
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
    H = 8
    fdj = np.array([S[min(t + H, T - 1), J4] - S[t, J4] for t in range(T)])
    desc_frames = sorted(int(t) for t in np.argsort(fdj)[:5] if t < T - H)
    need = {t_g, max(0, t_g - 30), max(0, t_g - 60)} | set(desc_frames)
    high_mp4  = f"{DS}/videos/chunk-000/observation.images.cam_high/episode_{epi:06d}.mp4"
    wrist_mp4 = f"{DS}/videos/chunk-000/observation.images.cam_wrist/episode_{epi:06d}.mp4"
    high  = decode_frames(high_mp4, need)
    wrist = decode_frames(wrist_mp4, need)

    def infer(t):
        return np.asarray(policy.infer({
            "observation/image": high[t], "observation/wrist_image": wrist[t],
            "observation/state": S[t], "prompt": PROMPT})["actions"])

    for label, t in [("approach-60", max(0, t_g-60)), ("approach-30", max(0, t_g-30)), ("grasp", t_g)]:
        if t not in high or t not in wrist: continue
        ch = infer(t)
        # 复合位姿: chunk 内任一帧 j4<71 & j7<45 同帧
        comp_mask = (ch[:, J4] < J4_THR) & (ch[:, J7] < J7_THR)
        grasp_rows.append({"ep": epi, "label": label, "t": t,
            "train_j4": float(A[t,J4]), "pred_j4_0": float(ch[0,J4]),
            "train_j7": float(A[t,J7]), "pred_j7_0": float(ch[0,J7]),
            "state_grip": float(S[t,GRIP]), "train_grip": float(A[t,GRIP]),
            "pred_grip_0": float(ch[0,GRIP]), "pred_grip_hmax": float(ch[:,GRIP].max()),
            "comp_frames": int(comp_mask.sum()),       # 复合帧数 (v7=0)
            "pred_j4_min": float(ch[:,J4].min()), "pred_j7_min": float(ch[:,J7].min())})
        comp_rows.append({"ep":epi,"label":label,"comp":int(comp_mask.sum()),"H":int(ch.shape[0])})

    for t in desc_frames:
        if t not in high or t not in wrist: continue
        ch = infer(t)
        desc_rows.append({"ep": epi, "t": t,
            "train_descent": float(A[min(t+15,T-1),J4]-A[t,J4]),
            "pred_descent": float(ch[-1,J4]-ch[0,J4]),
            "train_j4": float(A[t,J4]), "pred_j4_0": float(ch[0,J4])})
    print(f"ep{epi}: t_g={t_g} done")

G = pd.DataFrame(grasp_rows); D = pd.DataFrame(desc_rows)
G.to_json(f"{os.environ.get('NERO_ROOT', '.')}/realrobot/tf_offline_grasp_{TAG}.json", orient="records")
D.to_json(f"{os.environ.get('NERO_ROOT', '.')}/realrobot/tf_offline_desc_{TAG}.json", orient="records")
gg = G[G.label=="grasp"]; ap = G[G.label.str.startswith("approach")]
ratio = D.pred_descent.mean()/D.train_descent.mean() if len(D) and D.train_descent.mean() else float("nan")
n_comp_grasp = int((gg.comp_frames>0).sum())
print("\n"+"="*74)
print(f"  TF-OFFLINE v8  tag={TAG}  ckpt={CKPT.split('/')[-1]}")
print("="*74)
print(f"  ★[复合grasp位姿合成] grasp帧里输出chunk含 j4<{J4_THR:.0f}&j7<{J7_THR:.0f}同帧: "
      f"{n_comp_grasp}/{len(gg)} ep  (v7此处=0)  总复合帧 grasp={int(gg.comp_frames.sum())}")
print(f"  [grasp j4] 训练={gg.train_j4.mean():.1f} 模型chunk[0]={gg.pred_j4_0.mean():.1f} 模型min={gg.pred_j4_min.mean():.1f}")
print(f"  [grasp j7] 训练={gg.train_j7.mean():.1f} 模型chunk[0]={gg.pred_j7_0.mean():.1f} 模型min={gg.pred_j7_min.mean():.1f}")
print(f"  [下降比] 训练={D.train_descent.mean():.1f} 模型={D.pred_descent.mean():.1f} → {ratio:.0%} (n={len(D)})")
print(f"  [grasp 夹爪] state={gg.state_grip.mean():.0f} 模型chunk[0]={gg.pred_grip_0.mean():.0f} chunk-max={gg.pred_grip_hmax.mean():.0f}")
print(f"  [approach 夹爪] state={ap.state_grip.mean():.0f} 模型chunk[0]={ap.pred_grip_0.mean():.0f} (应~0保持开)")
print("="*74)
