#!/usr/bin/env python3
"""离线 teacher-forcing B (pi05_nero_b1) — 喂 B 训练真 obs 逐帧, 看 grasp 帧模型能否合成 j4→96.
判读: 离线能合成 j4(到~96) → 在线 j4=46 是部署/欠训差距; 离线也合成不了 → 模型3k还没学会grasp协调(欠训)。
用法: CUDA_VISIBLE_DEVICES=0 uv 环境用 openpi-agilex venv
"""
import os
import sys, glob
import numpy as np, pandas as pd
import av
from openpi.policies import policy_config as pc
from openpi.training import config as _config

CKPT = sys.argv[1]
TAG  = sys.argv[2] if len(sys.argv) > 2 else "b1_3k"
CFG  = "pi05_nero_b1"
DS   = f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_b1"
PROMPT = "pick the pink sponge and place it in the blue bucket"
J4, J7, GRIP = 3, 6, 7   # B 夹爪: 0mm=open/78mm=closed → closed=high

def decode_frames(mp4, idxs):
    want = set(int(i) for i in idxs); out = {}; i = 0
    c = av.open(mp4)
    for fr in c.decode(c.streams.video[0]):
        if i in want:
            out[i] = fr.to_ndarray(format="rgb24"); want.discard(i)
            if not want: break
        i += 1
    c.close(); return out

print(f"[load] {CKPT}")
policy = pc.create_trained_policy(_config.get_config(CFG), CKPT, default_prompt=PROMPT)
print(f"[ready] tag={TAG}\n")

eps = sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))
rows = []
for ep_path in eps:
    epi = int(ep_path.split("episode_")[-1].split(".")[0])
    df = pd.read_parquet(ep_path)
    S = np.stack(df["observation.state"].values).astype(np.float32)
    A = np.stack(df["action"].values).astype(np.float32)
    T = len(df)
    closed = np.where(A[:, GRIP] > 50)[0]   # 闭爪=high mm
    if len(closed) == 0: continue
    t_g = int(closed[0])
    probe = [("approach-30", max(0,t_g-30)), ("grasp", t_g)]
    hi = f"{DS}/videos/chunk-000/observation.images.cam_high/episode_{epi:06d}.mp4"
    wr = f"{DS}/videos/chunk-000/observation.images.cam_wrist/episode_{epi:06d}.mp4"
    need = {t for _,t in probe}
    H = decode_frames(hi, need); W = decode_frames(wr, need)
    for label, t in probe:
        if t not in H or t not in W: continue
        ch = np.asarray(policy.infer({
            "observation/image": H[t], "observation/wrist_image": W[t],
            "observation/state": S[t], "prompt": PROMPT})["actions"])
        rows.append({"ep":epi,"label":label,"t":t,
            "train_j4":float(A[t,J4]),"pred_j4_0":float(ch[0,J4]),"pred_j4_min":float(ch[:,J4].min()),"pred_j4_max":float(ch[:,J4].max()),
            "train_j7":float(A[t,J7]),"pred_j7_0":float(ch[0,J7]),
            "train_grip":float(A[t,GRIP]),"pred_grip_max":float(ch[:,GRIP].max())})
    print(f"ep{epi}: t_g={t_g} done")

R = pd.DataFrame(rows)
R.to_json(f"{os.environ.get('NERO_ROOT', '.')}/realrobot/tf_offline_b1_{TAG}.json", orient="records")
g = R[R.label=="grasp"]
print("\n"+"="*70)
print(f"  TF-OFFLINE B  tag={TAG}  ckpt={CKPT.split('/')[-1]}  (n={len(R)}, {len(eps)}ep)")
print("="*70)
print(f"  ★[grasp j4] 训练={g.train_j4.mean():.1f}  模型chunk[0]={g.pred_j4_0.mean():.1f}  chunk内min={g.pred_j4_min.mean():.1f} max={g.pred_j4_max.mean():.1f}")
print(f"   → 离线能否合成 j4→训练值? 差={abs(g.pred_j4_0.mean()-g.train_j4.mean()):.1f}° (真机在线 j4=46 vs 训练96 差50)")
print(f"  [grasp j7] 训练={g.train_j7.mean():.1f}  模型chunk[0]={g.pred_j7_0.mean():.1f}")
print(f"  [grasp 夹爪] 训练={g.train_grip.mean():.0f}  模型chunk-max={g.pred_grip_max.mean():.0f} (闭=high)")
print("="*70)
