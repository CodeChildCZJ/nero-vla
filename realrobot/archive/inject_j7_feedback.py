#!/usr/bin/env python3
"""验 windows 的 j7 闭环正反馈假设 (离线, 用 v8 训练数据).
喂 in-dist grasp/approach 帧, 但把 state 的 j7 注入偏移 Δ∈{-10,0,+10,+15,+20}° (图像保持正确),
量模型命令的【绝对 j7】(chunk[0] & chunk-mean) 随 Δ 的斜率。
  因 openpi absolute_out = predicted_delta(image,state) + injected_state:
  斜率≈0 → 模型靠图像锚定把 j7 拉回流形 = 负反馈稳 → 真实漂来自图像 → v9 crop/图像稳定是解
  斜率≈1 → 模型跟随 state 不纠正 = 正反馈漂 → 病在过度信 state
用法: CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 uv run python inject_j7_feedback.py <ckpt> <tag>
"""
import os
import sys, glob
import numpy as np, pandas as pd
import av
from openpi.policies import policy_config as pc
from openpi.training import config as _config

CKPT = sys.argv[1]
TAG  = sys.argv[2] if len(sys.argv) > 2 else "v8_3k"
CFG  = "pi05_nero_pick_pink_sponge_v8"
DS   = f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v8"
PROMPT = "pick the pink sponge and place it in the blue bucket"
J4, J7, GRIP = 3, 6, 7
DELTAS = [-10.0, 0.0, 10.0, 15.0, 20.0]
N_EP = 20  # 抽样 ep 数

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
print("[ready]\n")

eps = sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))[:N_EP]
rows = []
for ep_path in eps:
    epi = int(ep_path.split("episode_")[-1].split(".")[0])
    df = pd.read_parquet(ep_path)
    S = np.stack(df["observation.state"].values).astype(np.float32)
    A = np.stack(df["action"].values).astype(np.float32)
    T = len(df)
    closed = np.where(A[:, GRIP] > 50)[0]
    if len(closed) == 0: continue
    t_g = int(closed[0])
    # 探测帧: grasp 前 approach 窗 (windows 说漂在 approach 后期发生)
    probe = [t for t in (t_g, max(0,t_g-15), max(0,t_g-30)) if t < T]
    high_mp4  = f"{DS}/videos/chunk-000/observation.images.cam_high/episode_{epi:06d}.mp4"
    wrist_mp4 = f"{DS}/videos/chunk-000/observation.images.cam_wrist/episode_{epi:06d}.mp4"
    hi = decode_frames(high_mp4, set(probe)); wr = decode_frames(wrist_mp4, set(probe))

    for t in probe:
        if t not in hi or t not in wr: continue
        base_j7 = float(S[t, J7]); train_j7 = float(A[t, J7])
        cmds = {}
        for d in DELTAS:
            st = S[t].copy(); st[J7] = base_j7 + d
            ch = np.asarray(policy.infer({
                "observation/image": hi[t], "observation/wrist_image": wr[t],
                "observation/state": st, "prompt": PROMPT})["actions"])
            cmds[d] = float(ch[0, J7])   # 命令的绝对 j7 (chunk[0])
        rows.append({"ep":epi,"t":t,"tg":t_g,"base_j7":base_j7,"train_j7":train_j7,
                     **{f"cmd_d{int(d)}":cmds[d] for d in DELTAS}})
    print(f"ep{epi} done (t_g={t_g})")

R = pd.DataFrame(rows)
R.to_json(f"{os.environ.get('NERO_ROOT', '.')}/realrobot/inject_j7_{TAG}.json", orient="records")
# 斜率: (cmd_j7(+20) - cmd_j7(0)) / 20, 以及逐 Δ
print("\n"+"="*70)
print(f"  INJECT-J7 正反馈测试  tag={TAG}  (n={len(R)} 帧, {len(eps)}ep)")
print("="*70)
base = R["cmd_d0"]
for d in DELTAS:
    if d==0: continue
    dd = (R[f"cmd_d{int(d)}"] - base)
    slope = (dd/d).mean()
    print(f"  注入 j7 {d:+.0f}°: 命令绝对j7 平均变化 {dd.mean():+5.1f}°  → 斜率 {slope:+.2f}  "
          f"({'拉回/稳' if slope<0.35 else '跟随/漂' if slope>0.65 else '中性'})")
# 总斜率 (线性拟合 over 所有 Δ)
alld, allc = [], []
for _,r in R.iterrows():
    for d in DELTAS:
        alld.append(d); allc.append(r[f"cmd_d{int(d)}"]-r["cmd_d0"])
slope_fit = np.polyfit(alld, allc, 1)[0]
print(f"\n  ★总斜率(线性拟合) ∂cmd_j7/∂注入 = {slope_fit:+.2f}")
print(f"   斜率≈0=图像锚定拉回(负反馈稳→v9 crop/图像稳定是解); ≈1=跟随state(正反馈漂)")
print("="*70)
