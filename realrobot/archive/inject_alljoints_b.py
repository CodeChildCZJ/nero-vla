#!/usr/bin/env python3
"""B 全7关节 inject 自反馈 slope 测 (确认 j7 是否【唯一】失控轴, 为 b2 single-vs-multi-axis recovery 定调)。
对每个关节 j∈0..6 单独注入 Δ∈{-10,0,10,15,20}° 到 state[j](图像保持正确, 锚真 approach@35%+grasp@48% 帧),
量 cmd[0,j] 随 Δ 的 slope。slope≈1=该轴零视觉纠正正反馈(闭环会漂); ≈0=视觉锚定拉回。
用法: CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
      HF_HUB_OFFLINE=1 ${NERO_ROOT}/third_party/openpi-agilex/.venv/bin/python3 inject_alljoints_b.py <ckpt> <tag>
"""
import os
import sys, glob
import numpy as np, pandas as pd, av
from openpi.policies import policy_config as pc
from openpi.training import config as _config

CKPT = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else "b1"
CFG = "pi05_nero_b1"
DS  = f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_b1"
PROMPT = "pick the pink sponge and place it in the blue bucket"
GRIP = 7; DELTAS = [-10., 0., 10., 15., 20.]; N_EP = 10

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
print("[ready]")
eps = sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))[:N_EP]
recs = []
for ep_path in eps:
    epi = int(ep_path.split("episode_")[-1].split(".")[0])
    df = pd.read_parquet(ep_path)
    S = np.stack(df["observation.state"].values).astype(np.float32)
    A = np.stack(df["action"].values).astype(np.float32)
    g = A[:, GRIP]; lo, hi = float(g.min()), float(g.max()); rng = max(1e-6, hi - lo)
    op = np.where(g > lo + 0.75 * rng)[0]; cl = np.where(g < lo + 0.25 * rng)[0]
    if len(op) == 0: continue
    t_o = int(op[0]); t_g = -1
    for t in cl:
        if t > t_o: t_g = int(t); break
    if t_g < 0: continue
    frames = [t_o, t_g]
    hi_mp4 = f"{DS}/videos/chunk-000/observation.images.cam_high/episode_{epi:06d}.mp4"
    wr_mp4 = f"{DS}/videos/chunk-000/observation.images.cam_wrist/episode_{epi:06d}.mp4"
    H = decode_frames(hi_mp4, set(frames)); W = decode_frames(wr_mp4, set(frames))
    for t in frames:
        if t not in H or t not in W: continue
        for j in range(7):
            for d in DELTAS:
                st = S[t].copy(); st[j] = S[t, j] + d
                ch = np.asarray(policy.infer({"observation/image": H[t], "observation/wrist_image": W[t],
                    "observation/state": st, "prompt": PROMPT})["actions"])
                recs.append({"ep": epi, "t": t, "joint": j, "delta": d, "cmd": float(ch[0, j])})
    print(f"ep{epi} done (t_o={t_o},t_g={t_g})")

R = pd.DataFrame(recs)
R.to_json(f"{os.environ.get('NERO_ROOT', '.')}/realrobot/inject_alljoints_{TAG}.json", orient="records")
print("\n" + "=" * 60)
print(f"  全7关节 inject slope  tag={TAG}  ({len(eps)}ep, frames=approach+grasp)")
print("=" * 60)
for j in range(7):
    sub = R[R.joint == j]
    alld, allc = [], []
    for (ep, t), grp in sub.groupby(["ep", "t"]):
        c0 = grp[grp.delta == 0]["cmd"]
        if len(c0) == 0: continue
        c0 = c0.values[0]
        for _, r in grp.iterrows():
            alld.append(r.delta); allc.append(r.cmd - c0)
    s = np.polyfit(alld, allc, 1)[0] if alld else float("nan")
    tag = "★失控(正反馈漂)" if s > 0.65 else "稳/视觉拉回" if s < 0.35 else "中性"
    print(f"  j{j+1}: slope={s:+.2f}  {tag}")
print("=" * 60)
print("  期望: 只有 j7≈1 失控, 其余<0.65 → b2 单 j7 recovery 够; 若别轴也~1 → 需多轴 recovery")
