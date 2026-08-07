#!/usr/bin/env python3
"""全 horizon inject: 同 inject_alljoints_b 协议, 但记录 chunk 全 10 步(非只 chunk[0])。
回答: client 每次 query 执行~10 步开环, 后面的 horizon 步会不会用视觉锚定(slope 随 k 降)?
若 slope 全 k≈1 → 整段 horizon 纯 dead-reckoning, receding-horizon(执行1步重规划)也救不了。
若后段 k slope 降 → 晚步视觉锚定 → 缩短开环执行步数=便宜的缓解。
用法: CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
      HF_HUB_OFFLINE=1 ${NERO_ROOT}/third_party/openpi-agilex/.venv/bin/python3 inject_horizon_b.py <ckpt> <tag>
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
GRIP = 7; DELTAS = [-10., 0., 10., 15., 20.]; N_EP = 10; H = 10

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
    Hd = decode_frames(hi_mp4, set(frames)); Wd = decode_frames(wr_mp4, set(frames))
    for t in frames:
        if t not in Hd or t not in Wd: continue
        for j in range(7):
            for d in DELTAS:
                st = S[t].copy(); st[j] = S[t, j] + d
                ch = np.asarray(policy.infer({"observation/image": Hd[t], "observation/wrist_image": Wd[t],
                    "observation/state": st, "prompt": PROMPT})["actions"])
                rec = {"ep": epi, "t": t, "joint": j, "delta": d}
                for k in range(H):
                    rec[f"k{k}"] = float(ch[k, j])
                recs.append(rec)
    print(f"ep{epi} done (t_o={t_o},t_g={t_g})")

R = pd.DataFrame(recs)
R.to_json(f"{os.environ.get('NERO_ROOT', '.')}/realrobot/inject_horizon_{TAG}.json", orient="records")
print("\n" + "=" * 70)
print(f"  全 horizon inject slope  tag={TAG}  ({len(eps)}ep)  行=每个 chunk 步 k 的 slope")
print("=" * 70)
print("  关节 | " + " ".join(f"k{k}" for k in range(H)))
for j in range(7):
    sub = R[R.joint == j]
    row = []
    for k in range(H):
        alld, allc = [], []
        for (ep, t), grp in sub.groupby(["ep", "t"]):
            c0 = grp[grp.delta == 0][f"k{k}"]
            if len(c0) == 0: continue
            c0 = c0.values[0]
            for _, r in grp.iterrows():
                alld.append(r.delta); allc.append(r[f"k{k}"] - c0)
        s = np.polyfit(alld, allc, 1)[0] if alld else float("nan")
        row.append(f"{s:+.2f}")
    print(f"  j{j+1}   | " + " ".join(row))
print("=" * 70)
print("  读法: 一行内 slope 随 k 降=晚步视觉锚定(缩短开环可缓解); 全 k≈1=整段纯跟随(hz 救不了)")
