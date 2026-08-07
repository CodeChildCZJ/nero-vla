#!/usr/bin/env python3
"""量化真机帧 vs 训练grasp帧 视觉距离 (确诊病B多重: 亮度/色偏/直方图/结构).
用法: uv run python image_diff.py <obs_dump.npz> [real_frame_idx]
对照训练 ep25 frame164 (grasp帧). 真机帧默认取中段.
"""
import os, sys
os.environ.setdefault("HF_LEROBOT_HOME", os.environ.get('NERO_DATA_ROOT', 'data'))
import numpy as np
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

NPZ = sys.argv[1]
z = np.load(NPZ, allow_pickle=True)
keys = list(z.keys())
def pick(*subs):
    for k in keys:
        if all(s in k.lower() for s in subs): return k
    return None
kh, kw, ks = pick("high") or pick("base"), pick("wrist"), pick("state")
print(f"[diff] NPZ keys={[(k,np.asarray(z[k]).shape) for k in keys]}")
highs, wrists = np.asarray(z[kh]), np.asarray(z[kw])
state = np.asarray(z[ks]) if ks else None
T = len(highs)
rf = int(sys.argv[2]) if len(sys.argv) > 2 else T // 2  # 真机代表帧
print(f"[diff] 真机帧数={T}, 取代表帧#{rf}" + (f" (state.grip={state[rf][7]:.0f} j4={state[rf][3]:.0f})" if state is not None else ""))

# 训练 grasp 帧 ep25 f164
ds = lerobot_dataset.LeRobotDataset("local/pick_pink_sponge_v3")
import pandas as pd
df = pd.read_parquet(f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v3/data/chunk-000/episode_000025.parquet")
g164 = int(df["index"].values[164])
tf = ds[g164]
def to_hwc(x):
    a = np.asarray(x.numpy() if hasattr(x, "numpy") else x)
    if a.ndim == 3 and a.shape[0] == 3: a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = (255*a).astype(np.uint8) if a.max() <= 1.01 else a.astype(np.uint8)
    return a
tr_high, tr_wrist = to_hwc(tf["observation.images.cam_high"]), to_hwc(tf["observation.images.cam_wrist"])

def cmp(name, real, train):
    real = real.astype(np.float32); train = train.astype(np.float32)
    if real.shape != train.shape:
        # resize训练帧到真机帧尺寸 (简单最近邻)
        from numpy import linspace
        h, w = real.shape[:2]
        yi = np.clip((linspace(0, train.shape[0]-1, h)).astype(int), 0, train.shape[0]-1)
        xi = np.clip((linspace(0, train.shape[1]-1, w)).astype(int), 0, train.shape[1]-1)
        train = train[yi][:, xi]
    print(f"\n=== {name} (real{real.shape} vs train{train.shape}) ===")
    print(f"  每通道均值 real RGB: {[round(real[...,c].mean(),1) for c in range(3)]}")
    print(f"  每通道均值 train RGB: {[round(train[...,c].mean(),1) for c in range(3)]}")
    print(f"  亮度差(real-train): {real.mean()-train.mean():+.1f} (|>15|=明显光照差)")
    print(f"  色偏(各通道均值差): {[round(real[...,c].mean()-train[...,c].mean(),1) for c in range(3)]}")
    mse = ((real-train)**2).mean()
    print(f"  整图 MSE: {mse:.0f} (0=同图; >2000=差异大)  RMSE={mse**0.5:.1f}/255")
    # 直方图相关 (每通道)
    hc = []
    for c in range(3):
        hr,_ = np.histogram(real[...,c], 32, (0,255)); ht,_ = np.histogram(train[...,c], 32, (0,255))
        hr = hr/hr.sum(); ht = ht/ht.sum()
        corr = np.corrcoef(hr, ht)[0,1]
        hc.append(round(corr,2))
    print(f"  直方图相关(每通道, 1=同分布): {hc}  均值{np.mean(hc):.2f}")

cmp("cam_high (俯视)", highs[rf], tr_high)
cmp("cam_wrist (腕)", wrists[rf], tr_wrist)
print("\n[判读] 亮度差|>15|或直方图相关<0.7 = 真机视觉明显OOD, 病B重, 需覆盖真机条件的数据/域随机.")
print("[判读] 若各项都接近(亮度差<10, 直方图>0.85) = 视觉其实in-dist, 失败更可能是闭环动力学(执行层修).")
