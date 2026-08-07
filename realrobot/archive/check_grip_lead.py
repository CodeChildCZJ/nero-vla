#!/usr/bin/env python3
"""数据侧验 action.grip 是 leader命令 还是 follower state拷贝:
- 过渡帧(夹爪开合, |Δstate.grip|大) 上 action.grip vs state.grip 差多少
- lead-lag: action.grip[t] 跟 state.grip[t+k] 哪个lag误差最小
  lag=0 → action==当前state = 拷贝/主从同步 (copycat根源, mask是对的)
  lag>0 → action领先state k帧 = 真命令/目标 (改记leader命令可根治, 比mask好)
"""
import os
import glob, sys
import numpy as np
import pandas as pd

# argv[1] = 数据集 chunk 目录 或 数据集根(自动找 data/chunk-*); 默认 v3
ARG = sys.argv[1] if len(sys.argv) > 1 else f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v3/data/chunk-000"
files = sorted(glob.glob(f"{ARG}/episode_*.parquet"))
if not files:  # 传的是数据集根目录
    files = sorted(glob.glob(f"{ARG}/data/chunk-*/episode_*.parquet"))
print(f"[输入] {ARG} → {len(files)} episodes")
GRIP = 7
all_sg, all_ag = [], []
trans_same, trans_next = [], []   # 过渡帧 |a-s_t| 和 |a-s_{t+1}|
lag_errs = {k: [] for k in range(-2, 6)}
for fp in files:
    df = pd.read_parquet(fp)
    sg = np.stack(df["observation.state"].values)[:, GRIP].astype(np.float64)
    ag = np.stack(df["action"].values)[:, GRIP].astype(np.float64)
    all_sg.append(sg); all_ag.append(ag)
    ds = np.abs(np.diff(sg))
    trans = np.where(ds > 5.0)[0]   # 夹爪移动>5mm的帧
    for t in trans:
        trans_same.append(abs(ag[t] - sg[t]))
        if t + 1 < len(sg):
            trans_next.append(abs(ag[t] - sg[t + 1]))
    # lead-lag: action[t] vs state[t+k]
    for k in lag_errs:
        if k >= 0:
            a = ag[:len(ag) - k] if k > 0 else ag
            s = sg[k:]
        else:
            a = ag[-k:]; s = sg[:len(sg) + k]
        n = min(len(a), len(s))
        if n > 0:
            lag_errs[k].append(np.mean(np.abs(a[:n] - s[:n])))

sg_all = np.concatenate(all_sg); ag_all = np.concatenate(all_ag)
print(f"[全帧] action.grip vs state.grip: mean|a-s|={np.mean(np.abs(ag_all-sg_all)):.3f}mm, max={np.max(np.abs(ag_all-sg_all)):.2f}")
print(f"[全帧] state.grip范围 [{sg_all.min():.0f},{sg_all.max():.0f}], action.grip范围 [{ag_all.min():.0f},{ag_all.max():.0f}]")
print(f"\n[过渡帧 |Δstate|>5mm] 共{len(trans_same)}帧")
if trans_same:
    print(f"  action vs 当前state |a-s_t|:   mean={np.mean(trans_same):.3f} max={np.max(trans_same):.2f}")
    print(f"  action vs 下一state |a-s_t+1|: mean={np.mean(trans_next):.3f} max={np.max(trans_next):.2f}")
print(f"\n[lead-lag] action[t] vs state[t+k] 平均误差(越小越贴):")
for k in sorted(lag_errs):
    if lag_errs[k]:
        print(f"  lag={k:+d}: {np.mean(lag_errs[k]):.3f}mm")
best = min((k for k in lag_errs if lag_errs[k]), key=lambda k: np.mean(lag_errs[k]))
print(f"\n[判决] 最贴lag={best:+d}")
if best == 0 and (not trans_same or np.mean(trans_same) < 2.0):
    print("  → action.grip ≡ 当前state (过渡帧也重合) = **拷贝/主从同步, copycat根源** → v7走(A)mask")
else:
    print(f"  → action.grip 领先state {best}帧 / 过渡帧差大 = **真命令/目标** → v7可走(B)记leader命令根治")
