#!/usr/bin/env python3
"""查训练数据每个episode第0帧的夹爪态(state[7] + action[7]) — 验windows#3:
部署SAFE_HOME grip=0(闭) 是否 == 训练ep第0帧夹爪态。失配=t=0分布失配。
也看每ep最后一帧(收尾态)和全程grip范围。"""
import glob, os
import numpy as np
import pandas as pd

DATA = f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v3/data/chunk-000"
files = sorted(glob.glob(f"{DATA}/episode_*.parquet"))
print(f"[frame0] {len(files)} episodes")
f0_state, f0_act, flast_state = [], [], []
gmins, gmaxs = [], []
for fp in files:
    df = pd.read_parquet(fp)
    st = np.stack(df["observation.state"].values)   # (T,8?)
    ac = np.stack(df["action"].values)              # (T,8?)
    g_st = st[:, 7]; g_ac = ac[:, 7]
    f0_state.append(g_st[0]); f0_act.append(g_ac[0]); flast_state.append(g_st[-1])
    gmins.append(g_st.min()); gmaxs.append(g_st.max())
f0_state, f0_act, flast_state = map(np.array, (f0_state, f0_act, flast_state))
print(f"\n[每ep第0帧] state.grip: mean={f0_state.mean():.1f} min={f0_state.min():.1f} max={f0_state.max():.1f}")
print(f"[每ep第0帧] action.grip: mean={f0_act.mean():.1f} min={f0_act.min():.1f} max={f0_act.max():.1f}")
print(f"[每ep最后帧] state.grip: mean={flast_state.mean():.1f} min={flast_state.min():.1f} max={flast_state.max():.1f}")
print(f"[全程grip范围] 各ep min均值={np.mean(gmins):.1f}, max均值={np.mean(gmaxs):.1f}")
print(f"\n[前8个ep第0帧 state.grip] {np.round(f0_state[:8],1).tolist()}")
print(f"[判据] 部署SAFE_HOME grip=0(闭). 若训练第0帧grip>40(开) = t=0失配坐实, v7采集起步态要对齐")
