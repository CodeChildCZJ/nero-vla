#!/usr/bin/env python3
"""重核 B 夹爪约定 (windows 纠正: 0mm=闭合 / ~76mm=张开, 跟我旧记的反了)。
dump B 数据集每条 episode 的 action[GRIP=7] 原始 mm 轨迹形状, 定位【真抓取帧=mm掉到接近min(闭)】
及其相位%, 看 phase-shortcut 结论方向是否反。
用法: ${NERO_ROOT}/third_party/openpi-agilex/.venv/bin/python3 dump_grip_b.py
"""
import os
import glob
import numpy as np, pandas as pd

DS = f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_b1"
GRIP = 7
eps = sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))
print(f"{len(eps)} episodes  (GRIP idx={GRIP})\n")

close_phases = []   # 真闭(mm掉到低带)首帧相位
for ep_path in eps:
    epi = int(ep_path.split("episode_")[-1].split(".")[0])
    df = pd.read_parquet(ep_path)
    A = np.stack(df["action"].values).astype(np.float32)
    S = np.stack(df["observation.state"].values).astype(np.float32)
    g = A[:, GRIP]; sg = S[:, GRIP]; T = len(g)
    lo, hi = float(g.min()), float(g.max())
    rng = max(1e-6, hi - lo)
    thr_close = lo + 0.25 * rng     # 接近闭(低 mm)
    thr_open  = lo + 0.75 * rng     # 接近张(高 mm)
    open_fr  = np.where(g > thr_open)[0]
    close_fr = np.where(g < thr_close)[0]
    first_open = int(open_fr[0]) if len(open_fr) else -1
    # 真抓取 = 第一次【张开之后】掉进闭带
    grasp = -1
    for t in close_fr:
        if first_open >= 0 and t > first_open:
            grasp = int(t); break
    gp = grasp / T * 100 if grasp > 0 else -1
    if grasp > 0:
        close_phases.append(gp)
    # 12点降采样看形状
    ds = g[:: max(1, T // 12)]
    traj = " ".join(f"{x:4.0f}" for x in ds)
    print(f"ep{epi:02d} T={T:3d} g0={g[0]:3.0f} gT={g[-1]:3.0f} min={lo:3.0f} max={hi:3.0f} | "
          f"firstOPEN={first_open:3d}({first_open/T*100:3.0f}%) GRASP(闭)={grasp:3d}({gp:4.0f}%) "
          f"| state_g0={sg[0]:3.0f}")
    print(f"     traj(mm): {traj}")

cp = np.array(close_phases)
print("\n" + "=" * 70)
print(f"真抓取(闭)相位: n={len(cp)}  mean={cp.mean():.1f}%  std={cp.std():.1f}%  "
      f"min={cp.min():.0f}%  max={cp.max():.0f}%")
print("  -> std 窄(<~5%)=相位捷径; 宽=随姿态/视觉变 (对照旧'31-39%闭'是否其实是张爪相位)")
print("=" * 70)
