#!/usr/bin/env python3
"""Result 2: b1 训练集 grasp-window 统计 (windows 规格, 纠正约定 0mm=闭/76mm=张)。
open>50mm / close<15mm。每 ep 第一次 open->close transition:
  t_close = 首个 t 满足 grip[t-1]>50 且 grip[t]<15 且 min(grip[t:t+5])<15 (防噪)。
窗口 [t_close-10, t_close+10] (30fps 数据≈前后各 0.33s; windows 说10Hz 前后各1s——我们数据30fps, 用±10帧)。
检测信号=action(leader) gripper(=抓取意图, 跟 rollout 命令端可比); 构型检查 j4≈93±8 & j7≈23±8 用 state(follower 实际臂位, 阈值来自 grasp_pose_b)。
输出 6 通道 mean/std/min/max + 构型达标帧比例 + 无 transition 的 ep 数。纯 CPU。
用法: ${NERO_ROOT}/third_party/openpi-agilex/.venv/bin/python3 grasp_window_b.py
"""
import os
import glob, json
import numpy as np, pandas as pd

DS = f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_b1"
J4, J7, GRIP = 3, 6, 7
OPEN_TH, CLOSE_TH, W = 50.0, 15.0, 10
J4_C, J7_C, TOL = 93.0, 23.0, 8.0

eps = sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))
print(f"[grasp-window] {len(eps)} episodes, open>{OPEN_TH} close<{CLOSE_TH}, window ±{W}帧")

# 池化所有窗口帧
pool = {k: [] for k in ["s_j4","s_j7","s_grip","a_j4","a_j7","a_grip"]}
t_close_phase = []           # t_close / len 相位
no_trans = []                # 无 transition 的 ep (判据B)
n_strict = 0                 # 判据A (相邻帧) 检出数
config_hit, config_tot = 0, 0      # 全窗口 j4&j7 同时达标
config_hit_pre, config_tot_pre = 0, 0   # 仅 close 前 [t_close-10, t_close]
per_ep = []

for ep_path in eps:
    epi = int(ep_path.split("episode_")[-1].split(".")[0])
    df = pd.read_parquet(ep_path)
    S = np.stack(df["observation.state"].values).astype(np.float32)
    A = np.stack(df["action"].values).astype(np.float32)
    g = A[:, GRIP]; n = len(g)
    # 判据A (windows 字面: 相邻帧 >50->-<15) — 30fps 下太严, 仅作对照
    t_strict = -1
    for t in range(1, n):
        if g[t-1] > OPEN_TH and g[t] < CLOSE_TH and float(np.min(g[t:t+5])) < CLOSE_TH:
            t_strict = t; break
    # 判据B (鲁棒: 张过(>50)之后【首次进入】闭带 <15, 不要求相邻 + 防噪) — 主用
    opened = False; t_close = -1
    for t in range(n):
        if g[t] > OPEN_TH: opened = True
        if opened and g[t] < CLOSE_TH and float(np.min(g[t:t+5])) < CLOSE_TH:
            t_close = t; break
    if t_strict >= 0: n_strict += 1
    if t_close < 0:
        no_trans.append(epi); per_ep.append({"ep": epi, "t_close": None}); continue
    t_close_phase.append(t_close / n)
    lo, hi = max(0, t_close - W), min(n, t_close + W + 1)
    sl = slice(lo, hi)
    pool["s_j4"]  += S[sl, J4].tolist();  pool["s_j7"]  += S[sl, J7].tolist();  pool["s_grip"] += S[sl, GRIP].tolist()
    pool["a_j4"]  += A[sl, J4].tolist();  pool["a_j7"]  += A[sl, J7].tolist();  pool["a_grip"] += A[sl, GRIP].tolist()
    # 构型达标 (state j4&j7) 全窗口
    hit = (np.abs(S[sl, J4] - J4_C) <= TOL) & (np.abs(S[sl, J7] - J7_C) <= TOL)
    config_hit += int(hit.sum()); config_tot += int(hit.size)
    # close 前
    slp = slice(lo, t_close + 1)
    hitp = (np.abs(S[slp, J4] - J4_C) <= TOL) & (np.abs(S[slp, J7] - J7_C) <= TOL)
    config_hit_pre += int(hitp.sum()); config_tot_pre += int(hitp.size)
    per_ep.append({"ep": epi, "t_close": t_close, "phase": round(t_close/n, 3),
                   "s_j4@close": round(float(S[t_close, J4]),1), "s_j7@close": round(float(S[t_close, J7]),1),
                   # 完整 8 维 (windows NN-to-set 用): [j1..j7 deg, gripper mm]
                   "state_at_close": [round(float(x),3) for x in S[t_close]],
                   "action_at_close": [round(float(x),3) for x in A[t_close]]})

def stats(a):
    a = np.array(a)
    return {"mean": round(float(a.mean()),2), "std": round(float(a.std()),2),
            "min": round(float(a.min()),2), "max": round(float(a.max()),2), "n": int(a.size)}

out = {
    "n_episodes": len(eps),
    "fps": 30,
    "n_detected_strict_adjacent (windows字面判据A)": n_strict,
    "n_with_transition (鲁棒判据B)": len(eps) - len(no_trans),
    "n_no_transition": len(no_trans),
    "no_transition_eps": no_trans,
    "t_close_phase_mean": round(float(np.mean(t_close_phase)),3) if t_close_phase else None,
    "t_close_phase_std": round(float(np.std(t_close_phase)),3) if t_close_phase else None,
    "window_stats": {k: stats(v) for k, v in pool.items()},
    "config_frac_fullwin (j4=93±8 & j7=23±8, state)": round(config_hit / max(1, config_tot), 3),
    "config_frac_preclose": round(config_hit_pre / max(1, config_tot_pre), 3),
}
json.dump({"summary": out, "per_ep": per_ep}, open(f"{os.environ.get('NERO_ROOT', '.')}/realrobot/grasp_window_b1.json","w"), indent=2)

print("\n" + "="*64)
print(f"  ep 总数={out['n_episodes']}  (数据 30fps, 窗口±10帧=±0.33s)")
print(f"  判据A 相邻帧>50->-<15 (windows字面): 检出 {n_strict}/32  ← 30fps 渐变闭合下太严, 假象")
print(f"  判据B 张过后首次进闭带<15 (鲁棒, 主用): 有 transition={len(eps)-len(no_trans)}/32  无={len(no_trans)} {no_trans if no_trans else ''}")
print(f"  t_close 相位: mean={out['t_close_phase_mean']}  std={out['t_close_phase_std']}")
print("-"*64)
print("  窗口 ±10帧 池化统计 (mean / std / min / max / n):")
for k,v in out["window_stats"].items():
    print(f"    {k:8s}: {v['mean']:8.2f} / {v['std']:7.2f} / {v['min']:8.2f} / {v['max']:8.2f}  (n={v['n']})")
print("-"*64)
print(f"  构型达标(state j4=93±8 且 j7=23±8 同时):")
print(f"    全窗口帧比例   = {out['config_frac_fullwin (j4=93±8 & j7=23±8, state)']}")
print(f"    close前帧比例   = {out['config_frac_preclose']}")
print("="*64)
