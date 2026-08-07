#!/usr/bin/env python3
"""测 windows 执行层假设:
A) 模型 chunk 里有没有命令真抓取位姿(下扎 j4 低 + 腕 j7 到位)
B) 臂追不追得上 — 命令跳变 vs 单 infer 间隔实际位移
索引: j4(下扎)=idx3, j7(腕)=idx6, gripper=idx7  (state[:8]=右臂)
"""
import json, sys, numpy as np

path = sys.argv[1]
rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass

print(f"== {path}  ({len(rows)} infers) ==")
if not rows:
    sys.exit(0)

# infer cadence
ts = [r["ts"] for r in rows if "ts" in r]
dts = np.diff(ts)
print(f"infer 间隔: median={np.median(dts)*1000:.0f}ms  p90={np.percentile(dts,90)*1000:.0f}ms  (=re-infer 节奏)")

J4, J7, GR = 3, 6, 7
states = np.array([r["state_8"] for r in rows])          # (N,8) 实际到达
chunks = [np.array(r["chunk_8"]) for r in rows if r.get("chunk_8")]  # list of (H,8)
H = chunks[0].shape[0]
print(f"horizon H={H}, chunk dims={chunks[0].shape[1]}")

# ---- A) 模型命令的位姿范围 (跨所有 chunk 所有 step) vs 实际到达的 state 范围 ----
all_cmd = np.concatenate(chunks, axis=0)   # (N*H, 8)
def rng(a, i): return f"min={a[:,i].min():6.1f} max={a[:,i].max():6.1f} mean={a[:,i].mean():6.1f}"
print("\n[A] 命令(chunk 全帧) vs 实际到达(state):")
print(f"  j4  命令 {rng(all_cmd,J4)}")
print(f"  j4  到达 {rng(states,J4)}")
print(f"  j7  命令 {rng(all_cmd,J7)}")
print(f"  j7  到达 {rng(states,J7)}")
print(f"  grip命令 {rng(all_cmd,GR)}")
print(f"  grip到达 {rng(states,GR)}")

# 真抓取位姿 = 下扎到位(j4 低) 同帧 腕到抓取值. 用 grip 闭合(<阈)定位抓取意图帧.
gmax = all_cmd[:,GR].max()
grip_close = all_cmd[:,GR] < 0.35*gmax
print(f"\n[A2] 命令闭爪(<0.35*max={0.35*gmax:.1f})的帧: {grip_close.sum()}/{len(all_cmd)}")
if grip_close.sum() > 0:
    cc = all_cmd[grip_close]
    print(f"  闭爪同帧 j4: {rng(cc,J4)}")
    print(f"  闭爪同帧 j7: {rng(cc,J7)}")
    # 复合抓取位姿: 闭爪 且 j4 低于到达分布的中位 (=真下扎而非半空)
    j4_med = np.median(states[:,J4])
    compound = grip_close & (all_cmd[:,J4] < j4_med)
    print(f"  闭爪+j4<到达中位({j4_med:.0f}) 的复合下扎帧: {compound.sum()}  ← 模型有没有命令真下扎抓")

# ---- B) 臂追不追得上: 命令首waypoint跳变 vs 实际单间隔位移 ----
print("\n[B] 命令跳变 vs 实际位移 (臂能否到位):")
N = min(len(chunks), len(states)-1)
cmd_jump_j4, cmd_jump_j7, ach_j4, ach_j7, gap_j4, gap_j7 = [],[],[],[],[],[]
for i in range(N):
    c0 = chunks[i][0]                 # 这次命令的首 waypoint
    s_now = states[i]                 # 当前 state
    s_next = states[i+1]              # 下次 infer 时实际到达
    cmd_jump_j4.append(abs(c0[J4]-s_now[J4]))
    cmd_jump_j7.append(abs(c0[J7]-s_now[J7]))
    ach_j4.append(abs(s_next[J4]-s_now[J4]))
    ach_j7.append(abs(s_next[J7]-s_now[J7]))
    gap_j4.append(abs(s_next[J4]-c0[J4]))   # 到下次还差命令多少
    gap_j7.append(abs(s_next[J7]-c0[J7]))
for nm,cj,aj,gj in [("j4",cmd_jump_j4,ach_j4,gap_j4),("j7",cmd_jump_j7,ach_j7,gap_j7)]:
    cj,aj,gj = np.array(cj),np.array(aj),np.array(gj)
    follow = aj/np.maximum(cj,1e-3)   # 实际/命令 跟随比
    print(f"  {nm}: 命令跳变 median={np.median(cj):5.1f}  实际位移 median={np.median(aj):5.1f}  "
          f"跟随比 median={np.median(follow):.2f}  残余gap median={np.median(gj):5.1f}")

# 最大命令帧(整 chunk 内最大目标位移) — 模型一个 chunk 想走多远
intra = np.array([np.abs(c[-1]-c[0]) for c in chunks])  # (N,8)
print(f"\n[B2] 单 chunk 内 chunk[-1]-chunk[0] 位移 (模型一块想走多远):")
print(f"  j4 median={np.median(intra[:,J4]):.1f} max={intra[:,J4].max():.1f}")
print(f"  j7 median={np.median(intra[:,J7]):.1f} max={intra[:,J7].max():.1f}")
