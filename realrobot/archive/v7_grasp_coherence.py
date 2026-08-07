#!/usr/bin/env python3
"""v7 训练集 grasp-moment 合成度对照 (vs windows v8 audit r(j4,j7)=-0.63).
问题: v7 训练数据本身在抓取瞬间 j4(下扎,idx3) 和 j7(腕,idx6) 是否已经双模/反相关?
- 若 v7 也反相关双模 -> 病根一直在数据, v8 必须统一打法
- 若 v7 连贯(单模, 两轴同帧到 grasp 值) -> v8 引入了新双模, 统一回 v7
state[:8] = [right_j1..j7, right_gripper]. gripper open=高(~99) close=低(~13).
"""
import os
import glob, numpy as np, pyarrow.parquet as pq

J4, J7, GRIP = 3, 6, 7
# 与 coord_check.py 一致的 grasp 判据
TH4, TH7 = 71.0, 45.0   # 下扎到位 j4<71, 腕到位 j7<45

files = sorted(glob.glob(f'{os.environ.get("NERO_DATA_ROOT", "data")}/pick_pink_sponge_v7/data/chunk-000/episode_*.parquet'))
rows = []
for f in files:
    t = pq.read_table(f)
    st = np.array(t['observation.state'].to_pylist(), dtype=np.float32)  # (T,16)
    ep = int(t['episode_index'][0].as_py())
    g = st[:, GRIP]
    amax = float(g.max())
    op_th, cl_th = 0.6 * amax, 0.35 * amax   # 开/闭阈 (相对峰值)
    # 1) 找首次"主动张开"(>op_th) = 抓前 ready-open, 跳过头部静止0
    open_idx = np.where(g > op_th)[0]
    grasp_i = None
    if len(open_idx):
        start = int(open_idx[0])
        # 2) 从张开后, 找首次持续闭合(<cl_th 连续>=8帧) = 抓取瞬间闭合
        run = 0
        for i in range(start, len(g)):
            if g[i] < cl_th:
                run += 1
                if run >= 8:
                    grasp_i = i - run + 1
                    break
            else:
                run = 0
    if grasp_i is None:
        grasp_i = int(np.argmin(g))
    gop, gcl = op_th, cl_th
    # 取 grasp 起点起 15 帧的中位姿(HOLD 段代表姿)
    seg = st[grasp_i:grasp_i + 15]
    j4v, j7v = float(np.median(seg[:, J4])), float(np.median(seg[:, J7]))
    rows.append((ep, grasp_i, len(st), j4v, j7v, float(gop), float(gcl)))

rows.sort()
print("ep  graspF/len   j4(下扎)  j7(腕)   open/close   合成?")
both = 0
j4s, j7s = [], []
for ep, gi, T, j4v, j7v, gop, gcl in rows:
    syn = (j4v < TH4) and (j7v < TH7)
    both += syn
    j4s.append(j4v); j7s.append(j7v)
    print(f"{ep:2d}  {gi:4d}/{T:4d}   {j4v:6.1f}   {j7v:6.1f}   {gop:5.1f}/{gcl:4.1f}   {'✅同帧到位' if syn else ''}")

j4s, j7s = np.array(j4s), np.array(j7s)
r = float(np.corrcoef(j4s, j7s)[0, 1])
print(f"\n=== v7 训练集 grasp 瞬间汇总 ({len(rows)} eps) ===")
print(f"j4(下扎): mean {j4s.mean():.1f}  std {j4s.std():.1f}  range {j4s.min():.0f}-{j4s.max():.0f}")
print(f"j7(腕)  : mean {j7s.mean():.1f}  std {j7s.std():.1f}  range {j7s.min():.0f}-{j7s.max():.0f}")
print(f"r(j4,j7) = {r:+.2f}   (windows v8 = -0.63)")
print(f"同帧合成(j4<{TH4:.0f} AND j7<{TH7:.0f}) = {both}/{len(rows)}")
# 双模检测: 高j4(>=100) vs 低j4(<50) 时 j7 均值
hi = j7s[j4s >= 100]; lo = j7s[j4s < 50]
print(f"高j4(>=100,{len(hi)}条) 时 j7均值 {hi.mean():.1f}" if len(hi) else "高j4(>=100): 0条")
print(f"低j4(<50,{len(lo)}条)  时 j7均值 {lo.mean():.1f}" if len(lo) else "低j4(<50): 0条")
