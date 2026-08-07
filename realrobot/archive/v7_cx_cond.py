#!/usr/bin/env python3
"""v7 独立复验 windows 的 conditioned-multimodal 发现:
顶视海绵水平位置 cx 是否预测 grasp 打法 (j4 下扎深度)?
cx 从 cam_high 早期帧抠粉色海绵(magenta score), grasp (j4,j7) 从 parquet 抓取帧。
对齐窗: windows v8 报 r(cx,j4)=-0.93, r(cx,j7)=+0.66。
"""
import os
import glob, numpy as np, pyarrow.parquet as pq
from PIL import Image

J4, J7, GRIP = 3, 6, 7

def grasp_pose(st):
    g = st[:, GRIP]; amax = float(g.max())
    op_th, cl_th = 0.6 * amax, 0.35 * amax
    oi = np.where(g > op_th)[0]; gi = None
    if len(oi):
        run = 0
        for i in range(int(oi[0]), len(g)):
            if g[i] < cl_th:
                run += 1
                if run >= 8: gi = i - run + 1; break
            else: run = 0
    if gi is None: gi = int(np.argmin(g))
    seg = st[gi:gi + 15]
    return float(np.median(seg[:, J4])), float(np.median(seg[:, J7]))

def sponge_cx(png):
    im = np.asarray(Image.open(png).convert('RGB')).astype(np.int32)
    H, W, _ = im.shape
    R, G, B = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    score = (R - G) + (B - G)        # 粉/品红: R,B 高于 G
    for thr in (60, 50, 40, 30):
        ys, xs = np.where(score > thr)
        if len(xs) >= 30:
            # 抗散点: 取离中位最近的 80% 像素再求中位
            cx0, cy0 = np.median(xs), np.median(ys)
            d = np.abs(xs - cx0) + np.abs(ys - cy0)
            keep = d < np.percentile(d, 80)
            return float(np.median(xs[keep]) / W), float(np.median(ys[keep]) / H), int(keep.sum()), float(xs[keep].std())
    return None, None, 0, 0.0

files = sorted(glob.glob(f'{os.environ.get("NERO_DATA_ROOT", "data")}/pick_pink_sponge_v7/data/chunk-000/episode_*.parquet'))
rows = []
for f in files:
    t = pq.read_table(f)
    st = np.array(t['observation.state'].to_pylist(), dtype=np.float32)
    ep = int(t['episode_index'][0].as_py())
    j4, j7 = grasp_pose(st)
    cx, cy, npx, sx = sponge_cx(f'/tmp/v7frames/ep{ep}.png')
    rows.append((ep, cx, cy, npx, sx, j4, j7))

rows.sort()
print("ep   cx     cy   npx spread   j4(下扎) j7(腕)")
good = []
for ep, cx, cy, npx, sx, j4, j7 in rows:
    flag = '' if (cx is not None and sx < 30) else '  ⚠检测可疑'
    cxs = f'{cx:.3f}' if cx is not None else ' NA  '
    cys = f'{cy:.3f}' if cy is not None else ' NA '
    print(f"{ep:2d}  {cxs}  {cys}  {npx:4d}  {sx:5.1f}   {j4:6.1f}  {j7:6.1f}{flag}")
    if cx is not None and sx < 30:
        good.append((cx, j4, j7))

g = np.array(good)
cx, j4, j7 = g[:, 0], g[:, 1], g[:, 2]
print(f"\n=== v7 复验 ({len(g)}/{len(rows)} 条有效检测) ===")
print(f"r(cx, j4) = {np.corrcoef(cx, j4)[0,1]:+.2f}   (windows v8 = -0.93)")
print(f"r(cx, j7) = {np.corrcoef(cx, j7)[0,1]:+.2f}   (windows v8 = +0.66)")
print(f"r(j4, j7) = {np.corrcoef(j4, j7)[0,1]:+.2f}   (双模反相关基线)")
print(f"cx range {cx.min():.2f}-{cx.max():.2f}  j4 range {j4.min():.0f}-{j4.max():.0f}")
# 分位: 左半 vs 右半海绵的 j4
med = np.median(cx)
print(f"cx<中位({med:.2f}) 时 j4 均值 {j4[cx<med].mean():.0f} (左/近) ; cx>中位 时 j4 均值 {j4[cx>=med].mean():.0f} (右/远)")
