#!/usr/bin/env python3
"""grasp 模式分类 + 单模筛选 — v8 数据一到立刻跑, 零延迟。
用法:
  python3 grasp_mode_classify.py <dataset_dir> [keep_mode]
  dataset_dir: 含 data/chunk-000/episode_*.parquet 的 LeRobot 目录
  keep_mode:   可选 '甲'/'乙'/'A'/'B' — 给出则打印该模式要 KEEP 的 episode_index 列表 + 筛后 r/std

逻辑: 每 ep 取抓取瞬间(gripper 闭合帧)的 state[:8] -> (j4=idx3 下扎, j7=idx6 腕).
判别量 d=j4-j7 (甲: 高j4低j7 -> d 大; 乙: 低j4高j7 -> d 小).
对 d 做数据驱动 2 簇分裂(最大间隙), 不写死阈值, v7/v8 通用。
"""
import os
import sys, glob, numpy as np, pyarrow.parquet as pq

J4, J7, GRIP = 3, 6, 7
TH4, TH7 = 71.0, 45.0   # 同帧合成判据 (coord_check 口径)

def grasp_pose(st):
    """返回抓取瞬间 (j4, j7); st=(T,16)"""
    g = st[:, GRIP]; amax = float(g.max())
    op_th, cl_th = 0.6 * amax, 0.35 * amax
    oi = np.where(g > op_th)[0]
    gi = None
    if len(oi):
        run = 0
        for i in range(int(oi[0]), len(g)):
            if g[i] < cl_th:
                run += 1
                if run >= 8: gi = i - run + 1; break
            else: run = 0
    if gi is None: gi = int(np.argmin(g))
    seg = st[gi:gi + 15]
    return float(np.median(seg[:, J4])), float(np.median(seg[:, J7])), gi

def load(d):
    files = sorted(glob.glob(f'{d.rstrip("/")}/data/chunk-000/episode_*.parquet'))
    out = []
    for f in files:
        t = pq.read_table(f)
        st = np.array(t['observation.state'].to_pylist(), dtype=np.float32)
        ep = int(t['episode_index'][0].as_py())
        j4, j7, gi = grasp_pose(st)
        out.append({'ep': ep, 'j4': j4, 'j7': j7, 'd': j4 - j7, 'gi': gi, 'T': len(st)})
    return out

def cluster(rows, margin=10.0):
    """对 d=j4-j7 做 Otsu 阈值 2 簇分裂(最大化类间方差, 抗离群).
    高 d 簇=甲(肩够), 低 d 簇=乙(肘探). 界 ±margin 内标 borderline(不强分)."""
    ds = np.array(sorted(r['d'] for r in rows), dtype=float)
    if len(ds) < 2:
        return None
    best_var, split = -1.0, ds.mean()
    cands = 0.5 * (ds[1:] + ds[:-1])   # 相邻中点为候选阈
    for t in cands:
        lo, hi = ds[ds <= t], ds[ds > t]
        if len(lo) == 0 or len(hi) == 0:
            continue
        w0, w1 = len(lo) / len(ds), len(hi) / len(ds)
        between = w0 * w1 * (lo.mean() - hi.mean()) ** 2
        if between > best_var:
            best_var, split = between, t
    for r in rows:
        if r['d'] > split + margin:
            r['mode'] = '甲'
        elif r['d'] < split - margin:
            r['mode'] = '乙'
        else:
            r['mode'] = '中间'   # borderline, 筛选时默认丢
    return split, best_var

def stats(sub, tag):
    if len(sub) < 2:
        print(f"  [{tag}] {len(sub)} 条 (太少不算 r)"); return
    j4 = np.array([r['j4'] for r in sub]); j7 = np.array([r['j7'] for r in sub])
    r = float(np.corrcoef(j4, j7)[0, 1])
    syn = sum((r_['j4'] < TH4) and (r_['j7'] < TH7) for r_ in sub)
    print(f"  [{tag}] {len(sub)}条  r={r:+.2f}  j4 {j4.mean():.0f}±{j4.std():.0f}  j7 {j7.mean():.0f}±{j7.std():.0f}  同帧合成 {syn}/{len(sub)}")

if __name__ == '__main__':
    d = sys.argv[1] if len(sys.argv) > 1 else f'{os.environ.get("NERO_DATA_ROOT", "data")}/pick_pink_sponge_v7'
    keep = sys.argv[2] if len(sys.argv) > 2 else None
    keep = {'A': '甲', 'B': '乙'}.get(keep, keep)
    rows = load(d)
    if not rows:
        print(f"无 parquet: {d}"); sys.exit(1)
    cl = cluster(rows)
    rows.sort(key=lambda r: r['ep'])
    print(f"dataset: {d}   ({len(rows)} eps)")
    print("ep  graspF/len   j4(下扎) j7(腕)  d=j4-j7  mode")
    for r in rows:
        print(f"{r['ep']:2d}  {r['gi']:4d}/{r['T']:4d}   {r['j4']:6.1f}  {r['j7']:6.1f}  {r['d']:6.1f}   {r.get('mode','?')}")
    if cl:
        split, bvar = cl
        print(f"\nOtsu 分裂点 d={split:.1f} (类间方差 {bvar:.0f}); 界±10 内记'中间'(筛选默认丢)")
    print("=== 全体 ===");  stats(rows, '全体')
    print("=== 分簇 ===")
    stats([r for r in rows if r.get('mode') == '甲'], '甲 高j4低j7 肩够')
    stats([r for r in rows if r.get('mode') == '乙'], '乙 低j4高j7 肘探')
    if keep in ('甲', '乙'):
        klist = sorted(r['ep'] for r in rows if r.get('mode') == keep)
        print(f"\n>>> KEEP 模式 [{keep}] = {len(klist)} 条: {klist}")
        print(f">>> 丢弃 {len(rows) - len(klist)} 条另一模式/中间. 用这批 + czj 后采同模式 = unimodal v8。")
