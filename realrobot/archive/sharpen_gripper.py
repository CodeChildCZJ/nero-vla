#!/usr/bin/env python3
"""Fix A: 锐化 lerobot 数据集 gripper 通道的开合过渡 (治 grasp-close 糊/不commit).
段式方法: 把 grip 轨迹分成 稳定段(open/closed-on-sponge/bucket-0), 每段保留其 median 值,
段间过渡压成 RAMP 帧线性. 保留海绵厚度闭合值(非0), 只动 dim7, 关节(0-6)完全不碰.
剔除没有干净抓取的坏 ep.

用法: python sharpen_gripper.py <src_dataset_dir> <dst_dataset_dir>
(目前先 --dry-run 在 src 上验证, 不写)
"""
import os
import sys, glob, json, shutil
import numpy as np
import pandas as pd

GRIP = 7
RAMP = 3          # 过渡压成几帧
MID = 25          # open/closed 分界 mm
MIN_SEG = 4       # 短于此的段并入邻段 (去噪)

def segments(grip):
    """把 grip 分成 open(>MID)/closed(<MID) 段, 返回 [(start,end,level)]."""
    state = (grip < MID).astype(int)  # 1=closed
    segs = []
    i = 0; T = len(grip)
    while i < T:
        j = i
        while j < T and state[j] == state[i]:
            j += 1
        segs.append([i, j, float(np.median(grip[i:j]))])
        i = j
    # 并掉过短段 (噪声)
    merged = True
    while merged and len(segs) > 1:
        merged = False
        for k in range(len(segs)):
            if segs[k][1] - segs[k][0] < MIN_SEG:
                # 并入较长的邻段
                if k == 0:
                    segs[1][0] = segs[0][0]; segs.pop(0)
                elif k == len(segs) - 1:
                    segs[-2][1] = segs[-1][1]; segs.pop()
                else:
                    if segs[k-1][1]-segs[k-1][0] >= segs[k+1][1]-segs[k+1][0]:
                        segs[k-1][1] = segs[k][1]
                    else:
                        segs[k+1][0] = segs[k][0]
                    segs.pop(k)
                merged = True
                break
    # 重算 level
    for s in segs:
        s[2] = float(np.median(grip[s[0]:s[1]]))
    return segs

def sharpen(grip):
    segs = segments(grip)
    out = np.empty_like(grip, dtype=float)
    # 先各段填 level
    for s, e, lv in segs:
        out[s:e] = lv
    # 段间过渡压成 RAMP 帧 (跨边界, 前段尾 + 后段头)
    for i in range(len(segs)-1):
        b = segs[i][1]  # 边界帧 (后段起点)
        lv0, lv1 = segs[i][2], segs[i+1][2]
        s = max(segs[i][0], b - RAMP//2)
        e = min(segs[i+1][1], b + RAMP - RAMP//2)
        out[s:e] = np.linspace(lv0, lv1, e - s)
    return out

def has_clean_grasp(grip):
    """有没有干净抓取: 开过(>40) 且 闭到海绵厚度(<25) 且闭合保持>=15帧."""
    oi = np.where(grip > 40)[0]
    if len(oi) == 0: return False
    ci = np.where((grip < 25) & (np.arange(len(grip)) > oi[0]))[0]
    if len(ci) == 0: return False
    # 闭合段够长 (真抓不是抖)
    return (grip[ci[0]:] < 25).sum() >= 15

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else f'{os.environ.get("NERO_DATA_ROOT", "data")}/pick_pink_sponge_v2_clean'
    dry = '--dry-run' in sys.argv or len(sys.argv) < 3
    files = sorted(glob.glob(f'{src}/data/chunk-000/episode_*.parquet'))
    kept, dropped = 0, []
    for fp in files:
        ep = int(fp.split('_')[-1].split('.')[0])
        df = pd.read_parquet(fp)
        act = np.stack(df['action'].values)
        grip = act[:, GRIP]
        if not has_clean_grasp(grip):
            dropped.append(ep); continue
        new = sharpen(grip.copy())
        if dry:
            # 报过渡前后
            oi = np.where(grip > 40)[0]
            ci = np.where((grip < 25) & (np.arange(len(grip)) > oi[0]))[0]
            if len(ci):
                c = ci[0]; po = oi[oi < c][-1] if (oi < c).any() else c
                span = c - po
                print(f"ep{ep:03d}: 闭爪过渡 {span}帧→压成≤{RAMP}帧; settle={np.median(grip[c:c+30][grip[c:c+30]<25]):.0f}mm 保留")
        kept += 1
    print(f"\n保留 {kept} ep, 剔除 {len(dropped)}: {dropped}")
    print(f"(dry-run, 没写. 真跑: python sharpen_gripper.py {src} <dst>)")

if __name__ == '__main__':
    main()
