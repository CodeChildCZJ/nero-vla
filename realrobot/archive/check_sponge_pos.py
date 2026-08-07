#!/usr/bin/env python3
"""量 v7 23ep 训练数据的海绵位置散布 (cam_high). 判 v7 是否还是'死一个点'(位置OOD根因)。
每ep取抓取前的早帧(t_g-90, 桌面海绵可见臂未挡), HSV mask pink/magenta, 取最大连通块质心(x,y)。
散布大=位置diverse(好); 散布小=单点(需v7.1补数据, 30Hz也救不了位置泛化)。
"""
import os
import glob
import numpy as np, pandas as pd, cv2
import av

DS = f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_v7"
GRIP = 7

def decode_one(mp4, idx):
    i = 0
    c = av.open(mp4)
    for frame in c.decode(c.streams.video[0]):
        if i == idx:
            img = frame.to_ndarray(format="rgb24"); c.close(); return img
        i += 1
    c.close(); return None

def pink_centroid(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # pink/magenta: 高S 中V, hue 兼顾 wrap(近0/180) 与紫红(140-175)
    m1 = cv2.inRange(hsv, (140, 60, 60), (179, 255, 255))
    m2 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None, 0
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[k, cv2.CC_STAT_AREA])
    if area < 80:
        return None, area
    return (float(cent[k][0]), float(cent[k][1])), area

eps = sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))
pts = []
print("ep |  frame | centroid(x,y) | area")
for ep_path in eps:
    epi = int(ep_path.split("episode_")[-1].split(".")[0])
    df = pd.read_parquet(ep_path)
    A = np.stack(df["action"].values).astype(np.float32)
    closed = np.where(A[:, GRIP] > 50)[0]
    t_g = int(closed[0]) if len(closed) else 90
    t = max(5, t_g - 90)  # 抓取前 ~3s, 桌面海绵可见
    mp4 = f"{DS}/videos/chunk-000/observation.images.cam_high/episode_{epi:06d}.mp4"
    img = decode_one(mp4, t)
    if img is None:
        print(f"{epi:2d} | frame{t:4d} | decode fail"); continue
    c, area = pink_centroid(img)
    if c is None:
        print(f"{epi:2d} | frame{t:4d} | no pink (area={area})"); continue
    pts.append(c)
    print(f"{epi:2d} | frame{t:4d} | ({c[0]:6.1f},{c[1]:6.1f}) | {area}")

pts = np.array(pts)
print("\n" + "=" * 60)
print(f"  海绵位置散布 (n={len(pts)}/{len(eps)} ep 检出)  [cam_high 640x480]")
print("=" * 60)
if len(pts) >= 3:
    print(f"  x: mean={pts[:,0].mean():.0f} std={pts[:,0].std():.0f} range=[{pts[:,0].min():.0f},{pts[:,0].max():.0f}] 跨度{pts[:,0].ptp():.0f}px")
    print(f"  y: mean={pts[:,1].mean():.0f} std={pts[:,1].std():.0f} range=[{pts[:,1].min():.0f},{pts[:,1].max():.0f}] 跨度{pts[:,1].ptp():.0f}px")
    print(f"  对比 06-14 旧数据'死一个点': x136±10 y278±14 (std~10-14)")
    sx, sy = pts[:,0].std(), pts[:,1].std()
    if sx < 25 and sy < 25:
        print(f"  → 判: 仍偏单点 (std x{sx:.0f} y{sy:.0f} < 25px) → 位置覆盖薄, v7.1 需补 diverse 位置")
    else:
        print(f"  → 判: 有一定散布 (std x{sx:.0f} y{sy:.0f}) → 比旧数据强, 但看跨度够不够铺满工作区")
print("=" * 60)
