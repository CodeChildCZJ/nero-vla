#!/usr/bin/env python3
"""windows 要的精确检验: 链尾残差 = |state[infer_k] - chunk[-1] of infer_{k-1}| 逐关节.
state 在 re-infer 时刻(chunk 边界)记 → 即臂执行完 chunk_{k-1} 后实际落点 vs 该 chunk 末 waypoint.
残差小 → 臂走到了上一块末(执行层 OK, 假设弱); 残差大(尤其 j2/j4/j7) → 臂没跟上(假设强).
索引: j1..j7 = idx0..6, gripper=idx7.  windows 关注 j2=idx1, j4=idx3, j7=idx6.
"""
import json, sys, numpy as np

path = sys.argv[1]
rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: rows.append(json.loads(line))
        except Exception: pass

states = np.array([r["state_8"] for r in rows])
chunks = [np.array(r["chunk_8"]) for r in rows if r.get("chunk_8")]
N = min(len(chunks), len(states))
print(f"== {path}  (N={N} infer 对) ==")

names = ["j1","j2","j3","j4","j5","j6","j7","grip"]
# 链尾残差: state[k] vs chunk[k-1][-1]
res = np.array([np.abs(states[k] - chunks[k-1][-1]) for k in range(1, N)])  # (N-1, 8)
# 对照: 同时也看臂这一段实际走了多远 (state[k]-state[k-1]) — 残差要相对运动幅度看
moved = np.array([np.abs(states[k] - states[k-1]) for k in range(1, N)])

print(f"{'joint':>5} | {'链尾残差 med':>11} {'p90':>6} {'max':>6} | {'实走 med':>8} {'p90':>6} | 残差/实走比")
for i,nm in enumerate(names):
    r = res[:,i]; m = moved[:,i]
    ratio = np.median(r)/max(np.median(m),1e-3)
    star = "  <<<" if nm in ("j2","j4","j7") else ""
    print(f"{nm:>5} | {np.median(r):11.2f} {np.percentile(r,90):6.1f} {r.max():6.1f} | "
          f"{np.median(m):8.2f} {np.percentile(m,90):6.1f} | {ratio:5.2f}{star}")

# 聚焦下扎/抓取段: 选 grip 命令闭合(chunk[-1] grip 低)的那些 infer, 看残差是否更大
gr_tail = np.array([chunks[k-1][-1,7] for k in range(1,N)])
gmax = gr_tail.max()
grasp_seg = gr_tail < 0.35*gmax
print(f"\n[抓取段 only] chunk 末命令闭爪(<{0.35*gmax:.0f})的 {grasp_seg.sum()} 个 infer 的链尾残差:")
if grasp_seg.sum() > 3:
    for i in (1,3,6):
        r = res[grasp_seg, i]
        print(f"  {names[i]}: med={np.median(r):.2f} p90={np.percentile(r,90):.1f} max={r.max():.1f}")
