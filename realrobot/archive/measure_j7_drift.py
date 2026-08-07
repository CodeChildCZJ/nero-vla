#!/usr/bin/env python3
"""量 server_trace 里腕 j7 (=state_8[6]) 的闭环漂移。
判据: 下扎最低 j7 (奔训练抓取 37.8?) + 卡住吸引子 (掉 80 以下?)。
用法: python3 measure_j7_drift.py <trace.jsonl> [tag]
基线 6k: 下扎最低~51, 卡住吸引子~80, 训练抓取=37.8。
"""
import json, sys

TRAIN_HOME_J7 = 95.8
TRAIN_GRASP_J7 = 37.8
J7_IDX = 6  # state_8[6] = 腕 j7 (我0-indexed) = windows j7(1-indexed)

def load_j7(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    seq = []
    for i, r in enumerate(rows):
        st = r.get('state_8')
        if not st or len(st) < 7:
            continue
        if all(abs(x) < 1e-6 for x in st):  # warmup/dummy
            continue
        seq.append(round(float(st[J7_IDX]), 1))
    return seq, len(rows)

def segment_runs(seq):
    """按 home-reset 切 run: j7 回升到 >88 且之前已下扎到 <70。"""
    runs, cur = [], []
    descended = False
    for v in seq:
        if v > 88 and descended and len(cur) > 5:
            runs.append(cur); cur = [v]; descended = False
        else:
            cur.append(v)
            if v < 70:
                descended = True
    if cur:
        runs.append(cur)
    return runs

def analyze_run(run):
    mn = min(run)
    mn_i = run.index(mn)
    tail = run[max(mn_i, len(run)-20):]  # 下扎后的尾段 = 卡住段
    parked = round(sorted(tail)[len(tail)//2], 1) if tail else None  # median
    return mn, parked

def main():
    if len(sys.argv) < 2:
        print("用法: python3 measure_j7_drift.py <trace.jsonl> [tag]"); return
    path = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else path.split('/')[-1]
    seq, nrow = load_j7(path)
    print(f"=== {tag} === ({nrow} rows, {len(seq)} 有效 j7 帧)")
    if not seq:
        print("无有效 j7 帧"); return
    print(f"训练: home={TRAIN_HOME_J7}  抓取目标={TRAIN_GRASP_J7}")
    runs = segment_runs(seq)
    print(f"检出 {len(runs)} 条 run:")
    for k, run in enumerate(runs):
        mn, parked = analyze_run(run)
        gap_min = round(mn - TRAIN_GRASP_J7, 1)
        gap_park = round(parked - TRAIN_GRASP_J7, 1) if parked else None
        home = "✓home" if run[0] > 88 else f"起手{run[0]}"
        print(f"  run{k}: {home} | 下扎最低 j7={mn} (差抓取{gap_min:+}) | 卡住吸引子={parked} (差{gap_park:+}) | len={len(run)}")
    # 全局摘要
    g_min = min(seq)
    print(f">>> 全局下扎最低 j7 = {g_min} (差抓取 {round(g_min-TRAIN_GRASP_J7,1):+})")
    print(f">>> 判据: 下扎最低↓50奔38 且 吸引子↓80 = 多训步治住; 仍~51+~80 = 偏移没破→补v8")

if __name__ == '__main__':
    main()
