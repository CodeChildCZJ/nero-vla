"""验 windows 假设: 同一帧 j7(腕roll,idx6) 和 j4(下扎,idx3) 是否从不同时到 grasp 值。
grasp: j7~37.8, j4~70.9。windows box: j7<45 且 j4<71 同时成立 = 完整 grasp pose。
windows 称该帧数=0 (6k/15k), 两轴反相关, 闭爪闭半空。
用法: python3 coord_check.py <trace.jsonl> [tag]
"""
import json, sys, math

J7, J4 = 6, 3
GJ7, GJ4 = 37.8, 70.9          # 训练 grasp 值
TH7, TH4 = 45.0, 71.0          # windows 的 grasp-box 阈值

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    fr = [(r['state_8'][J7], r['state_8'][J4]) for r in rows
          if r.get('state_8') and not all(abs(x) < 1e-6 for x in r['state_8'])]
    return fr

def pearson(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x-mx)**2 for x in xs)); sy = math.sqrt(sum((y-my)**2 for y in ys))
    return cov/(sx*sy) if sx*sy else float('nan')

def med(v):
    v = sorted(v); n = len(v)
    return v[n//2] if n else float('nan')

def main():
    path = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else path.split('/')[-1]
    fr = load(path)
    j7 = [a for a, b in fr]; j4 = [b for a, b in fr]
    n = len(fr)
    print(f"=== {tag} === ({n} 真机帧)")
    print(f"训练 grasp: j7={GJ7} j4={GJ4} | box: j7<{TH7} 且 j4<{TH4}")
    print(f"j7: min={min(j7):.1f} | j4: min={min(j4):.1f}")
    n7 = [i for i in range(n) if j7[i] < TH7]          # 腕到位
    n4 = [i for i in range(n) if j4[i] < TH4]          # 手扎深
    both = [i for i in range(n) if j7[i] < TH7 and j4[i] < TH4]
    print(f"腕到位帧(j7<{TH7}): {len(n7)}  |  手扎深帧(j4<{TH4}): {len(n4)}")
    print(f">>> 【同时成立帧数 (完整grasp pose)】= {len(both)}  <<<")
    if n7:
        j4_when_wrist = [j4[i] for i in n7]
        print(f"  腕到位时 j4: 中位={med(j4_when_wrist):.1f} 最低={min(j4_when_wrist):.1f} (训练grasp要≤71; >71=手还抬着)")
    if n4:
        j7_when_deep = [j7[i] for i in n4]
        print(f"  手扎深时 j7: 中位={med(j7_when_deep):.1f} 最低={min(j7_when_deep):.1f} (训练grasp要≤45; >45=腕转回去了)")
    print(f"  Pearson r(j7,j4) = {pearson(j7, j4):+.2f} (正=同向, 负=反相关)")
    # 最接近完整 grasp 的帧 (归一化距离)
    dist = [((j7[i]-GJ7)/10)**2 + ((j4[i]-GJ4)/30)**2 for i in range(n)]
    bi = dist.index(min(dist))
    print(f"  最接近grasp的单帧: idx={bi} j7={j7[bi]:.1f} j4={j4[bi]:.1f} (目标 {GJ7}/{GJ4})")

if __name__ == '__main__':
    main()
