import os
import glob
import numpy as np, pandas as pd
DS=f"{os.environ.get('NERO_DATA_ROOT', 'data')}/pick_pink_sponge_b1"; GRIP=7
eps=sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet"))
poses=[]; jg=[]
for ep_path in eps:
    df=pd.read_parquet(ep_path)
    S=np.stack(df["observation.state"].values).astype(np.float32)
    A=np.stack(df["action"].values).astype(np.float32)
    g=A[:,GRIP]; lo,hi=float(g.min()),float(g.max()); rng=max(1e-6,hi-lo)
    op=np.where(g>lo+0.75*rng)[0]; cl=np.where(g<lo+0.25*rng)[0]
    if len(op)==0: continue
    t_o=int(op[0]); t_g=-1
    for t in cl:
        if t>t_o: t_g=int(t); break
    if t_g<0: continue
    poses.append(S[t_g,0:7]); jg.append(float(g[t_g]))
P=np.array(poses); m=P.mean(0); s=P.std(0)
print(f"n={len(P)}  抓取帧(真闭~48%) follower-state 臂姿 j1..j7 (deg):")
print("  mean: "+" ".join(f"j{i+1}={m[i]:.1f}" for i in range(7)))
print("  std : "+" ".join(f"{s[i]:.1f}" for i in range(7)))
