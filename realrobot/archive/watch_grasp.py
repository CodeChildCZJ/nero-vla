#!/usr/bin/env python3
"""Tail v4 trace, emit gripper-grasp-relevant frames in real time.
Each REAL frame (non-zero state): track state.grip + chunk gripper trajectory.
Emit when model commands close (chunk min grip <20mm) or state closing, + heartbeat."""
import json, time, sys, os

P = f"{os.environ.get('NERO_ROOT', '.')}/realrobot/server_trace_v4_4999.jsonl"
# start AFTER current end so we only see czj's new run
with open(P) as f:
    f.seek(0, 2)
    pos = f.tell()

real_n = 0
hb = 0
last_state_grip = None
print(f"[watch] armed at byte {pos}, waiting czj real-robot frames...", flush=True)
while True:
    with open(P) as f:
        f.seek(pos)
        lines = f.readlines()
        pos = f.tell()
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
            s = d["state_8"]; c = d["chunk_8"]
        except Exception:
            continue
        # zero-obs ping? skip (warmup/RTT)
        if all(abs(x) < 1e-6 for x in s):
            continue
        real_n += 1
        sg = s[7]
        cg = [c[i][7] for i in range(len(c))]
        cmin = min(cg); c0 = cg[0]; c15 = cg[-1]
        close_cmd = cmin < 20.0          # model wants close
        closed = sg < 8.0                # physically closed
        # state.grip 上下抖动检测
        jitter = (last_state_grip is not None and abs(sg - last_state_grip) > 15.0)
        last_state_grip = sg
        emit = close_cmd or closed or jitter or (real_n - hb >= 20)
        if emit:
            hb = real_n
            tag = []
            if closed: tag.append("CLOSED<8mm")
            if close_cmd: tag.append("MODEL-CLOSE-CMD")
            if jitter: tag.append("STATE-JITTER")
            tagstr = (" " + " ".join(tag)) if tag else " (heartbeat)"
            print(f"#{real_n} state.grip={sg:.1f}mm chunk.grip[0/7/15]={c0:.1f}/{cg[7] if len(cg)>7 else -1:.1f}/{c15:.1f} min={cmin:.1f}{tagstr}", flush=True)
    time.sleep(0.5)
