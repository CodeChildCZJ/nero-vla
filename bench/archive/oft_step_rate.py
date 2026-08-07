#!/usr/bin/env python3
"""Instantaneous s/step from the tqdm stream in the OFT training log.

tqdm's own `X.XXs/it` field is a SMOOTHED running average (EMA over the whole run for
this bar), so it cannot resolve a 7-minute perturbation -- it would dilute a 2x slowdown
into a rounding change. This parses the raw (step, elapsed) pairs and differences them,
which is what a co-location cost measurement needs.
"""
import re, sys, pathlib
log = pathlib.Path(sys.argv[1]); win = int(sys.argv[2]) if len(sys.argv) > 2 else 50
txt = log.read_bytes().decode("utf-8", "replace").replace("\r", "\n")
# tqdm: " 4%|  | 451/10500 [29:06<10:50:22,  3.88s/it]"   elapsed may be H:MM:SS or MM:SS
pat = re.compile(r"\|\s*(\d+)/(\d+) \[(\d+:\d+(?::\d+)?)<")
pts = []
for m in pat.finditer(txt):
    h = [int(x) for x in m.group(3).split(":")]
    sec = h[0]*3600 + h[1]*60 + h[2] if len(h) == 3 else h[0]*60 + h[1]
    pts.append((int(m.group(1)), sec))
if len(pts) < 2:
    sys.exit("not enough tqdm points")
step, el = pts[-1]
print(f"step {step}/{pts[-1] and m.group(2)}  elapsed {el/3600:.3f}h  mean {el/step:.4f} s/step")
for w in (win, 200, 500):
    older = [p for p in pts if p[0] <= step - w]
    if not older:
        continue
    s0, e0 = older[-1]
    print(f"  last {step-s0:4d} steps: {(el-e0)/(step-s0):.4f} s/step")
