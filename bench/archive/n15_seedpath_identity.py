#!/usr/bin/env python3
"""Prove the --dry-run-seeds edit did not touch ANY statement the seed values depend on.

The dry run is only evidence about the delivered run if the code that computes seeds is
the same code. So: extract the seed-load-bearing statements from both files by AST and
require byte-identical source. Anything else (obs building, policy/dataset construction,
the forward) may differ -- it cannot move a seed.
"""
import pathlib
import ast, hashlib, sys

DELIVERED = str(pathlib.Path(__file__).resolve().parents[1] / "logs" / "provenance" /
                "predict_gr00t_delivered_7de0fcb5.py")
CURRENT = str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "predict" / "predict_gr00t.py")


def parts(path):
    src = open(path).read()
    tree = ast.parse(src)
    seg = lambda n: ast.get_source_segment(src, n)
    out = {}
    # 1. the pinner class in full (holds the seed arithmetic)
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == "PinnedNoise":
            out["class PinnedNoise"] = seg(n)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    # 2. argparse defaults that parameterise the seeds
    for n in ast.walk(main):
        if (isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_argument"
                and n.args and isinstance(n.args[0], ast.Constant)):
            flag = n.args[0].value
            if flag in ("--seed", "--repeats", "--batch-size", "--limit", "--subset", "--every"):
                out[f"arg {flag}"] = seg(n)
    # 3. every assignment that defines or re-indexes gidx / n, and the pinner construction
    for n in ast.walk(main):
        if isinstance(n, ast.Assign):
            tgt = ast.unparse(n.targets[0])
            if tgt == "gidx" or "gidx" in tgt or tgt == "n" or tgt == "pinner":
                out[f"assign {tgt} @L{n.lineno - 0}"] = seg(n)
    # 4. the two loop headers + the build call site (the CALLER -- where n17's bug lived).
    # Disambiguate by CONTENT, not by loop-variable name: the dry-run reshape block reuses
    # `for lo` / `for r` verbatim, so a name match alone silently compares the wrong loop
    # (this extractor did exactly that on its first run and reported a false MISMATCH).
    def has_build(node):
        return any(isinstance(x, ast.Call) and getattr(x.func, "attr", "") == "build"
                   and getattr(getattr(x.func, "value", None), "id", "") == "pinner"
                   for x in ast.walk(node))

    cand = [n for n in ast.walk(main)
            if isinstance(n, ast.For) and ast.unparse(n.target) == "lo" and has_build(n)]
    assert len(cand) == 1, f"expected exactly 1 anchor loop containing pinner.build, got {len(cand)}"
    n = cand[0]
    out["for lo: iter"] = ast.unparse(n.iter)
    for s in n.body:
        if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "hi":
            out["hi ="] = seg(s)
    inner = [s for s in ast.walk(n)
             if isinstance(s, ast.For) and ast.unparse(s.target) == "r" and has_build(s)]
    assert len(inner) == 1, f"expected exactly 1 repeat loop containing pinner.build, got {len(inner)}"
    out["for r: iter"] = ast.unparse(inner[0].iter)
    for b in inner[0].body:
        if (isinstance(b, ast.Expr) and isinstance(b.value, ast.Call)
                and getattr(b.value.func, "attr", "") == "build"):
            out["pinner.build call site"] = seg(b)
    return out


a, b = parts(DELIVERED), parts(CURRENT)
# keys carry line numbers for assigns; compare on the sorted VALUE multiset per logical group
ka = {k.split(" @L")[0]: v for k, v in a.items()}
kb = {k.split(" @L")[0]: v for k, v in b.items()}
assert set(ka) == set(kb), f"statement set changed: {set(ka) ^ set(kb)}"
bad = 0
print(f"{'seed-load-bearing statement':<34}{'identical':>10}  bytes")
for k in sorted(ka):
    same = ka[k] == kb[k]
    bad += (not same)
    print(f"{k:<34}{str(same):>10}  {len(ka[k])}")
    if not same:
        print("   DELIVERED:", ka[k][:200])
        print("   CURRENT  :", kb[k][:200])
print()
print("delivered md5:", hashlib.md5(open(DELIVERED, 'rb').read()).hexdigest())
print("current   md5:", hashlib.md5(open(CURRENT, 'rb').read()).hexdigest())
# power control: the proof must FAIL if a seed statement is perturbed
mut = open(DELIVERED).read().replace("pinner.build(gidx[lo:hi], r)", "pinner.build(gidx[lo:hi], 0)")
assert mut != open(DELIVERED).read(), "power control could not find the call site to mutate"
open("/tmp/_mut_probe.py", "w").write(mut)
m = {k.split(" @L")[0]: v for k, v in parts("/tmp/_mut_probe.py").items()}
ctrl_caught = m["pinner.build call site"] != ka["pinner.build call site"]
print(f"POWER CONTROL (drop `repeat` at the call site, n17's real bug): caught={ctrl_caught}")
print("VERDICT:", "SEED PATH IDENTICAL" if bad == 0 and ctrl_caught else "MISMATCH")
sys.exit(0 if bad == 0 and ctrl_caught else 1)
