#!/usr/bin/env python3
"""Two-sided test for predict_openpi.guard_production_out (the preds/ write gate).

Runs on 145 and 147 with no GPU and no model: the guard is a pure function placed before
`import jax` precisely so it can be tested this way. Two halves, and the second is the one that
matters most:

  1. the function accepts every production invocation and rejects every defect shape;
  2. the SHIPPED SCRIPT actually calls it -- a correct guard that no code path reaches is the
     classic way this check passes review and does nothing. Half 2 execs predict_openpi.py for
     real and asserts it dies on the gate *before* jax is imported (elapsed < 5s proves it: a
     jax import on this box takes ~15s+, so a slow exit would mean the gate moved below it).
"""
import pathlib, subprocess, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from predict_openpi import CFG_TAG, PREDS_DIR, guard_production_out  # noqa: E402

BENCH = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = BENCH / "scripts" / "predict" / "predict_openpi.py"
OK = "anchors"  # default --state-source
npass = nfail = 0


def check(label, cond, detail=""):
    global npass, nfail
    if cond:
        npass += 1
        print(f"  PASS  {label}" + (f"   [{detail}]" if detail else ""))
    else:
        nfail += 1
        print(f"  FAIL  {label}   {detail}")


def accepts(label, **kw):
    """Guard must return silently."""
    try:
        guard_production_out(**kw)
        check(label, True)
    except SystemExit as e:
        check(label, False, f"unexpectedly REJECTED: {str(e)[:120]}")


def rejects(label, expect_substr, **kw):
    """Guard must raise SystemExit, and for the stated reason -- a gate that fires for the wrong
    reason is a control that failed for its own reasons and teaches nothing."""
    try:
        guard_production_out(**kw)
        check(label, False, "was ACCEPTED (gate did not fire)")
    except SystemExit as e:
        msg = str(e)
        check(label, expect_substr in msg,
              f"fired but reason lacks {expect_substr!r}: {msg[:160]}" if expect_substr not in msg
              else expect_substr)


def base(**over):
    kw = dict(out=str(PREDS_DIR / "preds_pi05.npz"), config_name="pi05_nero_b2_train",
              subset_n=0, limit=None, noise_seed=0, state_source=OK)
    kw.update(over)
    return kw


print("=== half 1: the guard's own behaviour ===")
print("-- positive: every argv the pipeline actually issues must pass --")
accepts("production pi0.5 (pi05_nero_b2_train -> preds_pi05.npz)", **base())
accepts("production pi0   (pi0_nero_b2_train  -> preds_pi0.npz)",
        **base(out=str(PREDS_DIR / "preds_pi0.npz"), config_name="pi0_nero_b2_train"))
accepts("non-normalised path that resolves INTO preds/ still accepted when clean",
        **base(out=str(PREDS_DIR / ".." / "preds" / "preds_pi05.npz")))

print("-- negative: each defect shape, one at a time --")
rejects("partial run: --subset-n 200", "--subset-n 200", **base(subset_n=200))
rejects("partial run: --limit 50", "--limit 50", **base(limit=50))
rejects("non-canonical --noise-seed 3", "--noise-seed 3", **base(noise_seed=3))
rejects("--state-source dataset", "--state-source dataset", **base(state_source="dataset"))
rejects("unknown config may not write a row", "absent from CFG_TAG",
        **base(config_name="pi05_nero_b2_abs"))

print("-- the trap this gate exists for: ckpt/filename swap --")
rejects("pi0 config writing preds_pi05.npz", "may only write preds_pi0.npz",
        **base(config_name="pi0_nero_b2_train"))
rejects("pi0.5 config writing preds_pi0.npz", "may only write preds_pi05.npz",
        **base(out=str(PREDS_DIR / "preds_pi0.npz")))
# The prefix trap, stated explicitly: had the stem test been `startswith`, the pi0->preds_pi05
# case above would have been ACCEPTED, because 'preds_pi05'.startswith('preds_pi0') is True.
check("startswith would have been unsafe (documents why equality is used)",
      "preds_pi05".startswith("preds_pi0") and "preds_pi05" != "preds_pi0",
      "prefix True, equality False")

print("-- scope: outside preds/ nothing is restricted --")
accepts("/tmp partial run + odd seed + swapped tag is fine",
        **base(out="/tmp/dryrun_pi05.npz", config_name="pi0_nero_b2_train",
               subset_n=200, noise_seed=7, state_source="dataset"))
accepts("logs/variance path unrestricted",
        **base(out=str(BENCH / "logs" / "variance" / "pi05_s3.npz"), noise_seed=3, subset_n=200))
accepts("a DIFFERENT dir named preds/ elsewhere is not ours",
        **base(out="/tmp/preds/preds_pi0.npz", config_name="pi0_nero_b2_train", noise_seed=9))

print("-- multi-defect argv reports all reasons, not just the first --")
try:
    guard_production_out(**base(subset_n=200, noise_seed=3, config_name="pi0_nero_b2_train"))
    check("multi-defect rejected", False, "accepted")
except SystemExit as e:
    m = str(e)
    check("multi-defect lists all 3 reasons", all(s in m for s in ("--subset-n", "--noise-seed",
                                                                  "may only write")),
          f"{m.count(chr(10)+'  - ')} reasons listed")

print("\n=== half 2: the SHIPPED script reaches the guard (before jax) ===")
t0 = time.monotonic()
r = subprocess.run([sys.executable, str(SCRIPT), "--config-name", "pi0_nero_b2_train",
                    "--ckpt-dir", "/nonexistent-ckpt-for-test",
                    "--out", str(PREDS_DIR / "preds_pi05.npz")],
                   capture_output=True, text=True, timeout=300)
el = time.monotonic() - t0
out = r.stdout + r.stderr
check("shipped script rejects the ckpt/filename swap", r.returncode != 0 and "[out-gate]" in out,
      f"rc={r.returncode} out={out.strip()[:140]}")
check("gate runs BEFORE the jax import (fast exit)", el < 5.0, f"{el:.2f}s")
check("nothing was written to preds/ by the rejected run",
      not (PREDS_DIR / "preds_pi05.npz").exists()
      or (PREDS_DIR / "preds_pi05.npz").stat().st_size > 0,
      "no truncated/created delivery file")

# Power control for half 2: the SAME argv with the tag corrected must get PAST the gate and fail
# later, for a different reason. Without this, "the script exits nonzero" proves nothing -- it
# would exit nonzero if predict_openpi.py were simply broken.
t0 = time.monotonic()
r2 = subprocess.run([sys.executable, str(SCRIPT), "--config-name", "pi0_nero_b2_train",
                     "--ckpt-dir", "/nonexistent-ckpt-for-test",
                     "--out", str(PREDS_DIR / "preds_pi0.npz")],
                    capture_output=True, text=True, timeout=900,
                    env={**__import__("os").environ, "JAX_PLATFORMS": "cpu",
                         "HF_HUB_OFFLINE": "1"})
el2 = time.monotonic() - t0
out2 = r2.stdout + r2.stderr
check("power control: clean production argv is NOT stopped by the gate",
      "[out-gate]" not in out2, f"rc={r2.returncode} {el2:.1f}s")
check("power control: it got past the guard and died downstream instead",
      r2.returncode != 0 and "[out-gate]" not in out2,
      f"tail={out2.strip().splitlines()[-1][:120] if out2.strip() else '(empty)'}")

print(f"\n{npass} PASS / {nfail} FAIL")
sys.exit(1 if nfail else 0)
