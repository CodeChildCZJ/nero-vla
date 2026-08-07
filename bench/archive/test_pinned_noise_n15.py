"""CPU-only test of N1.5's PinnedNoise against a fake single-draw ODE head.

Structure borrowed from gr00t-n17's test_pinned_noise.py, including the control
group: "subset rows == full rows" only proves something if the naive alternative
(global seeding) demonstrably FAILS the same assertion on the same fake head.
Without the control, a head that ignores its noise would pass trivially.

Run: $NERO_ROOT/third_party/Isaac-GR00T-n1d5/.venv/bin/python bench/archive/test_pinned_noise_n15.py
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "predict"))
from predict_gr00t import PinnedNoise  # noqa: E402

H, D = 16, 32          # N1.5: config.action_horizon, config.action_dim
fails = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")
    if not cond:
        fails.append(name)


def head(batch, draws=1, shape=None):
    """Fake get_action: draw the prior, then 'denoise' deterministically."""
    shape = shape or (batch, H, D)
    out = None
    for _ in range(draws):
        z = torch.randn(size=shape, dtype=torch.float32, device="cpu")
        out = z if out is None else out + z
    return (out * 3.0).sum(dim=(1, 2)).tolist()


def run(idxs, seed=0, repeat=0, batch=None):
    """Predict `idxs` in batches, returning {global_idx: value}."""
    batch = batch or len(idxs)
    p = PinnedNoise(H, D, seed)
    got = {}
    for lo in range(0, len(idxs), batch):
        chunk = idxs[lo:lo + batch]
        torch.rand(lo + 1)                    # deliberately disturb global RNG history
        p.build(chunk, repeat)
        with p:
            vals = head(len(chunk))
        got.update(dict(zip(chunk, vals)))
    return got


print("1. exactly-once enforcement (the part n17 caught)")
p = PinnedNoise(H, D, 0)
p.build([0], 0)
try:
    with p:
        head(1, draws=2)                      # ODE -> SDE would look like this
    check("second prior-shaped draw raises", False, "no exception")
except RuntimeError as e:
    check("second prior-shaped draw raises", "SECOND prior-shaped" in str(e))

p = PinnedNoise(H, D, 0)
p.build([0], 0)
try:
    with p:
        head(1, shape=(1, H, 8))              # wrong shape => pin never fires
    check("zero prior-shaped draws raises", False, "no exception")
except RuntimeError as e:
    check("zero prior-shaped draws raises", "fired 0x" in str(e))
    check("  diagnostic names observed shape", "(1, 16, 8)" in str(e), str(e)[-60:])

p = PinnedNoise(H, D, 0)
p.build([0], 0)
try:
    with p:
        raise ValueError("model blew up")
    check("real exception is not masked", False)
except ValueError:
    check("real exception is not masked", True)
except RuntimeError:
    check("real exception is not masked", False, "fired-count check masked it")
check("torch.randn restored after raise", torch.randn is p._orig)

print("2. same (seed, index) -> same value, independent of call order and batching")
full = run(list(range(20)))
rev = run(list(reversed(range(20))))
check("order-independent", all(full[i] == rev[i] for i in range(20)))
b1 = run(list(range(20)), batch=1)
check("batch-size-independent (fake head is batch-invariant)",
      all(full[i] == b1[i] for i in range(20)))
sub = run([7, 19])
check("subset rows == full rows", sub[7] == full[7] and sub[19] == full[19])
check("distinct anchors differ", len({round(v, 6) for v in full.values()}) == 20)

print("3. seed and repeat actually move the noise")
check("different seed -> different", run([5], seed=1)[5] != full[5])
check("different repeat -> different", run([5], repeat=1)[5] != full[5])
check("repeat is reproducible", run([5], repeat=1)[5] == run([5], repeat=1)[5])

print("4. CONTROL: the naive global-seed approach fails the same assertions")


def run_global(idxs, seed=0, batch=None):
    batch = batch or len(idxs)
    got = {}
    for lo in range(0, len(idxs), batch):
        chunk = idxs[lo:lo + batch]
        torch.manual_seed(seed)
        vals = head(len(chunk))
        got.update(dict(zip(chunk, vals)))
    return got


g_full = run_global(list(range(20)))
g_sub = run_global([7, 19])
check("global seeding BREAKS subset==full",
      not (g_sub[7] == g_full[7] and g_sub[19] == g_full[19]),
      "(if this PASSes as equal, test 2 proves nothing)")

print()
if fails:
    print(f"{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("all checks passed")
