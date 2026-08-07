"""CPU-only test of PinnedNoise against a fake single-draw ODE head.

The property that matters is ORDER-INDEPENDENCE: a given anchor's noise must be a
function of (seed, global index) alone, so a 200-anchor subset run reproduces the
1397-anchor run's rows. Seeding the global RNG gives order-DEPENDENCE, which looks
identical until you actually compare the two runs -- so test it, don't assert it.
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "predict"))
from predict_gr00t_n17 import PinnedNoise  # noqa: E402

WANT = (1, 16, 32)
fails = []


def check(name, cond, extra=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + extra if extra else ''}")
    if not cond:
        fails.append(name)


def head(draws=1, shape=WANT):
    """Fake get_action: draws the prior, then does deterministic 'denoising'."""
    out = None
    for _ in range(draws):
        z = torch.randn(size=shape, dtype=torch.float32, device="cpu")
        out = z if out is None else out + z
    return (out * 3.0).sum().item()   # deterministic function of the draw


print("1. probe learns the prior shape from the model, not from attribute names")
p = PinnedNoise(0)
p.probe(lambda: head())
check("probe -> (1,16,32)", p.want == WANT, str(p.want))

print("2. same (seed, index) -> identical output, regardless of call order")
def run(order, seed=0):
    pin = PinnedNoise(seed)
    pin.want = WANT
    out = {}
    for i in order:
        torch.rand(int(i) + 1)       # perturb the global RNG between calls
        pin.build(i)
        with pin:
            out[i] = head()
    return out

a = run([3, 7, 11, 19])
b = run([19, 11, 7, 3])              # reversed order = different global RNG history
c = run([7, 19])                     # a "subset run"
check("order-independent", all(a[k] == b[k] for k in a))
check("subset rows == full rows", all(c[k] == a[k] for k in c))
check("distinct anchors differ", len({round(v, 9) for v in a.values()}) == len(a))

print("3. a different seed moves every anchor")
d = run([3, 7, 11, 19], seed=1)
check("seed changes output", all(d[k] != a[k] for k in a))

print("4. control: global-RNG seeding is order-DEPENDENT (what we are avoiding)")
def run_global(order, seed=0):
    torch.manual_seed(seed)
    return {i: head() for i in order}
g_full = run_global([3, 7, 11, 19])
g_sub = run_global([7, 19])
check("global seeding breaks subset==full", g_sub[7] != g_full[7],
      "so the pinning is doing real work")

print("5. upstream drift is caught, not silently ignored")
try:
    PinnedNoise(0).probe(lambda: head(draws=2))
    check("two-draw head raises", False)
except AssertionError as e:
    check("two-draw head raises", "single-draw ODE" in str(e))

p2 = PinnedNoise(0); p2.want = WANT; p2.build(0)
try:
    with p2:
        head(shape=(1, 16, 64))      # prior shape changed -> pin never fires
    check("unconsumed noise raises", False)
except AssertionError as e:
    check("unconsumed noise raises", "never consumed" in str(e))

p3 = PinnedNoise(0); p3.want = WANT; p3.build(0)
try:
    with p3:
        head(draws=2)                # second prior-shaped draw
    check("second draw raises", False)
except RuntimeError as e:
    check("second draw raises", "second prior-shaped" in str(e))

print("6. torch.randn is restored even when the body raises")
before = torch.randn
p4 = PinnedNoise(0); p4.want = WANT; p4.build(0)
try:
    with p4:
        raise ValueError("boom")
except ValueError:
    pass
check("torch.randn restored after exception", torch.randn is before)

print()
print("VERDICT:", "ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
