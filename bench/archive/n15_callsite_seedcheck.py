#!/usr/bin/env python3
"""Layer-3 (production CALL SITE) seed check for the gr00t_n15 variance footnote.

Layer 1 (n15_seed_collision_check.py) tests the shipped PinnedNoise CLASS; layer 2 tests
the landed noise TENSORS. Neither can see a caller that never passes `repeat` -- which is
the bug gr00t-n17 actually shipped. This closes layer 3 without a GPU and without redoing
inference: `predict_gr00t.py --dry-run-seeds` runs the REAL anchor/repeat loop, short-
circuited before the model forward, and taps torch.Generator so the exported seeds are the
ones that reached the RNG rather than a re-derivation of the formula.

Two premises make the dry run evidence about the DELIVERED run, and both are checked here:
  (a) the seed-load-bearing statements are byte-identical to the frozen delivered copy
      (n15_seedpath_identity.py), so the --dry-run-seeds edit could not move a seed; and
  (b) the delivered script predates the artifacts it produced (mtime ordering), so the
      frozen copy is the code that actually ran.
"""
import ast, hashlib, json, pathlib, subprocess, sys
import numpy as np

B = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = B / "scripts/predict/predict_gr00t.py"
FROZEN = B / "logs/provenance/predict_gr00t_delivered_7de0fcb5.py"
SEEDS = B / "logs/variance/seeds_gr00t_n15_dryrun.npz"
SUBSET = B / "logs/variance_subset200.npy"
MUT = {"drop_repeat_n17_real_bug": B / "logs/variance/seeds_mutA_droprepeat.npz",
       "stride_1_diagonal_aliasing": B / "logs/variance/seeds_mutB_stride1.npz"}
ARTIFACTS = [B / "preds/preds_gr00t_n15.npz",
             B / "logs/variance/var_gr00t_n15_200x5.npz",
             B / "logs/variance/preds_gr00t_n15_subset200_seed0.npz"]
md5 = lambda p: hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def collisions(a):
    f = a.ravel().tolist()
    return len(f) - len(set(f))


# (a) seed path identical to the delivered code
ident = subprocess.run([sys.executable, str(B / "archive/n15_seedpath_identity.py")],
                       capture_output=True, text=True)
if ident.returncode != 0:
    raise SystemExit(f"seed-path identity proof FAILED — the dry run is not evidence about "
                     f"the delivered run:\n{ident.stdout}\n{ident.stderr}")
n_stmt = sum(1 for l in ident.stdout.splitlines() if l.endswith(tuple("0123456789")) and "True" in l)

# (b) the frozen copy is the code that ran: it predates every artifact it produced
t_script = FROZEN.stat().st_mtime
order = {p.name: {"artifact_mtime": p.stat().st_mtime,
                  "script_older": bool(t_script < p.stat().st_mtime)} for p in ARTIFACTS}
if not all(v["script_older"] for v in order.values()):
    raise SystemExit(f"delivered script is NEWER than an artifact it supposedly produced: {order}")

Z = np.load(SEEDS)
seeds, g, R = Z["seeds"], Z["global_idx"], int(Z["repeats"])
if not np.array_equal(g, np.load(SUBSET)):
    raise SystemExit("exported global_idx != the board's shared 200-anchor subset")
# base seed per draw, MEASURED: seeds[r,j] - g[j] must be constant in j
base = seeds - g[None, :]
if not (base == base[:, :1]).all():
    raise SystemExit("per-draw base seed is not constant across anchors — formula is not "
                     "base + global_idx and the footnote's seed fields are wrong")
per_draw = [int(v) for v in base[:, 0]]
stride = sorted({per_draw[r + 1] - per_draw[r] for r in range(R - 1)})
col = collisions(seeds)

# power controls: both caller-side bug shapes must produce collisions on the SAME anchors
ctrl = {k: collisions(np.load(p)["seeds"]) for k, p in MUT.items()}
if not all(v > 0 for v in ctrl.values()):
    raise SystemExit(f"a power control did not trip — the check has no power: {ctrl}")

out = {
    "backbone": "gr00t_n15",
    "check": "layer3_production_call_site_seed_export",
    "why": ("layer 1 tests the pinner CLASS and layer 2 the landed TENSORS; neither can see "
            "a caller that never passes `repeat` (gr00t-n17's actual bug). This runs the real "
            "loop with the forward short-circuited and taps torch.Generator."),
    "measured_at": "torch.Generator.manual_seed (the RNG sink), not re-derived from the formula",
    "gpu_used": False, "reinference": False, "preds_written": False,
    "delivered_script": {
        "frozen_copy": str(FROZEN), "md5": md5(FROZEN),
        "md5_at_delivery_recorded_from_live_file": "7de0fcb51e86905a19058ab00aba3f22",
        "current_md5_after_dryrun_branch": md5(SCRIPT),
        "seed_path_identical_to_delivered": True,
        "seed_load_bearing_statements_compared": n_stmt,
        "identity_proof": "archive/n15_seedpath_identity.py (AST-extracted, byte-compared, "
                          "with a must-trip mutation of the call site)",
        "mtime_ordering_vs_artifacts": order,
    },
    "export": {"file": str(SEEDS), "md5": md5(SEEDS), "shape": list(seeds.shape),
               "argv": str(Z["argv"]), "batch_size": int(Z["batch_size"])},
    "measured_per_draw_seeds": per_draw,
    "measured_per_draw_seed_stride": stride[0] if len(stride) == 1 else stride,
    "measured_stride_exceeds_anchor_count": (len(stride) == 1 and stride[0] > 1397),
    "measured_seed_collisions": col,
    "n_seeds": int(seeds.size),
    "min_seed_gap_across_draws": int(min(
        np.abs(seeds[i][:, None] - seeds[j][None, :]).min()
        for i in range(R) for j in range(i + 1, R))),
    "power_controls": ctrl,
    "power_controls_note": ("stride_1 gives 254 collisions, the same number layer 1's synthetic "
                            "control produced — NOT independent corroboration: collision count "
                            "is a deterministic function of the same 200 anchor indices and the "
                            "same stride, so this only shows the two harnesses agree."),
    "verdict": "PASS" if col == 0 else "FAIL",
}
p = B / "logs/variance/seedcheck_gr00t_n15_callsite.json"
p.write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("measured_per_draw_seeds", "measured_per_draw_seed_stride",
                                      "measured_seed_collisions", "n_seeds", "power_controls",
                                      "verdict")}, indent=2))
print("wrote", p)
