"""Replay openpi's OWN RunningStats object over my TRAIN-111 frame set -- all four stats.

Follow-up to openpi_normstats_port_arithmetic.py, which showed the mean/std floor is exactly
0.00e+00 once the recompute uses float32 AND the bs16 streaming tree. That covered mean/std
because those are closed-form. q01/q99 are NOT: openpi estimates them from a 5000-bin
histogram with a lossy rebinning step (`_adjust_histograms`), so comparing the shipped
quantile against `np.quantile` compares two DIFFERENT ESTIMATORS -- the ~4e-2 "floor" I
measured there is estimator mismatch, not numerical noise. That distinction predicts the
floor is removable by porting the *estimator*, not just the dtype/order.

So instead of re-implementing, this instantiates the upstream `normalize.RunningStats` and
feeds it my frame set in the loader's order. The frame set (which episodes, which anchors,
the delta transform) is still independently mine -- that is the part the leak question turns
on -- but every line of arithmetic downstream of it is upstream's.

Runs both candidates:
  TRAIN-111 -> should reproduce the shipped file (0 = bit-exact positive identification)
  FULL-131  -> the leak counterfactual; its separation from the shipped file, measured on
               the SAME estimator, is the honest signal (no estimator mismatch inflating it)

CPU only, no GPU, no model.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from openpi_leak_lattice import BS, VAL_EPS, build  # noqa: E402
from openpi.shared import normalize as _normalize  # noqa: E402

STATS_PATH = str(pathlib.Path(__file__).resolve().parents[2] / "third_party" / "openpi-agilex" /
                 "assets" / "pi05_nero_b2_train" / "local" /
                 "pick_pink_sponge_b2_train" / "norm_stats.json")
OUT = pathlib.Path(__file__).resolve().parents[1] / "logs" / "openpi_normstats_upstream_replay.json"

H = 10
STATS = ("mean", "std", "q01", "q99")


def replay(chunked, n_frames):
    """Feed batches of BS through upstream RunningStats exactly as compute_norm_stats does."""
    rs = _normalize.RunningStats()
    for i in range(0, n_frames, BS):
        rs.update(np.asarray(chunked[i:i + BS], dtype=np.float32))
    return rs.get_statistics()


def run(repo_id, restrict_eps=None):
    d = build(repo_id, restrict_eps=restrict_eps)
    act_flat = d["actions"]
    n_frames = act_flat.shape[0] // H
    assert d["state"].shape[0] == n_frames
    n_batches = (n_frames // BS) * BS
    out = {
        "actions": replay(act_flat.reshape(n_frames, H, -1), n_batches),
        "state": replay(d["state"], n_batches),
        "meta": d["meta"],
    }
    return out


def main():
    shipped = json.loads(pathlib.Path(STATS_PATH).read_text())["norm_stats"]

    print("replaying upstream RunningStats over TRAIN-111 ...", flush=True)
    train = run("local/pick_pink_sponge_b2_train")
    print("  ", train["meta"], flush=True)
    print("replaying upstream RunningStats over FULL-131 (leak counterfactual) ...", flush=True)
    full = run("local/pick_pink_sponge_b2")
    print("  ", full["meta"], flush=True)
    # VAL-20: the second, more extreme counterfactual. gr00t-n15's point (2026-08-05): the number
    # of "dead cells" in the DIRECT direction (does the counterfactual move off the shipped value)
    # is NOT the same quantity as the COLLISION rate in the must-differ direction, and one cannot
    # be inferred from the other -- a dim can have plenty of resolving power overall and still have
    # an individual component collide. So measure my own collision rate rather than infer it.
    print("replaying upstream RunningStats over VAL-20 (collision counterfactual) ...", flush=True)
    val = run("local/pick_pink_sponge_b2", restrict_eps=VAL_EPS)
    print("  ", val["meta"], flush=True)

    report = {"meta": {"train111": train["meta"], "full131": full["meta"], "val20": val["meta"]},
              "keys": {}}
    power = {"full131": {"collide": 0, "total": 0}, "val20": {"collide": 0, "total": 0}}

    for key in ("actions", "state"):
        shp = {s: np.asarray(shipped[key][s], dtype=np.float64) for s in STATS}
        st = {s: np.asarray(getattr(train[key], s), dtype=np.float64) for s in STATS}
        sf = {s: np.asarray(getattr(full[key], s), dtype=np.float64) for s in STATS}
        sv = {s: np.asarray(getattr(val[key], s), dtype=np.float64) for s in STATS}

        print(f"\n===== {key} =====")
        print(f"  {'stat':<6}{'max|train111-shipped|':>24}{'min|full131-shipped|':>23}"
              f"{'sep':>9}{'val20 collisions':>19}")
        entry = {}
        for s in STATS:
            repl = np.abs(st[s] - shp[s])
            sep = np.abs(sf[s] - shp[s])
            sepv = np.abs(sv[s] - shp[s])
            # with a zero floor, "separating" == "moved at all"
            nsep = int((sep > repl).sum()) if repl.max() > 0 else int((sep > 0).sum())
            ncol_f = int((sep == 0).sum())   # cells where the leak counterfactual is INVISIBLE
            ncol_v = int((sepv == 0).sum())
            power["full131"]["collide"] += ncol_f
            power["full131"]["total"] += len(repl)
            power["val20"]["collide"] += ncol_v
            power["val20"]["total"] += len(repl)
            entry[s] = {"repl_max": float(repl.max()), "repl": repl.tolist(),
                        "sep_min": float(sep.min()), "sep": sep.tolist(),
                        "dims_separating": nsep, "ndim": int(len(repl)),
                        "val20_sep": sepv.tolist(), "val20_sep_min": float(sepv.min()),
                        "collisions_full131": ncol_f, "collisions_val20": ncol_v}
            print(f"  {s:<6}{repl.max():>24.3e}{sep.min():>23.3e}{nsep:>6}/{len(repl)}"
                  f"{ncol_v:>16}/{len(repl)}")
        report["keys"][key] = entry

    print("\n===== TEST POWER (how often a contaminated stat would be INVISIBLE) =====")
    for cf, d in power.items():
        c, t = d["collide"], d["total"]
        print(f"  vs {cf:<8} collisions {c}/{t} = {100.0 * c / t:5.2f}%  "
              f"-> power {100.0 * (t - c) / t:5.2f}%")
    report["power"] = power
    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
