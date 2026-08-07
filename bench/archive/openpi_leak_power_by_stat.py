"""Per-statistic resolving power of the openpi leak falsification.

gr00t-n15 measured, on the LeRobot/GR00T stacks, that leak-detection power is ordered
    full-frame aggregates (mean/std, 8/8 dims) > order statistics (q01/q99, 6-7/8) > extrema (max 3/8, min 1-2/8)
and pointed out that my headline evidence was the LATTICE test, which rides on q01/q99+min/max
-- i.e. the weaker end -- even though openpi ships mean/std, the strong end.

The claim is about *which dims move at all*. It does not automatically transfer here, for two
reasons worth measuring rather than assuming:
  1. my action rows are DELTA-transformed (make_bool_mask(7,-1)), so the "does adding val
     episodes move this dim" question is asked in a different space than n15's absolute one;
  2. power = signal / noise, and my noise floor is NOT n15's. Their clean-vs-clean floor was
     ~2e-14 (pure reordering). Mine is the replication error |shipped - my recomputation|,
     which is limited by float64-vs-float32 and batch-order differences, ~1e-6 on mean.
     A statistic that moves on 8/8 dims but sits on a 1e-6 floor is not automatically better
     than one that moves on 4/8 dims but sits on a 1e-13 floor.

So this reports BOTH axes per statistic: how many dims separate train-111 from full-131
(coverage), and by how much relative to the replication floor (SNR).

CPU-only, no model. Reuses the builders from openpi_leak_lattice.py so the frame set is
identical to the one already validated against LeRobot's own _get_query_indices.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from openpi_leak_lattice import VAL_EPS, build  # noqa: E402

STATS_PATH = str(pathlib.Path(__file__).resolve().parents[2] / "third_party" / "openpi-agilex" /
                 "assets" / "pi05_nero_b2_train" / "local" /
                 "pick_pink_sponge_b2_train" / "norm_stats.json")
OUT = pathlib.Path(__file__).resolve().parents[1] / "logs" / "openpi_leak_power_by_stat.json"


def all_stats(arr):
    return {
        "mean": arr.mean(0),
        "std": arr.std(0),
        "q01": np.quantile(arr, 0.01, axis=0),
        "q99": np.quantile(arr, 0.99, axis=0),
        "min": arr.min(0),
        "max": arr.max(0),
    }


def main():
    shipped = json.loads(pathlib.Path(STATS_PATH).read_text())["norm_stats"]

    print("building TRAIN-111 ...", flush=True)
    train = build("local/pick_pink_sponge_b2_train")
    print("building FULL-131 (the leaked counterfactual) ...", flush=True)
    full = build("local/pick_pink_sponge_b2")
    print("building VAL-20 ...", flush=True)
    val = build("local/pick_pink_sponge_b2", restrict_eps=VAL_EPS, truncate=False)

    report = {"meta": {k: v["meta"] for k, v in
                       (("train111", train), ("full131", full), ("val20", val))}, "keys": {}}

    for key in ("actions", "state"):
        st, sf = all_stats(train[key]), all_stats(full[key])
        ndim = st["mean"].shape[0]

        # replication floor: only mean/std are actually shipped, so only those have a
        # measurable floor. For the rest we can only report the train-vs-full separation.
        floor = {}
        for s in ("mean", "std", "q01", "q99"):
            floor[s] = np.abs(np.asarray(shipped[key][s], dtype=np.float64) - st[s])

        entry = {"ndim": int(ndim), "per_stat": {}}
        print(f"\n===== {key} ({ndim} dims) =====")
        print(f"  {'stat':<6}{'dims separating':>17}{'min|sep|':>12}{'median|sep|':>13}"
              f"{'repl floor':>12}{'worst-dim SNR':>15}")
        for s in ("mean", "std", "q01", "q99", "min", "max"):
            sep = np.abs(st[s] - sf[s])
            fl = floor.get(s)
            # a dim "separates" if the train-vs-full gap is >=100x the replication floor on
            # that dim (for shipped stats), else simply nonzero in float64.
            if fl is not None:
                # guard against a zero floor: use the key-wide max floor as the scale
                scale = np.maximum(fl, fl.max() if fl.max() > 0 else 1e-15)
                sepd = int((sep > 100 * scale).sum())
                snr = float((sep / np.maximum(scale, 1e-300)).min())
                flr = float(fl.max())
            else:
                sepd = int((sep > 0).sum())
                snr = float("nan")
                flr = float("nan")
            entry["per_stat"][s] = dict(
                dims_separating=sepd, sep_min=float(sep.min()), sep_median=float(np.median(sep)),
                sep_max=float(sep.max()), repl_floor_max=flr, worst_dim_snr=snr,
                sep_per_dim=sep.tolist(),
            )
            print(f"  {s:<6}{f'{sepd}/{ndim}':>17}{sep.min():>12.3e}{np.median(sep):>13.3e}"
                  f"{flr:>12.3e}{snr:>15.3e}")
        report["keys"][key] = entry

    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
