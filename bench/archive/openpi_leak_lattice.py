"""Leakage falsification for the openpi (pi0 / pi0.5) norm_stats.

Team-lead's broadcast test is `val_max > stat_max` per joint: if val leaked into the
normalization stats then `stat.max >= val.max` on EVERY joint, so any joint where val
exceeds proves val was not in the stats.

openpi does NOT serialize max. `RunningStats.get_statistics()` (src/openpi/shared/
normalize.py) returns only mean/std/q01/q99; `_max` is tracked and then discarded. So the
literal test is unrunnable here. But the max is *encoded* in the quantiles:

    _adjust_histograms():  _bin_edges[i] = linspace(_min[i], _max[i], 5001)
    _compute_quantiles():  q_values.append(edges[idx])        # a RAW BIN EDGE

q01/q99 are therefore snapped to a 5000-interval lattice whose two endpoints ARE min and
max. Given a candidate episode set we can recompute (min, max), rebuild that lattice, and
ask whether the shipped q01/q99 land exactly on it. A leaked (full-131) stats object would
carry a different min/max => a different lattice => the shipped quantiles would sit off-grid.

This is a two-sided identification, strictly stronger than the one-sided max test: it does
not merely fail to contradict train-only, it excludes full-131.

Robustness note: the histogram COUNTS drift (\_adjust_histograms redistributes lossily), so
the quantile VALUE is approximate. The bin EDGES are exact. The lattice test only uses the
edges, so it is immune to that drift.

Also reported: whether val actually exceeds the train range at all, per joint. If val is a
subset of the train range then min/max coincide for both candidates and the test has no
discriminating power -- that is the blind spot to declare, not to hide.
"""

import argparse
import json
import pathlib

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

H = 10  # action_horizon for both bench configs
BS = 16  # compute_norm_stats batch_size (2376 batches x 16 == 38016 <= 38025 frames)
NBINS = 5000

VAL_EPS = [0, 2, 3, 4, 20, 24, 25, 33, 37, 51, 56, 68, 70, 80, 90, 91, 92, 101, 106, 111]


def load_columns(repo_id):
    meta = LeRobotDatasetMetadata(repo_id)
    ds = LeRobotDataset(repo_id, delta_timestamps={"action": [t / meta.fps for t in range(H)]})
    hf = ds.hf_dataset.select_columns(["observation.state", "action", "episode_index"]).with_format("numpy")
    cols = hf[:]
    state = np.asarray(cols["observation.state"], dtype=np.float64)
    action = np.asarray(cols["action"], dtype=np.float64)
    ep = np.asarray(cols["episode_index"], dtype=np.int64)
    frm = np.asarray(ds.episode_data_index["from"], dtype=np.int64)
    to = np.asarray(ds.episode_data_index["to"], dtype=np.int64)
    return ds, state, action, ep, frm, to


def verify_clamp(ds, ep, frm, to, n, rng):
    """Cross-check the numpy clamp against LeRobot's own _get_query_indices."""
    idxs = rng.choice(n, size=min(500, n), replace=False)
    bad = 0
    for i in idxs:
        qi_ref, _ = ds._get_query_indices(int(i), int(ep[i]))
        ref = np.asarray(qi_ref["action"], dtype=np.int64)
        mine = np.clip(i + np.arange(H), frm[ep[i]], to[ep[i]] - 1)
        if not np.array_equal(ref, mine):
            bad += 1
    return len(idxs), bad


def build(repo_id, restrict_eps=None, truncate=True):
    """Return the delta-transformed action rows + state rows that RunningStats saw."""
    ds, state, action, ep, frm, to = load_columns(repo_id)
    n = len(state)
    rng = np.random.default_rng(0)
    nchk, nbad = verify_clamp(ds, ep, frm, to, n, rng)

    limit = (n // BS) * BS if truncate else n
    idx = np.arange(limit)
    if restrict_eps is not None:
        idx = idx[np.isin(ep[idx], restrict_eps)]

    qi = np.clip(idx[:, None] + np.arange(H)[None, :], frm[ep[idx]][:, None], (to[ep[idx]] - 1)[:, None])
    chunks = action[qi]                       # (M, H, 8) absolute
    chunks[:, :, :7] -= state[idx][:, None, :7]   # DeltaActions(make_bool_mask(7, -1))
    return {
        "actions": chunks.reshape(-1, chunks.shape[-1]),
        "state": state[idx],
        "meta": dict(repo=repo_id, n_items=int(len(idx)), n_total=int(n), limit=int(limit),
                     clamp_checked=nchk, clamp_mismatch=nbad,
                     n_eps=int(len(np.unique(ep[idx])))),
    }


def stats_of(arr):
    return dict(mean=arr.mean(0), std=arr.std(0), min=arr.min(0), max=arr.max(0))


def lattice_residual(q, lo, hi, eps_pad=0.0):
    """How far the shipped quantile sits from the nearest node of linspace(lo,hi,NBINS+1)."""
    lo = lo - eps_pad
    hi = hi + eps_pad
    t = (q - lo) / (hi - lo) * NBINS
    return np.abs(t - np.round(t)), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=str(pathlib.Path(__file__).resolve().parents[2] / "third_party" / "openpi-agilex" /
                                           "assets" / "pi05_nero_b2_train" / "local" /
                                           "pick_pink_sponge_b2_train" / "norm_stats.json"))
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[1] /
                                        "logs" / "openpi_leak_lattice.json"))
    args = ap.parse_args()

    shipped = json.loads(pathlib.Path(args.stats).read_text())["norm_stats"]

    print("building TRAIN-111 (the repo the configs point at) ...", flush=True)
    train = build("local/pick_pink_sponge_b2_train")
    print("  ", train["meta"], flush=True)
    print("building FULL-131 (counterfactual: what leaked stats would be) ...", flush=True)
    full = build("local/pick_pink_sponge_b2")
    print("  ", full["meta"], flush=True)
    print("building VAL-20 (for discriminating power) ...", flush=True)
    val = build("local/pick_pink_sponge_b2", restrict_eps=VAL_EPS, truncate=False)
    print("  ", val["meta"], flush=True)

    report = {"meta": {"train": train["meta"], "full131": full["meta"], "val": val["meta"]}, "keys": {}}

    for key in ("actions", "state"):
        st, sf, sv = stats_of(train[key]), stats_of(full[key]), stats_of(val[key])
        shp = shipped[key]
        q01 = np.asarray(shp["q01"], dtype=np.float64)
        q99 = np.asarray(shp["q99"], dtype=np.float64)
        mean = np.asarray(shp["mean"], dtype=np.float64)
        std = np.asarray(shp["std"], dtype=np.float64)

        # (0) validate the replication: shipped mean/std are exact running computations
        val_mean_err = np.abs(mean - st["mean"])
        val_std_err = np.abs(std - st["std"])

        # (1) lattice test, both grid variants (post-adjust exact, and never-adjusted +-1e-10)
        res = {}
        for cand, s in (("train111", st), ("full131", sf)):
            r99, t99 = lattice_residual(q99, s["min"], s["max"])
            r01, t01 = lattice_residual(q01, s["min"], s["max"])
            r99p, _ = lattice_residual(q99, s["min"], s["max"], 1e-10)
            r01p, _ = lattice_residual(q01, s["min"], s["max"], 1e-10)
            res[cand] = dict(
                q99_resid=r99.tolist(), q01_resid=r01.tolist(),
                q99_resid_pad=r99p.tolist(), q01_resid_pad=r01p.tolist(),
                q99_binidx=t99.tolist(), q01_binidx=t01.tolist(),
                min=s["min"].tolist(), max=s["max"].tolist(),
            )

        # (2) discriminating power: does val exceed the train range at all?
        exceed_hi = sv["max"] > st["max"]
        exceed_lo = sv["min"] < st["min"]

        report["keys"][key] = dict(
            shipped_q01=q01.tolist(), shipped_q99=q99.tolist(),
            repl_mean_abserr=val_mean_err.tolist(), repl_std_abserr=val_std_err.tolist(),
            train_min=st["min"].tolist(), train_max=st["max"].tolist(),
            full131_min=sf["min"].tolist(), full131_max=sf["max"].tolist(),
            val_min=sv["min"].tolist(), val_max=sv["max"].tolist(),
            val_exceeds_train_hi=exceed_hi.tolist(), val_exceeds_train_lo=exceed_lo.tolist(),
            lattice=res,
        )

        print(f"\n===== {key} =====")
        print(f"  replication check: max|mean-shipped|={val_mean_err.max():.3e}  "
              f"max|std-shipped|={val_std_err.max():.3e}")
        print(f"  {'j':<3}{'train_max':>12}{'full131_max':>13}{'val_max':>12}"
              f"{'val>train?':>11}{'r99|train':>11}{'r99|full':>11}")
        for j in range(len(q99)):
            print(f"  {j:<3}{st['max'][j]:>12.4f}{sf['max'][j]:>13.4f}{sv['max'][j]:>12.4f}"
                  f"{('YES' if exceed_hi[j] else '-'):>11}"
                  f"{res['train111']['q99_resid'][j]:>11.2e}{res['full131']['q99_resid'][j]:>11.2e}")

    pathlib.Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
