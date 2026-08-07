"""Port openpi's own RunningStats arithmetic, to collapse the mean/std replication floor.

Context. My leak evidence is |shipped norm_stats - my recomputation over TRAIN-111|. That
residual has been ~3.8e-06 (actions mean) / ~1.6e-05 (actions std), which is a *floor*, not
a signal: it limits how small a real contamination I could still resolve, and it is the
reason I demoted std to corroboration (its floor is ~4x the mean's, so its worst-dim SNR is
4.7x vs the mean's 140x).

gr00t-n15's attribution: that floor belongs to my RECOMPUTE IMPLEMENTATION, not to the
statistic. n15 ported GR00T's arithmetic verbatim and got dev = 0.00e+00 on 24/24 cells,
std included. Their falsifiable prediction for openpi: replay `RunningStats.update` in the
trainer's dtype and batch order and my floor should collapse to ~1e-7 or 0 -- and if it
collapses, the reason to demote std disappears.

Two candidate mechanisms, and this script ABLATES them separately rather than changing both
at once (which would prove a number but not a cause):

  dtype   -- the loader hands float32; my recompute cast to float64.
  order   -- `compute_norm_stats.py` streams batches of 16 through
             `_mean += (batch_mean - _mean) * (n/count)` (normalize.py:65).
             That is algebraically the plain mean for equal-size batches, but a different
             floating-point summation tree than one global `arr.mean(0)`.

4 cells: {f64,f32} x {global, streamed bs16}. Whichever cell hits 0.00e+00 names the cause.

Note the batch shapes differ per key, and that matters for the streaming tree:
  actions -> (16, H=10, 8) reshaped by update() to (160, 8), so 160 elements per step
  state   -> (16, 8), 16 elements per step

CPU only, no GPU, no model. Reuses openpi_leak_lattice.build() so the frame set is the one
already cross-checked against LeRobot's own _get_query_indices.
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from openpi_leak_lattice import BS, build  # noqa: E402

STATS_PATH = str(pathlib.Path(__file__).resolve().parents[2] / "third_party" / "openpi-agilex" /
                 "assets" / "pi05_nero_b2_train" / "local" /
                 "pick_pink_sponge_b2_train" / "norm_stats.json")
OUT = pathlib.Path(__file__).resolve().parents[1] / "logs" / "openpi_normstats_port_arithmetic.json"

H = 10  # action_horizon; must match openpi_leak_lattice.H


def global_stats(rows, dtype):
    a = rows.astype(dtype, copy=False)
    m = a.mean(0)
    mos = (a ** 2).mean(0)
    return m, np.sqrt(np.maximum(0, mos - m ** 2))


def streamed_stats(batches, dtype):
    """Bit-faithful replay of normalize.RunningStats mean / mean_of_squares."""
    count = 0
    mean = None
    mos = None
    for b in batches:
        b = np.asarray(b, dtype=dtype)
        b = b.reshape(-1, b.shape[-1])          # update() line 1
        n = b.shape[0]
        if count == 0:
            mean = np.mean(b, axis=0)
            mos = np.mean(b ** 2, axis=0)
            count = n
            continue
        count += n
        bm = np.mean(b, axis=0)
        bmos = np.mean(b ** 2, axis=0)
        mean += (bm - mean) * (n / count)
        mos += (bmos - mos) * (n / count)
    var = mos - mean ** 2
    return mean, np.sqrt(np.maximum(0, var))


def main():
    shipped = json.loads(pathlib.Path(STATS_PATH).read_text())["norm_stats"]

    print("building TRAIN-111 ...", flush=True)
    train = build("local/pick_pink_sponge_b2_train")
    print("  ", train["meta"], flush=True)

    # build() returns actions already flattened to (M*H, 8); re-fold to per-frame chunks so
    # the batch boundaries land where the dataloader put them.
    act_flat = train["actions"]
    n_frames = act_flat.shape[0] // H
    act_chunks = act_flat.reshape(n_frames, H, act_flat.shape[-1])
    state = train["state"]
    assert state.shape[0] == n_frames, (state.shape, n_frames)
    assert n_frames % BS == 0, n_frames

    report = {"meta": train["meta"], "n_frames": int(n_frames), "batch_size": BS,
              "n_batches": int(n_frames // BS), "keys": {}}

    for key, rows, chunked in (("actions", act_flat, act_chunks), ("state", state, state)):
        shp = {s: np.asarray(shipped[key][s], dtype=np.float64) for s in ("mean", "std")}
        batches = [chunked[i:i + BS] for i in range(0, n_frames, BS)]

        cells = {
            "f64_global": global_stats(rows, np.float64),
            "f32_global": global_stats(rows, np.float32),
            "f64_streamed_bs16": streamed_stats(batches, np.float64),
            "f32_streamed_bs16": streamed_stats(batches, np.float32),
        }

        print(f"\n===== {key} =====")
        print(f"  {'cell':<20}{'max|mean-shipped|':>20}{'max|std-shipped|':>20}")
        entry = {}
        for name, (m, s) in cells.items():
            dm = float(np.abs(m.astype(np.float64) - shp["mean"]).max())
            ds = float(np.abs(s.astype(np.float64) - shp["std"]).max())
            entry[name] = {"mean_abserr_max": dm, "std_abserr_max": ds,
                           "mean_abserr": np.abs(m.astype(np.float64) - shp["mean"]).tolist(),
                           "std_abserr": np.abs(s.astype(np.float64) - shp["std"]).tolist()}
            print(f"  {name:<20}{dm:>20.3e}{ds:>20.3e}")
        report["keys"][key] = entry

    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
