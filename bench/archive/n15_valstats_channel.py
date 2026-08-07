"""Corruption-counterfactual for the THIRD leakage channel on the N1.5 leg.

Upgrades "predict_gr00t.py:151 passes transforms=None, so the val dataset's own
stats can't reach the observations" from a CODE-LEVEL ARGUMENT to a MEASUREMENT.

Method (template from gr00t-n17, archive/n17_valstats_channel.py):
  NaN-ify every numeric leaf of the val loader's own statistics, re-fetch the same
  anchors through the exact call predict uses (dataset.get_step_data), and require
  the observation bytes to be BIT-IDENTICAL to the clean pass.

Two controls, and the measurement is worthless without BOTH:

  POWER   -- the same corruption, pushed through an entry point that DOES read the
             statistics, must visibly break.  Here that entry point is the real
             leak pathway itself: the same dataset built WITH data_config.transform()
             attached, i.e. exactly what predict would look like if it had passed
             transforms instead of None.  Without this control, "bytes unchanged"
             and "my corruption never took effect" are indistinguishable -- a probe
             whose output is identical under both hypotheses has zero power.
  DECODER -- two CLEAN passes must already be bit-identical before any corruption.
             Otherwise a diff in the corrupt pass cannot be attributed: stats leak
             vs. nondeterministic video decode.

Prints PASS/FAIL lines that post_train can grep.  CPU only, no GPU, no 3B model.
"""
import argparse, json, pathlib, sys
import numpy as np

BENCH = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH / "configs"))
from gr00t.data.dataset import LeRobotSingleDataset  # noqa: E402


def load_data_config(spec):
    mod, _, cls = spec.partition(":")
    import importlib
    return getattr(importlib.import_module(mod), cls)()


def nan_leaves(obj, path="", hits=None):
    """Recursively NaN-ify every numeric leaf, returning the leaf paths touched."""
    hits = [] if hits is None else hits
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fiu":
            obj[...] = np.nan if obj.dtype.kind == "f" else np.iinfo(obj.dtype).min
            hits.extend(f"{path}[{i}]" for i in range(obj.size))
        return hits
    if isinstance(obj, dict):
        for k, v in obj.items():
            nan_leaves(v, f"{path}.{k}", hits)
        return hits
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            nan_leaves(v, f"{path}[{i}]", hits)
        return hits
    # pydantic model / plain object with __dict__
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for k, v in d.items():
            if k.startswith("__"):
                continue
            nan_leaves(v, f"{path}.{k}", hits)
    return hits


def fingerprint(step, keys):
    """Byte-level fingerprint of the observation dict predict actually consumes."""
    out = {}
    for k in keys:
        v = step[k]
        a = np.asarray(v)
        out[k] = a.tobytes() if a.dtype != object else repr(v).encode()
    return out


def diff(fa, fb):
    return sorted(k for k in fa if fa[k] != fb[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", default=str(BENCH / "data/b2_gr00t_val"))
    ap.add_argument("--data-config", default="nero_data_config:NeroDualCamDataConfig")
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--n-anchors", type=int, default=3)
    ap.add_argument("--json-out", default=str(BENCH / "logs/n15_valstats_channel.json"))
    args = ap.parse_args()

    dc = load_data_config(args.data_config)
    obs_keys = dc.video_keys + dc.state_keys + dc.language_keys
    A = np.load(BENCH / "data" / "val_anchors.npz")
    eps, frs = A["episodes"][: args.n_anchors], A["frames"][: args.n_anchors]

    def build(transforms):
        return LeRobotSingleDataset(
            dataset_path=args.dataset_path,
            modality_configs=dc.modality_config(),
            video_backend="decord",
            video_backend_kwargs=None,
            transforms=transforms,
            embodiment_tag=args.embodiment_tag,
        )

    print("=== the channel under test: predict's loader (transforms=None) ===")
    ds = build(None)
    grab = lambda: [fingerprint(ds.get_step_data(int(e), int(f)), obs_keys)
                    for e, f in zip(eps, frs)]

    clean_a = grab()
    clean_b = grab()
    dec_bad = [i for i, (a, b) in enumerate(zip(clean_a, clean_b)) if diff(a, b)]
    dec_ok = not dec_bad
    print(f"[DECODER control] two clean passes bit-identical: {dec_ok}"
          + ("" if dec_ok else f"  <-- anchors differing: {dec_bad}"))

    leaves = nan_leaves(ds.metadata.statistics, "statistics")
    print(f"[corruption] NaN-ified {len(leaves)} numeric leaves of the loader's own statistics")

    corrupt = grab()
    changed = {i: diff(a, c) for i, (a, c) in enumerate(zip(clean_a, corrupt)) if diff(a, c)}
    unreachable = not changed
    print(f"[MEASUREMENT] observations bit-identical after corruption: {unreachable}"
          + ("" if unreachable else f"  <-- {changed}"))

    print("\n=== POWER control: same corruption via the pathway that DOES read stats ===")
    ds2 = build(dc.transform())
    def n_nan(d):
        return sum(int(np.isnan(np.asarray(v, dtype=np.float64)).sum())
                   for k, v in d.items()
                   if k.startswith(("state", "action"))
                   and np.asarray(v).dtype.kind == "f")
    before = n_nan(ds2.transforms(ds2.get_step_data(int(eps[0]), int(frs[0]))))
    leaves2 = nan_leaves(ds2.metadata.statistics, "statistics")
    ds2.set_transforms_metadata(ds2.metadata)          # propagate into the transforms
    after = n_nan(ds2.transforms(ds2.get_step_data(int(eps[0]), int(frs[0]))))
    power_ok = before == 0 and after > 0
    print(f"[POWER control] NaNs in transformed state/action: {before} -> {after}"
          f"   corruption is detectable: {power_ok}")

    ok = dec_ok and unreachable and power_ok
    print("\n" + ("PASS - val-dataset statistics are UNREACHABLE from predict's observations"
                  if ok else "!!! FAIL - see the failing control above"))
    print("  Blind spot: proves the val dir's stats cannot reach the OBSERVATIONS predict "
          "feeds the policy.\n  It says nothing about the ckpt-baked stats (channel 1) or "
          "the training sampler (channel 2).")

    res = dict(dataset=args.dataset_path, n_anchors=int(args.n_anchors),
               decoder_control_ok=bool(dec_ok), leaves_corrupted=len(leaves),
               observations_bit_identical=bool(unreachable),
               power_nan_before=int(before), power_nan_after=int(after),
               power_control_ok=bool(power_ok), verdict="PASS" if ok else "FAIL")
    pathlib.Path(args.json_out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.json_out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
