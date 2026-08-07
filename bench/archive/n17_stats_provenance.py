#!/usr/bin/env python3
"""Which statistics does the N1.7 postprocessor ACTUALLY hold at inference?

Motivated by openpi-base's self-reported blind spot: every leak/floor check so
far was run on a *source asset* file, while inference un-normalizes with
whatever got baked into the checkpoint.  Those can differ, and the failure is
silent -- a wrong table still produces a plausible MAE.

There is a second, sharper worry specific to this leg: `predict_gr00t_n17.py`
constructs a `LeRobotEpisodeLoader` over the **val** dataset (to fetch
observations), and that dataset carries its own `meta/stats.json` +
`meta/relative_stats.json`.  If those ever reached the processor, predictions
would be normalized with val-derived statistics -- a leak.

So: load the processor (no 3B model needed), read the live statistics object,
and identify it against every candidate table on disk.

  candidate A  <ckpt>/statistics.json
  candidate B  <ckpt>/experiment_cfg/dataset_statistics.json
  candidate C  TRAIN meta/stats.json + meta/relative_stats.json  (sliced)
  candidate D  VAL   meta/stats.json + meta/relative_stats.json  (sliced)  <- must NOT match

Reports bit-level equality per (modality, key, stat) cell.  Also re-runs the
leakage falsification on `relative_stats.json`, which the earlier absolute-space
check never covered.
"""

import pathlib
import argparse
import json
import sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "third_party" / "Isaac-GR00T"))


def flatten(stats, prefix=""):
    """Flatten a nested statistics dict to {dotted.path: np.ndarray}.

    Non-numeric leaves (e.g. the `sha256:...` cache-invalidation sidecar that
    gr00t.data.stats writes into relative_stats.json) are dropped, not coerced.
    """
    out = {}
    for k, v in stats.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, p + "."))
        elif isinstance(v, str):
            continue
        else:
            arr = np.asarray(v, dtype=np.float64)
            out[p] = arr
    return out


def unwrap_embodiment(d, tag="new_embodiment"):
    """The live processor table is keyed by embodiment tag (it also carries the 8
    pretrain embodiments).  On-disk candidates may or may not be.  Normalize both
    to the single embodiment we finetune."""
    if isinstance(d, dict) and tag in d and isinstance(d[tag], dict):
        return d[tag]
    return d


def compare(live, cand, name):
    """Bit-compare two flattened tables; return (verdict, detail)."""
    lk, ck = set(live), set(cand)
    only_live, only_cand = sorted(lk - ck), sorted(ck - lk)
    shared = sorted(lk & ck)
    exact, mismatch = [], []
    for k in shared:
        a, b = live[k], cand[k]
        if a.shape != b.shape:
            mismatch.append((k, f"shape {a.shape} vs {b.shape}"))
        elif np.array_equal(a, b):
            exact.append(k)
        else:
            mismatch.append((k, f"max|d|={np.abs(a - b).max():.6g}"))
    verdict = "IDENTICAL" if (not mismatch and not only_live and not only_cand) else (
        "SUBSET-EXACT" if not mismatch and not only_cand else "DIFFERS")
    print(f"\n--- candidate {name} ---")
    print(f"  cells shared {len(shared)} | exact {len(exact)} | mismatch {len(mismatch)}"
          f" | only-in-live {len(only_live)} | only-in-candidate {len(only_cand)}")
    if mismatch:
        for k, d in mismatch[:6]:
            print(f"    MISMATCH {k}: {d}")
        if len(mismatch) > 6:
            print(f"    ... {len(mismatch) - 6} more")
    if only_live:
        print(f"    only-live e.g. {only_live[:4]}")
    if only_cand:
        print(f"    only-candidate e.g. {only_cand[:4]}")
    print(f"  => {verdict}")
    return verdict


def sliced_from_dataset(ds_path):
    """Rebuild the {state,action,relative_action} table the loader would hand up,
    straight from meta/*.json -- no gr00t imports, so this is independent."""
    meta = Path(ds_path) / "meta"
    stats = json.loads((meta / "stats.json").read_text())
    modality = json.loads((meta / "modality.json").read_text())
    mapping = {"state": "observation.state", "action": "action"}
    out = {}
    for mod, col in mapping.items():
        out[mod] = {}
        for key, spec in modality[mod].items():
            src = spec.get("original_key") or col
            s, e = spec["start"], spec["end"]
            out[mod][key] = {st: v[s:e] for st, v in stats[src].items()}
    rel = meta / "relative_stats.json"
    if rel.exists():
        r = json.loads(rel.read_text())
        r.pop("_cache_key", None)
        r.pop("cache_key", None)
        out["relative_action"] = r
    return out


def relative_leak_check(train_ds, val_ds):
    """Falsification on the RELATIVE table (never covered by the absolute check).

    If train relative_stats were computed over the 111 train eps only, then val's
    own relative extremes must exceed the train table somewhere -- val data the
    train stats never saw.  Any single joint/step exceeding is proof of exclusion.
    """
    print("\n=== relative_stats leakage falsification ===")
    tr = json.loads((Path(train_ds) / "meta/relative_stats.json").read_text())
    va = json.loads((Path(val_ds) / "meta/relative_stats.json").read_text())
    for k in ("_cache_key", "cache_key"):
        tr.pop(k, None)
        va.pop(k, None)
    ft, fv = flatten(tr), flatten(va)
    common = sorted(set(ft) & set(fv))
    if not common:
        print("  no comparable keys -- CANNOT CONCLUDE")
        return
    n_exceed_max = n_exceed_min = 0
    worst = None
    for k in common:
        if k.endswith(".max"):
            d = (fv[k] - ft[k]).max()
            if d > 0:
                n_exceed_max += 1
                if worst is None or d > worst[1]:
                    worst = (k, d)
        elif k.endswith(".min"):
            d = (ft[k] - fv[k]).max()
            if d > 0:
                n_exceed_min += 1
    print(f"  comparable cells: {len(common)}")
    print(f"  val max EXCEEDS train max in {n_exceed_max} cells")
    print(f"  val min BELOW   train min in {n_exceed_min} cells")
    if worst:
        print(f"  worst exceedance: {worst[0]} by {worst[1]:.4f}")
    if n_exceed_max or n_exceed_min:
        print("  => CLEAN: train relative_stats provably did NOT see the val episodes")
    else:
        print("  => INCONCLUSIVE (val strictly inside train range -- possible but "
              "no positive proof of exclusion; NOT evidence of leakage either)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--train-ds", default=str(BENCH / "data/b2_n17_train"))
    ap.add_argument("--val-ds", default=str(BENCH / "data/b2_n17_val"))
    args = ap.parse_args()

    import gr00t.model  # noqa: F401  (registers the N1.7 processor with AutoProcessor)
    from transformers import AutoProcessor  # noqa: E402

    model_dir = Path(args.model_path)
    proc_dir = model_dir / "processor" if (model_dir / "processor").is_dir() else model_dir
    print(f"loading processor from {proc_dir}")
    processor = AutoProcessor.from_pretrained(str(proc_dir))
    sap = processor.state_action_processor
    print(f"LIVE statistics embodiments = {sorted(sap.statistics.keys())}")
    print(f"  use_relative_action={getattr(sap, 'use_relative_action', 'n/a')} "
          f"use_percentiles={getattr(sap, 'use_percentiles', 'n/a')}")
    live_emb = unwrap_embodiment(sap.statistics)
    live = flatten(live_emb)
    print(f"LIVE new_embodiment: {len(live)} numeric leaf cells, "
          f"groups = {sorted(live_emb.keys())}")

    cands = {}
    for name, p in (("A ckpt/statistics.json", model_dir / "statistics.json"),
                    ("B ckpt/experiment_cfg/dataset_statistics.json",
                     model_dir / "experiment_cfg/dataset_statistics.json")):
        if p.exists():
            cands[name] = flatten(unwrap_embodiment(json.loads(p.read_text())))
        else:
            print(f"  (absent: {p})")
    cands["C TRAIN meta/ (independent slice)"] = flatten(sliced_from_dataset(args.train_ds))
    cands["D VAL meta/ (MUST NOT MATCH)"] = flatten(sliced_from_dataset(args.val_ds))

    verdicts = {n: compare(live, c, n) for n, c in cands.items()}

    print("\n=== VERDICT ===")
    for n, v in verdicts.items():
        print(f"  {n}: {v}")
    val_v = verdicts.get("D VAL meta/ (MUST NOT MATCH)")
    trn_v = verdicts.get("C TRAIN meta/ (independent slice)")
    if val_v == "DIFFERS" and trn_v in ("IDENTICAL", "SUBSET-EXACT"):
        print("  PASS: live table is the TRAIN one and is NOT the val one")
    else:
        print(f"  ATTENTION: train={trn_v} val={val_v} -- inspect above")

    relative_leak_check(args.train_ds, args.val_ds)


if __name__ == "__main__":
    main()
