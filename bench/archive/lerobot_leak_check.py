#!/usr/bin/env python
"""Falsification-style leak check for the LeRobot leg's normalization stats.

If val episodes had leaked into the stats used by the (un)normalizer, then for EVERY
joint `stat.max >= val.max`. So any joint where `val_max > stat_max` PROVES the stats
were computed without those val frames. No-exceedance is uninformative (val may simply
sit inside train range) -- it is "needs another check", never "leaked".

Two sources of stats, and the CKPT one is the one that matters at predict time:
  --stats-json  meta/stats.json of the working dataset (what training was handed)
  --ckpt        .../pretrained_model, reads the normalizer safetensors that
                `make_pre_post_processors(pretrained_path=...)` actually loads back,
                i.e. exactly the numbers that unnormalize the predictions.

Blind spot, stated up front: this covers ONLY the norm-stats channel. It says nothing
about whether val frames entered the training sampler (gradients) -- that needs the
episode-id list / frame-count arithmetic from the train log, a separate argument.
"""

import os
import argparse
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

BENCH = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_ROOT = pathlib.Path(os.environ.get("NERO_DATA_V30", os.path.expanduser("~/.cache/huggingface/lerobot/local/pick_pink_sponge_b2_bench")))
KEYS = ("action", "observation.state")


def _tie_counts(a, ep, mval, j, value):
    """How many frames/episodes sit EXACTLY on `value`, counted per split.

    Both sides matter. A tie value held by 1 train frame but 50 val frames is a shared
    attractor, not a fingerprint -- counting only the train side calls that a unique
    peak and false-alarms (gr00t-n15 hit exactly this on state j5.min).
    """
    hit = a[:, j] == value
    out = {}
    for lab, m in (("train", hit & ~mval), ("val", hit & mval)):
        out[lab] = (int(m.sum()), len(set(ep[m].tolist())))
    return out


def _lattice(a, cols, step):
    """Fraction of values sitting on a `step` lattice, judged at float32 resolution.

    Tolerance must scale with magnitude: these are float32, so a fixed 1e-6 absolute
    tolerance is *below* the representation error at ~100 deg and makes a perfectly
    quantized column look 73% quantized.
    """
    x = a[:, cols]
    dev = np.abs(x / step - np.round(x / step)) * step
    tol = 2 * np.maximum(np.abs(x) * np.finfo(np.float32).eps, np.finfo(np.float32).tiny)
    return (dev <= tol).mean()


def val_extrema(root: pathlib.Path, val_eps: list[int]):
    """Per-joint min/max over ALL frames of the val episodes (not just anchors).

    Returns the raw arrays too, so ties can be counted per split and the column's
    quantization lattice measured (both are needed to read a tie correctly).
    """
    cols = {k: [] for k in KEYS}
    eps_all = []
    for f in sorted(root.glob("data/**/*.parquet")):
        t = pq.read_table(f, columns=["episode_index", *KEYS])
        eps_all.append(t.column("episode_index").to_numpy())
        for k in KEYS:
            cols[k].append(np.stack(t.column(k).to_pylist()))
    ep = np.concatenate(eps_all)
    mval = np.isin(ep, val_eps)
    out = {}
    for k in KEYS:
        a = np.concatenate(cols[k]).astype(np.float64)
        v = a[mval]
        arm = list(range(min(7, a.shape[1])))
        out[k] = {"min": v.min(0), "max": v.max(0), "n": len(v), "all": a, "ep": ep, "mval": mval,
                  "support": [len(np.unique(a[:, j])) for j in range(a.shape[1])],
                  "lattice": {"1 mrad": _lattice(a, arm, 1e-3 * 180.0 / np.pi),
                              "0.001 deg": _lattice(a, arm, 1e-3)}}
    return out


def stats_from_json(p: pathlib.Path) -> dict:
    s = json.loads(p.read_text())
    return {k: {kk: np.asarray(vv, dtype=np.float64).reshape(-1) for kk, vv in s[k].items()}
            for k in KEYS if k in s}


def stats_from_ckpt(ckpt: pathlib.Path) -> dict:
    """Read the normalizer state the postprocessor actually loads back."""
    from safetensors.numpy import load_file
    files = sorted(ckpt.glob("*normalizer*.safetensors")) or sorted(ckpt.glob("*.safetensors"))
    files = [f for f in files if f.name != "model.safetensors"]
    if not files:
        raise SystemExit(f"no processor safetensors under {ckpt}")
    merged: dict[str, dict[str, np.ndarray]] = {}
    for f in files:
        for flat, arr in load_file(f).items():
            key, stat = flat.rsplit(".", 1)
            if key in KEYS:
                merged.setdefault(key, {})[stat] = np.asarray(arr, dtype=np.float64).reshape(-1)
    print(f"[ckpt] read {len(files)} processor file(s): {[f.name for f in files]}")
    return merged


def report(tag: str, stats: dict, vx: dict) -> None:
    print(f"\n===== {tag} =====")
    for k in KEYS:
        if k not in stats:
            print(f"  [{k}] absent from these stats"); continue
        st, v = stats[k], vx[k]
        lat = max(v["lattice"].items(), key=lambda kv: kv[1])
        sup = v["support"]
        role = "PRIMARY evidence" if k == "action" else "SECONDARY (coarser lattice -> ties collide by construction)"
        print(f"  [{k}] stat keys: {sorted(st)}  (val frames n={v['n']})  -- {role}")
        print(f"      arm quantized {lat[1] * 100:.1f}% onto a {lat[0]} lattice; "
              f"distinct values/joint {min(sup[:7])}-{max(sup[:7])}")
        if not ("max" in st and "min" in st):
            print("    no min/max baked -> this test cannot run on these stats")
            continue
        hi = v["max"] - st["max"]
        lo = st["min"] - v["min"]
        sides_clean = sides_total = 0
        for j in range(len(hi)):
            marks = []
            # min_max-style stats give a two-sided test: 2 independent checks per joint.
            for lab, gap, sv, tv in (("max", hi[j], v["max"][j], st["max"][j]),
                                     ("min", lo[j], v["min"][j], st["min"][j])):
                sides_total += 1
                if gap > 0:
                    sides_clean += 1
                    marks.append(f"val_{lab} {sv:.3f} beyond stat_{lab} {tv:.3f} (+{gap:.3f}) PROVES-CLEAN")
                elif abs(gap) < 1e-12:
                    # Equality is a NECESSARY consequence of leak, never a sufficient one.
                    # Count BOTH splits: either side being a plateau makes it a shared
                    # attractor (hardware stop / saturation), which any clean split hits.
                    c = _tie_counts(v["all"], v["ep"], v["mval"], j, sv)
                    plateau = max(c["train"][0], c["val"][0]) > 10 or max(c["train"][1], c["val"][1]) > 2
                    marks.append(f"{lab} TIE at {sv:.6f} [train {c['train'][0]}f/{c['train'][1]}ep, "
                                 f"val {c['val'][0]}f/{c['val'][1]}ep] "
                                 + ("= shared attractor, uninformative" if plateau
                                    else "= near-unique BOTH sides, WORTH INVESTIGATING"))
            print(f"    j{j} " + (" | ".join(marks) if marks
                                  else f"silent  val[{v['min'][j]:.3f},{v['max'][j]:.3f}] "
                                       f"within stat[{st['min'][j]:.3f},{st['max'][j]:.3f}]"))
        print(f"    => {sides_clean}/{sides_total} one-sided checks prove val was NOT in these stats")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--stats-json", default=None, help="default: <root>/meta/stats.json")
    ap.add_argument("--ckpt", default=None, help=".../pretrained_model (the stats that matter)")
    ap.add_argument("--skip-json", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    val_eps = json.loads((BENCH / "data" / "split.json").read_text())["val_episodes"]
    print(f"[split] {len(val_eps)} val episodes: {val_eps}")
    vx = val_extrema(root, val_eps)

    if not args.skip_json:
        sj = pathlib.Path(args.stats_json) if args.stats_json else root / "meta" / "stats.json"
        report(f"dataset stats.json  {sj}", stats_from_json(sj), vx)
    if args.ckpt:
        report(f"CKPT baked stats  {args.ckpt}", stats_from_ckpt(pathlib.Path(args.ckpt)), vx)
    print("\n[blind spot] norm-stats channel only; says nothing about the training sampler.")


if __name__ == "__main__":
    main()
