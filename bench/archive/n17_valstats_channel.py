#!/usr/bin/env python3
"""Leak channel 3, MEASURED: can the VAL directory's own normalization metadata
reach a prediction?

predict_gr00t_n17.py deliberately opens a LeRobotEpisodeLoader over
data/b2_n17_val to rebuild the observations, and that directory ships its own
meta/stats.json + meta/relative_stats.json computed over the HELD-OUT episodes.
If any of those numbers reached the model, predictions would be normalized with
held-out statistics -- no exception, just a quietly flattered MAE.

Reading the source says they cannot:
  * LeRobotEpisodeLoader.__init__ has no `transforms` parameter at all
    (dataset_path / modality_configs / decoder_kwargs);
  * `self.stats` is referenced only where it is loaded and inside
    get_dataset_statistics(), which predict never calls;
  * extract_step_data() takes a DataFrame and never sees the loader object.
That is an argument, not a measurement. This script measures it:

  corrupt EVERY numeric leaf of loader.stats to NaN, redo the whole
  loader[pos] -> extract_step_data() -> obs-dict path, and require the bytes to
  be identical.

Two controls, because a probe whose output is identical under both hypotheses is
worse than blind (it looks like evidence):
  POWER    the same corrupted loader is asked for get_dataset_statistics(); that
           MUST come back NaN. It proves the corruption is real and would have
           been visible had anything on the predict path read it.
  DECODER  the clean pass is run TWICE before corrupting. If two clean passes do
           not agree byte-for-byte the video decode is nondeterministic and the
           whole comparison is inconclusive -- say so instead of reporting PASS.

Part 2 answers openpi-base's warning that `episodes=` filters frames but NOT
`meta.stats` (so a merged-dataset loader would expose all-131 statistics): our
val directory is a separate physical tree, so this recomputes its stats.json
from the source parquet over VAL-20, TRAIN-111 and ALL-131 and reports which one
it actually is. Scope is moot if part 1 shows the numbers are unreachable, but
"unreachable AND val-only" is cheap to state and leaves nothing to argue about.

CPU only, no GPU, no 3B model: the processor alone supplies the modality config.
"""
import argparse
import json
import hashlib
import os
import pathlib
from copy import deepcopy
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parents[1]
VAL_DIR = BENCH / "data/b2_n17_val"


def nanify(o):
    """Replace every numeric leaf with NaN, preserving structure and keys."""
    if isinstance(o, dict):
        return {k: nanify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [nanify(v) for v in o]
    if isinstance(o, bool):
        return o
    if isinstance(o, (int, float)):
        return float("nan")
    return o


def count_leaves(o):
    if isinstance(o, dict):
        return sum(count_leaves(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(count_leaves(v) for v in o)
    return 1 if isinstance(o, (int, float)) and not isinstance(o, bool) else 0


def nan_fraction(o):
    """(n_nan, n_numeric) over a nested structure."""
    if isinstance(o, dict):
        parts = [nan_fraction(v) for v in o.values()]
    elif isinstance(o, (list, tuple)):
        parts = [nan_fraction(v) for v in o]
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        return (1 if np.isnan(float(o)) else 0, 1)
    else:
        return (0, 0)
    return (sum(p[0] for p in parts), sum(p[1] for p in parts))


def digest_step(step):
    """Deterministic hash of everything predict feeds into the observation dict.

    predict builds obs from exactly step.states / step.images / step.text, so
    hashing those three covers the whole loader -> model surface.
    """
    h = hashlib.sha256()
    for k in sorted(step.states):
        arr = np.ascontiguousarray(step.states[k])
        h.update(f"state.{k}|{arr.shape}|{arr.dtype}|".encode())
        h.update(arr.tobytes())
    for k in sorted(step.images):
        arr = np.ascontiguousarray(np.asarray(step.images[k]))
        h.update(f"video.{k}|{arr.shape}|{arr.dtype}|".encode())
        h.update(arr.tobytes())
    h.update(f"text|{step.text}".encode())
    return h.hexdigest()


def pass_over(loader, pos_of_ep, by_ep, frames, obs_modality, tag, extract_step_data):
    out = {}
    for ep in sorted(by_ep):
        traj = loader[pos_of_ep[ep]]          # decodes this episode's video
        for i in by_ep[ep]:
            step = extract_step_data(traj, int(frames[i]), obs_modality, tag)
            out[i] = digest_step(step)
        del traj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(pathlib.Path(os.environ.get("NERO_CKPT", str(pathlib.Path(__file__).resolve().parents[2] / "checkpoints"))) / "n17_smoke" / "smoke" / "checkpoint-10"),
                    help="only used for its modality config; no weights are loaded")
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--anchors", type=int, default=6,
                    help="how many val anchors to route through the full path")
    ap.add_argument("--json-out", default=str(BENCH / "logs/n17_valstats_channel.json"))
    ap.add_argument("--skip-scope", action="store_true")
    args = ap.parse_args()

    import gr00t.model  # noqa: F401  registers the processor with AutoProcessor
    from transformers import AutoProcessor
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag

    results = {"val_dir": str(VAL_DIR), "ckpt": args.ckpt}

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    ck = Path(args.ckpt)
    proc_dir = ck / "processor" if (ck / "processor").is_dir() and not (
        ck / "processor_config.json").exists() else ck
    processor = AutoProcessor.from_pretrained(str(proc_dir))
    modality = {k: v for k, v in processor.get_modality_configs()[tag.value].items()
                if k != "rl_info"}
    obs_modality = deepcopy(modality)
    obs_modality.pop("action", None)

    # ---- part 1: reachability ------------------------------------------------
    print("=== part 1: can loader.stats reach the observation? ===")
    loader = LeRobotEpisodeLoader(dataset_path=str(VAL_DIR), modality_configs=modality)
    n_leaves = count_leaves(loader.stats)
    print(f"loader.stats loaded: {list(loader.stats)} -> {n_leaves} numeric leaves")

    A = np.load(BENCH / "data" / "val_anchors.npz")
    episodes, frames = A["episodes"], A["frames"]
    pos_of_ep = {m["episode_index"]: i for i, m in enumerate(loader.episodes_metadata)}
    # spread the probe anchors across episodes rather than taking a contiguous head,
    # so more than one video decode is covered
    sel = np.linspace(0, len(episodes) - 1, args.anchors).astype(int)
    by_ep = {}
    for i in sel:
        by_ep.setdefault(int(episodes[i]), []).append(int(i))
    print(f"probing {len(sel)} anchors over {len(by_ep)} episodes: "
          + ", ".join(f"ep{e}(x{len(v)})" for e, v in sorted(by_ep.items())))

    clean1 = pass_over(loader, pos_of_ep, by_ep, frames, obs_modality, tag, extract_step_data)
    clean2 = pass_over(loader, pos_of_ep, by_ep, frames, obs_modality, tag, extract_step_data)
    decoder_ok = clean1 == clean2
    print(f"[control DECODER] two clean passes identical: {decoder_ok}")

    stats_before = deepcopy(loader.get_dataset_statistics())
    loader.stats = nanify(loader.stats)
    corrupted = pass_over(loader, pos_of_ep, by_ep, frames, obs_modality, tag, extract_step_data)
    stats_after = loader.get_dataset_statistics()

    n_nan, n_num = nan_fraction(stats_after)
    b_nan, b_num = nan_fraction(stats_before)
    power_ok = n_num > 0 and n_nan == n_num and b_nan == 0
    print(f"[control POWER  ] get_dataset_statistics(): before {b_nan}/{b_num} NaN, "
          f"after {n_nan}/{n_num} NaN -> corruption is visible: {power_ok}")

    same = corrupted == clean1
    diff = [i for i in clean1 if clean1[i] != corrupted.get(i)]
    print(f"[result         ] observation bytes unchanged under NaN stats: {same}"
          + (f" (differing anchors {diff})" if diff else ""))

    reach_verdict = "UNREACHABLE" if (same and decoder_ok and power_ok) else (
        "INCONCLUSIVE" if not (decoder_ok and power_ok) else "REACHABLE")
    results["part1"] = {
        "anchors": [int(i) for i in sel], "episodes": sorted(by_ep),
        "stat_leaves": n_leaves, "decoder_deterministic": decoder_ok,
        "corruption_visible_in_get_dataset_statistics": power_ok,
        "obs_identical_under_corruption": same, "verdict": reach_verdict,
        "digests_clean": {str(k): v for k, v in sorted(clean1.items())},
    }
    if reach_verdict == "UNREACHABLE":
        print("PASS: val-directory normalization metadata cannot reach the observation")
    else:
        print(f"!!! {reach_verdict}: do not quote the n17 row until this is understood")

    # ---- part 2: what scope is the val dir's stats.json anyway? -------------
    if not args.skip_scope:
        print("\n=== part 2: scope of the val directory's own stats.json ===")
        import sys
        sys.path[:0] = [str(BENCH / "scripts" / "predict"), str(BENCH / "scripts" / "score"),
                        str(BENCH / "archive")]   # was one flat scripts/ dir
        from n17_leak_direct_recompute import calculate_dataset_statistics, ep_parquet

        split = json.loads((BENCH / "data" / "split.json").read_text())
        train_eps, val_eps = split["train_episodes"], split["val_episodes"]
        all_eps = sorted(train_eps + val_eps)
        on_disk = json.loads((VAL_DIR / "meta/stats.json").read_text())

        def maxdev(rec):
            worst = 0.0
            for le in ("observation.state", "action"):
                for st in ("mean", "std", "min", "max", "q01", "q99"):
                    a = np.asarray(rec[le][st], dtype=np.float64)
                    b = np.asarray(on_disk[le][st], dtype=np.float64)
                    worst = max(worst, float(np.max(np.abs(a - b))))
            return worst

        scope = {}
        for name, eps in (("VAL-20", val_eps), ("TRAIN-111", train_eps), ("ALL-131", all_eps)):
            rec, nrows = calculate_dataset_statistics([ep_parquet(e) for e in eps])
            d = maxdev(rec)
            scope[name] = {"rows": int(nrows), "max_abs_dev": d, "match": d == 0.0}
            print(f"  {name:9s} {nrows:6d} rows  max|dev| = {d:.6e}"
                  + ("   <== MATCHES the on-disk file" if d == 0.0 else ""))
        results["part2"] = scope
        matches = [k for k, v in scope.items() if v["match"]]
        print(f"verdict: val dir stats.json == {matches or 'NONE of the three'}")

    Path(args.json_out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
