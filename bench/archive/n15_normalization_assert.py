#!/usr/bin/env python3
"""Assert the N1.5 leg's action space and normalization are what the board believes.

Board pin (team-lead 2026-08-05): a leg's parameterization (absolute vs delta) and its
normalization mode decide how its row may be compared, and OFT measured that a q99-as-bounds
choice crosses the clip boundary on only 0.63% of values -- small enough that nothing
downstream looks wrong while the row is silently on a different footing. A `flag` does not
catch it, because the flag is a restatement; the checkpoint and the shipped data config are
the two places the truth actually lives.

This leg is FINISHED, so the value here is retrospective: it pins what the delivered
checkpoint-10000 row was produced under, and gates any rerun. Two independent sources must
agree, and neither is a restatement of the other:

  1. the CHECKPOINT's experiment_cfg/metadata.json -- `modalities[grp][key]["absolute"]`
  2. the SHIPPED data config that predict_gr00t.py imports -- `normalization_modes`

Note why the checkpoint alone cannot answer the normalization question: its statistics block
saves min/max/mean/std/q01/q99 for EVERY key regardless of mode, so what was saved tells you
nothing about what is read. The mode lives only in the transform.

Power control (required, per the board META-rule that a gate must be shown to reject): run
with --control to flip the expectation and confirm the assertions actually fire. A gate only
ever exercised in the passing direction is indistinguishable from one that cannot fail.
"""

import os
import argparse
import ast
import hashlib
import json
import pathlib
import sys

BENCH = pathlib.Path(__file__).resolve().parents[1]
CKPT = pathlib.Path(os.environ.get("NERO_CKPT", str(pathlib.Path(__file__).resolve().parents[2] / "checkpoints"))) / "gr00t_n15_nero_b2" / "checkpoint-10000"
DATA_CONFIG = BENCH / "configs" / "nero_data_config.py"

EXPECT_ABSOLUTE = True
EXPECT_NORM = "min_max"


def md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()


def modes_from_source(path):
    """Read every `normalization_modes={...}` literal out of the shipped config by AST.

    Deliberately does NOT import the module: importing runs it, and a config that computes its
    mode at runtime would then report whatever the import environment produced rather than what
    the file says. Every value in this file is a dict-comprehension over a key list with a
    constant mode, so the constant is what we extract -- and if that ever stops being true the
    extraction returns nothing and the caller fails loudly rather than guessing.
    """
    tree = ast.parse(pathlib.Path(path).read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "normalization_modes":
            v = node.value
            if isinstance(v, ast.DictComp) and isinstance(v.value, ast.Constant):
                found.append(v.value.value)
            elif isinstance(v, ast.Dict):
                found += [e.value for e in v.values if isinstance(e, ast.Constant)]
            else:
                found.append(f"<unparsed:{type(v).__name__}>")
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--control", action="store_true",
                    help="flip the expectations; both assertions MUST fire (power control)")
    ap.add_argument("--out", default=str(BENCH / "logs" / "n15_normalization_assert.json"))
    args = ap.parse_args()

    exp_abs = (not EXPECT_ABSOLUTE) if args.control else EXPECT_ABSOLUTE
    exp_norm = "q99" if args.control else EXPECT_NORM

    meta = json.loads((CKPT / "experiment_cfg" / "metadata.json").read_text())
    emb = sorted(meta.keys())
    assert len(emb) == 1, f"expected exactly one embodiment, got {emb}"
    mods = meta[emb[0]]["modalities"]

    absolute = {f"{g}.{k}": v.get("absolute")
                for g in ("state", "action") for k, v in mods.get(g, {}).items()}
    modes = modes_from_source(DATA_CONFIG)

    stats_saved = sorted({s for g in ("state", "action")
                          for v in meta[emb[0]]["statistics"].get(g, {}).values() for s in v})

    out = {
        "leg": "gr00t_n15",
        "checkpoint": str(CKPT),
        "data_config": str(DATA_CONFIG),
        "data_config_md5": md5(DATA_CONFIG),
        "metadata_md5": md5(CKPT / "experiment_cfg" / "metadata.json"),
        "expected_absolute": exp_abs,
        "expected_normalization": exp_norm,
        "measured_absolute_by_key": absolute,
        "measured_normalization_modes": modes,
        "stats_saved_in_checkpoint": stats_saved,
        "why_stats_do_not_answer_it": (
            "the checkpoint saves min/max/mean/std/q01/q99 for every key regardless of mode, "
            "so what is saved says nothing about what the transform reads"
        ),
        "control_mode": args.control,
    }

    fail = []
    if not absolute:
        fail.append("no modality keys found in checkpoint metadata")
    for k, v in absolute.items():
        if v is not exp_abs:
            fail.append(f"{k}: absolute={v}, expected {exp_abs}")
    if not modes:
        fail.append(f"no normalization_modes literal found in {DATA_CONFIG}")
    for m in modes:
        if m != exp_norm:
            fail.append(f"normalization mode {m!r}, expected {exp_norm!r}")

    out["failures"] = fail
    out["verdict"] = "FAIL" if fail else "PASS"
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))

    print(f"absolute by key : {absolute}")
    print(f"normalization   : {modes}  (from the shipped config, AST-read, not imported)")
    print(f"stats saved     : {stats_saved}  <- present for every key in BOTH modes")
    print(f"verdict         : {out['verdict']}"
          + (f"  ({len(fail)} failure(s))" if fail else ""))
    for f in fail:
        print(f"   - {f}")

    if args.control:
        # under --control the expectations are wrong on purpose; a silent PASS means the
        # comparison is not actually reached and the gate is decorative.
        if not fail:
            sys.exit("POWER CONTROL FAILED: flipped expectations still PASS")
        print("power control OK: flipped expectations are rejected")
        return
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
