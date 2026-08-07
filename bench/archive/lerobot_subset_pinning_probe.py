#!/usr/bin/env python3
"""Does a SUBSET run reproduce the corresponding rows of a FULL run, bit-for-bit?

Why before SmolVLA exists: team-lead made per-draw provenance a board-wide footnote
requirement -- the variance npz must bit-match the delivered row at the same global
anchor indices, proving footnote and row came from the same checkpoint. For a
DETERMINISTIC backbone (ACT) that is free. For a flow-matching one it is not, and
gr00t-n15 named the exact failure: if the prior is drawn off the GLOBAL RNG, each
row's noise is tied to its position INSIDE ITS BATCH/RUN, so a 200-anchor run and a
1397-anchor run disagree on every row -- same checkpoint, same seed, same code.
That failure is indistinguishable from a wrong-checkpoint footnote and is not one.

`lerobot_predict.py` reseeds per anchor with `seed * 1_000_003 + glob_idx[n]`, where
glob_idx is the position in the FULL 1397 array, and runs one anchor per forward. So
the claim is that pinning is already correct. This probe MEASURES that claim instead
of reading the source, and it does so now -- once SmolVLA has numbers, a failure here
would arrive tangled up with "is the checkpoint right", which is the expensive
question. A randomly-initialised SmolVLA exercises the identical sampling path.

The probe is only worth running because of the CONTROL. A run of the production code
alone passes trivially if the head ignores its injected noise (then everything is
bit-exact for the wrong reason). So the same test runs against a copy of the script
with ONE line changed -- seeding by LOOP POSITION instead of global index, i.e. the
bug n15 described -- which must FAIL. Prod passes + control fails is evidence; prod
passing alone is not.

Third control: a WEIGHT FINGERPRINT printed by both copies. With --random the weights
come from `torch.manual_seed(args.seed)` before make_policy, so if anything RNG-ish
happened between the runs the weights would differ and every row would differ -- which
looks exactly like a seeding failure. All runs share one val-episode set precisely so
that this cannot happen, and the fingerprint proves it did not.

CPU-only, no checkpoint, no GPU. ~30 SmolVLA forwards.
"""

import argparse
import atexit
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
VENV_PY = str(pathlib.Path(__file__).resolve().parents[2] / "third_party" / "lerobot" / ".venv" / "bin" / "python")
SRC = BENCH / "scripts" / "predict" / "lerobot_predict.py"
OUT = BENCH / "logs" / "pinprobe"

PROD_SEED_LINE = "        torch.manual_seed(args.seed * 1_000_003 + int(glob_idx[n]))"
CTRL_SEED_LINE = "        torch.manual_seed(args.seed * 1_000_003 + int(n))  # CONTROL: loop position"
WFP_ANCHOR = "    policy.eval()"
WFP_LINE = ('    policy.eval()\n'
            '    print("[wfp] %.12e" % float(sum(p.double().abs().sum() for p in policy.parameters())), flush=True)')


def make_variant(ctrl: bool, tmpdir: pathlib.Path) -> pathlib.Path:
    src = SRC.read_text()
    if src.count(PROD_SEED_LINE) != 1:
        sys.exit("lerobot_predict.py's per-anchor seed line moved -- probe would test nothing")
    if src.count(WFP_ANCHOR) != 1:
        sys.exit("policy.eval() anchor not unique -- refusing to patch blind")
    out = src.replace(WFP_ANCHOR, WFP_LINE)
    if ctrl:
        out = out.replace(PROD_SEED_LINE, CTRL_SEED_LINE)
    p = tmpdir / ("predict_ctrl.py" if ctrl else "predict_prod.py")
    p.write_text(out)
    return p


def run(script: pathlib.Path, sel_file: pathlib.Path, out_npz: pathlib.Path) -> tuple[np.ndarray, str, int]:
    cmd = [VENV_PY, str(script), "--backbone", "smolvla", "--random", "--device", "cpu",
           "--seed", "0", "--subset", str(sel_file), "--out", str(out_npz)]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:]); print(r.stderr[-4000:])
        sys.exit(f"run failed rc={r.returncode}")
    wfp = re.search(r"\[wfp\] (\S+)", r.stdout)
    neps = re.search(r"over (\d+) val episodes", r.stdout)
    print("  " + "\n  ".join(l for l in r.stdout.splitlines() if l.startswith(("[anchors]", "[wfp]", "[done]"))))
    return np.load(out_npz)["pred"], wfp.group(1), int(neps.group(1))


def main() -> None:
    # A script with no argparse EXECUTES on `--help`. Mine did, during a sweep whose whole
    # point was to check that every tool's flags still parsed -- so the verification step
    # silently launched a second 50-core copy of this probe writing to the same fixed
    # output paths as the live one. Both hazards are real and separate: `--help` must be
    # inert, and a probe with fixed output paths must refuse to run twice. Fix both.
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=None,
                    help="verdict json (default logs/subset_pinning_probe.json); "
                         "point at /tmp for a pre-flight (openpi's board rule)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "RUNNING.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit(f"another instance is running (or died leaving {lock}); its outputs are the "
                 f"same fixed paths as mine -- refusing to race it. Remove the lock if stale.")
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    atexit.register(lambda: lock.unlink(missing_ok=True))
    anchors = np.load(BENCH / "data" / "val_anchors.npz")
    ep = anchors["episodes"]
    # ALL anchors from a single val episode: the two runs then open the identical
    # LeRobotDataset view, so weight init and video decode cannot differ between them
    # and the only thing left varying is anchor position. (In production both the
    # 1397-row run and the 200-row subset already span all 20 val episodes, so this
    # matches the real case rather than dodging it.)
    first_ep = int(ep[0])
    pool = np.flatnonzero(ep == first_ep)
    selA = pool[:12].astype(int)
    selB = selA[[1, 5, 9]]                     # positions 1,5,9 in A -> 0,1,2 in B
    assert set(selB) <= set(selA) and not np.array_equal(
        np.flatnonzero(np.isin(selA, selB)), np.arange(len(selB))), \
        "subset must sit at DIFFERENT positions in A than in B or the control has no power"
    fA, fB = OUT / "selA.npy", OUT / "selB.npy"
    np.save(fA, selA); np.save(fB, selB)
    print(f"[probe] episode {first_ep}: A={selA.tolist()}  B={selB.tolist()} "
          f"(positions in A {np.flatnonzero(np.isin(selA, selB)).tolist()} vs in B [0,1,2])")

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        prod, ctrl = make_variant(False, td), make_variant(True, td)
        pA, wA, eA = run(prod, fA, OUT / "A_prod.npz")
        pB, wB, eB = run(prod, fB, OUT / "B_prod.npz")
        cA, wcA, ecA = run(ctrl, fA, OUT / "A_ctrl.npz")
        cB, wcB, ecB = run(ctrl, fB, OUT / "B_ctrl.npz")

    rows = np.flatnonzero(np.isin(selA, selB))
    fps = {wA, wB, wcA, wcB}
    eps_seen = {eA, eB, ecA, ecB}
    prod_diff = float(np.abs(pA[rows] - pB).max())
    ctrl_diff = float(np.abs(cA[rows] - cB).max())
    # Scale to judge ctrl_diff against: SAME anchors, SAME weights, only the seed
    # differs (prod seeds B by global index, ctrl by position 0,1,2). That is a pure
    # noise-substitution difference, so it says how big "a different draw" is here --
    # without it, a nonzero ctrl_diff could be read as float jitter.
    noise_scale = float(np.abs(cB - pB).max())
    # Free internal consistency check: the control can only diverge at anchors whose
    # LOOP POSITION differs from their GLOBAL INDEX. Where the two coincide it must be
    # bit-identical to prod -- if it is not, the patch changed something else too.
    same_pos = np.flatnonzero(selA == np.arange(len(selA)))
    ctrl_clean = bool(len(same_pos) == 0 or np.array_equal(cA[same_pos], pA[same_pos]))

    res = {
        "weight_fingerprints_identical": len(fps) == 1,
        "weight_fingerprint": sorted(fps),
        "val_episode_sets_identical": len(eps_seen) == 1,
        "prod_subset_vs_full_maxdiff": prod_diff,
        "prod_bit_exact": prod_diff == 0.0,
        "control_subset_vs_full_maxdiff": ctrl_diff,
        "control_differs_as_required": ctrl_diff > 0.0,
        "scale_same_anchor_different_seed": noise_scale,
        "control_touches_only_position_shifted_rows": ctrl_clean,
        "selA": selA.tolist(), "selB": selB.tolist(),
    }
    ok = (res["weight_fingerprints_identical"] and res["val_episode_sets_identical"]
          and res["prod_bit_exact"] and res["control_differs_as_required"] and ctrl_clean)
    print("\n=== subset-vs-full noise pinning ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print(f"\n  PROD  : subset reproduces full rows bit-for-bit      -> {res['prod_bit_exact']}")
    print(f"  CONTROL: position-seeded copy breaks it (maxdiff {ctrl_diff:.4g}) -> "
          f"{res['control_differs_as_required']}")
    print(f"\n{'PASS' if ok else 'FAIL'} -- subset_vs_full_bit_exact is a REAL property of this "
          f"predict path, not a vacuous one")
    res["verdict"] = "PASS" if ok else "FAIL"
    vp = pathlib.Path(args.out) if args.out else BENCH / "logs" / "subset_pinning_probe.json"
    vp.write_text(json.dumps(res, indent=2))
    print(f"[write] {vp}")   # the path actually written, not the default one
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
