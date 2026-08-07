#!/usr/bin/env python3
"""Exercise the SHIPPED variance_openpi functions against REAL files, CPU only, no ckpt.

Every check is two-sided: the positive says "it works on the real inputs", the negative injects
the exact failure the check exists to catch and requires it to fire. A check that cannot fail
reports PASS for free.
"""
import glob, hashlib, json, pathlib, sys, tempfile
import numpy as np

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BENCH / "scripts" / "predict"), str(BENCH / "scripts" / "score"),
                str(BENCH / "archive")]   # was one flat scripts/ dir
import variance_openpi as V  # the shipped module, not a copy

ok = fail = 0
def check(label, cond, extra=""):
    global ok, fail
    print(("  PASS  " if cond else "  FAIL  ") + label + ("   " + extra if extra else ""))
    ok, fail = ok + bool(cond), fail + (not cond)

A = np.load(BENCH / "data" / "val_anchors.npz")
eps, state, gt = A["episodes"], A["state"].astype(np.float32), A["gt"].astype(np.float32)
total, K = len(eps), gt.shape[1]

print("\n== 1. resolve_subset: identity gate ==")
p, sel, m, std, src = V.resolve_subset(str(BENCH / "logs/variance_subset200.npy"))
check("real primary resolves + md5 matches", std and m == V.SUBSET_MD5 and len(sel) == 200, src)
p2, sel2, m2, std2, src2 = V.resolve_subset("/nonexistent/variance_subset200.npy")
check("missing primary -> falls back to logs/ and still finds the shared file",
      std2 and np.array_equal(sel2, sel), src2)

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    # CONTROL WITH TEETH: right filename, right shape, right dtype, WRONG CONTENT.
    # An existence check passes this; only an md5 gate rejects it.
    decoy = td / "variance_subset200.npy"
    np.save(decoy, np.unique(np.linspace(0, total - 1, 200).round().astype(int)))
    check("decoy is shape/dtype-indistinguishable from the real subset",
          np.load(decoy).shape == sel.shape and np.load(decoy).dtype == sel.dtype,
          f"md5 {V.md5_of(decoy)[:8]} != {V.SUBSET_MD5[:8]}")
    orig_bench = V.BENCH
    try:
        V.BENCH = td   # redirect BOTH fallbacks so only the decoy is reachable
        try:
            V.resolve_subset(str(decoy)); check("decoy REJECTED (fatal)", False, "no SystemExit!")
        except SystemExit as e:
            check("decoy REJECTED (fatal)", "WRONG CONTENT" in str(e))
        try:
            V.resolve_subset(str(td / "absent.npy")); check("all-missing REJECTED", False)
        except SystemExit as e:
            check("all-missing REJECTED (fatal)", "not found" in str(e))
        pp, ss, mm, sd, sc = V.resolve_subset(str(decoy), allow_nonstandard=True)
        check("--allow-nonstandard-subset opens the door and MARKS it",
              (not sd) and len(ss) == 200 and "NON-STANDARD" in sc)
        pp, ss, mm, sd, sc = V.resolve_subset(str(td / "absent.npy"), allow_nonstandard=True)
        check("...and falls back to linspace only under the flag",
              (not sd) and mm is None and "linspace" in sc,
              f"overlap with shared = {len(set(ss.tolist()) & set(sel.tolist()))}/200")
    finally:
        V.BENCH = orig_bench

print("\n== 2. pinned_noise / measure_noise_collisions ==")
H, ADIM = 10, 32
n0 = V.pinned_noise(0, 8, H, ADIM)
check("shape + dtype", n0.shape == (H, ADIM) and n0.dtype == np.float32)
check("deterministic (same seed,anchor -> same bytes)",
      V.pinned_noise(0, 8, H, ADIM).tobytes() == n0.tobytes())
check("varies with anchor", V.pinned_noise(0, 9, H, ADIM).tobytes() != n0.tobytes())
check("varies with seed", V.pinned_noise(1, 8, H, ADIM).tobytes() != n0.tobytes())
sc_ = V.measure_noise_collisions(sel, [0, 1, 2, 3, 4], H, ADIM)
print("   " + json.dumps({k: sc_[k] for k in
      ("collisions", "power_control_A_additive_stride1", "power_control_B_anchor_not_folded_in",
       "verdict", "n_noise_tensors")}))
check("0 collisions on the real subset x 5 seeds", sc_["collisions"] == 0)
check("power control A (additive stride 1) TRIPS", sc_["power_control_A_additive_stride1"] > 0)
# key(s) ignores the anchor, so each draw emits ONE distinct tensor: 199 dupes per draw.
check("power control B (anchor not folded in) TRIPS",
      sc_["power_control_B_anchor_not_folded_in"] == (len(sel) - 1) * 5,
      f"{sc_['power_control_B_anchor_not_folded_in']} == 199*5")
check("verdict PASS", sc_["verdict"].startswith("PASS"))
# NEGATIVE: an aliasing subset (duplicate anchors) must be detected as collisions
bad = np.concatenate([sel[:100], sel[:100]])
sc_bad = V.measure_noise_collisions(bad, [0, 1], H, ADIM)
check("injected aliasing (duplicated anchors) -> FAIL verdict",
      sc_bad["collisions"] == 200 and sc_bad["verdict"].startswith("FAIL"),
      f"collisions={sc_bad['collisions']}")

print("\n== 3. subset_design: weighting + recomputed pass line ==")
# Any landed full-1397 preds file works as the reference for the offset block. Deliberately do
# NOT ship another leg's preds to a host that lacks them -- a foreign preds_*.npz sitting in
# preds/ is exactly the confusion the board's --out rule exists to prevent. Synthesize instead.
_cands = [f for f in sorted(glob.glob(str(BENCH / "preds/preds_*.npz")))
          if np.load(f)["pred"].shape[0] == total]
if _cands:
    ref, REF_IS_REAL = np.load(_cands[0])["pred"].astype(np.float32), True
    print(f"   ref preds: {pathlib.Path(_cands[0]).name} (real)")
else:
    ref = (gt + np.random.default_rng(0).normal(0, 1.5, gt.shape)).astype(np.float32)
    REF_IS_REAL = False
    print("   ref preds: SYNTHETIC (no full-1397 preds on this host) -- offset SIGNS not asserted")
d = V.subset_design(sel, eps, state, gt, K, ref)
print("   " + json.dumps({k: d[k] for k in ("sampling_scheme", "episode_balanced",
      "anchors_per_episode", "pass_line_subset200", "pass_line_full1397")}))
print("   weighting " + json.dumps(d["weighting_measured"]))
print("   offset    " + json.dumps(d["subset_vs_full_offset"]))
check("stratification measured, not declared",
      d["sampling_scheme"] == "stratified by episode" and d["episode_balanced"]
      and d["anchors_per_episode"] == {"min": 10, "median": 10, "max": 10})
check("subset pass line != full pass line (this is the whole point)",
      d["pass_line_subset200"]["mae"] != d["pass_line_full1397"]["mae"],
      f"{d['pass_line_subset200']['mae']} vs {d['pass_line_full1397']['mae']}")
# NB: the reported values are already rounded to 4 dp, so re-rounding to 3 double-rounds
# (6.8295 -> 6.83). Compare with the same tolerance the shipped tripwire uses.
check("full1397 pass line reproduces n15's + score.py's 3.072/2.535/6.829",
      all(abs(d["pass_line_full1397"][k] - v) < 1e-3
          for k, v in zip(("mae", "arm", "grip"), (3.072, 2.535, 6.829))),
      str([d["pass_line_full1397"][k] for k in ("mae", "arm", "grip")]))
check("subset200 pass line reproduces n15's 3.198/2.591/7.449",
      [round(d["pass_line_subset200"][k], 3) for k in ("mae", "arm", "grip")] == [3.198, 2.591, 7.449])
w = d["weighting_measured"]
# float32 summation order, not a real difference: the two agree to 2.4e-07 (7e-08 relative).
check("episode-balanced subset: frame-weighted == episode-equal (measured)",
      abs(w["subset200_hold_frame_weighted"] - w["subset200_hold_episode_equal"]) < 1e-5,
      f"diff {abs(w['subset200_hold_frame_weighted'] - w['subset200_hold_episode_equal']):.2e}")
check("full population: the two weightings DIFFER (so the field is load-bearing)",
      abs(w["full1397_hold_frame_weighted"] - w["full1397_hold_episode_equal"]) > 1e-3,
      f"{w['full1397_hold_frame_weighted']:.4f} vs {w['full1397_hold_episode_equal']:.4f}")
o = d["subset_vs_full_offset"]
check("offset block present and numeric",
      all(isinstance(o["preds"][k], float) for k in ("mae", "arm", "grip")))
if REF_IS_REAL:
    check("model and pass line move in OPPOSITE directions on the subset",
          o["preds"]["mae"] * o["hold"]["mae"] < 0,
          f"preds {o['preds']['mae']} hold {o['hold']['mae']}")
else:
    print("  SKIP  opposite-direction check (synthetic ref has no model structure)")
# BOUNDARY: grip recomputes to 6.82950, exactly on the round-3 cliff. The tripwire must NOT
# fire on a float-noise-sized shift, and MUST fire on a real one. Both sides, on the shipped fn.
_tiny = gt.copy(); _tiny[0, 0, 0] += np.float32(1e-4)   # ~1e-8 on the mean: pure noise scale
try:
    V.subset_design(sel, eps, state, _tiny, K, None)
    check("tripwire does NOT fire on float-noise-scale drift (no spurious unattended death)", True)
except SystemExit as e:
    check("tripwire does NOT fire on float-noise-scale drift", False, str(e)[:80])
# NEGATIVE: perturb val_anchors so the recomputed pass line drifts -> must die
gt_bad = gt.copy(); gt_bad[:, :, 0] += 0.5
try:
    V.subset_design(sel, eps, state, gt_bad, K, None)
    check("perturbed val_anchors -> pass-line tripwire fires", False, "no SystemExit!")
except SystemExit as e:
    check("perturbed val_anchors -> pass-line tripwire fires", "disagrees with the board" in str(e))

print("\n== 4. build_summary on real-shaped inputs (the post-GPU block) ==")
class Args: pass
a = Args(); a.tag = "pi05_TEST"; a.ckpt_dir = "/tmp/fake"; a.config_name = "pi05_nero_b2_train"
class M: action_horizon = 10; action_dim = 32
class C: model = M(); batch_size = 16
rng = np.random.default_rng(0)
fake = np.stack([ref[sel] + rng.normal(0, 0.01, ref[sel].shape) for _ in range(5)]).astype(np.float32)
s = V.build_summary(a, fake, sel, [0, 1, 2, 3, 4], gt, eps, state, K, C(), d, sc_,
                    p, m, True, "seed0 vs ...: max|diff|=0.000e+00 (IDENTICAL)")
json.dumps(s)  # must be JSON-serializable end to end
check("summary is JSON-serializable and complete",
      set(("std_ddof1", "mae", "arm", "grip", "subset_design", "seed_check",
           "per_anchor_noise_collisions_measured", "train_batch_size")) <= set(s))
check("train_batch_size recorded (n17: norm-stats bins depend on it)", s["train_batch_size"] == 16)
check("collision count is the MEASURED one", s["per_anchor_noise_collisions_measured"] == 0)
check("std is nonzero for perturbed draws (a real spread survives the pipeline)",
      s["mae_std"] > 0 and not s["bit_exact_across_draws"], f"std={s['mae_std']:.5f}")
check("SE scale factor = sqrt(200/1397)", abs(s["SE_sampling_scale_factor"] - np.sqrt(200/1397)) < 1e-12)
check("declared stride field is null, not a true-by-construction lie",
      s["seed_stride_exceeds_anchor_count"] is None)

print("\n== 5. power-control KEY superset (board contract; cross-leg reader bug) ==")
# n17's reader looked for its own key name in n15's json, missed, and silently accepted the bare
# measured 0 -- turning a controlled check into an uncontrolled one. The fix is to publish control
# A under every name any leg looks for. The failure mode of THAT fix is alias drift: a name in the
# advertised list that is not actually emitted, or emitted with a different value. Assert both, at
# both levels -- I nearly shipped exactly that drift (an alias listed but absent at top level).
KNOWN_READER_KEYS = ("power_control_collisions", "seed_check_power_control_collisions",
                     "seed_collision_power_control")
ctlA = sc_["power_control_A_additive_stride1"]
check("control A is nonzero (else every alias below publishes a meaningless 0)", ctlA > 0,
      f"A={ctlA}")
for k in KNOWN_READER_KEYS:
    check(f"top-level alias {k!r} present and == control A", s.get(k) == ctlA, f"{s.get(k)} vs {ctlA}")
    check(f"seed_check alias {k!r} present and == control A", sc_.get(k) == ctlA,
          f"{sc_.get(k)} vs {ctlA}")
check("advertised alias list is honest at TOP level (no drift)",
      all(s.get(k) == ctlA for k in s["power_control_key_aliases"]),
      str({k: s.get(k) for k in s["power_control_key_aliases"]}))
check("advertised alias list is honest inside seed_check",
      all(sc_.get(k) == ctlA for k in sc_["power_control_key_aliases"]))
check("control B is construction-fixed at (n-1)*draws, so it is anchor-set immune",
      sc_["power_control_B_matches_construction"]
      and sc_["power_control_B_expected_by_construction"] == (len(sel) - 1) * 5,
      f"B={sc_['power_control_B_anchor_not_folded_in']} "
      f"expected={sc_['power_control_B_expected_by_construction']}")
check("A and B are DIFFERENT bug shapes (that is the independent coverage, not the shared 254)",
      ctlA != sc_["power_control_B_anchor_not_folded_in"],
      f"A={ctlA} B={sc_['power_control_B_anchor_not_folded_in']}")

# Read-side control: the SHARED reader must find a power control in this schema. Injecting the
# n17 failure (strip every alias, keep the measured 0) must make it complain -- otherwise this
# whole section is decoration.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import importlib
_ps = importlib.import_module("paired_signif")
if hasattr(_ps, "_seed_stride_audit"):
    import io, contextlib
    def _audit(j, stem):
        # Distinct stem per call, and clear the memo: _seed_stride_audit dedupes on
        # (stem,"stride") via _ZERO_WARNED, so reusing a stem would make the SECOND call return
        # early and print nothing -- the injected control would read as "no teeth" for a reason
        # that has nothing to do with the injection.
        getattr(_ps, "_ZERO_WARNED", set()).clear()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _ps._seed_stride_audit(pathlib.Path(f"variance_{stem}.json"), j, stem)
        return buf.getvalue()
    clean = _audit(dict(s), "pi05_clean")
    stripped = dict(s)
    for k in list(KNOWN_READER_KEYS) + ["power_control_key_aliases", "power_control_ok",
                                        "power_control_A_additive_stride1",
                                        "seed_check_power_control_B", "seed_check"]:
        stripped.pop(k, None)
    injected = _audit(stripped, "pi05_stripped")
    check("shared reader accepts my schema silently (no note/warn)", clean.strip() == "",
          repr(clean[:200]))
    check("shared reader NOTICES when the aliases are stripped (control has teeth)",
          injected.strip() != "", repr(injected[:200]) or "(silent -- control failed)")
else:
    check("paired_signif._seed_stride_audit present", False, "reader API changed; re-verify by hand")

print("\n== 5b. bit_exact_across_draws is measured on bytes, not on a float == 0 ==")
# Two-sided: identical draws must report True, and a run that differs must report False. The
# reason this is not `m_std == 0.0` is the FALSE-NEGATIVE direction -- bit-identical draws whose
# std lands on ~1e-17 would report False, making a collapsed run look healthy.
ident = np.repeat(fake[:1], 5, axis=0)
s_id = V.build_summary(a, ident, sel, [0, 1, 2, 3, 4], gt, eps, state, K, C(), d, sc_,
                       p, m, True, "x")
check("identical draws -> bit_exact True", s_id["bit_exact_across_draws"] is True)
check("perturbed draws -> bit_exact False", s["bit_exact_across_draws"] is False)
# One draw differing in a SINGLE float must break bit-exactness even though the MAE (and hence
# any std-derived proxy) barely moves -- this is the gap the bytes-based test closes.
near = np.repeat(fake[:1], 5, axis=0).copy()
near[3, 0, 0, 0] = np.float32(near[3, 0, 0, 0] + np.float32(1e-6))
s_near = V.build_summary(a, near, sel, [0, 1, 2, 3, 4], gt, eps, state, K, C(), d, sc_,
                         p, m, True, "x")
check("a single-float difference is NOT reported as bit-exact",
      s_near["bit_exact_across_draws"] is False,
      f"std-derived proxy would have said {s_near['bit_exact_via_zero_std']}")

print("\n== 6. verify_driver_noise: the gate/driver gap (n17's actual defect) ==")
# The gate hashes pinned_noise(s, g) over `sel`. It cannot see WHICH arguments the production loop
# passes. n17's bug was exactly there: correct generator, correct-looking declared field, wrong
# call site. So build the ledger the driver builds, and inject the call-site defects.
SEEDS5 = [0, 1, 2, 3, 4]
gate_fp = sc_["noise_multiset_fingerprint"]
ledger = np.array([[hashlib.md5(V.pinned_noise(s, int(g), H, ADIM).tobytes()).hexdigest()
                    for g in sel] for s in SEEDS5], dtype="<U32")
try:
    dv = V.verify_driver_noise(ledger, gate_fp)
    check("honest ledger reconciles with the gate", dv["driver_matches_gate_multiset"]
          and dv["driver_noise_collisions_measured"] == 0
          and dv["driver_noise_chunks"] == len(SEEDS5) * len(sel),
          f"{dv['driver_noise_chunks']} chunks, fp={dv['driver_noise_multiset_fingerprint'][:8]}")
except SystemExit as e:
    check("honest ledger reconciles with the gate", False, str(e)[:200])

check("fingerprint is order-independent (driver walks episodes, gate walks seeds)",
      V.noise_fingerprint(ledger.reshape(-1).tolist())
      == V.noise_fingerprint(list(np.random.default_rng(0).permutation(ledger.reshape(-1)))),
      "shuffled ledger fingerprints identically")


def must_raise(label, ledger_, expect, fp=gate_fp):
    try:
        V.verify_driver_noise(ledger_, fp)
        check(label, False, "ACCEPTED a defective ledger")
    except SystemExit as e:
        check(label, expect in str(e), f"raised for the wrong reason: {str(e)[:160]}"
              if expect not in str(e) else expect)


# n17's defect verbatim: the loop passes the LOCAL row index r where the gate assumed the GLOBAL
# anchor index g. Collision-free, right count, completely different tensors -- fingerprint only.
local_idx = np.array([[hashlib.md5(V.pinned_noise(s, r, H, ADIM).tobytes()).hexdigest()
                       for r in range(len(sel))] for s in SEEDS5], dtype="<U32")
check("the n17 defect is invisible to a collision count alone (motivates the fingerprint)",
      len(set(local_idx.reshape(-1).tolist())) == local_idx.size,
      "local-index ledger is ALSO collision-free")
must_raise("driver passing local r instead of global g is caught", local_idx, "MISMATCH")

aliased = np.array([[ledger[0, 0]] * len(sel)] * len(SEEDS5), dtype="<U32")
must_raise("driver feeding one tensor everywhere is caught", aliased, "duplicate noise chunks")

holed = ledger.copy(); holed[2, 7] = ""
must_raise("unwritten ledger cell (loop skipped a pair) is caught", holed, "unwritten cells")

# Control with teeth for the fingerprint comparison itself: an HONEST ledger against a WRONG gate
# fingerprint must also fail, else the comparison is only checking the ledger's self-consistency.
must_raise("honest ledger vs a foreign gate fingerprint is caught", ledger, "MISMATCH",
           fp="0" * 32)

print(f"\n==== {ok} passed, {fail} failed ====")
sys.exit(1 if fail else 0)
