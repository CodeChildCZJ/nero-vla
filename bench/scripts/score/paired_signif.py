#!/usr/bin/env python3
"""Is the gap between two leaderboard rows bigger than this val set can resolve?

Does NOT modify or replace score.py (frozen harness) -- it reads the same
preds_*.npz files afterwards and adds an uncertainty column.

The subtlety this exists for: the delete-one-episode jackknife SE of an ABSOLUTE
MAE on b2 is 0.1042 (3.4% of the pass line), which would suggest "gaps under ~0.2
are noise". That is the WRONG ruler. Every stack is scored on the SAME 20 episodes,
so episode-composition wobble is largely COMMON to both rows and cancels in the
difference. Measured on b2 with two closely-related predictors the paired SE came
out 11x tighter (0.00907), i.e. the naive threshold would have dismissed almost
every real difference on the board.

So: always jackknife the PAIRED DIFFERENCE, per pair.

  python scripts/paired_signif.py preds/preds_a.npz preds/preds_b.npz
  python scripts/paired_signif.py --all            # every pair in preds/

Caveat baked into the output: the shrinkage factor depends on how CORRELATED the
two stacks' per-episode errors are. Two similar predictors shrink a lot; two very
different backbones shrink less. That is exactly why this must be computed per
pair rather than quoted as one global threshold.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

# 仓库自包含:从本文件位置推导。bench/scripts/score/*.py -> parents[2] == bench/
BENCH = Path(__file__).resolve().parents[2]


def load(path, K, n):
    p = np.load(path)["pred"]
    assert len(p) == n, f"{path}: {len(p)} rows, expected {n}"
    assert p.shape[1] >= K, f"{path}: horizon {p.shape[1]} < K={K}"
    # FINITENESS GATE AT INGESTION -- a non-finite row's blast radius is the whole
    # comparison, not its own line. Measured before this gate existed: an all-NaN
    # preds file sorted to rank #1 (NaN breaks Timsort's transitivity) AND every
    # pair against it printed the verdict "RESOLVED". The `if se>0 else inf`
    # divide-by-zero guard was the amplifier: NaN>0 is False, so NaN fell into the
    # inf branch and inf cleared |z|>=2. A defensive guard manufactured a confident
    # false positive. Excluding matters more than warning -- see score.py's own
    # version of this bug (gr00t-n15).
    if not np.isfinite(p[:, :K, :]).all():
        bad = int((~np.isfinite(p[:, :K, :])).sum())
        print(f"!! EXCLUDING {Path(path).stem}: {bad} non-finite values in the K={K} "
              f"prefix. A NaN row misorders the WHOLE board, it does not just lose "
              f"its own line.")
        return None
    return p[:, :K, :]


def per_anchor_mae(pred, gt, dims=None):
    if dims is not None:
        pred, gt = pred[..., dims], gt[..., dims]
    return np.abs(pred - gt).mean(axis=(1, 2))


# The board's headline is a mean over 8 dims, but 7 of them are the arm and the arm
# moves ~1 deg/frame, so hold-state is already strong there while the gripper has a
# 6.8 pass line to win back. Averaging the two channels dilutes the only one that
# discriminates by ~7x, and it can inverts a row's story: measured on b2, ACT is
# 11.5 sigma better than hold-state on the gripper and 0.7 sigma WORSE on the arm,
# which the 1.9-sigma total shows as a single lukewarm number. Always print both.
CHANNELS = {"TOTAL (8 dims)": None,
            "ARM  (j0-j6)": list(range(7)),
            "GRIP (j7)": [7]}

# json key for each channel inside logs/variance/variance_<bb>.json
VAR_KEY = {"TOTAL (8 dims)": "mae", "ARM  (j0-j6)": "arm", "GRIP (j7)": "grip"}


def sampling_std(stem, channel):
    """Seed-to-seed std of THIS row's MAE, or None if the stack posted no footnote.

    Two schemas are in use on the board and they must both be understood -- act
    nests {"mae": {"mean":..,"std":..}}, gr00t_n15 uses a flat {"std_ddof1": {...}}.
    A reader that knows only one and falls back to 0.0 would silently DROP the
    sampling term for the other stack, which is indistinguishable from a stack that
    is genuinely deterministic. So an unparseable file RAISES; only a missing file
    is treated as "no footnote posted yet".

    Uses the per-DRAW std, not std/sqrt(R): the board row is one realization, so the
    question is "how far could this row have landed", not "how well do we know the
    mean of 5 draws".
    """
    f = BENCH / "logs" / "variance" / f"variance_{stem.replace('preds_', '')}.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    k = VAR_KEY[channel]
    if "std_ddof1" in j:                        # gr00t_n15 schema
        std = float(j["std_ddof1"][k])
    elif isinstance(j.get(k), dict) and "std" in j[k]:   # act schema
        std = float(j[k]["std"])
    elif f"{k}_std" in j:                       # flat schema: gr00t_n17, openpi
        std = float(j[f"{k}_std"])
    else:
        raise SystemExit(f"{f.name}: cannot find a '{k}' std in this schema "
                         f"(keys={sorted(j)}). Refusing to assume 0.0 -- a dropped "
                         f"sampling term looks exactly like a deterministic stack.")
    if std == 0.0:
        _zero_std_audit(f, j, stem)
    else:
        _seed_stride_audit(f, j, stem)
    return std


def _seed_stride_audit(f, j, stem):
    """A NONZERO std can still be understated, and the giveaway is not in the std.

    The seed that matters is per (draw, ANCHOR), not per draw. With the natural
    formula `base + stride*draw + anchor_index`, a stride smaller than the anchor
    count makes draw d anchor g collide with draw d+1 anchor g-stride: the same
    seed, the same noise tensor, reused across draws. The draws are then positively
    correlated and the measured spread is too SMALL -- the direction that reports
    noise as signal. gr00t-n15 measured 796/800 reused at stride 1; my own variance
    script fed the draw index in as the BASE seed (stride 1) and reused 254/1000 on
    the shared subset, while its `per_draw_seeds` list [0,1,2,3,4] stayed perfectly
    distinct. So a distinctness check on the per-draw seeds -- the check one layer
    up in this file -- passes this bug green. Only the stride can catch it.

    ...except the stride field is itself DECLARED, and that is not good enough
    either (gr00t-n15, 2026-08-05). `seed_stride_exceeds_anchor_count` is derived
    from the stride CONSTANT, and on the day my variance script had the stride-1
    bug that constant was already 1000003 -- the caller, not the constant, was
    wrong. So the field would have read True on 254/1000 aliased draws. Prefer a
    MEASURED collision count when the leg posts one; fall back to the declared
    field with an explicit note that it is weaker evidence. A leg posting neither
    still gets warned, because silence and evidence must not look alike.
    """
    key = (stem, "stride")
    if key in _ZERO_WARNED:
        return
    m = j.get("per_anchor_noise_collisions_measured")
    if m is not None:
        # A measured 0 is only worth anything if the measurement could have come
        # back nonzero, so require the leg's power control alongside it. The legs
        # converged on the same name for the load-bearing count but NOT for its
        # power control (n17 `seed_collision_power_control`, n15
        # `seed_check_power_control_collisions`), so accept both -- and if neither
        # is present, SAY SO rather than treating the bare 0 as evidence. Missing
        # power control was exactly the hole the measured count exists to close;
        # silently accepting it here would reopen it one layer up.
        # Hardcoding the union does not scale: four legs invented four names, and
        # the fifth reader to be written would have to be taught all of them again.
        # Prefer the artifact's OWN advertised alias list, then fall back to the
        # names known today. Note the self-describing route can only ADD names --
        # a file that advertises nothing still gets checked against the known set,
        # so a leg cannot dodge the audit by omitting the list.
        known = ("seed_collision_power_control", "seed_check_power_control_collisions",
                 "power_control_collisions", "collision_power_control")
        advertised = j.get("power_control_key_aliases") or []
        names = list(dict.fromkeys([str(k) for k in advertised] + list(known)))
        pc = next((j[k] for k in names if k in j), None)
        if m == 0 and pc is None:
            _ZERO_WARNED.add(key)
            print(f"[note] {f.name}: 0 measured collisions, but no power-control count "
                  f"under any known key (looked for {' / '.join(names)}). A 0 from a "
                  f"check that cannot fail is not evidence -- ask the leg which key it "
                  f"posts, and have it emit `power_control_key_aliases`.")
            return
        if m == 0 and pc > 0:
            return
        _ZERO_WARNED.add(key)
        if m != 0:
            print(f"[warn] {f.name}: {m} MEASURED per-(draw,anchor) noise collisions -- "
                  f"draws are aliased and this sampling std is UNDERSTATED.")
        else:
            print(f"[warn] {f.name}: reports 0 measured collisions but its power control "
                  f"returned {pc} -- a check that cannot fail reporting 0 is not evidence.")
        return
    v = j.get("seed_stride_exceeds_anchor_count")
    if v is True:
        _ZERO_WARNED.add(key)
        print(f"[note] {f.name}: non-aliasing is DECLARED (stride "
              f"{j.get('per_draw_seed_stride')} > {j.get('n_anchors_full')} anchors), not "
              f"measured. That field is arithmetic over the stride constant, and the "
              f"constant was already correct during a real stride-1 aliasing bug -- the "
              f"fault was in the caller. Ask the leg for a measured collision count.")
        return
    _ZERO_WARNED.add(key)
    if v is False:
        print(f"[warn] {f.name}: seed_stride_exceeds_anchor_count is FALSE -- draws share "
              f"noise tensors, so this sampling std is UNDERSTATED (stride "
              f"{j.get('per_draw_seed_stride')}, anchors {j.get('n_anchors_full')}).")
    else:
        print(f"[warn] {f.name}: nonzero sampling std with no "
              f"`seed_stride_exceeds_anchor_count` field. Distinct per-DRAW seeds do "
              f"not rule out per-(draw,anchor) seed collisions, which shrink the std. "
              f"Ask the leg for the stride and its per-anchor seed formula.")


_ZERO_WARNED = set()


def _zero_std_audit(f, j, stem):
    """A std of EXACTLY 0.0 has two causes that produce identical artifacts.

    (a) a genuinely deterministic decoder (ACT's argmax-style forward), and
    (b) a variance harness that never varied anything -- same seed reused for all R
    draws, or the noise pinned outside the draw loop. Both write std 0.0, and both
    also write `bit_exact_across_draws: true`, so that field CANNOT discriminate
    them: it is computed from the same R numbers, downstream of the same bug. It
    restates the 0.0, it does not corroborate it.

    Two DIFFERENT kinds of upstream evidence discriminate them, and they sit on
    opposite sides of the harness:

    (1) a recorded list of DISTINCT per-draw seeds -- proves the draws COULD have
        differed (the input was perturbed);
    (2) a `harness_power_control` -- proves the comparator would have NOTICED if
        they had (a deliberately perturbed draw is rejected, measured).

    Accepting only (1) mis-scores a whole class of legs: OFT's predict path takes
    `do_sample=False` through an L1 regression head and consumes NO seed at all, so
    it is not that they forgot to record seeds, there are none to record. Demanding
    (1) there would have rewarded generating seeds that touch nothing -- satisfying
    the audit while adding no information -- and meanwhile it PASSES a lazy harness
    that reuses one seed R times but dutifully writes down R distinct-looking ones.
    (2) is what a lazy harness cannot fake, and it is the only one a seedless
    deterministic stack can supply. Either alone clears the row; a lazy harness
    clears neither (openvla-oft, 2026-08-05).

    Warn, never raise -- a real 0.0 is legitimate and common and the row is still
    usable; the reader just cannot tell which of the two it is holding.
    """
    seeds = j.get("per_draw_seeds") or j.get("seeds") or []
    if len(seeds) > 1 and len(set(map(str, seeds))) == len(seeds):
        return                                  # distinct seeds recorded -> real 0
    if j.get("harness_power_control"):
        return                                  # comparator demonstrated to have teeth
    if stem in _ZERO_WARNED:
        return
    _ZERO_WARNED.add(stem)
    print(f"[warn] {f.name}: sampling std is exactly 0.0 and the footnote records "
          f"NEITHER a list of DISTINCT per-draw seeds (proving the draws could "
          f"differ) NOR a `harness_power_control` (proving the comparator would "
          f"notice if they did), so 'deterministic decoder' and 'harness reused one "
          f"seed' are indistinguishable here. Using 0.0 (the quadrature term drops "
          f"out). `bit_exact_across_draws` does not settle it -- it is derived from "
          f"the same draws. A seedless stack should send the power control.")


def jackknife_se(values, eps, uniq):
    """delete-one-episode jackknife SE of the mean of `values`."""
    loo = np.array([values[eps != e].mean() for e in uniq])
    n = len(uniq)
    return float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))


def jackknife_se_eq(values, eps, uniq):
    """Same delete-one-episode scheme, but for the EPISODE-EQUAL mean.

    The SE has to belong to the number it is attached to: `jackknife_se` above is the
    SE of the frame-weighted mean, so comparing it against an episode-equal gap would
    mix two estimands. This resamples the same unit (the episode) and re-estimates the
    same functional the gapEQ column prints -- an unweighted mean of per-episode means.

    Why the board needs both: reweighting is a DETERMINISTIC shift that moves a model
    and its baseline in opposite directions (measured corr +0.34 model vs -0.62 against
    the hold-state line), so it threatens DIFFERENCES far more than ratios -- and it
    moves the SE too, which means a 2-sigma VERDICT can flip while the SIGN is stable.
    Measured board-wide: exactly 1 of 9 pairs is affected (act vs the pass line on
    TOTAL, 1.89 sigma frame vs 2.97 sigma episode-equal). gr00t-n15 then swept the
    caliber as a continuous family w_e ~ L_e**alpha and found that pair crosses 2 sigma
    at alpha=0.892 -- i.e. 89% of the family resolves it and the board sits in the
    unresolved 11% next to the frame-weighted endpoint. So the frame-weighted reading
    is the most pessimistic member of the family, not a neutral one, and reporting it
    alone would be as one-sided as reporting only the episode-equal number.
    """
    loo = np.array([np.mean([values[(eps == e2)].mean() for e2 in uniq if e2 != e])
                    for e in uniq])
    n = len(uniq)
    return float(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="*")
    ap.add_argument("--all", action="store_true", help="every pair in preds/")
    ap.add_argument("--channel", default="", choices=["", "total", "arm", "grip"],
                    help="restrict to one channel (default: report all three)")
    args = ap.parse_args()

    A = np.load(BENCH / "data" / "val_anchors.npz")
    eps, gt, K = A["episodes"], A["gt"], int(A["K"])
    gt = gt[:, :K, :]
    uniq = sorted(set(int(e) for e in eps))
    n = len(gt)

    files = ([sorted((BENCH / "preds").glob("preds_*.npz"))] if args.all
             else [[Path(p) for p in args.preds]])[0]
    if len(files) < 2:
        raise SystemExit(f"need >=2 preds files, got {len(files)}: {files}")

    # The hold-state (delta=0) predictor is score.py's pass line. Include it as a
    # virtual row so every run answers "is this stack significantly better than
    # doing nothing?" -- score.py's headline check, but with an SE attached.
    hold = np.repeat(A["state"][:, None, :], K, axis=1)
    loaded = {f.stem: v for f in files if (v := load(f, K, n)) is not None}
    loaded["_hold_state(pass line)"] = hold
    if len(loaded) < 2:
        raise SystemExit("fewer than 2 usable rows after the finiteness gate")

    for cname, dims in CHANNELS.items():
        if args.channel and cname.split()[0] != args.channel.upper():
            continue
        print(f"\n{'='*72}\n=== {cname}\n{'='*72}")
        per = {k: per_anchor_mae(v, gt, dims) for k, v in loaded.items()}
        mae = {k: v.mean() for k, v in per.items()}
        # Episode-equal weighting alongside the board's frame-length weighting.
        # Both are legitimate; they are not a robustness check of each other, they
        # answer different questions ("average anchor" vs "average episode"), and
        # on this val set they DISAGREE about the arm column (gr00t-n15). Printing
        # it makes the disagreement a measurement instead of a caveat somebody has
        # to remember -- the same reason the seed fields had to become measured.
        epw = {k: float(np.mean([v[eps == e].mean() for e in uniq]))
               for k, v in per.items()}
        for k in sorted(mae, key=lambda k: mae[k]):
            print(f"{k:<28} MAE {mae[k]:.4f}   (absolute jackknife SE "
                  f"{jackknife_se(per[k], eps, uniq):.4f})   ep-equal {epw[k]:.4f}")
        order_f = sorted(mae, key=lambda k: mae[k])
        order_e = sorted(epw, key=lambda k: epw[k])
        if order_f != order_e:
            print(f"[weighting] this channel REORDERS under episode-equal weighting:\n"
                  f"            frame-weighted {order_f}\n"
                  f"            episode-equal  {order_e}\n"
                  f"            => any prose claim of the form 'X beats Y on this "
                  f"channel' must name its weighting.")

        # sampling footnote per row; hold-state is exact (no decoder, no seed)
        samp = {k: (0.0 if k.startswith("_hold_state")
                    else sampling_std(k, cname)) for k in loaded}

        print(f"\n{'pair':<46} {'gap':>9} {'gapEQ':>9} {'SEepis':>8} {'SEsamp':>7} "
              f"{'SEtot':>8} {'gap/SE':>7} {'rho':>6} {'shrink':>7}  verdict")
        for a, b in itertools.combinations(sorted(mae, key=lambda k: mae[k]), 2):
            d = per[b] - per[a]
            # rows are sorted best-first under frame weighting, so `gap` is >=0 by
            # construction; gapEQ is the same difference with every episode given
            # equal weight, and a NEGATIVE gapEQ means the two rows swap.
            gap_eq = float(np.mean([d[eps == e].mean() for e in uniq]))
            se = jackknife_se(d, eps, uniq)
            sa, sb = jackknife_se(per[a], eps, uniq), jackknife_se(per[b], eps, uniq)
            # per-episode correlation of the two error profiles: this is what decides
            # whether pairing helps. rho>0 shrinks the SE, rho<0 INFLATES it.
            ma = np.array([per[a][eps == e].mean() for e in uniq])
            mb = np.array([per[b][eps == e].mean() for e in uniq])
            rho = float(np.corrcoef(ma, mb)[0, 1])
            naive = float(np.hypot(sa, sb))
            # Two INDEPENDENT uncertainty sources (board policy): episode composition
            # (the jackknife) and seed-to-seed sampling. The two stacks' decoders are
            # independent of each other, so their sampling variances simply add.
            # A row with no footnote yet is marked "?" rather than silently 0.
            miss = [k for k in (a, b) if samp[k] is None]
            s_pair = float(np.sqrt(sum((samp[k] or 0.0) ** 2 for k in (a, b))))
            se_tot = float(np.hypot(se, s_pair))
            z = d.mean() / se_tot if se_tot > 0 else float("inf")
            # Defence in depth behind the ingestion gate: never let a non-finite
            # statistic reach either verdict. Both branches would LIE -- inf reads
            # "RESOLVED" (false positive) and NaN fails `>= 2` and so reads
            # "NOT resolved", a plausible English result for a corrupt file.
            if not np.isfinite(z):
                verdict = "UNUSABLE (non-finite statistic -- do not read as a result)"
            else:
                verdict = ("RESOLVED" if abs(z) >= 2
                           else "NOT resolved by this val set")
            if miss:
                verdict += "  [no variance footnote: " + ",".join(miss) + "]"
            # A sign flip is not a second significance test and it does not move a
            # single SE; it says the DIRECTION was never pinned by this val set, so
            # it belongs on rows that are already NOT resolved and is worth shouting
            # about on any row that is.
            if gap_eq < 0:
                verdict += "  [SIGN FLIPS under episode-equal weighting]"
            # CALIBER-DEPENDENT VERDICT (team-lead's ruling, board rule).
            # The sign flag above only fires when the DIRECTION reverses. A verdict can
            # flip with the sign perfectly stable, because episode-equal weighting moves
            # the gap and its SE at once. That case is invisible to every other column
            # here, so it gets its own flag rather than a footnote somewhere else.
            # PRIMARY stays frame-weighted (it must match the estimand score.py ranks);
            # this does not collapse to one caliber, because the DISAGREEMENT is itself
            # the finding.
            se_eq = jackknife_se_eq(d, eps, uniq)
            se_tot_eq = float(np.hypot(se_eq, s_pair))
            z_eq = gap_eq / se_tot_eq if se_tot_eq > 0 else float("inf")
            if np.isfinite(z) and np.isfinite(z_eq) and (abs(z) >= 2) != (abs(z_eq) >= 2):
                verdict += (f"  [VERDICT IS CALIBER-DEPENDENT: frame {abs(z):.2f}sigma "
                            f"vs ep-equal {abs(z_eq):.2f}sigma -- not robustly resolved]")
            # No "row A is behind on this channel" flag is possible here: rows are
            # re-sorted best-first WITHIN each channel, so the gap is >=0 by
            # construction. The channel-level reversal shows up as a reordering
            # instead -- on ARM the pass line sorts ABOVE act, which is the finding.
            print(f"{(a+' vs '+b):<46} {d.mean():>+9.4f} {gap_eq:>+9.4f} {se:>8.5f} "
                  f"{s_pair:>7.5f} {se_tot:>8.5f} {z:>7.1f} "
                  f"{rho:>+6.2f} {naive/max(se,1e-12):>6.2f}x  {verdict}")
    print("\nrho = per-episode correlation of the two error profiles; shrink = "
          "sqrt(SEa^2+SEb^2)/pairedSE.")
    print("shrink > 1 means pairing HELPED (shared episode difficulty cancels); "
          "shrink < 1 means it")
    print("HURT -- which happens against the hold-state line, because that line is "
          "the delta=0")
    print("predictor and its per-episode difficulty runs OPPOSITE to a real model's "
          "(still episodes")
    print("are easy for it and hard for a model, and vice versa). There is therefore "
          "NO global")
    print("significance threshold in either direction -- compute it per pair.")

    print("\nverdict uses |gap| >= 2 * SEtot, SEtot = sqrt(SEepis^2 + SEsamp^2):")
    print("  SEepis = paired episode jackknife (which 20 episodes were held out)")
    print("  SEsamp = seed-to-seed, read from logs/variance/variance_<bb>.json, the")
    print("           per-DRAW std (the board row is ONE draw, not a mean of R), and")
    print("           summed over the pair since the two decoders are independent.")
    print("NOTE the footnotes are measured on the shared 200-anchor subset while the")
    print("board scores 1397. If per-anchor sampling noise is independent, the true")
    print("full-set SEsamp is ~sqrt(200/1397)=0.38x what is printed, so leaving it")
    print("unscaled is the CONSERVATIVE choice (errs toward calling signal noise).")


if __name__ == "__main__":
    main()
