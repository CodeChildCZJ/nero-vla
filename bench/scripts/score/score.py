#!/usr/bin/env python3
"""Common offline open-loop action-error leaderboard for NERO b2 backbone bench.
Each backbone writes preds/preds_<bb>.npz with key 'pred' shape (A,K,8) aligned to
val_anchors.npz anchor order. This scores all present preds uniformly."""
import glob, numpy as np, pathlib, sys
# 仓库自包含:从本文件位置推导,不依赖任何环境变量或绝对路径。
# bench/scripts/score/score.py -> parents[2] == bench/
BENCH = pathlib.Path(__file__).resolve().parents[2]
A = np.load(BENCH/"data"/"val_anchors.npz")
gt = A["gt"].astype(np.float32)            # (N,K,8)
state = A["state"].astype(np.float32)      # (N,8) obs@t
N, K, D = gt.shape
# Poisoned-REFERENCE gate (gr00t-n15 2026-08-05). The finiteness gate in the preds loop below
# covers the glob, but HOLD_MAE — the pass line itself — is computed from gt/state here and its
# row is appended UNCONDITIONALLY, never passing any gate. A single non-finite gt/state cell is
# STRICTLY WORSE than a poisoned preds file: HOLD_MAE goes NaN, so `mae >= HOLD_MAE` is False for
# EVERY row (the "learned nothing" flag goes globally dead, not one row), AND the hold row's NaN
# sort key breaks Timsort transitivity on the one row that CANNOT be excluded. There is no
# degraded mode without a pass line → hard exit, not a skip. (Measured clean on the live file:
# gt/state 0 NaN / 0 Inf, HOLD_MAE 3.0719962 finite — this is a latent guard for anyone who
# regenerates val_anchors.npz, not a live defect.)
if not (np.isfinite(gt).all() and np.isfinite(state).all()):
    sys.exit("[FATAL] non-finite cell in val_anchors gt/state: the pass line itself would be NaN, "
             "silently disabling the do-nothing flag and corrupting the whole ranking. "
             "Regenerate val_anchors.npz — do not trust any board produced from this file.")
rows = []
# trivial "hold current state" baseline — pass line: any backbone worse than this
# has learned nothing (pred = obs state, broadcast over K steps).
_hold = np.broadcast_to(state[:, None, :], (N, K, D))
_herr = np.abs(_hold - gt)
HOLD_MAE = _herr.mean()
rows.append(("[hold-state]", HOLD_MAE, _herr[..., :7].mean(), _herr[..., 7].mean(), _herr.mean(axis=(0,1))))
for f in sorted(glob.glob(str(BENCH/"preds"/"preds_*.npz"))):
    name = pathlib.Path(f).stem.replace("preds_","")
    P = np.load(f)["pred"].astype(np.float32)   # (N,Kp,8) Kp>=K
    if P.shape[0] != N:
        print(f"[skip {name}] anchor count {P.shape[0]} != {N}"); continue
    P = P[:, :K, :]                              # compare shared prefix
    if not np.isfinite(P).all():                 # NaN/Inf finiteness gate (scoped to the K-prefix
        # actually scored). A non-finite row does NOT corrupt just its own row — it silently
        # misorders the WHOLE board: `mae >= HOLD_MAE` is False for NaN so the "learned nothing"
        # flag never fires, AND a NaN sort key breaks Timsort's comparison transitivity so the
        # final ranking depends on this file's glob position (measured: NaN mid-glob pushes the
        # pass line to #1). The `continue` matters more than the print — warning without excluding
        # leaves the sort broken. (gr00t-n15/n17/openvla-oft 2026-08-05; task#1 shared layer.)
        print(f"[skip {name}] non-finite preds in K-prefix (NaN/Inf) — excluded from board"); continue
    err = np.abs(P - gt)                         # (N,K,8)
    mae = err.mean()
    arm = err[..., :7].mean()                    # joints 0-6
    grip = err[..., 7].mean()                    # gripper j7
    per_joint = err.mean(axis=(0,1))             # (8,)
    rows.append((name, mae, arm, grip, per_joint))
rows.sort(key=lambda r: r[1])
if not rows:
    print("no preds_*.npz yet in", BENCH/"preds"); sys.exit(0)
print(f"\n=== NERO b2 held-out ({N} anchors, K={K}, per-joint MAE, raw joint units) ===")
print(f"{'backbone':<18}{'MAE':>8}{'arm(0-6)':>10}{'grip(7)':>9}   per-joint j0..j7")
for name, mae, arm, grip, pj in rows:
    pjs = " ".join(f"{v:5.2f}" for v in pj)
    flag = ""
    if name == "[hold-state]":
        flag = "  <-- pass line (do-nothing baseline)"
    elif mae >= HOLD_MAE:
        flag = "  !! worse than do-nothing = learned nothing"
    print(f"{name:<18}{mae:8.3f}{arm:10.3f}{grip:9.3f}   {pjs}{flag}")
print("\n(lower=better; arm in deg, gripper in its unit; open-loop first-K prediction vs GT)")
print(f"(hold-state pass line: MAE {HOLD_MAE:.3f} — a real backbone must beat this)")
