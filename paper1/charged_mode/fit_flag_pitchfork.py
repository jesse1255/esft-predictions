"""
Location of the 0 ↔ 2 symmetry-restoring pitchfork on the F₂ branch.

Reads the gauge-invariant continuations (run_flag_continue.py --disc align) and
fits (D₀ − D₂)² = A (κ₃ − κ₃_p) to the asymmetric points closest to the
transition (a supercritical pitchfork gives D₀ − D₂ ∝ √(κ₃ − κ₃_p)).  Also
prints the linearity check: the fit residuals and the value of (D₀ − D₂)² that a
straight line through the outer points predicts for the inner ones.

Usage: python fit_flag_pitchfork.py   → data/flag_pitchfork_fit.json
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def rows_for(ne):
    rows = []
    for fn in sorted(glob.glob(os.path.join(DATA, f"flag_continue_ne{ne}_align*.json"))):
        if "up_large" in fn:
            continue
        for r in json.load(open(fn))["rows"]:
            if "D_all" in r:          # the first ne = 16 run stored D₂ only
                rows.append((r["kappa3"], r["D_all"][0] - r["D_all"][2], r["dE"]))
    # one entry per κ₃ (the fine scans repeat some points).  Just above κ₃_p the
    # upward scan can stay on the symmetric saddle; keep the asymmetric minimum.
    uniq = {}
    for k, d, e in rows:
        key = round(k, 4)
        if key not in uniq or abs(d) > uniq[key][0]:
            uniq[key] = (abs(d), e)
    return sorted((k, d, e) for k, (d, e) in uniq.items())


def main():
    out = {}
    for ne in (16, 24, 32):
        rows = rows_for(ne)
        if not rows:
            continue
        asym = [(k, d) for k, d, e in rows if d > 1e-2 and k < 0.77]
        sym = [k for k, d, e in rows if d <= 1e-2]
        if len(asym) < 3:
            print(f"ne={ne}: only {len(asym)} asymmetric points below 0.77 — waiting for the fine scan")
            continue
        k = np.array([a[0] for a in asym])
        d2 = np.array([a[1] ** 2 for a in asym])
        A, B = np.polyfit(k, d2, 1)
        kp = -B / A
        res = d2 - (A * k + B)
        out[ne] = dict(points=[dict(kappa3=float(x), D0_minus_D2=float(np.sqrt(y))) for x, y in zip(k, d2)],
                       slope=float(A), kappa3_p=float(kp), max_abs_residual=float(np.abs(res).max()),
                       highest_symmetric_kappa3=float(max(s for s in sym if s < 0.77)) if sym else None)
        print(f"ne={ne}: κ3_p = {kp:.4f}  slope {A:.1f}  max residual {np.abs(res).max():.3f} "
              f"(on (D0−D2)² ≤ {d2.max():.1f});  highest symmetric point below 0.77: "
              f"{out[ne]['highest_symmetric_kappa3']}")
    with open(os.path.join(DATA, "flag_pitchfork_fit.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
