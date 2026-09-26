"""
Tables for HOPF_PAIR_R5.md from the run_hopf_pair.py outputs.

For the constrained (relaxed) scans, a point is kept only if the relaxation
converged and the Hopf charge stayed at −2 (|Q + 2| < 0.02); points where the
lattice let the pair unwind (Q → 0) are listed separately.  For every d with a
complete, valid α circle the relaxed E_int(α) is Fourier-analysed like the
superposition map: E_int = c0 − B1 cos(α − α1) + ….

Usage: python summarize_hopf_pair.py   → data/pair_summary.json
"""

import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def harmonics(alphas, E):
    F = np.fft.rfft(E) / len(alphas)
    return dict(c0=float(F[0].real), B1=float(2 * abs(F[1])),
                alpha1_deg=float(np.degrees(np.angle(-F[1].conj())) % 360),
                B2=float(2 * abs(F[2])), higher=float(2 * np.abs(F[3:]).sum()))


def main():
    out = {}
    for fn in sorted(glob.glob(os.path.join(DATA, "pair_constrained_*.json"))):
        d = json.load(open(fn))
        key = os.path.basename(fn)[len("pair_constrained_"):-5]
        by_d = {}
        for r in d["rows"]:
            by_d.setdefault(round(r["d"], 3), []).append(r)
        rows = []
        print(f"== {key}")
        for dd, rs in sorted(by_d.items()):
            rs.sort(key=lambda r: r["alpha"])
            ok = [r["converged"] and abs(r["Q"] + 2) < 0.02 for r in rs]
            al = np.array([r["alpha"] for r in rs])
            Er = np.array([r["E_relaxed_int"] for r in rs])
            Es = np.array([r["E_sup_int"] for r in rs])
            row = dict(d=dd, n=len(rs), n_valid=int(sum(ok)),
                       invalid_alpha_deg=[round(math.degrees(r["alpha"]), 1) for r, o in zip(rs, ok) if not o],
                       E_relaxed=[float(x) if o else None for x, o in zip(Er, ok)],
                       E_superposition=Es.tolist(), alpha_deg=np.degrees(al).tolist())
            if all(ok) and len(rs) >= 8:
                row["relaxed"] = harmonics(al, Er)
                row["superposition"] = harmonics(al, Es)
            valid = [(a, e) for a, e, o in zip(al, Er, ok) if o]
            if valid:
                amin = min(valid, key=lambda t: t[1])
                amax = max(valid, key=lambda t: t[1])
                row["relaxed_min"] = dict(alpha_deg=float(np.degrees(amin[0])), E=float(amin[1]))
                row["relaxed_max"] = dict(alpha_deg=float(np.degrees(amax[0])), E=float(amax[1]))
            rows.append(row)
            h = row.get("relaxed")
            hs = row.get("superposition")
            print(f"d = {dd:5.2f}  valid {row['n_valid']}/{row['n']}  "
                  f"min {row.get('relaxed_min', {}).get('E', float('nan')):+9.4f} at "
                  f"{row.get('relaxed_min', {}).get('alpha_deg', float('nan')):5.1f}°  "
                  f"max {row.get('relaxed_max', {}).get('E', float('nan')):+9.4f} at "
                  f"{row.get('relaxed_max', {}).get('alpha_deg', float('nan')):5.1f}°"
                  + (f"   relaxed B1 {h['B1']:.4f} α1 {h['alpha1_deg']:.1f}° c0 {h['c0']:+.4f}"
                     f" | superposition B1 {hs['B1']:.4f} α1 {hs['alpha1_deg']:.1f}°" if h else ""))
        out[key] = rows
    with open(os.path.join(DATA, "pair_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
