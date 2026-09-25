"""
Look-elsewhere check of the Paper 7 quark-mass formula (2026-09-25).

Paper 7 writes m(l) = Λ ε^{n(l)} with Λ = ħc/(0.003 fm) = 65.78 GeV and
ε = √σ/Λ, √σ = 459 MeV, and builds the exponents n from a colour ladder plus
a "Fibonacci" correction F_k/(N_c D_k) with F = 2, 5, 7 and D = 5, 3, 5.

For fixed Λ and ε, the exponent a mass needs is n* = ln(m/Λ)/ln ε.  All of
Paper 7's exponents are fractions whose denominators divide 45 (1, 3, 5, 9,
15), so the natural baseline is: round n* to the nearest multiple of 1/45.
This script compares the two, and gives the deviation that rounding produces
for arbitrary masses.  Masses are the ones quoted in Paper 7 (PDG 2024).

Usage: python paper7_grid_check.py   → paper7_grid_check.json
"""

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

L = 197.327 / 0.003            # MeV
eps = 459.0 / L
lne = math.log(eps)
quarks = {"u,d": (3.45, 2.0), "s": (93.4, 4 / 3), "c": (1270.0, 4 / 5), "b": (4180.0, 5 / 9),
          "t": (172690.0, -1 / 5)}


def dev(n, nstar):
    return abs(math.exp((n - nstar) * lne) - 1)


def main():
    rows = []
    for q, (m, n) in quarks.items():
        nstar = math.log(m / L) / lne
        g = round(nstar * 45) / 45
        rows.append(dict(quark=q, n_needed=nstar, n_paper7=n, dev_paper7=dev(n, nstar),
                         n_nearest_k_over_45=g, dev_nearest=dev(g, nstar),
                         paper7_equals_nearest=abs(n - g) < 1e-12))
        print(f"{q:4s} n* = {nstar:+.4f}   Paper 7: {n:+.4f} ({100 * dev(n, nstar):.1f}%)   "
              f"nearest k/45: {g:+.4f} ({100 * dev(g, nstar):.1f}%)   same: {abs(n - g) < 1e-12}")
    rng = np.random.default_rng(1)
    t = rng.uniform(-1, 3, 200000)
    d = np.abs(np.exp(np.abs(t - np.round(t * 45) / 45) * (-lne)) - 1)
    base = dict(median=float(np.median(d)), q90=float(np.quantile(d, 0.9)), max=float(d.max()))
    print(f"arbitrary masses rounded to the 1/45 grid: median {100 * base['median']:.2f}%, "
          f"90% {100 * base['q90']:.2f}%, max {100 * base['max']:.2f}%")
    F = [2, 5, 7]
    print(f"Fibonacci terms that enter the masses: {F}; ratios {F[1] / F[0]}, {F[2] / F[1]} (golden ratio 1.618)")
    out = dict(description=__doc__, Lambda_MeV=L, epsilon=eps, rows=rows, grid_baseline=base,
               fibonacci_terms_used=F)
    with open(os.path.join(HERE, "paper7_grid_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
