"""
Non-axisymmetric stability of the charged Hopfion: soliton spectrum (gauge
field relaxed, Schur complement) in the azimuthal sectors k = 0..4.

Usage: python run_nonaxisym.py ne [k ...]      (default k = 0..4)
Writes data/nonaxisym_ne{ne}.json.
"""

import json
import os
import sys
import time

import numpy as np

from hopfion_axisym import Model
from nonaxisym import SectorHessian
from run_checks import load_state, mirror_to_full

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main(ne, ks=(0, 1, 2, 3, 4), nev=6, state=None, e=0.3, N=1.0, mu=1.0, tag=""):
    Gh, d = load_state(ne)
    V = d["V_full"] if state is None else np.load(state)["V"]
    Gf, Uf = mirror_to_full(Gh, V)
    M = Model(Gf, e=e, N=N, mu=mu, kappa=1.0)
    out = dict(ne=ne, e=e, N=N, mu=mu, sectors=[],
               description="Lowest eigenvalues of the soliton Hessian with the gauge field "
                           "relaxed (Schur complement), generalised with the L² mass of the "
                           "matter perturbation; the vacuum continuum starts at μ² = 1. "
                           "Local part of the fixed-charge term only (the rest is PSD).")
    for k in ks:
        t0 = time.time()
        S = SectorHessian(M, Uf, k)
        g0 = S.check_stationary()
        H, asym = S.assemble()
        t1 = time.time()
        vals, vecs, Mm = S.lowest_schur(H, nev=nev, sigma=-1e-2)
        row = dict(k=k, nfree=S.nfree, M_phi=S.M, grad_at_background=g0, asym=asym,
                   eigenvalues=vals.tolist(), t_assemble=t1 - t0, t_eig=time.time() - t1)
        if k == 1:
            gens = S.generators()
            ov = []
            for i in range(nev):
                x = vecs[:, i]
                ov.append({n: float(abs(x @ (Mm @ g)) / np.sqrt((x @ (Mm @ x)) * (g @ (Mm @ g))))
                           for n, g in gens.items()})
            row["overlaps"] = ov
        out["sectors"].append(row)
        print(f"ne={ne} k={k}: " + " ".join(f"{v:.4e}" for v in vals)
              + f"  (assemble {t1 - t0:.0f}s, eig {time.time() - t1:.0f}s)", flush=True)
        with open(os.path.join(DATA, f"nonaxisym{tag}_ne{ne}.json"), "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    ne = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    ks = tuple(int(k) for k in sys.argv[2:]) or (0, 1, 2, 3, 4)
    main(ne, ks=ks)
