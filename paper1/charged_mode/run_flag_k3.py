"""
Stability of the embedded Hopfion at the S₃ point (λ = 1) as a function of the
three-cycle quartic coefficient κ₃ (κ₃ = 0: model M2; κ₃ = κ = 2: model M2').

The normal Hessian is linear in κ₃,  H(κ₃) = H_a + κ₃ H_C,  H_C ⪰ 0, so the
embedded state is stable for κ₃ ≥ κ₃* and unstable below.  κ₃* is found by
bisection on the LDLᵀ inertia; the critical mode (null vector at κ₃*) is saved
for run_flag_branch.py.

Usage: python run_flag_k3.py ne [l3 ...]
Writes data/flag_k3_ne{ne}.json and data/flag_mode_k3_ne{ne}_l{l3}.npy.
"""

import json
import os
import sys
import time

import numpy as np
import jax.numpy as jnp

from hopfion_axisym import Grid
from flag_axisym import FlagModel, FlagProblem, inertia, NT

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def threshold(G, U, l3, lam=1.0, tol=1e-6):
    prob = FlagProblem(FlagModel(G, L=(0, 1, l3)), U, types=(2, 3, 4, 5))
    prob.set_couplings(r=(2.0, 2.0 * lam, 2.0 * lam), kappa3=0.0)
    Ha = prob.hessian()
    prob.set_couplings(kappa3=1.0)
    HC = (prob.hessian() - Ha).tocsr()
    # H_C ⪰ 0 analytically (second variation of |C|² where C = 0); it has empty rows
    # far from the core, so no inertia count is taken for it.
    res = dict(l3=l3, lam=lam, nfree=prob.nfree,
               inertia={f"{k:g}": inertia(Ha + k * HC) for k in (0.0, 0.25, 0.5, 1.0, 2.0)})
    lo, hi = 0.0, 2.0
    if inertia(Ha + hi * HC)[1] > 0:
        res["kappa3_star"] = None
        return res, None
    if inertia(Ha)[1] == 0:
        res["kappa3_star"] = 0.0
        return res, None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if inertia(Ha + mid * HC)[1] > 0:
            lo = mid
        else:
            hi = mid
    res["kappa3_star"] = 0.5 * (lo + hi)
    vals, vecs = prob.lowest((Ha + hi * HC).tocsr(), nev=4, sigma=-1e-3)
    res["lowest_at_star"] = vals.tolist()
    x6 = np.zeros((G.nn, NT))
    for t in (2, 3, 4, 5):
        sel = prob.dof[:, t] >= 0
        x6[sel, t] = vecs[prob.dof[sel, t], 0]
    return res, x6 / np.abs(x6).max()


def main(ne, l3_list=(0, -1, -2)):
    G = Grid(ne, ne, p=2, a=3.0, half=False)
    U = np.load(os.path.join(DATA, f"flag_embedded_ne{ne}.npy"))
    out = dict(ne=ne, lam=1.0, couplings=dict(r=2.0, kappa=2.0, m2=1.0), sectors=[])
    for l3 in l3_list:
        t0 = time.time()
        res, x6 = threshold(G, U, l3)
        if x6 is not None:
            np.save(os.path.join(DATA, f"flag_mode_k3_ne{ne}_l{l3}.npy"), x6)
        out["sectors"].append(res)
        print(f"  l3={l3:+d}: κ3* = {res['kappa3_star']}  inertia {res['inertia']}  "
              f"lowest@κ3* {res.get('lowest_at_star')}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        with open(os.path.join(DATA, f"flag_k3_ne{ne}.json"), "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), tuple(int(x) for x in a[1:]) or (0, -1, -2))
