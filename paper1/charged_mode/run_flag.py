"""
A₂ flag-manifold test of the embedded CP¹ Hopfion.

1. Relax the embedded Q = 1 Hopfion (Z_3 = e_3 fixed) in the S₃-symmetric
   flag model with r = 2, κ = 2, m² = 1 (= Faddeev–Skyrme with f = g = μ = 1).
2. Normal Hessian (directions that rotate Z_3 into span(Z_1, Z_2)) for the
   axial sectors L = (0, 1, l_3) and the coupling ratio λ = r_13/r_12 = r_23/r_12.
   The embedded state does not depend on λ or l_3, and H(λ) = H0 + λ H1 exactly,
   with H1 ≻ 0 (gradient energy of the normal components).  Hence the lowest
   normal eigenvalue is monotone in λ and the linear threshold is
       λ_c = −θ_min,   H0 v = θ H1 v,
   obtained by shift-invert on the pencil and checked with LDLᵀ inertia counts.
3. The critical modes are saved for run_flag_branch.py, which follows the
   genuinely-F₂ branch.

Usage: python run_flag.py ne [--m2 1.0] [--kappa3 0] [--l3 -2,-1,0,1,2,3]
Writes data/flag[_m2X][_k3Y]_ne{ne}.json (+ .npy states and critical modes, not committed).
"""

import json
import os
import sys
import time

import numpy as np
import jax.numpy as jnp

from hopfion_axisym import Grid
from hansatz import HProfile
from flag_axisym import (FlagModel, FlagProblem, embedded_hopfion, newton_relax, inertia,
                         critical_ratio, NT)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
L3_LIST = (-2, -1, 0, 1, 2, 3)
INERTIA_LAMBDAS = (0.1, 0.25, 0.5, 0.75, 1.0)


def embedded_state(ne, m2=1.0, kappa3=0.0):
    G = Grid(ne, ne, p=2, a=3.0, half=False)
    cf = np.load(os.path.join(DATA, "state_ne32.npz"))["cf"]
    U0 = embedded_hopfion(G, HProfile(G).h_nodes(cf))
    fm = FlagModel(G, L=(0, 1, 2), m2=(m2, m2, m2), kappa3=kappa3)
    U, E, hist, conv = newton_relax(fm, U0, types=(0, 1), max_iter=60, verbose=False)
    return G, U, E, conv, hist


def normal_scan(G, U, l3, nev=4, m2=1.0, kappa3=0.0):
    prob = FlagProblem(FlagModel(G, L=(0, 1, l3), m2=(m2, m2, m2), kappa3=kappa3), U,
                       types=(2, 3, 4, 5))
    Hs = {}
    for lam in (0.0, 1.0):
        prob.set_couplings((2.0, 2.0 * lam, 2.0 * lam))
        g = np.asarray(prob.grad(jnp.zeros(prob.nfree)))
        Hs[lam] = prob.hessian()
    H0, H1 = Hs[0.0], (Hs[1.0] - Hs[0.0]).tocsr()
    res = dict(l3=l3, nfree=prob.nfree, normal_grad_max=float(np.abs(g).max()),
               H1_inertia=inertia(H1),
               inertia={f"{lam:g}": inertia(H0 + lam * H1) for lam in INERTIA_LAMBDAS})
    lam_hi = 1.5
    while inertia(H0 + lam_hi * H1)[1] > 0:
        lam_hi *= 2.0
    theta, vecs = critical_ratio(H0, H1, lam_hi, nev)
    lam_c = float(-theta[0])
    res.update(lam_hi=lam_hi, theta=theta.tolist(), lambda_c=lam_c)
    if lam_c > 0:
        res["inertia_below"] = inertia(H0 + lam_c * (1 - 1e-3) * H1)
        res["inertia_above"] = inertia(H0 + lam_c * (1 + 1e-3) * H1)
    if lam_c < 1:
        vals, _ = prob.lowest((H0 + H1).tocsr(), nev=nev, sigma=-5e-2)
        res["lowest_at_S3"] = vals.tolist()
    x6 = np.zeros((G.nn, NT))
    for t in (2, 3, 4, 5):
        sel = prob.dof[:, t] >= 0
        x6[sel, t] = vecs[prob.dof[sel, t], 0]
    return res, x6 / np.abs(x6).max()


def main(ne, m2=1.0, l3_list=L3_LIST, kappa3=0.0):
    t0 = time.time()
    tag = ("" if m2 == 1.0 else f"_m2{m2:g}") + ("" if kappa3 == 0.0 else f"_k3{kappa3:g}")
    G, U, E, conv, hist = embedded_state(ne, m2, kappa3)
    fm = FlagModel(G, L=(0, 1, 2), m2=(m2, m2, m2), kappa3=kappa3)
    out = dict(ne=ne, E_embedded=E, embedded_converged=conv, embedded_newton=hist,
               parts=fm.parts(U),
               couplings=dict(r12=2.0, kappa=2.0, kappa3=kappa3, m2=m2,
                              lambda_def="r13 = r23 = 2 λ"),
               sectors=[])
    print(f"ne={ne}: embedded E = {E:.8f} conv={conv} it={len(hist)} "
          f"|g/sqrtH|={hist[-1]['gscaled']:.2e} ({time.time() - t0:.0f}s)", flush=True)
    np.save(os.path.join(DATA, f"flag_embedded{tag}_ne{ne}.npy"), U)
    for l3 in l3_list:
        t1 = time.time()
        res, x6 = normal_scan(G, U, l3, m2=m2, kappa3=kappa3)
        np.save(os.path.join(DATA, f"flag_mode{tag}_ne{ne}_l{l3}.npy"), x6)
        out["sectors"].append(res)
        print(f"  l3={l3:+d}: λ_c = {res['lambda_c']:.6f}  θ = {res['theta'][:4]}  "
              f"inertia(λ=1) = {res['inertia']['1']}  below/above λ_c = "
              f"{res.get('inertia_below')}/{res.get('inertia_above')}  "
              f"S3 lowest = {res.get('lowest_at_S3')}  ({time.time() - t1:.0f}s)", flush=True)
        with open(os.path.join(DATA, f"flag{tag}_ne{ne}.json"), "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ne", type=int, nargs="?", default=24)
    ap.add_argument("--m2", type=float, default=1.0, help="potential m_a² (all three)")
    ap.add_argument("--kappa3", type=float, default=0.0,
                    help="three-cycle quartic coefficient (0: model M2; 2 = κ: full commutator)")
    ap.add_argument("--l3", type=str, default=",".join(str(l) for l in L3_LIST))
    a = ap.parse_args()
    main(a.ne, m2=a.m2, l3_list=tuple(int(x) for x in a.l3.split(",")), kappa3=a.kappa3)
