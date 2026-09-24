"""
Follow the genuinely-F₂ branch that bifurcates from the embedded Hopfion.

Start from  U_emb · Cay(ε·ξ)  with ξ a critical normal mode, relax the full F₂
problem (all six tangent coordinates) at a point where the embedded state is
unstable, then continue the solution in one coupling (λ = r_13/r_12 = r_23/r_12,
or the three-cycle coefficient κ₃) in both directions.  At the linear threshold p*
  * continuous (supercritical):  D → 0 as p → p* and E_F − E_emb ∝ −(p − p*)²;
  * first order (subcritical):   the branch survives beyond p* up to a fold, with
    E_F < E_emb on part of it (coexistence and hysteresis).

    D = ∫ (1 − |Z_33|²) d³x   (0 on the embedded state),   Q = Hopf degree.

Usage:
  python run_flag_branch.py ne l3 --param k3 --start 0.3 --lam 1.0 --mode data/flag_mode_k3_ne16_l0.npy
  python run_flag_branch.py ne l3 --param lam --start 0.02 --k3 2 --mode data/flag_mode_k32_ne16_l0.npy
Writes data/flag_branch_{param}_ne{ne}_l{l3}[_tag].json.
"""

import argparse
import json
import os
import time

import numpy as np
import jax.numpy as jnp

from hopfion_axisym import Grid
from flag_axisym import FlagModel, FlagProblem, newton_relax, inertia, NT
from flag_path import degree_parts

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def normal_amplitude(G, U):
    Uq = G.Pv @ U[:, 2, 2]
    return float(G.W @ (1.0 - np.abs(Uq) ** 2))


def embed_x(prob, x6):
    x = np.zeros(prob.nfree)
    for t in range(NT):
        sel = prob.dof[:, t] >= 0
        x[prob.dof[sel, t]] = x6[sel, t]
    return jnp.asarray(x)


def main(ne, l3, param, start, mode, lam=1.0, k3=0.0, eps=0.3, step=0.1, pmin=0.0, pmax=3.0,
         tag=""):
    G = Grid(ne, ne, p=2, a=3.0, half=False)
    Uemb = np.load(os.path.join(DATA, f"flag_embedded_ne{ne}.npy"))
    x6 = np.load(mode)
    fm = FlagModel(G, L=(0, 1, l3), kappa3=1.0)        # κ₃ ≠ 0 so the term is built
    E_emb = float(fm.energy_U(jnp.asarray(Uemb), None, 0.0))
    prob = FlagProblem(fm, Uemb, types=range(NT))

    def couple(p):
        lm, kk = (p, k3) if param == "lam" else (lam, p)
        prob.set_couplings(r=(2.0, 2.0 * lm, 2.0 * lm), kappa3=kk)

    out = dict(ne=ne, l3=l3, param=param, start=start, lam=lam, k3=k3, eps=eps,
               E_embedded=E_emb, mode=os.path.basename(mode), branch=[])
    fname = os.path.join(DATA, f"flag_branch_{param}_ne{ne}_l{l3}{tag}.json")
    print(f"ne={ne} l3={l3} param={param}: E_emb = {E_emb:.8f}", flush=True)

    def solve(U0, p):
        couple(p)
        t0 = time.time()
        U, E, hist, conv = newton_relax(fm, U0, types=range(NT), max_iter=120, verbose=False,
                                        prob=prob)
        prob.set_base(U)
        H = prob.hessian()
        Q, Qd, Q3 = degree_parts(fm, U)
        row = dict(p=float(p), E=E, dE=E - E_emb, D=normal_amplitude(G, U), Q=Q, Q_diag=Qd,
                   Q_3cycle=Q3, conv=conv, it=len(hist), gscaled=hist[-1]["gscaled"],
                   inertia=inertia(H), parts=fm.parts(U, prob.r, prob.k3), t=time.time() - t0)
        print(f"  {param}={p:.4f}: E={E:.8f} ΔE={E - E_emb:+.4e} D={row['D']:.4e} Q={Q:+.4f} "
              f"(3-cycle {Q3:+.4f}) conv={conv} it={row['it']} inertia={row['inertia']} "
              f"({row['t']:.0f}s)", flush=True)
        return U, row

    couple(start)
    Ustart = np.asarray(prob.U_of(eps * embed_x(prob, x6)))
    U, row = solve(Ustart, start)
    out["branch"].append(row)
    Ubase = U
    np.save(os.path.join(DATA, f"flag_F2_{param}_ne{ne}_l{l3}{tag}_{start:g}.npy"), U)

    for direction in (+1, -1):
        U, p, h = Ubase, start, step
        while h >= step / 16:
            p_new = round(p + direction * h, 6)
            if p_new < pmin or p_new > pmax:
                break
            Un, row = solve(U, p_new)
            out["branch"].append(row)
            with open(fname, "w") as fh:
                json.dump(out, fh, indent=1)
            if row["E"] < 1.0:                     # unwound to the vacuum on the lattice
                break
            if row["D"] < 1e-6 or not row["conv"]:
                h /= 2.0                           # the branch ended between p and p_new
                continue
            U, p = Un, p_new
    out["branch"].sort(key=lambda r: r["p"])
    with open(fname, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ne", type=int)
    ap.add_argument("l3", type=int)
    ap.add_argument("--param", choices=("lam", "k3"), required=True)
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--k3", type=float, default=0.0)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--pmin", type=float, default=0.0)
    ap.add_argument("--pmax", type=float, default=3.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.ne, a.l3, a.param, a.start, a.mode, a.lam, a.k3, a.eps, a.step, a.pmin, a.pmax,
         a.tag)
