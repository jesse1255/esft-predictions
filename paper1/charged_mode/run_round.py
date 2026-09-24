"""
Main computation of this round, one grid at a time.

For every grid:
  1. h(r) family with complete EM relaxation (reproduces the 2026-09-08 note);
  2. release both tangential components of u on the meridional plane and
     relax u, A_ρ, A_z, A_φ jointly (β re-solved at every trial point) with a
     damped Newton method;
  3. diagnostics for both: energy split, I_eff, ω, charge moments, form
     factors, magnetic moment, Hopf degree, the note's shape probes;
  4. gauge check for the relaxed state: re-solve A at fixed u with only the
     1e-8 regulariser (the gradient part of A free, as in the note) and
     compare with the Coulomb-gauge energy.

Usage:  python run_round.py 32 48 64 96      (element counts per direction)
Writes data/round_ne{ne}.json and (not committed) data/state_ne{ne}.npz.
"""

import json
import os
import sys
import time

import numpy as np

from hopfion_axisym import Grid, Model, hopf_ansatz, probe_field
from hansatz import HProfile, optimize_h
from newton import NewtonRelaxer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PARAMS = dict(e=0.3, N=1.0, m=1, f=1.0, g=1.0, mu=1.0)
PROBES = [(ell, R) for ell in (0, 2, 4) for R in (0.7, 1.4, 2.8)]


def probes(M, V):
    _, g = M.energy(V)
    return {f"l{ell}_R{R}": float(np.sum(g * probe_field(M.G, V, ell, R)))
            for ell, R in PROBES}


def contributions(M, V):
    """Energy with A = 0 and with A_φ only, at fixed u (fixed-u solves)."""
    U0 = V.copy()
    U0[:, 3:] = 0.0
    Uaz = V.copy()
    Uaz[:, 3:5] = 0.0
    E0 = M.energy(U0, want_grad=False)[0]
    Eaz = M.energy(Uaz, want_grad=False)[0]
    E = M.energy(V, want_grad=False)[0]
    return dict(E_A0=E0, dE_az=Eaz - E0, dE_mer=E - Eaz)


def main(ne_list, a=3.0, p=2, ncoef=60):
    os.makedirs(DATA, exist_ok=True)
    cf_prev, prof_prev = None, None
    rr = np.linspace(0.0, 8.0, 33)
    for ne in ne_list:
        t0 = time.time()
        print(f"=== grid ne = {ne} (p = {p}, a = {a})", flush=True)
        G = Grid(ne, ne, p=p, a=a, half=True)
        prof = HProfile(G, ncoef=ncoef)
        if cf_prev is None:
            cf0 = prof.fit(lambda r: np.pi * np.exp(-r / 1.1))
        else:
            cf0 = prof.fit(lambda r: prof_prev.h_of_r(cf_prev, r))
        cf, Vh, Mh, info_h = optimize_h(G, prof, cf0, **{k: PARAMS[k] for k in ("e", "N")})
        t_h = time.time() - t0
        print(f"  h-family: E = {info_h['E']:.10f} ({info_h['nit']} it, {t_h:.0f}s)", flush=True)
        res_h = dict(opt=info_h, breakdown=Mh.breakdown(Vh), diag=Mh.diagnostics(Vh),
                     contributions=contributions(Mh, Vh), probes=probes(Mh, Vh),
                     h_profile=dict(r=rr.tolist(), h=prof.h_of_r(cf, rr).tolist()))

        # ---- full relaxation of u and A --------------------------------
        t1 = time.time()
        M = Model(G, e=PARAMS["e"], N=PARAMS["N"], kappa=1.0)
        U_h, _ = hopf_ansatz(G, prof.h_nodes(cf))
        V1 = M.solve_A_fixed_u(U_h)
        E_start = M.energy(V1, want_grad=False)[0]
        R = NewtonRelaxer(M, V1)
        V, E = R.run(max_iter=60, tol=1e-10)
        t_f = time.time() - t1
        print(f"  full relaxation: E = {E:.10f}  ({len(R.history)} Newton it, {t_f:.0f}s)", flush=True)
        # gauge check: A re-solved at fixed final u, gradient part free
        Mg = Model(G, e=PARAMS["e"], N=PARAMS["N"], kappa=1e-8)
        Vg = Mg.solve_A_fixed_u(V)
        E_gauge = Mg.energy(Vg, want_grad=False)[0]
        res_f = dict(E=E, E_start_coulomb_gauge_fixed_u=E_start, converged=R.converged,
                     newton=R.history, breakdown=M.breakdown(V), diag=M.diagnostics(V),
                     probes=probes(M, V),
                     gauge_check=dict(E_fixed_u_kappa_1e8=E_gauge, diff=E_gauge - E),
                     contributions_fixed_u=contributions(Mg, Vg))
        out = dict(grid=dict(ne=ne, p=p, a=a, half=True, nodes=G.nn), params=PARAMS,
                   h_family=res_h, full=res_f,
                   delta=dict(E=E - info_h["E"],
                              rms=res_f["diag"]["rms_charge"] - res_h["diag"]["rms_charge"],
                              Dzz=res_f["diag"]["Dzz_charge"] - res_h["diag"]["Dzz_charge"],
                              mu_volume=res_f["diag"]["mu_volume"] - res_h["diag"]["mu_volume"]),
                   seconds=time.time() - t0)
        with open(os.path.join(DATA, f"round_ne{ne}.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        np.savez_compressed(os.path.join(DATA, f"state_ne{ne}.npz"), V_full=V, V_h=Vh, cf=cf,
                            ne=ne, a=a, p=p)
        print(f"  ΔE = {E - info_h['E']:.6f}, rms {res_h['diag']['rms_charge']:.6f} -> "
              f"{res_f['diag']['rms_charge']:.6f}, gauge diff {E_gauge - E:.2e}", flush=True)
        cf_prev, prof_prev = cf, prof


if __name__ == "__main__":
    ne_list = [int(x) for x in sys.argv[1:]] or [32, 48, 64]
    main(ne_list)
