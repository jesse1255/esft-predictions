"""
Non-axial stability of the axial rings of prime_charges.py (the user's "next step", 2026-10-04).
Prediction written before the computation: the rings that prime charges are forced into (Q × 1) are
unstable to bending or twisting, while the balanced composite rings (2 × 2, 3 × 2, 4 × 2, …) are more
stable.

Lowest eigenvalues of the Hessian, sector by sector in the azimuthal number k (perturbations ∝ cos kφ,
sin kφ in the frame co-rotating with the winding m), generalised with the L² mass of the perturbation;
the vacuum continuum starts at μ² = 1.  k = 1 holds the 4 zero modes (2 translations, 2 tilts); on the
grid they split by about ±0.05–0.1, so only an eigenvalue clearly below that counts there.  Matter only
(e = 0: the gauge field decouples), axis regularity for winding m (nonaxisym.SectorHessian, axis_k = m).
A negative eigenvalue in sector k means the axial ring lowers its energy by a k-fold deformation of its
shape (k = 2: oval, k = 3: three-lobed, …): it is not a minimum and will bend, twist or knot.

Usage: python prime_nonaxial.py [mxn ...] [--procs P] [--kmax K]
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SCRATCH = os.environ.get("FLAGPAIR_SCRATCH", "/tmp/flagpair")
REF = os.path.join(DATA, "prime_charges_ne32x48_a5.json")


def sectors(args):
    key, kmax = args
    import scipy.sparse.linalg as spla
    from run_hopf_pair import freeze_A, mirror_full
    from hopfion_axisym import Grid, Model
    from nonaxisym import SectorHessian
    d = json.load(open(REF))
    r = d["per_factorization"][key]
    m, n = r["m"], r["n"]
    Gh = freeze_A(Grid(d["ne_r"], d["ne_z"], p=2, a=d["a"], half=True))
    Uh = np.load(os.path.join(SCRATCH, f"prime_mn_state_m{m}_n{n}_{r['seed']}_ne{d['ne_r']}x{d['ne_z']}_a{d['a']:g}.npz"))["U"]
    Gf, Uf = mirror_full(Gh, Uh)
    M = Model(Gf, e=0.0, N=0.0, m=m, mu=1.0, kappa=1.0)
    out = dict(key=key, m=m, n=n, Q=m * n, E=r["E"], sectors={})
    for k in range(1, kmax + 1):
        cache = os.path.join(SCRATCH, f"prime_nonaxial_{key}_k{k}.json")
        if os.path.exists(cache):
            out["sectors"][k] = json.load(open(cache))
            print(f"   {key:5s} (Q = {m * n:2d}) k = {k}: lowest {np.round(out['sectors'][k]['lowest'], 4).tolist()}  (cached)", flush=True)
            continue
        t0 = time.time()
        S = SectorHessian(M, Uf, k, axis_k=m, matter_only=True)
        g0 = S.check_stationary()
        H, asym = S.assemble()
        Mm = S.mass(1.0)
        lu = spla.splu((H + 1.0 * Mm).tocsc())
        op = spla.LinearOperator(H.shape, matvec=lu.solve, dtype=float)
        nev = 2      # the lowest (cos, sin) pair: with σ = −1 a negative mode comes before the zero modes.
                     # Asking for more than lie below μ² = 1 reaches into the degenerate cluster of the
                     # far-field nodes and stalls ARPACK (20 min for one sector with nev = 4)
        note = ""
        try:
            vals = np.sort(spla.eigsh(H, k=nev, M=Mm, sigma=-1.0, which="LM", OPinv=op, tol=1e-7, maxiter=400,
                                      return_eigenvectors=False))
        except spla.ArpackNoConvergence as e:
            # an isolated eigenvalue near σ = −1 (a negative or bound mode) converges in a few dozen steps;
            # no convergence in 400 means the lowest pair sits in the degenerate μ² = 1 cluster: no bound mode
            vals = np.sort(np.asarray(e.eigenvalues)) if len(e.eigenvalues) else np.array([np.nan])
            note = "no convergence in 400 ARPACK steps: lowest pair in the μ² = 1 cluster (no bound mode)"
        out["sectors"][k] = dict(lowest=[float(v) for v in vals], stationarity=float(g0), asym=float(asym),
                                 seconds=time.time() - t0, note=note)
        json.dump(out["sectors"][k], open(cache, "w"))
        print(f"   {key:5s} (Q = {m * n:2d}) k = {k}: lowest {np.round(vals, 4).tolist()}  ({time.time() - t0:.0f} s) {note}", flush=True)
    return out


def verdict(o):
    neg = {}
    for k, s in o["sectors"].items():
        vals = s["lowest"]
        if int(k) == 1:
            bad = [v for v in s["lowest"] if v < -0.1]          # the 4 zero modes split by ±0.05–0.1 on the grid
        else:
            bad = [v for v in vals if v < -0.02]
        if bad:
            neg[int(k)] = min(bad)
    return neg


def main(keys, procs=3, kmax=4):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(procs) as pool:
        res = pool.map(sectors, [(k, kmax) for k in keys], chunksize=1)
    print(f"\n{'ring':>6} {'Q':>3} {'prime':>6}   lowest eigenvalue per sector k = 1 … {kmax}      verdict")
    summary = {}
    for o in sorted(res, key=lambda o: (o["Q"], -o["m"])):
        Q = o["Q"]
        prime = Q > 1 and all(Q % j for j in range(2, Q))
        low = [np.nanmin(o["sectors"][k]["lowest"]) if np.isfinite(o["sectors"][k]["lowest"]).any() else 1.0
               for k in range(1, kmax + 1)]
        neg = verdict(o)
        v = "unstable in k = " + ", ".join(f"{k} ({neg[k]:+.3f})" for k in sorted(neg)) if neg else "stable (k ≤ %d)" % kmax
        print(f"{o['key']:>6} {Q:>3} {'yes' if prime else '':>6}   " + "  ".join(f"{x:+.3f}" if x is not None else "  —   " for x in low) + f"   {v}")
        summary[o["key"]] = dict(Q=Q, prime=prime, negative=neg, low=low)
    json.dump(dict(results=res, summary=summary, seconds=time.time() - t0), open(os.path.join(DATA, "prime_nonaxial.json"), "w"), indent=1)
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    args = sys.argv[1:]
    procs = int(args[args.index("--procs") + 1]) if "--procs" in args else 3
    kmax = int(args[args.index("--kmax") + 1]) if "--kmax" in args else 4
    keys = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
    main(keys or ["3x1", "2x2", "4x1", "5x1", "3x2", "6x1", "7x1", "4x2", "8x1"], procs=procs, kmax=kmax)
