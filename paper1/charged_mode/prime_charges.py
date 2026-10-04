"""
Prime charges (the user's question, 2026-10-04).  In our axial solver the ring charge is multiplicative,
Q = m × n: m windings around the symmetry axis (n = R₃(mφ) u) times the degree n of the meridional map
u: half plane → S² (the windings around the core).  A ring of prime charge p can only be p × 1 or 1 × p;
a composite charge can share its windings (4 = 2 × 2, 6 = 2 × 3 = 3 × 2).  Does a balanced
factorization cost less than a one-directional one?  If it does, prime charges are "frustrated".

Sector: the embedded single-link sector of the flag model = the CP¹ Faddeev–Skyrme model of
run_hopf_pair.py (f = g = μ = 1), on the half plane with the reflection symmetry z → −z.
Seed for (m, n): the rational map W = iⁿ Z₁ᵐ/Z₀ⁿ (Sutcliffe), at two sizes; damped Newton relaxation
(energy decreasing).  On a grid the charge is not protected: a relaxation that loses its charge is
reported and discarded.  (A first try composed the degree-1 seed with W → Wⁿ; with its wrong behaviour
on the axis for m ≥ 2 most relaxations unwound to the vacuum — see FLAG_PAIR_R6.md §5.20.)
Prediction written down before the computation (from Q = 2: A₂,₁ = 510.6 ≪ A₁,₂ = 603.7): the pure
azimuthal winding (Q, 1) is the cheapest for every Q, i.e. no prime frustration in this sector.

Usage: python prime_charges.py [Qmax] [ne_r ne_z a] [procs]     (default 8, 32 48 5, 4)
       python prime_charges.py check                               (key cases on 40 × 64, a = 7)
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SCRATCH = os.environ.get("FLAGPAIR_SCRATCH", "/tmp/flagpair")


def seed(G, n):
    from run_hopf_pair import initial
    from hopfion_axisym import U1, U2, U3
    U = initial(G).copy()
    u1, u2, u3 = U[:, U1], U[:, U2], U[:, U3]
    th = np.arccos(np.clip(u3, -1.0, 1.0))
    ph = np.arctan2(u2, u1)
    th2 = 2.0 * np.arctan(np.tan(0.5 * th) ** n)
    ph2 = n * ph
    U[:, U1], U[:, U2], U[:, U3] = np.sin(th2) * np.cos(ph2), np.sin(th2) * np.sin(ph2), np.cos(th2)
    return U


def seed_rational(G, m, n, lam):
    """The rational-map seed W = iⁿ Z₁ᵐ / Z₀ⁿ (Sutcliffe's ansatz; Q = m·n) on the meridional half plane:
    Z₁ = 2ρ̃/(1 + r̃²) (its e^{iφ} is the model's azimuthal winding), Z₀ = (2z̃ + i(r̃² − 1))/(1 + r̃²),
    x̃ = x/λ.  |W| ~ ρᵐ on the axis (smooth for winding m), W = ∞ on the core circle ρ = λ, z = 0, where it
    winds n times; W → 0 at infinity.  The factor iⁿ makes W(ρ, −z) = conj W(ρ, z), the reflection
    symmetry imposed on the half plane."""
    from hopfion_axisym import U1, U2, U3
    U = G.vacuum().copy()
    rho = np.asarray(G.rho_n, float) / lam
    z = np.asarray(G.z_n, float) / lam
    fin = np.isfinite(rho) & np.isfinite(z)
    r2 = np.where(fin, rho ** 2 + z ** 2, 0.0)
    Z1 = np.where(fin, 2 * np.where(fin, rho, 0.0) / (1 + r2), 0.0)
    Z0 = np.where(fin, (2 * np.where(fin, z, 0.0) + 1j * (r2 - 1)) / (1 + r2), 1j)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = (1j ** n) * Z1 ** m / Z0 ** n
        # the massive model decays like e^{−μr}; the rational map only like r^{−m}: damp it outside the ring
        # (changes |W| far away only, not the core or the windings, so not the charge)
        W = W * np.exp(-np.maximum(lam * np.sqrt(r2) - 1.5 * lam, 0.0))
    big = ~np.isfinite(W) | (np.abs(W) > 1e8)
    a2 = np.where(big, 0.0, np.abs(W) ** 2)
    u3 = np.where(big, 1.0, (a2 - 1) / (a2 + 1))
    c = np.where(big, 0.0, 2 * np.where(big, 0.0, W) / (a2 + 1))
    U[:, U1], U[:, U2], U[:, U3] = c.real, c.imag, u3
    return U


def cores(G, U, level=0.5):
    """Local maxima of u₃ (the cores, where u = +e₃) on the nodal grid of the half plane: (ρ, z, u₃)."""
    from hopfion_axisym import U3
    u3 = U[:, U3].reshape(G.nr, G.nz)
    rho = np.asarray(G.rho_n).reshape(G.nr, G.nz)
    zz = np.asarray(G.z_n).reshape(G.nr, G.nz)
    out = []
    for i in range(1, G.nr - 1):
        for j in range(0, G.nz - 1):
            v = u3[i, j]
            if v < level:
                continue
            nb = [u3[i + di, j + dj] for di in (-1, 0, 1) for dj in (-1, 0, 1)
                  if (di or dj) and 0 <= j + dj < G.nz]
            if all(v >= w for w in nb):
                out.append((float(rho[i, j]), float(zz[i, j]), float(v)))
    return out


def relax_mn(args):
    m, n, ne_r, ne_z, a = args[:5]
    lam = args[5] if len(args) > 5 else None
    from run_hopf_pair import freeze_A, model, parts_and_virial
    from hopfion_axisym import Grid
    from newton import NewtonRelaxer
    t0 = time.time()
    kind = "deg1" if lam is None else f"rat{lam:.2f}"
    fn = os.path.join(SCRATCH, f"prime_mn_state_m{m}_n{n}_{kind}_ne{ne_r}x{ne_z}_a{a:g}.npz")
    G = freeze_A(Grid(ne_r, ne_z, p=2, a=a, half=True))
    M = model(G, m=m)
    if os.path.exists(fn):
        U = np.load(fn)["U"]
        conv, iters = bool(np.load(fn)["converged"]), -1
    else:
        R = NewtonRelaxer(M, seed(G, n) if lam is None else seed_rational(G, m, n, lam), verbose=False)
        U, _ = R.run(max_iter=150, tol=1e-9)
        conv, iters = bool(R.converged), len(R.history)
        np.savez_compressed(fn, U=U, converged=conv)
    E = float(M.energy(U, want_grad=False)[0])
    d = M.diagnostics(U)
    c = cores(G, U)
    out = dict(m=m, n=n, Q=m * n, seed=kind, E=E, Q_H=float(d["Q_H"]), deg=float(d["deg_u"]), converged=conv, iterations=iters,
               cores=len(c), core_positions=c, parts=parts_and_virial(M, U), seconds=time.time() - t0)
    print(f"   ({m},{n}) Q = {m * n} [{kind}]: E = {E:9.3f}  Q_H = {d['Q_H']:+.3f}  cores {len(c)} at (ρ, z) "
          f"{[(round(x, 2), round(y, 2)) for x, y, _ in c]}  converged {conv}"
          f" ({iters} Newton steps, {time.time() - t0:.0f} s)", flush=True)
    return out


def factorizations(Q):
    return [(m, Q // m) for m in range(1, Q + 1) if Q % m == 0]


def size0(m, n):
    """Seed size from the relaxed rings already known (core radius 0.93, 1.36, 1.81 for m = 1, 2, 3)."""
    return 0.93 * m ** 0.55 * n ** 0.15


def main(Qmax=8, ne_r=32, ne_z=48, a=5.0, procs=4):
    from multiprocessing import Pool
    os.makedirs(SCRATCH, exist_ok=True)
    pairs = [(m, Q // m) for Q in range(1, Qmax + 1) for m in range(1, Q + 1) if Q % m == 0]
    jobs = [(m, n, ne_r, ne_z, a, round(size0(m, n) * f, 2)) for (m, n) in pairs for f in (1.0, 1.35)]
    jobs = sorted(jobs, key=lambda j: -j[0] * j[1])          # big ones first
    t0 = time.time()
    with Pool(procs) as pool:
        res = pool.map(relax_mn, jobs, chunksize=1)
    ok = lambda r: r["converged"] and abs(abs(r["Q_H"]) - r["Q"]) < 0.05 * r["Q"]
    per = {}
    for r in res:
        k = (r["m"], r["n"])
        if ok(r) and (k not in per or r["E"] < per[k]["E"]):
            per[k] = r
    E1 = per[(1, 1)]["E"]
    print(f"\n{'Q':>2} {'(m,n)':>7} {'E':>9} {'E/Q':>8} {'E/E1':>7}  cores  (best charge-preserving of two seed sizes)")
    for (m, n) in pairs:
        r = per.get((m, n))
        if r is None:
            print(f"{m * n:>2} {str((m, n)):>7}   lost its charge for both seeds (unwound on the grid)")
            continue
        print(f"{r['Q']:>2} {str((m, n)):>7} {r['E']:9.2f} {r['E'] / r['Q']:8.2f} {r['E'] / E1:7.3f}  {r['cores']:>5}")
    best = {}
    for (m, n), r in per.items():
        if r["Q"] not in best or r["E"] < best[r["Q"]]["E"]:
            best[r["Q"]] = r
    print("\ncheapest factorization per charge (among those that kept their charge):")
    for Q in sorted(best):
        b = best[Q]
        prime = Q > 1 and all(Q % k for k in range(2, Q))
        missing = [p for p in pairs if p[0] * p[1] == Q and p not in per]
        print(f"   Q = {Q} ({'prime' if prime else 'composite' if Q > 1 else 'unit'}): ({b['m']},{b['n']})  E = {b['E']:.2f}"
              f"  E/Q = {b['E'] / Q:.2f}  E/Q^0.75 = {b['E'] / Q ** 0.75:.2f}" + (f"   (not available: {missing})" if missing else ""))
    json.dump(dict(ne_r=ne_r, ne_z=ne_z, a=a, results=res, per_factorization={f"{m}x{n}": r for (m, n), r in per.items()},
                   best={str(k): v for k, v in best.items()}, seconds=time.time() - t0),
              open(os.path.join(DATA, f"prime_charges_ne{ne_r}x{ne_z}_a{a:g}.json"), "w"), indent=1)
    print(f"total {time.time() - t0:.0f} s")


def cmd_check(ne_r=40, ne_z=64, a=7.0, procs=4, ref="data/prime_charges_ne32x48_a5.json"):
    """Grid check of the key comparisons: the cheapest factorizations and the pure azimuthal rings, re-relaxed
    on a larger, finer grid (seed size = the best one of the reference run)."""
    from multiprocessing import Pool
    R = json.load(open(os.path.join(HERE, ref)))["per_factorization"]
    keys = ["1x1", "2x1", "3x1", "4x1", "2x2", "5x1", "6x1", "3x2", "7x1", "8x1", "4x2"]
    jobs = []
    for k in keys:
        r = R[k]
        lam = float(r["seed"][3:])
        jobs.append((r["m"], r["n"], ne_r, ne_z, a, lam))
    t0 = time.time()
    with Pool(procs) as pool:
        res = pool.map(relax_mn, sorted(jobs, key=lambda j: -j[0] * j[1]), chunksize=1)
    by = {f"{r['m']}x{r['n']}": r for r in res}
    print(f"\n{'(m,n)':>6} {'Q':>2} {'E 32x48,a5':>11} {'E 40x64,a7':>11} {'change':>8}  Q_H(new)")
    for k in keys:
        o, nw = R[k], by[k]
        print(f"{k:>6} {o['Q']:>2} {o['E']:11.2f} {nw['E']:11.2f} {100 * (nw['E'] / o['E'] - 1):+7.2f}%  {nw['Q_H']:+.3f}")
    json.dump(dict(ne_r=ne_r, ne_z=ne_z, a=a, results=res, seconds=time.time() - t0),
              open(os.path.join(DATA, f"prime_charges_check_ne{ne_r}x{ne_z}_a{a:g}.json"), "w"), indent=1)
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "check":
        cmd_check()
        sys.exit(0)
    Qmax = int(args[0]) if args else 8
    ne = (int(args[1]), int(args[2]), float(args[3])) if len(args) >= 4 else (32, 48, 5.0)
    procs = int(args[4]) if len(args) >= 5 else 4
    main(Qmax, *ne, procs=procs)
