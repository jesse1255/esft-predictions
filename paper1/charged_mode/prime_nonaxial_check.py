"""
Checks of the two anomalies in prime_nonaxial.py (2026-10-04): the k = 1 sectors of 2 × 2 and 4 × 2 show no
zero modes (lowest +0.156, +0.116) and 4 × 2 has an unusually deep k = 3 eigenvalue (−2.66).
1. axial criticality: the k = 0 sector gradient of each state;
2. the analytic translation mode δv = cos φ ∂ρu − (m/ρ) sin φ (e₃ × u) in sector k = 1: its Rayleigh quotient
   should be ≈ 0 for a critical point (control: 5 × 1, 3 × 2);
3. where the deepest k = 3 mode of 4 × 2 lives (weight near the axis, the core, the far field).

Usage: python prime_nonaxial_check.py            → checks 1–3
       python prime_nonaxial_check.py localize   → where every reported negative mode lives
"""

import json
import os

import numpy as np
import scipy.sparse.linalg as spla

from run_hopf_pair import freeze_A, mirror_full
from hopfion_axisym import Grid, Model
from nonaxisym import SectorHessian, CT1, CT2, ST1, ST2, NT

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("FLAGPAIR_SCRATCH", "/tmp/flagpair")
d = json.load(open(os.path.join(HERE, "data", "prime_charges_ne32x48_a5.json")))


def load(key):
    r = d["per_factorization"][key]
    Gh = freeze_A(Grid(d["ne_r"], d["ne_z"], p=2, a=d["a"], half=True))
    Uh = np.load(os.path.join(SCRATCH, f"prime_mn_state_m{r['m']}_n{r['n']}_{r['seed']}_ne{d['ne_r']}x{d['ne_z']}_a{d['a']:g}.npz"))["U"]
    Gf, Uf = mirror_full(Gh, Uh)
    return r, Gf, Uf, Model(Gf, e=0.0, N=0.0, m=r["m"], mu=1.0, kappa=1.0)


def translation_rq(key):
    r, G, U, M = load(key)
    m = r["m"]
    S0 = SectorHessian(M, U, 0, axis_k=m, matter_only=True)
    g0 = S0.check_stationary()
    S = SectorHessian(M, U, 1, axis_k=m, matter_only=True)
    H, _ = S.assemble()
    Mm = S.mass(1.0)
    u = U[:, :3]
    rho = np.asarray(G.rho_n).reshape(G.nr, G.nz)
    z = np.asarray(G.z_n).reshape(G.nr, G.nz)
    ur = np.zeros((G.nr, G.nz, 3))
    uu = u.reshape(G.nr, G.nz, 3)
    fin = np.isfinite(rho[:, 0])
    rr = rho[:, 0]
    for c in range(3):                                  # ∂ρ u on the nodal grid (finite rows only)
        ur[fin, :, c] = np.gradient(uu[fin, :, c], rr[fin], axis=0)
    ur = ur.reshape(-1, 3)
    e3xu = np.stack([-u[:, 1], u[:, 0], np.zeros(len(u))], -1)
    rho_f = np.asarray(G.rho_n)
    with np.errstate(divide="ignore", invalid="ignore"):
        sterm = np.where((rho_f > 0) & np.isfinite(rho_f), m / rho_f, 0.0)[:, None] * e3xu
    X = np.zeros((G.nn, NT))
    X[:, CT1] = np.einsum("ni,ni->n", ur, S.e1)
    X[:, CT2] = np.einsum("ni,ni->n", ur, S.e2)
    X[:, ST1] = -np.einsum("ni,ni->n", sterm, S.e1)
    X[:, ST2] = -np.einsum("ni,ni->n", sterm, S.e2)
    X[~np.isfinite(X)] = 0.0
    x = X.ravel()[S.free_flat]
    rq = float(x @ (H @ x) / (x @ (Mm @ x)))
    lu = spla.splu((H + Mm).tocsc())
    op = spla.LinearOperator(H.shape, matvec=lu.solve, dtype=float)
    vals = np.sort(spla.eigsh(H, k=2, M=Mm, sigma=-1.0, which="LM", OPinv=op, tol=1e-7, return_eigenvectors=False))
    print(f"{key}: k = 0 gradient {g0:.2e};  translation-mode Rayleigh quotient in k = 1: {rq:+.4f};  lowest k = 1 pair {np.round(vals, 4).tolist()}", flush=True)
    return dict(key=key, k0_gradient=float(g0), translation_rq=rq, k1_lowest=[float(v) for v in vals])


def deep_mode(key="4x2", k=3):
    r, G, U, M = load(key)
    S = SectorHessian(M, U, k, axis_k=r["m"], matter_only=True)
    H, _ = S.assemble()
    Mm = S.mass(1.0)
    lu = spla.splu((H + 3.0 * Mm).tocsc())
    op = spla.LinearOperator(H.shape, matvec=lu.solve, dtype=float)
    vals, vecs = spla.eigsh(H, k=2, M=Mm, sigma=-3.0, which="LM", OPinv=op, tol=1e-7)
    o = np.argsort(vals)
    v = vecs[:, o[0]]
    X = np.zeros(G.nn * NT)
    X[S.free_flat] = v
    X = X.reshape(G.nn, NT)
    w = (X[:, :4] ** 2).sum(1) * np.asarray(G.W_n if hasattr(G, "W_n") else np.ones(G.nn))
    rho, z = np.asarray(G.rho_n), np.asarray(G.z_n)
    fin = np.isfinite(rho) & np.isfinite(z)
    w = np.where(fin, w, 0.0)
    i = int(np.argmax(w))
    near_axis = float(w[fin & (rho < 0.5)].sum() / w.sum())
    rc = d["per_factorization"][key]["core_positions"][0][0]
    near_core = float(w[fin & (np.hypot(rho - rc, z) < 1.0)].sum() / w.sum())
    far = float(w[fin & (np.hypot(rho, z) > 5.0)].sum() / w.sum())
    print(f"{key} k = {k}: lowest {np.round(np.sort(vals), 4).tolist()}; mode maximum at (ρ, z) = ({rho[i]:.2f}, {z[i]:.2f});"
          f" weight near the axis (ρ < 0.5) {near_axis:.2f}, near the core {near_core:.2f}, far (r > 5) {far:.2f}", flush=True)
    return dict(key=key, k=k, lowest=[float(x) for x in np.sort(vals)], max_at=[float(rho[i]), float(z[i])],
                near_axis=near_axis, near_core=near_core, far=far)


def localize(args):
    """Where the lowest mode of one sector lives: weight near the axis (ρ < 0.5), on the core tube
    (within 1.0 of the core circle in the meridional plane), far away (r > 5)."""
    key, k = args
    r, G, U, M = load(key)
    S = SectorHessian(M, U, k, axis_k=r["m"], matter_only=True)
    H, _ = S.assemble()
    Mm = S.mass(1.0)
    lu = spla.splu((H + 3.0 * Mm).tocsc())
    op = spla.LinearOperator(H.shape, matvec=lu.solve, dtype=float)
    vals, vecs = spla.eigsh(H, k=2, M=Mm, sigma=-3.0, which="LM", OPinv=op, tol=1e-7, maxiter=400)
    o = np.argsort(vals)
    X = np.zeros(G.nn * NT)
    X[S.free_flat] = vecs[:, o[0]]
    X = X.reshape(G.nn, NT)
    rho, z = np.asarray(G.rho_n), np.asarray(G.z_n)
    fin = np.isfinite(rho) & np.isfinite(z)
    import scipy.sparse as sps
    vol = np.asarray((G.PvT @ sps.diags(G.W) @ G.Pv).diagonal()).ravel()     # ∫ φ_i² dV > 0: the L² weight of node i
    w = np.where(fin, (X[:, :4] ** 2).sum(1) * vol, 0.0)
    cores = r["core_positions"]
    tube = np.zeros(G.nn, bool)
    for (rc, zc, _) in cores:
        tube |= fin & (np.hypot(rho - rc, np.abs(z) - abs(zc)) < 1.0)
    res = dict(key=key, k=k, lowest=float(np.min(vals)), axis=float(w[fin & (rho < 0.5)].sum() / w.sum()),
               tube=float(w[tube].sum() / w.sum()), far=float(w[fin & (np.hypot(rho, z) > 5)].sum() / w.sum()))
    print(f"   {key:4s} k = {k}: lowest {res['lowest']:+.4f}   weight on the core tube {res['tube']:.2f},"
          f" near the axis {res['axis']:.2f}, far {res['far']:.2f}", flush=True)
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "localize":
        from multiprocessing import Pool
        cases = [("4x1", 2), ("4x1", 3), ("5x1", 2), ("5x1", 3), ("6x1", 2), ("6x1", 3), ("6x1", 4), ("7x1", 2),
                 ("7x1", 3), ("7x1", 4), ("8x1", 3), ("8x1", 4), ("4x2", 2), ("4x2", 3), ("2x2", 1), ("4x2", 1)]
        with Pool(4) as pool:
            res = pool.map(localize, cases, chunksize=1)
        json.dump(res, open(os.path.join(HERE, "data", "prime_nonaxial_localize.json"), "w"), indent=1)
        sys.exit(0)
    out = dict(translation=[translation_rq(k) for k in ("5x1", "3x2", "2x2", "4x2")], deep=deep_mode())
    json.dump(out, open(os.path.join(HERE, "data", "prime_nonaxial_check.json"), "w"), indent=1)
