"""
Numerical check of the lower-bound statement used in E1 (k = 0 sector).

E_N[n, A] = E_s[n, A] + N²/(2 I[n]),  I[n] = min_β F[β; n],
F = ∫ (β² K[n] + |∇β|²/e²).  The Hessian used in nonaxisym.py replaces the
charge term by its local part, i.e. it is the Hessian of

    E_loc[n, A] = E_s[n, A] − (ω₀²/2) ∫ β₀² K[n]  (+ const),   ω₀ = N/I₀,

which has the same value and gradient as E_N at the background.  The claim is

    d²/dt² (E_N − E_loc)(U₀ + t v) |₀ = (N²/I³)(δI)² + (N²/2I²)·Rᵀ(∂²_βF)⁻¹R ≥ 0

for every direction v (matter and gauge).  Here it is checked by second
differences of the exact E_N (β re-solved at every point) along random
axisymmetric directions on the half plane, together with the analytic
decomposition.

Usage: python check_lower_bound.py 32
Writes data/check_lower_bound_ne{ne}.json.
"""

import json
import os
import sys

import numpy as np

from hopfion_axisym import Grid, Model, U1, U2, U3, sparse_solve
from run_checks import load_state

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def K_of(M, U):
    Qv, Qr, Qz = M.interp(U)
    return M.c2 * (Qv[:, U1] ** 2 + Qv[:, U2] ** 2) + M.c4 * (Qr[:, U3] ** 2 + Qz[:, U3] ** 2)


def main(ne, n_dir=4, t=1e-3, seed=1):
    G, d = load_state(ne)
    U0 = d["V_full"]
    M = Model(G, e=0.3, N=1.0, mu=1.0, kappa=1.0)
    E0, _ = M.energy(U0, want_grad=False)
    beta0, I0 = M.last["beta"], M.last["I"]
    w0 = M.N / I0
    bq0 = G.Pv @ beta0

    f = G.beta_free

    dmask = ~f

    def I_precise(K):
        """min_β F with iterative refinement (‖A_ff‖ is large on the compactified grid)."""
        Ab = M.beta_matrix(K)
        Aff = Ab[f][:, f].tocsr()
        rhs = -(Ab[f][:, dmask] @ np.ones(dmask.sum()))
        x = sparse_solve(Aff, rhs)
        for _ in range(4):
            x = x + sparse_solve(Aff, rhs - Aff @ x)
        beta = np.ones(G.nn)
        beta[f] = x
        # sum of positive terms at the quadrature points (βᵀAβ cancels badly where β ≈ 1)
        bq, br, bz = G.Pv @ beta, G.Pr @ beta, G.Pz @ beta
        return float(G.W @ (K * bq * bq + (br * br + bz * bz) / M.e ** 2))

    def D(U):
        """E_N − E_loc up to a constant: only the charge sector differs."""
        K = K_of(M, U)
        return 0.5 * M.N ** 2 / I_precise(K) + 0.5 * w0 * w0 * G.integrate(bq0 * bq0 * K)

    # directions: dilation, z-stretch, and smoothed random fields damped near the axis
    r = np.sqrt(np.nan_to_num(G.rho_n, posinf=0.0) ** 2 + np.nan_to_num(G.z_n, posinf=0.0,
                                                                         neginf=0.0) ** 2)
    win = (r ** 2 / (1.0 + r ** 2) * np.exp(-r ** 2 / 16.0))[:, None]
    dirs = []
    Ur = U0.reshape(G.nr, G.nz, -1)
    rho2 = np.nan_to_num(G.rho_n, posinf=0.0).reshape(G.nr, G.nz)
    z2 = np.nan_to_num(G.z_n, posinf=0.0, neginf=0.0).reshape(G.nr, G.nz)
    dU_r = np.gradient(Ur, axis=0) / np.gradient(rho2, axis=0)[..., None]
    dU_z = np.gradient(Ur, axis=1) / np.gradient(z2, axis=1)[..., None]
    dil = (rho2[..., None] * dU_r + z2[..., None] * dU_z).reshape(G.nn, -1)
    dirs.append(("dilation", np.nan_to_num(dil) * win * G.free))
    dirs.append(("z-stretch", np.nan_to_num((z2[..., None] * dU_z).reshape(G.nn, -1)) * win * G.free))
    rng = np.random.default_rng(seed)
    for i in range(n_dir):
        v = rng.standard_normal((G.nr, G.nz, U0.shape[1]))
        for _ in range(20):
            v[1:-1, 1:-1] = 0.2 * (v[1:-1, 1:-1] + v[2:, 1:-1] + v[:-2, 1:-1]
                                   + v[1:-1, 2:] + v[1:-1, :-2])
        dirs.append((f"random {i}", v.reshape(G.nn, -1) * win * G.free))
    rows = []
    D0 = D(U0)
    Aff = M.beta_matrix(K_of(M, U0))[f][:, f]
    for name, v in dirs:
        v = v / np.abs(v).max()
        d2 = []
        for s in (t, t / 2):
            d2.append((D(U0 + s * v) + D(U0 - s * v) - 2 * D0) / s ** 2)
        fd = (4 * d2[1] - d2[0]) / 3
        dK = (K_of(M, U0 + t * v) - K_of(M, U0 - t * v)) / (2 * t)
        dI = G.integrate(bq0 * bq0 * dK)
        b = (G.PvT @ (G.W * dK * bq0))[f]
        resp = M.N ** 2 / I0 ** 2 * float(b @ sparse_solve(Aff, b))
        sq = M.N ** 2 / I0 ** 3 * dI ** 2
        rows.append(dict(direction=name, fd=fd, predicted=resp + sq, response_term=resp,
                         square_term=sq))
        print(f"{name:10s}: FD d²(E_N − E_loc) = {fd:+.6e}   predicted = {resp + sq:.6e}  "
              f"(response {resp:.3e}, square {sq:.3e})", flush=True)
    out = dict(ne=ne, E0=E0, I0=I0, omega0=w0, t=t, rows=rows, description=__doc__)
    with open(os.path.join(DATA, f"check_lower_bound_ne{ne}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
