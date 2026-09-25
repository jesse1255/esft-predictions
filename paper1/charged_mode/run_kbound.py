"""
High azimuthal sectors k ≥ 5 without assembling each one.

For k ≥ 1 the second variation in sector k is an exact quadratic polynomial
in k on a k-independent coefficient space (the φ quadrature is exact, the
background is φ-independent, and k enters only through ∂_φ = k·(swap c ↔ s)):

    H(k) = A + k B + k² C.

For k ≥ 2 the axis constraints are also k-independent, so A, B, C follow
exactly from H(2), H(3), H(4); a direct assembly at k = 5 checks this.
Then for every integer k from 5 to k_max the LDLᵀ inertia of H(k) is taken:
H_AA ≻ 0 (gauge-fixed Maxwell block), so by Haynsworth additivity the number
of negative eigenvalues of H(k) equals that of the gauge-relaxed Schur
complement.  For the tail, H(k) − H(k0) = (k − k0)[B + (k + k0)C]: if B + 2k0 C ⪰ 0
(and C ≻ 0) then H(k) ⪰ H(k0) for all k ≥ k0, so the finite check closes.

Usage: python run_kbound.py ne [--state f.npz --e 0.6 --N 0.5 --tag _e0.6_N0.5 --kmax 60]
Writes data/kbound{tag}_ne{ne}.json.
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp

from hopfion_axisym import Model
from nonaxisym import SectorHessian
from run_checks import load_state, mirror_to_full

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def inertia(A):
    """(n_pos, n_neg, n_perturbed_pivots) from PARDISO's symmetric indefinite LDLᵀ."""
    import pypardiso
    s = pypardiso.PyPardisoSolver()
    s.set_matrix_type(-2)
    s.factorize(sp.triu(sp.csr_matrix(A), format="csr"))
    ip = s.get_iparms()
    out = (int(ip[22]), int(ip[23]), int(ip[14]))
    s.free_memory(everything=True)
    return out


def main(ne, state=None, e=0.3, N=1.0, mu=1.0, tag="", kmax=60):
    Gh, d = load_state(ne)
    V = d["V_full"] if state is None else np.load(state)["V"]
    Gf, Uf = mirror_to_full(Gh, V)
    M = Model(Gf, e=e, N=N, mu=mu, kappa=1.0)
    out = dict(ne=ne, e=e, N=N, mu=mu, description=__doc__, rows=[])
    fname = os.path.join(DATA, f"kbound{tag}_ne{ne}.json")
    Hs, free, S4 = {}, None, None
    for k in (2, 3, 4, 5):
        t0 = time.time()
        S = SectorHessian(M, Uf, k)
        if free is None:
            free = S.free_flat
        assert np.array_equal(free, S.free_flat), "coefficient space changed with k"
        Hs[k], asym = S.assemble()
        if k == 4:
            S4 = S
        print(f"assembled k={k} ({time.time() - t0:.0f}s, asym {asym:.1e})", flush=True)
    # control: the method must see the continuum.  With the matter L² mass M_mm,
    # H(4) − σ diag(M_mm, 0) is positive for σ below the continuum edge and has
    # negative eigenvalues for σ above it.
    im, ia = S4.split()
    Mm = S4.mass(1.0)[im][:, im].tocoo()
    Mbig = sp.csr_matrix((Mm.data, (im[Mm.row], im[Mm.col])), shape=Hs[4].shape)
    out["control_k4"] = {f"{sig:g}": inertia((Hs[4] - sig * Mbig).tocsr())[1]
                         for sig in (0.99, 0.9995, 1.0005, 1.01)}
    out["inertia_k2_k4"] = {str(k): inertia(Hs[k])[1] for k in (2, 3, 4)}
    print(f"control (negatives of H(4) − σM): {out['control_k4']};  "
          f"negatives at k=2,3,4: {out['inertia_k2_k4']}", flush=True)
    C = (0.5 * (Hs[2] - 2.0 * Hs[3] + Hs[4])).tocsr()
    B = (Hs[3] - Hs[2] - 5.0 * C).tocsr()
    A = (Hs[2] - 2.0 * B - 4.0 * C).tocsr()
    H5 = (A + 5.0 * B + 25.0 * C).tocsr()
    rel = float(abs(H5 - Hs[5]).max() / abs(Hs[5]).max())
    out["polynomial_check_k5"] = rel
    out["inertia_C"] = inertia(C)
    out["nfree"] = int(C.shape[0])
    print(f"polynomial check at k=5: rel. max diff {rel:.2e};  inertia(C) = {out['inertia_C']}",
          flush=True)
    # Tail: H(k) − H(k0) = (k − k0)[B + (k + k0) C].  If B + 2k0·C ⪰ 0 then, since C ≻ 0,
    # H(k) ⪰ H(k0) for every k ≥ k0, so one PD check at k0 covers all larger k.
    out["tail"] = {}
    for k0 in (10, 20, 40, kmax):
        out["tail"][str(k0)] = inertia((B + (2.0 * k0) * C).tocsr())[1]
    print(f"tail test: negatives of B + 2k0·C for k0 = 10, 20, 40, {kmax}: {out['tail']}",
          flush=True)
    for k in range(5, kmax + 1):
        t0 = time.time()
        Hk = Hs[5] if k == 5 else (A + k * B + (k * k) * C).tocsr()
        npos, nneg, npert = inertia(Hk)
        out["rows"].append(dict(k=k, n_negative=nneg, n_positive=npos, perturbed_pivots=npert))
        print(f"k={k:3d}: negative eigenvalues {nneg}  (perturbed pivots {npert}, "
              f"{time.time() - t0:.0f}s)", flush=True)
        with open(fname, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ne", type=int)
    ap.add_argument("--state", default=None)
    ap.add_argument("--e", type=float, default=0.3)
    ap.add_argument("--N", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--kmax", type=int, default=60)
    a = ap.parse_args()
    main(a.ne, a.state, a.e, a.N, a.mu, a.tag, a.kmax)
