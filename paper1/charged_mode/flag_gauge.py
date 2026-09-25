"""
Smooth gauge for the nodal columns of a flag field.

The column phases of U are gauge: the projector discretisation does not see
them, but the Hopf degree Q_h is computed from an interpolated lift U, and a
rough phase field (accumulated along Newton paths) biases it.  For each column
a this chooses nodal phases θ_a(n) that make neighbouring columns as parallel as
possible,

    min_θ Σ_edges |w_nm| [1 − cos(θ_m − θ_n + arg w_nm)],   w_nm = ⟨Z_a(n), Z_a(m)⟩,

(an XY model with fixed bond angles; the discrete Coulomb gauge).  The start is
the tree gauge along a comb spanning tree (exactly parallel along the tree), so
that arbitrary nodal phases are removed; Newton's method then polishes it.  The
flags, and hence the projector energy, are unchanged.
"""

import numpy as np
import scipy.sparse as sp

from hopfion_axisym import sparse_solve


def _edges(G):
    idx = np.arange(G.nn).reshape(G.nr, G.nz)
    e1 = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    e2 = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    return np.concatenate([e1, e2])


def _tree_gauge(G, Z):
    """Phases making Z parallel along the comb tree: first J = 0 → nz−1 at I = 0, then I."""
    Zg = Z.reshape(G.nr, G.nz, 3)
    th = np.zeros((G.nr, G.nz))
    for j in range(1, G.nz):
        th[0, j] = th[0, j - 1] - np.angle(np.vdot(Zg[0, j - 1], Zg[0, j]))
    for i in range(1, G.nr):
        w = np.einsum("jk,jk->j", np.conj(Zg[i - 1]), Zg[i])
        th[i] = th[i - 1] - np.angle(w)
    return th.ravel()


def smooth_gauge(G, U, max_iter=30, tol=1e-12):
    """Return U·diag(e^{iθ}) with the smoothest nodal column phases, and the θ used."""
    U = np.asarray(U)
    E = _edges(G)
    n, m = E[:, 0], E[:, 1]
    theta = np.zeros((G.nn, 3))
    for a in range(3):
        w = np.einsum("ei,ei->e", np.conj(U[n, :, a]), U[m, :, a])
        J, phi = np.abs(w), np.angle(w)
        th = _tree_gauge(G, U[:, :, a])
        for _ in range(max_iter):
            d = th[m] - th[n] + phi
            s, c = J * np.sin(d), J * np.cos(d)
            g = np.zeros(G.nn)
            np.add.at(g, m, s)
            np.add.at(g, n, -s)
            if np.abs(g).max() < tol:
                break
            cc = np.maximum(c, 1e-3 * J)          # keep the Hessian positive
            H = sp.coo_matrix((np.concatenate([cc, cc, -cc, -cc]),
                               (np.concatenate([n, m, n, m]), np.concatenate([n, m, m, n]))),
                              shape=(G.nn, G.nn)).tocsr()
            H = H + 1e-10 * sp.identity(G.nn)
            th = th - sparse_solve(H.tocsr(), g)
        theta[:, a] = th - th[0]
    return U * np.exp(1j * theta)[:, None, :], theta
