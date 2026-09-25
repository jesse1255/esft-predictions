"""
The z-reflection symmetry R of the axisymmetric flag problem.

    R:  U(ρ, z) ↦ S · conj U(ρ, −z) · S,   S = diag(1, −1, 1).

R is an exact symmetry of the discrete energy (the grid is mirror symmetric and
every term is invariant under complex conjugation combined with z ↦ −z), and it
fixes the embedded m = 1 Hopfion (to 1e-12 on the stored states).  At an
R-symmetric base point U₀ the chart U₀·Cay(X) carries R to the signed mirror

    X_ab(ρ, z) ↦ s_a s_b · conj X_ab(ρ, −z),   (s_0, s_1, s_2) = (1, −1, 1).

The two continuum zero modes of the embedded Hopfion in the axisymmetric
sector (z-translation, isorotation diag(1, e^{iα}, 1)) are R-odd; so is the
Goldstone mode of the U(1) diag(1, 1, e^{iβ}) on the F₂ branch.  The R-even
sector is the slice that fixes the centre at z = 0 and these phases.
"""

import numpy as np
import scipy.sparse as sp

from flag_axisym import NT

S3 = np.diag([1.0, -1.0, 1.0])
# sign of (Re, Im) of x_01, x_02, x_12 under R
SIGN = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0])


def mirror_nodes(G):
    n = np.arange(G.nn)
    return (n // G.nz) * G.nz + (G.nz - 1 - n % G.nz)


def symmetry_defect(G, U):
    RU = S3 @ np.conj(U[mirror_nodes(G)]) @ S3
    return float(np.abs(RU - U).max())


def reflection(G, prob):
    """R on the free tangent coordinates of a FlagProblem: (R x)[i] = sign[i] · x[perm[i]]."""
    mir = mirror_nodes(G)
    perm = np.empty(prob.nfree, dtype=np.int64)
    sign = np.empty(prob.nfree)
    for t in range(NT):
        sel = np.nonzero(prob.dof[:, t] >= 0)[0]
        assert np.all(prob.dof[mir[sel], t] >= 0), "free mask not mirror symmetric"
        perm[prob.dof[sel, t]] = prob.dof[mir[sel], t]
        sign[prob.dof[sel, t]] = SIGN[t]
    return perm, sign


def restrict(perm, sign, idx):
    """R restricted to an R-invariant subset idx of the coordinates."""
    pos = -np.ones(len(perm), dtype=np.int64)
    pos[idx] = np.arange(len(idx))
    p = pos[perm[idx]]
    assert np.all(p >= 0), "subset not R-invariant"
    return p, sign[idx]


def sector_basis(perm, sign, parity):
    """Orthonormal basis (columns) of the R-even (parity +1) or R-odd (−1) subspace."""
    n = len(perm)
    i = np.arange(n)
    pi = i[i < perm]
    fx = i[(perm == i) & (sign == parity)]
    k = np.arange(len(pi))
    rows = np.concatenate([pi, perm[pi], fx])
    cols = np.concatenate([k, k, len(pi) + np.arange(len(fx))])
    vals = np.concatenate([np.full(len(pi), 2 ** -0.5), parity * sign[pi] * 2 ** -0.5,
                           np.ones(len(fx))])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, len(pi) + len(fx)))
