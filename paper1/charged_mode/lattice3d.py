"""
The three-line (flag) model on a three-dimensional cubic lattice, no symmetry assumed.

Purpose: the axisymmetric code (flag_axisym, run_flag_pair) can only follow coaxial
paths.  Its sector Hessian (flag_sector) shows that the coaxial fusion saddle has
non-axial (k = 1) downhill directions, so the true barrier is lower than the
coaxial one; how much lower needs a calculation without axial symmetry.  This is it.

Field: U(x) ∈ U(3) at the lattice sites x = h·(i − (N−1)/2, …), modulo the right
U(1)³ (the flag manifold).  Every energy term is built from gauge-invariant
lattice objects: the link overlaps O^i(x) = U(x)† U(x + h e_i), whose off-diagonal
entries are h·w_i^{ab} + O(h²) with w_i = U†∂_iU.

    sigma      h Σ_links Σ_{a<b} r_ab ½(|Õ_ab|² + |Õ_ba|²)               (O(h) terms cancel)
    Faddeev    h⁻¹ Σ_plaq Σ_a ½κ_a (arg Π_a)²,  Π_a = O^i_aa(x) O^j_aa(x+i) Ō^i_aa(x+j) Ō^j_aa(x)
               (arg Π_a = h² F^a_ij: the Berry flux of line a through the plaquette)
    Õ: off-diagonal entries rescaled from chord to arc, |Õ_ab| = asin|O_ab| (exact for a
    rotation within one pair).  Without these two corrections the quartic term comes out
    25 % low at h = 0.24 (the chord and |Π| < 1 saturate in the core).
    3-cycle    h⁻¹ Σ_plaq ¼ Σ_corners Σ_{a≠c} ½κ₃ |L1_ab L2_bc − L3_ab L4_bc|²
               (the two ways round the plaquette from one corner, b the third index;
                averaged over the four corners so that the O(h) errors cancel)
    potential  h³ Σ_sites Σ_a m_a² (1 − |U_aa|²)

In the continuum limit this is exactly the energy of flag_axisym.FlagModel
(σ: Σ r ½(|w^{ab}|² + |w^{ba}|²); quartic: ½κ_a (F^a_ij)² with
F^a_ij = 2 Im Σ_{b≠a} w̄_i^{ba} w_j^{ba}; three-cycle: ½κ₃|w_i^{ab}w_j^{bc} − w_j^{ab}w_i^{bc}|²;
potential Σ_a m_a² Σ_{c≠a}|U_ca|²).  The outermost layer of sites is held at the
vacuum U = 1.

Degrees of freedom: U = U_b Cay(X), X anti-Hermitian off-diagonal (6 reals per
interior site), minimised with L-BFGS; the base U_b is moved when X grows.
Constraints (augmented Lagrangian) on line centroids, 3D distances, line weights.

Initial data: an axisymmetric state U = R_L(φ) Û(ρ, z) of the spectral-element grid
(optionally perturbed by a sector mode Ŵ = Û Cay(ε(X_c cos kφ + X_s sin kφ))) is
carried over through the gauge-invariant projectors P_a = u_a u_a† (interpolated
with the element basis, top eigenvector, polar orthonormalisation).
"""

import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

PAIRS = ((0, 1), (0, 2), (1, 2))
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SCRATCH = os.environ.get("FLAGPAIR_SCRATCH", "/tmp/flagpair")


def _abs2(z):
    return jnp.real(z) ** 2 + jnp.imag(z) ** 2


def _dag(M):
    return jnp.conj(jnp.swapaxes(M, -1, -2))


def _sl(A, axis, start, stop):
    idx = [slice(None)] * A.ndim
    idx[axis] = slice(start, stop)
    return A[tuple(idx)]


def _arc_scale(s):
    """asin(√s)/√s, smooth in s = |O_ab|² ≥ 0 (the chord → arc correction of a link entry)."""
    small = s < 1e-6
    ss = jnp.where(small, 0.5, jnp.clip(s, 0.0, 1.0 - 1e-12))
    r = jnp.sqrt(ss)
    return jnp.where(small, 1.0 + s / 6.0 + 3.0 * s * s / 40.0, jnp.arcsin(r) / r)


def _arc(O):
    """Off-diagonal link entries rescaled from chord to arc: |Õ_ab| = asin|O_ab| (exact for a
    rotation inside one pair); the diagonal is kept (it carries the Berry phases)."""
    s = _abs2(O)
    eye = jnp.eye(3, dtype=bool)
    return jnp.where(eye, O, O * _arc_scale(s))


def cayley(X):
    I = jnp.eye(3, dtype=X.dtype)
    return jnp.linalg.solve(I - 0.5 * X, I + 0.5 * X)


def x6_to_X(x6):
    X = jnp.zeros(x6.shape[:-1] + (3, 3), dtype=jnp.complex128)
    for k, (a, b) in enumerate(PAIRS):
        z = x6[..., 2 * k] + 1j * x6[..., 2 * k + 1]
        X = X.at[..., a, b].set(z)
        X = X.at[..., b, a].set(-jnp.conj(z))
    return X


class Lattice:
    def __init__(self, N, h, r=(2.0, 2.0, 2.0), kappa=(2.0, 2.0, 2.0), m2=(1.0, 1.0, 1.0), kappa3=2.0):
        self.N, self.h = int(N), float(h)
        self.r, self.kappa, self.m2, self.kappa3 = tuple(r), tuple(kappa), tuple(m2), float(kappa3)
        self.x1 = (np.arange(self.N) - 0.5 * (self.N - 1)) * self.h
        X, Y, Z = np.meshgrid(self.x1, self.x1, self.x1, indexing="ij")
        self.X, self.Y, self.Z = X, Y, Z
        self.Xj = jnp.asarray(np.stack([X, Y, Z], axis=-1))
        n = self.N - 2
        self.nfree = 6 * n ** 3

    # --- energy -------------------------------------------------------------------
    def parts_U(self, U):
        h = self.h
        Uh = _dag(U)
        O = [jnp.matmul(Uh[:-1], U[1:]), jnp.matmul(Uh[:, :-1], U[:, 1:]), jnp.matmul(Uh[:, :, :-1], U[:, :, 1:])]
        O = [_arc(Oi) for Oi in O]
        Es = 0.0
        for Oi in O:
            for k, (a, b) in enumerate(PAIRS):
                Es = Es + self.r[k] * 0.5 * jnp.sum(_abs2(Oi[..., a, b]) + _abs2(Oi[..., b, a]))
        Es = Es * h
        EF, EC = 0.0, 0.0
        for d1, d2 in ((0, 1), (0, 2), (1, 2)):
            O1, O2 = O[d1], O[d2]
            A, Cc = _sl(O1, d2, 0, -1), _sl(O1, d2, 1, None)          # O^i(x), O^i(x + j)
            B, D = _sl(O2, d1, 1, None), _sl(O2, d1, 0, -1)           # O^j(x + i), O^j(x)
            for a in range(3):
                F = jnp.angle(A[..., a, a] * B[..., a, a] * jnp.conj(Cc[..., a, a]) * jnp.conj(D[..., a, a]))
                EF = EF + 0.5 * self.kappa[a] * jnp.sum(F ** 2)
            if self.kappa3 != 0.0:
                Ad, Bd, Cd, Dd = _dag(A), _dag(B), _dag(Cc), _dag(D)
                corners = ((A, B, D, Cc), (Ad, D, B, Cd), (Cc, Bd, Dd, A), (Cd, Dd, Bd, Ad))
                for a in range(3):
                    for c in range(3):
                        if c == a:
                            continue
                        b = 3 - a - c
                        for L1, L2, L3, L4 in corners:
                            Cv = L1[..., a, b] * L2[..., b, c] - L3[..., a, b] * L4[..., b, c]
                            EC = EC + 0.125 * self.kappa3 * jnp.sum(_abs2(Cv))
        EF, EC = EF / h, EC / h
        EV = 0.0
        for a in range(3):
            EV = EV + self.m2[a] * jnp.sum(1.0 - _abs2(U[..., a, a]))
        EV = EV * h ** 3
        return Es, EF, EC, EV

    def energy_U(self, U):
        return sum(self.parts_U(U))

    def parts(self, U):
        f = self._jparts if hasattr(self, "_jparts") else None
        if f is None:
            f = self._jparts = jax.jit(self.parts_U)
        vals = f(jnp.asarray(U))
        return dict(zip(("sigma", "faddeev", "three_cycle", "potential"), (float(v) for v in vals)))

    def energy(self, U):
        return sum(self.parts(U).values())

    # --- parametrisation ---------------------------------------------------------------
    def U_of(self, Ub, x):
        n = self.N - 2
        x6 = jnp.zeros((self.N, self.N, self.N, 6)).at[1:-1, 1:-1, 1:-1].set(x.reshape(n, n, n, 6))
        return Ub @ cayley(x6_to_X(x6))

    # --- line observables ----------------------------------------------------------------
    def q(self, U, c):
        return 1.0 - _abs2(U[..., c, c])

    def centroid(self, U, c):
        w = self.q(U, c) ** 2
        return jnp.einsum("ijk,ijkl->l", w, self.Xj) / jnp.sum(w)

    def weight(self, U, c):
        return self.h ** 3 * jnp.sum(self.q(U, c))

    def geometry(self, U):
        """Per line: weight, centroid, rms radius about the centroid, ring normal (smallest
        principal axis of the (1 − |U_cc|²)² distribution)."""
        U = np.asarray(U)
        out = {}
        P = np.stack([self.X, self.Y, self.Z], axis=-1)
        for c in range(3):
            q = 1.0 - np.abs(U[..., c, c]) ** 2
            w = q ** 2
            s = w.sum()
            if s < 1e-12:
                out[c] = dict(weight=float(self.h ** 3 * q.sum()))
                continue
            cen = np.einsum("ijk,ijkl->l", w, P) / s
            dP = P - cen
            S = np.einsum("ijk,ijkl,ijkm->lm", w, dP, dP) / s
            ev, evec = np.linalg.eigh(S)
            nrm = evec[:, 0] * np.sign(evec[2, 0] if abs(evec[2, 0]) > 1e-12 else 1.0)
            out[c] = dict(weight=float(self.h ** 3 * q.sum()), centroid=cen.tolist(),
                          radius=float(np.sqrt(ev[1] + ev[2])), normal=nrm.tolist(), moments=ev.tolist())
        g0, g2 = out.get(0), out.get(2)
        if g0 and g2 and "centroid" in g0 and "centroid" in g2:
            d = np.array(g0["centroid"]) - np.array(g2["centroid"])
            n0, n2 = np.array(g0["normal"]), np.array(g2["normal"])
            nm = (n0 + n2) / np.linalg.norm(n0 + n2)
            out["pair"] = dict(dist=float(np.linalg.norm(d)), dz_along_normal=float(d @ nm),
                               lateral=float(np.linalg.norm(d - (d @ nm) * nm)),
                               relative_tilt_deg=float(np.degrees(np.arccos(np.clip(abs(n0 @ n2), -1, 1)))),
                               axis_tilt_deg=float(np.degrees(np.arccos(np.clip(abs(nm[2]), -1, 1)))))
        return out


# ---------------------------------------------------------------------------------------
# carrying an axisymmetric state over to the lattice
# ---------------------------------------------------------------------------------------

def _lagrange2(t):
    return np.stack([0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)], axis=-1)


def _locate(s, s0, ds_el, ne):
    e = np.clip(np.floor((s - s0) / ds_el).astype(int), 0, ne - 1)
    t = 2.0 * (s - s0 - e * ds_el) / ds_el - 1.0
    return e, t


def hinge(theta, width=0.3):
    """Coordinate map for a 'hinge' seed: the upper half space is sampled rotated by +θ about
    the y axis, the lower by −θ (smoothly, tanh(z/width)), so the ring above tilts one way and
    the ring below the other; they approach on one side."""
    def f(x, y, z):
        t = theta * np.tanh(z / width)
        c, s = np.cos(t), np.sin(t)
        return c * x - s * z, y, s * x + c * z
    return f


def from_axial(lat, Gf, Uax, L=(0, 1, 2), mode=None, eps=0.0, k=1, chunk=40000, coords=None):
    """Lattice field from an axisymmetric spectral-element state (p = 2 elements).
    mode = (Xc, Xs): nodal (n, 3, 3) sector perturbation, applied as Û Cay(ε(Xc cos kφ + Xs sin kφ)).
    coords: optional map (x, y, z) → sample point (e.g. hinge(θ))."""
    assert Gf.p == 2 and not Gf.half
    a = Gf.a
    x, y, z = lat.X.ravel(), lat.Y.ravel(), lat.Z.ravel()
    if coords is not None:
        x, y, z = coords(x, y, z)
    rho, phi = np.hypot(x, y), np.arctan2(y, x)
    xi = (2.0 / np.pi) * np.arctan(rho / a)
    eta = (2.0 / np.pi) * np.arctan(z / a)
    ne_r = (Gf.nr - 1) // 2
    ne_z = (Gf.nz - 1) // 2
    er, tr = _locate(xi, 0.0, 1.0 / ne_r, ne_r)
    ez, tz = _locate(eta, -1.0, 2.0 / ne_z, ne_z)
    Br, Bz = _lagrange2(tr), _lagrange2(tz)
    Uax = np.asarray(Uax)
    Lv = np.asarray(L, float)
    out = np.empty((x.size, 3, 3), complex)
    I3 = np.eye(3)
    for s0 in range(0, x.size, chunk):
        sl = slice(s0, min(x.size, s0 + chunk))
        n = sl.stop - sl.start
        P = np.zeros((n, 3, 3, 3), complex)          # (point, a, row, col)
        for i in range(3):
            for j in range(3):
                node = (2 * er[sl] + i) * Gf.nz + (2 * ez[sl] + j)
                w = Br[sl, i] * Bz[sl, j]
                Wn = Uax[node]
                if mode is not None and eps != 0.0:
                    Xc, Xs = mode
                    Xn = eps * (np.cos(k * phi[sl])[:, None, None] * Xc[node] + np.sin(k * phi[sl])[:, None, None] * Xs[node])
                    Cay = np.linalg.solve(I3[None] - 0.5 * Xn, I3[None] + 0.5 * Xn)
                    Wn = Wn @ Cay
                P += w[:, None, None, None] * np.einsum("nra,nsa->nars", Wn, np.conj(Wn))
        ph = np.exp(1j * np.outer(phi[sl], Lv))                        # R_L(φ) diagonal
        P = P * ph[:, None, :, None] * np.conj(ph)[:, None, None, :]
        V = np.empty((n, 3, 3), complex)
        for c in range(3):
            ev, evec = np.linalg.eigh(P[:, c])
            V[:, :, c] = evec[:, :, -1]
        u, s, vh = np.linalg.svd(V)
        out[sl] = u @ vh                                               # polar factor
    U = out.reshape(lat.N, lat.N, lat.N, 3, 3)
    U[0], U[-1], U[:, 0], U[:, -1], U[:, :, 0], U[:, :, -1] = (np.eye(3),) * 6
    return U


def resample(U, h_old, lat, coords=None):
    """Carry a lattice state to another lattice (same centre): trilinear interpolation of the
    gauge-invariant flag matrix H = U diag(3, 2, 1) U†, then its ordered eigenbasis.  Points
    outside the old box get the vacuum.  coords: optional map of the new lattice points to the
    points sampled in the old state (e.g. an inverse rotation and shift: U'(x) = U(R⁻¹(x − a)))."""
    from scipy.ndimage import map_coordinates
    U = np.asarray(U)
    N_old = U.shape[0]
    H = np.einsum("...ra,a,...sa->...rs", U, np.array([3.0, 2.0, 1.0]), np.conj(U))
    P = (lat.X.ravel(), lat.Y.ravel(), lat.Z.ravel())
    if coords is not None:
        P = coords(*P)
    idx = [A / h_old + 0.5 * (N_old - 1) for A in P]
    Hn = np.empty((idx[0].size, 3, 3), complex)
    for r in range(3):
        for c in range(3):
            re = map_coordinates(H[..., r, c].real, idx, order=1, mode="nearest")
            im = map_coordinates(H[..., r, c].imag, idx, order=1, mode="nearest")
            Hn[:, r, c] = re + 1j * im
    ev, evec = np.linalg.eigh(Hn)
    Un = evec[..., ::-1].reshape(lat.N, lat.N, lat.N, 3, 3)
    Un[0], Un[-1], Un[:, 0], Un[:, -1], Un[:, :, 0], Un[:, :, -1] = (np.eye(3),) * 6
    return Un


# ---------------------------------------------------------------------------------------
# relaxation (L-BFGS in the local chart, augmented Lagrangian for constraints)
# ---------------------------------------------------------------------------------------

class Constraint:
    """c(U) − target, vectorised; built from the lattice observables."""

    def __init__(self, lat, kind, target):
        self.kind, self.target = kind, np.atleast_1d(np.asarray(target, float))
        if kind == "dz":            # z̄₀ − z̄₂  and the pair centre (x̄, ȳ, z̄ of lines 0 + 2) = 0
            f = lambda U: jnp.concatenate([jnp.stack([lat.centroid(U, 0)[2] - lat.centroid(U, 2)[2]]),
                                           0.5 * (lat.centroid(U, 0) + lat.centroid(U, 2))])
        elif kind == "dist":        # |X̄₀ − X̄₂| and the pair centre = 0
            f = lambda U: jnp.concatenate([jnp.stack([jnp.linalg.norm(lat.centroid(U, 0) - lat.centroid(U, 2))]),
                                           0.5 * (lat.centroid(U, 0) + lat.centroid(U, 2))])
        elif kind == "D1":          # line-1 weight and the pair centre = 0
            f = lambda U: jnp.concatenate([jnp.stack([lat.weight(U, 1)]),
                                           0.5 * (lat.centroid(U, 0) + lat.centroid(U, 2))])
        elif kind == "overlap":     # O = ∫ q₀ q₂ (0 apart, grows on contact and zipping) and the pair centre
            f = lambda U: jnp.concatenate([jnp.stack([lat.h ** 3 * jnp.sum(lat.q(U, 0) * lat.q(U, 2))]),
                                           0.5 * (lat.centroid(U, 0) + lat.centroid(U, 2))])
        elif kind == "centre":      # only the centre of line 0 (+ line 2 if present)
            f = lambda U: 0.5 * (lat.centroid(U, 0) + lat.centroid(U, 2))
        else:
            raise ValueError(kind)
        self.f = f
        if kind != "centre" and self.target.size == 1:
            self.target = np.concatenate([self.target, np.zeros(3)])
        elif kind == "centre" and self.target.size == 1:
            self.target = np.zeros(3)


class Precond:
    """S = (4h³(−Δ_h + c))^{-1/2} on the interior (Dirichlet), applied with the orthonormal
    DST-I.  4h³(−Δ + m²) is the Hessian of the lattice energy about the vacuum per real
    component (r = 2, m² = 1), so in the variables y = S⁻¹x the vacuum Hessian is 1."""

    def __init__(self, lat, c=1.0):
        from scipy.fft import dstn
        self.dstn = dstn
        n = lat.N - 2
        k = np.arange(1, n + 1)
        lam1 = (2.0 - 2.0 * np.cos(np.pi * k / (n + 1))) / lat.h ** 2
        L = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
        self.f = (4.0 * lat.h ** 3 * (L + c)) ** -0.5
        self.shape = (n, n, n, 6)

    def __call__(self, v):
        a = self.dstn(np.asarray(v).reshape(self.shape), type=1, axes=(0, 1, 2), norm="ortho")
        a *= self.f[..., None]
        return self.dstn(a, type=1, axes=(0, 1, 2), norm="ortho").ravel()


def relax(lat, U0, cons=None, K=200.0, outer=8, chunk=150, max_chunks=40, gtol=1e-4, verbose=True, log=None,
          precond=True, ctol=1e-4):
    """Minimise E (subject to cons(U) = target, augmented Lagrangian) starting from U0.
    L-BFGS in the chart U = U_b Cay(X(x)), x = S y (S: Precond), on y; the base moves to the
    current point after each chunk.  gtol: max-norm of ∂L/∂y (∼ √(2ΔE) per mode).
    Returns U, info."""
    Ub = jnp.asarray(U0)
    ncon = 0 if cons is None else cons.target.size
    lam = np.zeros(ncon)
    tgt = jnp.asarray(np.zeros(0) if cons is None else cons.target)
    S = Precond(lat) if precond else (lambda v: np.asarray(v))

    def L(Ub, x, lam):
        U = lat.U_of(Ub, x)
        E = lat.energy_U(U)
        if cons is None:
            return E, E
        c = cons.f(U) - tgt
        return E + jnp.dot(lam, c) + 0.5 * K * jnp.dot(c, c), E

    vg = jax.jit(jax.value_and_grad(L, argnums=1, has_aux=True))
    y0 = np.zeros(lat.nfree)
    hist, t0, it_total = [], time.time(), 0
    gmax, E = np.inf, np.nan
    for o in range(outer if cons is not None else 1):
        lj = jnp.asarray(lam)
        for ch in range(max_chunks):
            def fun(y):
                (Lv, E_), g = vg(Ub, jnp.asarray(S(y)), lj)
                return float(Lv), S(np.asarray(g))

            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options=dict(maxiter=chunk, maxcor=30, gtol=0.1 * gtol, ftol=1e-16))
            it_total += res.nit
            x = S(res.x)
            Ub = lat.U_of(Ub, jnp.asarray(x))
            (Lv, E), g = vg(Ub, jnp.zeros(lat.nfree), lj)
            gmax = float(np.abs(S(np.asarray(g))).max())
            row = dict(outer=o, chunk=ch, it=it_total, E=float(E), L=float(Lv), gmax=gmax,
                       step=float(np.abs(x).max()), t=round(time.time() - t0, 1))
            if cons is not None:
                row["c"] = (np.asarray(cons.f(Ub)) - cons.target).tolist()
            hist.append(row)
            if verbose:
                print(json.dumps(row), flush=True)
            if log:
                with open(log, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
            if gmax < gtol:
                break
        if cons is None:
            break
        c = np.asarray(cons.f(Ub)) - cons.target
        lam = lam + K * c
        # the first component is the reaction coordinate; the pair centre (a zero mode) only needs to
        # stay put roughly
        if abs(c[0]) < ctol and np.abs(c).max() < 20 * ctol and gmax < gtol:
            break
    info = dict(hist=hist, lam=lam.tolist(), E=float(E), gmax=gmax, iterations=it_total)
    return np.asarray(Ub), info


# ---------------------------------------------------------------------------------------
# the three-dimensional charge (π₃(F₂) = ℤ) on the lattice
# ---------------------------------------------------------------------------------------

def charge(lat, U, pad=2):
    """Q = (1/8π²) Σ_a ∫ A_a · B_a d³x, B_a the Berry curvature of line a (plaquette phases of the
    diagonal link overlaps), A_a its Coulomb-gauge vector potential (Fourier space, zero padding).
    For a field in one 2 × 2 block this is the Hopf invariant (1/4π²)∫A·B of that CP¹ (the two
    active lines contribute equally); in general the Chern–Simons forms of the three line bundles
    enter with equal weights, i.e. the quadratic form Σ_a x_a² = the Killing form on π₂(F₂).
    Returns (Q, [Q_0, Q_1, Q_2], max |div B| / max |B|)."""
    U = np.asarray(U)
    h = lat.h
    Uh = np.conj(np.swapaxes(U, -1, -2))
    O = [np.einsum("...ij,...jk->...ik", Uh[:-1], U[1:]),
         np.einsum("...ij,...jk->...ik", Uh[:, :-1], U[:, 1:]),
         np.einsum("...ij,...jk->...ik", Uh[:, :, :-1], U[:, :, 1:])]
    N = U.shape[0]
    Qs, divs = [], []
    for a in range(3):
        d = [Oi[..., a, a] for Oi in O]
        def flux(i, j):
            A = d[i]
            Bj = d[j]
            s = [slice(None)] * 3
            A0 = A[tuple(slice(0, -1) if k == j else slice(None) for k in range(3))]
            A1 = A[tuple(slice(1, None) if k == j else slice(None) for k in range(3))]
            B0 = Bj[tuple(slice(0, -1) if k == i else slice(None) for k in range(3))]
            B1 = Bj[tuple(slice(1, None) if k == i else slice(None) for k in range(3))]
            return np.angle(A0 * B1 * np.conj(A1) * np.conj(B0)) / h ** 2
        Fyz, Fzx, Fxy = flux(1, 2), -flux(0, 2), flux(0, 1)
        # to cell centres (N-1)^3: average the four parallel faces of each cell edge direction
        def centre(F, axis):
            # F has size N along `axis` and N-1 along the other two: average over the two ends of `axis`
            sl0 = [slice(None)] * 3; sl1 = [slice(None)] * 3
            sl0[axis] = slice(0, -1); sl1[axis] = slice(1, None)
            return 0.5 * (F[tuple(sl0)] + F[tuple(sl1)])
        Bx, By, Bz = centre(Fyz, 0), centre(Fzx, 1), centre(Fxy, 2)
        B = np.stack([Bx, By, Bz])
        M = B.shape[1] + 2 * pad * B.shape[1]
        Bp = np.zeros((3, M, M, M))
        o = pad * B.shape[1]
        Bp[:, o:o + B.shape[1], o:o + B.shape[1], o:o + B.shape[1]] = B
        k1 = 2 * np.pi * np.fft.fftfreq(M, d=h)
        KX, KY, KZ = np.meshgrid(k1, k1, k1, indexing="ij")
        K = np.stack([KX, KY, KZ])
        K2 = KX ** 2 + KY ** 2 + KZ ** 2
        K2[0, 0, 0] = 1.0
        Bk = np.fft.fftn(Bp, axes=(1, 2, 3))
        Ak = 1j * np.cross(K, Bk, axis=0) / K2
        A = np.real(np.fft.ifftn(Ak, axes=(1, 2, 3)))
        Qs.append(float(np.sum(A * Bp) * h ** 3 / (8 * np.pi ** 2)))
        div = np.abs(np.real(np.fft.ifftn(1j * np.sum(K * Bk, axis=0), axes=(0, 1, 2))))
        divs.append(float(div.max() / (np.abs(Bp).max() + 1e-300)))
    return sum(Qs), Qs, max(divs)
