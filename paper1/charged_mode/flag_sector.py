"""
Second variation of the three-line (flag) model about an axisymmetric state,
one azimuthal sector k at a time ("curved paths": is there a direction off the
symmetry axis that lowers the energy?).

Background: U = R_L(φ) Û(ρ, z), L = (0, 1, 2), stored as Û at the nodes.
Perturbed field: U = R_L(φ) Ŵ(ρ, φ, z),  Ŵ = Û · Cay(X),
    X(ρ, φ, z) = X_c(ρ, z) cos kφ + X_s(ρ, z) sin kφ     (anti-Hermitian, off-diagonal),
so that
    ω_ρ = Ŵ†∂_ρŴ,  ω_z = Ŵ†∂_zŴ,  ω_φ = ρ⁻¹ [Ŵ†(iL)Ŵ + Ŵ†∂_φŴ],
with ∂_φŴ = Û · dCay(X)[∂_φX] computed exactly.  The energy density is that of
flag_axisym.FlagModel ("align" discretisation: element-local phases taken from Ŵ
and applied to ∂_φŴ as well).  The φ integral is a uniform average over M samples,
exact for the quadratic form once M > 2k.  At X = 0 this is the axisymmetric
discrete energy, so an axisymmetric critical point is a critical point here and
the Hessian splits into sectors k.

Degrees of freedom per node: 12 real = (cos, sin) × (Re, Im) × (x₀₁, x₀₂, x₁₂).
Axis regularity: the lab-frame projectors must be smooth at ρ = 0.  With Û
diagonal on the axis, the (a, b) component carries e^{i(l_a − l_b)φ}·X_ab(φ), so
    k = 0 : x_ab free iff l_a = l_b (cos part only),
    k > 0 : x_ab free iff |l_b − l_a| = k, and only the combination
            X_s = +i X_c (l_b − l_a = +k)  or  X_s = −i X_c (l_b − l_a = −k);
            everything else vanishes on the axis.
The metric used to normalise eigenvalues is the L² form ⟨|X|²⟩_φ (Hessian of
mass_energy); the Morse index (number of negative eigenvalues of H) does not
depend on the metric (Sylvester).

Usage (validation and the fusion saddle):
    python flag_sector.py <state.npy> k [k ...]  → prints counts and lowest eigenvalues
"""

import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

from flag_axisym import FlagModel, PAIRS, _abs2, inertia
from newton import _share_1d

NS = 12            # types per node


def _x6_to_X(x6):
    """(n, 6) real → (n, 3, 3) anti-Hermitian off-diagonal X (same convention as flag_axisym.x_to_X)."""
    n = x6.shape[0]
    X = jnp.zeros((n, 3, 3), dtype=jnp.complex128)
    for k, (a, b) in enumerate(PAIRS):
        z = x6[:, 2 * k] + 1j * x6[:, 2 * k + 1]
        X = X.at[:, a, b].set(z)
        X = X.at[:, b, a].set(-jnp.conj(z))
    return X


class FlagSector:
    def __init__(self, fm, U0, k, M=None):
        assert fm.disc == "align"
        self.fm, self.G, self.k = fm, fm.G, int(k)
        self.M = M if M is not None else 2 * self.k + 3
        self.phis = 2 * np.pi * np.arange(self.M) / self.M
        self.U0 = jnp.asarray(np.asarray(U0))
        G = self.G
        L = fm.L
        interior = ~(G.inf_mask | G.axis_mask)
        axis = G.axis_mask & ~G.inf_mask
        free = np.zeros((G.nn, NS), bool)
        free[interior, :] = True
        # axis: cos part of an allowed pair is free; its sin part is derived
        self.axis_sign = np.zeros((G.nn, 3))
        for p, (a, b) in enumerate(PAIRS):
            dl = L[b] - L[a]
            if self.k == 0:
                if dl == 0:
                    free[axis, 2 * p] = free[axis, 2 * p + 1] = True
            elif abs(dl) == self.k:
                free[axis, 2 * p] = free[axis, 2 * p + 1] = True
                self.axis_sign[axis, p] = 1.0 if dl == self.k else -1.0
        if self.k == 0:
            free[:, 6:] = False          # sin 0 = 0: no sin part at all
        self.free = free
        self.free_flat = np.nonzero(free.ravel())[0]
        self.nfree = int(free.sum())
        tbl = -np.ones(G.nn * NS, dtype=np.int64)
        tbl[self.free_flat] = np.arange(self.nfree)
        self.dof = tbl.reshape(G.nn, NS)
        ff = jnp.asarray(self.free_flat)
        sgn = jnp.asarray(self.axis_sign)
        is_axis = jnp.asarray(axis.astype(float))
        nn = G.nn

        def fields(x):
            x12 = jnp.zeros(nn * NS).at[ff].set(x).reshape(nn, NS)
            xc, xs = x12[:, :6], x12[:, 6:]
            # axis: X_s = sgn * i * X_c for the allowed pair  →  (Re, Im)_s = sgn * (−Im_c, Re_c)
            der = jnp.stack([-xc[:, 1::2], xc[:, 0::2]], axis=2).reshape(nn, 6) * jnp.repeat(sgn, 2, axis=1)
            xs = xs * (1.0 - is_axis[:, None]) + der * is_axis[:, None]
            return xc, xs

        eye = jnp.eye(3, dtype=jnp.complex128)

        def energy(x):
            xc, xs = fields(x)
            Xc, Xs = _x6_to_X(xc), _x6_to_X(xs)
            tot = 0.0
            for ph in self.phis:
                c, s = np.cos(self.k * ph), np.sin(self.k * ph)
                X = c * Xc + s * Xs
                Xp = self.k * (-s * Xc + c * Xs)
                A = jnp.linalg.inv(eye[None] - 0.5 * X)
                Cay = A @ (eye[None] + 0.5 * X)
                W = self.U0 @ Cay
                Wp = self.U0 @ (A @ (0.5 * Xp) @ (Cay + eye[None]))
                sig, sk, pot = self._density(W, Wp)
                tot = tot + jnp.sum(fm.W * (sig + sk + pot))
            return tot / self.M

        def mass_energy(x):
            xc, xs = fields(x)
            tot = 0.0
            for ph in self.phis:
                c, s = np.cos(self.k * ph), np.sin(self.k * ph)
                x6 = c * xc + s * xs
                # consistent L² form: interpolate the six real fields to quadrature points
                q = fm.Pv @ x6
                tot = tot + jnp.sum(fm.W[:, None] * q ** 2)
            return tot / self.M

        self._energy = jax.jit(energy)
        self._grad = jax.jit(jax.grad(energy))
        self._hvp = jax.jit(lambda x, d: jax.jvp(jax.grad(energy), (x,), (d,))[1])
        self._mhvp = jax.jit(lambda x, d: jax.jvp(jax.grad(mass_energy), (x,), (d,))[1])

    # --- energy density with a non-axial φ derivative -----------------------------
    def _density(self, W, Wp):
        fm = self.fm
        Ug = W[fm.a_cols]
        Ur = W[fm.a_ref]
        w = jnp.sum(jnp.conj(Ur) * Ug, axis=1)
        ph = jnp.conj(w) * jax.lax.rsqrt(_abs2(w) + 1e-300)
        Wa = Ug * ph[:, None, :]
        Pa = Wp[fm.a_cols] * ph[:, None, :]

        def ip(v, F):
            return jax.ops.segment_sum(v[:, None, None] * F, fm.a_rows, num_segments=fm.nq)
        Z, Zr, Zz, Zp = ip(fm.a_vv, Wa), ip(fm.a_vr, Wa), ip(fm.a_vz, Wa), ip(fm.a_vv, Pa)
        Zh = jnp.conj(jnp.swapaxes(Z, 1, 2))
        w_r, w_z = Zh @ Zr, Zh @ Zz
        w_f = (Zh @ (1j * fm.Lvec[None, :, None] * Z) + Zh @ Zp) / fm.rho[:, None, None]
        r, k3 = fm.r, fm.kappa3
        sig = 0.0
        for kk, (a, b) in enumerate(PAIRS):
            s_ab = 0.0
            for wi in (w_r, w_z, w_f):
                s_ab = s_ab + 0.5 * (_abs2(wi[:, a, b]) + _abs2(wi[:, b, a]))
            sig = sig + r[kk] * s_ab
        sk = 0.0
        comps = ((w_r, w_z), (w_r, w_f), (w_z, w_f))
        for a in range(3):
            for (wi, wj) in comps:
                F = 0.0
                for b in range(3):
                    if b != a:
                        F = F + 2.0 * jnp.imag(jnp.conj(wi[:, b, a]) * wj[:, b, a])
                sk = sk + 0.5 * fm.kappa[a] * F ** 2
        for (wi, wj) in comps:
            for a in range(3):
                for c in range(3):
                    if c == a:
                        continue
                    b = 3 - a - c
                    C = wi[:, a, b] * wj[:, b, c] - wj[:, a, b] * wi[:, b, c]
                    sk = sk + 0.5 * k3 * _abs2(C)
        pot = 0.0
        for a in range(3):
            for c in range(3):
                if c != a:
                    pot = pot + fm.m2[a] * _abs2(Z[:, c, a])
        return sig, sk, pot

    # --- assembly by colouring (as FlagProblem.hessian, 12 types) -----------------
    def _assemble(self, hvp):
        G, p = self.G, self.G.p
        x = jnp.zeros(self.nfree)
        period = 2 * p + 1
        rows_n = np.repeat(np.arange(G.nn), NS)
        rd = self.dof.ravel()
        ok_r = rd >= 0
        rows_n, rd = rows_n[ok_r], rd[ok_r]
        ri, rj = G.I[rows_n], G.J[rows_n]
        R, C, V = [], [], []
        for ci in range(period):
            for cj in range(period):
                colour = ((G.I % period) == ci) & ((G.J % period) == cj)
                di = (ci - ri + p) % period - p
                dj = (cj - rj + p) % period - p
                cin, cjn = ri + di, rj + dj
                ok = (cin >= 0) & (cin < G.nr) & (cjn >= 0) & (cjn < G.nz)
                ok &= _share_1d(ri, cin, p) & _share_1d(rj, cjn, p)
                cnode = np.where(ok, cin * G.nz + np.where(ok, cjn, 0), 0)
                for t in range(NS):
                    sel = colour & (self.dof[:, t] >= 0)
                    if not sel.any():
                        continue
                    d = np.zeros(self.nfree)
                    d[self.dof[sel, t]] = 1.0
                    col = np.asarray(hvp(x, jnp.asarray(d)))
                    cd = self.dof[cnode, t]
                    good = ok & (cd >= 0) & colour[cnode]
                    R.append(rd[good]); C.append(cd[good]); V.append(col[rd[good]])
        H = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                          shape=(self.nfree, self.nfree))
        return (0.5 * (H + H.T)).tocsr()

    def hessian(self):
        return self._assemble(self._hvp)

    def mass(self):
        return self._assemble(self._mhvp)

    def energy0(self):
        return float(self._energy(jnp.zeros(self.nfree)))

    def grad0(self):
        return np.asarray(self._grad(jnp.zeros(self.nfree)))


def count_below(H, M, s):
    return inertia((H - s * M).tocsr())[1]


def lowest(H, M, lo=-10.0, hi=10.0, tol=1e-4):
    """Lowest generalised eigenvalue by inertia bisection."""
    while count_below(H, M, lo) > 0:
        lo = 2 * lo - 1
    while count_below(H, M, hi) == 0:
        hi = 2 * hi + 1
    while hi - lo > tol * max(1.0, abs(hi)):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if count_below(H, M, mid) == 0 else (lo, mid)
    return 0.5 * (lo + hi)


def analyse(fm, U, k, label=""):
    t0 = time.time()
    S = FlagSector(fm, U, k)
    E0 = S.energy0()
    g = S.grad0()
    H = S.hessian()
    Mm = S.mass()
    res = dict(label=label, k=k, M=S.M, nfree=S.nfree, E0=E0, grad_max=float(np.abs(g).max()))
    res["n_below"] = {f"{s:g}": count_below(H, Mm, s) for s in (-1e-1, -1e-2, -1e-3, -1e-4, 1e-4, 1e-3)}
    res["lowest"] = lowest(H, Mm)
    res["t"] = time.time() - t0
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    from run_flag_pair import setup, fmodel, energy as axial_energy
    Gf, _ = setup(24, 32, 4)
    fm = fmodel(Gf, 2.0)
    U = np.load(sys.argv[1])
    ks = [int(s) for s in sys.argv[2:]] or [0, 1]
    print(f"axial energy {axial_energy(fm, U):.8f}", flush=True)
    out = [analyse(fm, U, k, os.path.basename(sys.argv[1])) for k in ks]
