"""
The flag model with openable cores: each line gets an amplitude, ψ_a = ρ_a u_a.

Motivation (the user's objection, 2026-09-27): in the flag sigma model the three lines
are orthonormal with fixed length everywhere, so the cores of two different vortices
cannot pass through each other; the energy of the crossing point diverges like 1/h
(§5.6).  A divergence marks where a model is incomplete (as the ideal-MHD frozen-in
theorem is for magnetic reconnection).  Here the cores may open: a line's amplitude
can drop to zero, like the order parameter in a superfluid vortex core.

Continuum energy (ρ_a real, U = (u_0, u_1, u_2) unitary, gauge U(1)³ on the right):
    Σ_a |∇ρ_a|²                                     (amplitudes)
  + Σ_{a<b} (ρ_a² + ρ_b²) |ω^{ab}|²                 (= Σ_a |Dψ_a|² − |∇ρ_a|²: the linear kinetic
                                                       term of the lines; at ρ = 1 it is the
                                                       flag-model σ term with r = 2)
  + Σ_a ½κ_a ρ_a⁴ (F^{(a)}_ij)²                      (Faddeev term of line a, weighted by its amplitude)
  + Σ_{a≠c} ½κ₃ ρ_a²ρ_c² |C^{ac}_ij|²                (three-cycle term, weighted by its end lines)
  + Σ_a m_a² ρ_a² (1 − |U_aa|²)                      (mass term)
  + (λ/4) Σ_a (ρ_a² − 1)²                           (the one new coupling: core stiffness)
At ρ ≡ 1 this is exactly lattice3d.Lattice (M2′: r = 2, κ = 2, m² = 1, κ₃ = 2).  As
λ → ∞ the amplitudes freeze and the sigma model returns; the amplitude mode has mass² = λ
(so the core size is ε ~ 1/√λ).  With finite λ the configuration space is contractible:
the Hopf charge is no longer protected, and a vortex, or a crossing, costs a finite energy.

Lattice: the link/plaquette discretisation of lattice3d, each term multiplied by powers of
the amplitudes averaged over the link or the plaquette corners (so every weight is ≥ 0);
amplitude gradients on links.  Degrees of freedom per interior site: 6 (frame, Cayley chart) + 3 (ρ).
"""

import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from lattice3d import (Lattice, PAIRS, _abs2, _dag, _sl, _arc, cayley, x6_to_X, SCRATCH, DATA)

jax.config.update("jax_enable_x64", True)


class OpenLattice(Lattice):
    def __init__(self, N, h, lam=10.0, **kw):
        super().__init__(N, h, **kw)
        self.lam = float(lam)
        n = self.N - 2
        self.nfree = 9 * n ** 3

    def parts_UR(self, U, R):
        """U: (N,N,N,3,3) unitary frame; R: (N,N,N,3) amplitudes."""
        h = self.h
        Uh = _dag(U)
        O = [jnp.matmul(Uh[:-1], U[1:]), jnp.matmul(Uh[:, :-1], U[:, 1:]), jnp.matmul(Uh[:, :, :-1], U[:, :, 1:])]
        O = [_arc(Oi) for Oi in O]
        R2 = R ** 2
        # link weights: the square of the link-averaged amplitude (non-negative for any signs; a
        # product ρ(x)ρ(x + e) would turn negative across a sign change and the energy would be
        # unbounded below, which is what a first version did)
        Rl = [(0.5 * (R[:-1] + R[1:])) ** 2, (0.5 * (R[:, :-1] + R[:, 1:])) ** 2, (0.5 * (R[:, :, :-1] + R[:, :, 1:])) ** 2]
        Es, Er = 0.0, 0.0
        for i, Oi in enumerate(O):
            for k, (a, b) in enumerate(PAIRS):
                w = 0.5 * (Rl[i][..., a] + Rl[i][..., b])
                Es = Es + self.r[k] * 0.5 * jnp.sum(w * (_abs2(Oi[..., a, b]) + _abs2(Oi[..., b, a])))
        Es = Es * h
        for d in range(3):
            Er = Er + jnp.sum((_sl(R, d, 1, None) - _sl(R, d, 0, -1)) ** 2)
        Er = Er * h
        EF, EC = 0.0, 0.0
        for d1, d2 in ((0, 1), (0, 2), (1, 2)):
            O1, O2 = O[d1], O[d2]
            A, Cc = _sl(O1, d2, 0, -1), _sl(O1, d2, 1, None)
            B, D = _sl(O2, d1, 1, None), _sl(O2, d1, 0, -1)
            # amplitudes at the four plaquette corners
            R00 = _sl(_sl(R, d1, 0, -1), d2, 0, -1)
            R10 = _sl(_sl(R, d1, 1, None), d2, 0, -1)
            R01 = _sl(_sl(R, d1, 0, -1), d2, 1, None)
            R11 = _sl(_sl(R, d1, 1, None), d2, 1, None)
            Rm = 0.25 * (R00 + R10 + R01 + R11)                   # mean amplitude on the plaquette
            Rp4 = Rm ** 4                                         # ρ⁴ weight of each line (non-negative)
            for a in range(3):
                F = jnp.angle(A[..., a, a] * B[..., a, a] * jnp.conj(Cc[..., a, a]) * jnp.conj(D[..., a, a]))
                EF = EF + 0.5 * self.kappa[a] * jnp.sum(Rp4[..., a] * F ** 2)
            if self.kappa3 != 0.0:
                Ad, Bd, Cd, Dd = _dag(A), _dag(B), _dag(Cc), _dag(D)
                corners = ((A, B, D, Cc), (Ad, D, B, Cd), (Cc, Bd, Dd, A), (Cd, Dd, Bd, Ad))
                for a in range(3):
                    for c in range(3):
                        if c == a:
                            continue
                        b = 3 - a - c
                        wac = (Rm[..., a] * Rm[..., c]) ** 2
                        for L1, L2, L3, L4 in corners:
                            Cv = L1[..., a, b] * L2[..., b, c] - L3[..., a, b] * L4[..., b, c]
                            EC = EC + 0.125 * self.kappa3 * jnp.sum(wac * _abs2(Cv))
        EF, EC = EF / h, EC / h
        EV = 0.0
        for a in range(3):
            EV = EV + self.m2[a] * jnp.sum(R2[..., a] * (1.0 - _abs2(U[..., a, a])))
        EL = 0.25 * self.lam * jnp.sum((R2 - 1.0) ** 2)
        return Es, EF, EC, EV * h ** 3, Er, EL * h ** 3

    def energy_UR(self, U, R):
        return sum(self.parts_UR(U, R))

    def parts2(self, U, R):
        f = getattr(self, "_jp2", None)
        if f is None:
            f = self._jp2 = jax.jit(self.parts_UR)
        vals = f(jnp.asarray(U), jnp.asarray(R))
        return dict(zip(("sigma", "faddeev", "three_cycle", "potential", "amp_grad", "amp_pot"),
                        (float(v) for v in vals)))

    def UR_of(self, Ub, Rb, x):
        n = self.N - 2
        x = x.reshape(n, n, n, 9)
        x6 = jnp.zeros((self.N, self.N, self.N, 6)).at[1:-1, 1:-1, 1:-1].set(x[..., :6])
        dR = jnp.zeros((self.N, self.N, self.N, 3)).at[1:-1, 1:-1, 1:-1].set(x[..., 6:])
        return Ub @ cayley(x6_to_X(x6)), Rb + dR


class Precond9:
    """Frame components as lattice3d.Precond; amplitudes with 2h³(−Δ + λ)."""

    def __init__(self, lat):
        from scipy.fft import dstn
        self.dstn = dstn
        n = lat.N - 2
        k = np.arange(1, n + 1)
        lam1 = (2.0 - 2.0 * np.cos(np.pi * k / (n + 1))) / lat.h ** 2
        L = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
        f6 = (4.0 * lat.h ** 3 * (L + 1.0)) ** -0.5
        f3 = (2.0 * lat.h ** 3 * (L + max(lat.lam, 1.0))) ** -0.5
        self.f = np.concatenate([np.repeat(f6[..., None], 6, -1), np.repeat(f3[..., None], 3, -1)], -1)
        self.shape = (n, n, n, 9)

    def __call__(self, v):
        a = self.dstn(np.asarray(v).reshape(self.shape), type=1, axes=(0, 1, 2), norm="ortho")
        a *= self.f
        return self.dstn(a, type=1, axes=(0, 1, 2), norm="ortho").ravel()


def relax_open(lat, U0, R0=None, chunk=100, max_chunks=40, gtol=1e-3, log=None, verbose=True, describe=None):
    """Unconstrained L-BFGS for the open model (frame chart + amplitudes), base moved every chunk."""
    Ub = jnp.asarray(U0)
    Rb = jnp.ones(U0.shape[:3] + (3,)) if R0 is None else jnp.asarray(R0)
    S = Precond9(lat)
    vg = jax.jit(jax.value_and_grad(lambda Ub, Rb, x: lat.energy_UR(*lat.UR_of(Ub, Rb, x)), argnums=2))
    y0 = np.zeros(lat.nfree)
    hist, t0, it = [], time.time(), 0
    for ch in range(max_chunks):
        def fun(y):
            E, g = vg(Ub, Rb, jnp.asarray(S(y)))
            return float(E), S(np.asarray(g))
        res = minimize(fun, y0, jac=True, method="L-BFGS-B", options=dict(maxiter=chunk, maxcor=30, gtol=0.1 * gtol, ftol=1e-16))
        it += res.nit
        Ub, Rb = lat.UR_of(Ub, Rb, jnp.asarray(S(res.x)))
        E, g = vg(Ub, Rb, jnp.zeros(lat.nfree))
        gmax = float(np.abs(S(np.asarray(g))).max())
        row = dict(chunk=ch, it=it, E=float(E), gmax=gmax, rho_min=[float(v) for v in jnp.min(Rb, axis=(0, 1, 2))],
                   t=round(time.time() - t0, 1))
        if describe is not None:
            row.update(describe(np.asarray(Ub), np.asarray(Rb)))
        hist.append(row)
        if verbose:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        if log:
            with open(log, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        if gmax < gtol or (len(hist) > 1 and abs(hist[-1]["E"] - hist[-2]["E"]) < 1e-5):
            break
    return np.asarray(Ub), np.asarray(Rb), hist
