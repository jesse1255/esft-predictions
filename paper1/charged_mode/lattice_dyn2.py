"""
Time evolution with the complete kinetic term: a space-time lattice action (variational integrator).

lattice_dyn.py kept only the time component of the σ term.  That gives the solitons about a fifth of
their rest energy as inertia and misses the bound vibrations that §4.4 found with the full kinetic
metric.  Here every term of the static energy gets its time components, from the same gauge-invariant
lattice objects in the time direction:

    temporal links      O^t(x) = U_n(x)† U_{n+1}(x)                  (off-diagonal ≈ dt ω_t)
    temporal plaquettes (t, i): O^i_n(x), O^t(x + e_i), Ō^i_{n+1}(x), Ō^t(x)   (Berry phase ≈ h dt F_{0i})

    K(U_n, U_{n+1}) = h³/dt² Σ_x Σ_{a<b} r ½(|Õ^t_ab|² + |Õ^t_ba|²)                  σ
                    + h/dt²  Σ_{x,i} Σ_a ½κ_a (arg Π^t_a)²                               Faddeev
                    + h/dt²  Σ_{x,i} ¼ Σ_corners Σ_{a≠c} ½κ₃ |L1_ab L2_bc − L3_ab L4_bc|²   three-cycle

so that in the continuum L = K − V is the Lorentz-invariant completion of the static energy
(T = ∫ Σ r|ω_t^{ab}|² + ½κ_a Σ_i (F^a_{0i})² + ½κ₃ Σ_i |C^{ac}_{0i}|²).

Discrete Lagrangian L_d(q_n, q_{n+1}) = dt K(q_n, q_{n+1}) − ½dt (V(q_n) + V(q_{n+1})).  The discrete
Euler–Lagrange equations in position–momentum form:
    p̃ = p_n − ½dt ∇V(q_n)
    solve −dt D₁K(q_n, q_{n+1}) = p̃ for q_{n+1} = q_n Cay(X(y))      (Newton–GMRES; ≈ G y/dt = p̃)
    p_{n+1} = dt D₂K(q_n, q_{n+1}) − ½dt ∇V(q_{n+1})
All derivatives are taken in the right chart q Cay(X(δ)) at the point itself, so the momenta live in the
chart of the current field and nothing has to be transported between charts.  The scheme is symplectic
and second order; with K the σ part only it reduces to the leapfrog of lattice_dyn.py.

Energy monitor per step: E_{n+1/2} = K(q_n, q_{n+1}) + ½(V(q_n) + V(q_{n+1})).
Optional sponge: momenta multiplied by (1 − dt σ(x)) in a layer of width w at the box faces, so waves
leave instead of reflecting; the energy it removes is the radiated energy.

Usage:
    python lattice_dyn2.py test  N h dt            → rest; σ-only vs lattice_dyn; boosted vortex (E ≈ γM?)
    python lattice_dyn2.py collide N h V cfg dt T [sponge]  → two Lorentz-contracted rings at ±V
    python lattice_dyn2.py ring N h eps dt T                → free ringing of the relaxed Q = 2 ring
"""

import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from lattice3d import Lattice, charge, SCRATCH, DATA, PAIRS, _abs2, _dag, _sl, _arc

jax.config.update("jax_enable_x64", True)


def inv3(A):
    """Inverse of a batch of 3 × 3 matrices by the adjugate: elementwise arithmetic only.  (Many
    batched jnp.linalg.solve calls inside one compiled function hung the XLA CPU runtime here.)"""
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, hh, i = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    C00, C01, C02 = e * i - f * hh, -(d * i - f * g), d * hh - e * g
    C10, C11, C12 = -(b * i - c * hh), a * i - c * g, -(a * hh - b * g)
    C20, C21, C22 = b * f - c * e, -(a * f - c * d), a * e - b * d
    det = a * C00 + b * C01 + c * C02
    adj = jnp.stack([jnp.stack([C00, C10, C20], -1), jnp.stack([C01, C11, C21], -1), jnp.stack([C02, C12, C22], -1)], -2)
    return adj / det[..., None, None]


def cayley3(X):
    I = jnp.eye(3, dtype=X.dtype)
    return jnp.matmul(inv3(I - 0.5 * X), I + 0.5 * X)


def aligned_log(U, V):
    """Off-diagonal Y with U Cay(Y) ≅ V modulo the column phases of V (jnp version of
    lattice_dyn.log_map): the columns of V are first rotated so that diag(U†V) is real positive."""
    w = jnp.einsum("...ra,...ra->...a", jnp.conj(U), V)
    ph = jnp.conj(w) / jnp.maximum(jnp.abs(w), 1e-300)
    W = jnp.einsum("...ba,...bc->...ac", jnp.conj(U), V * ph[..., None, :])
    I = jnp.eye(3, dtype=W.dtype)
    Y = 2.0 * jnp.matmul(W - I, inv3(W + I))
    return Y - jnp.einsum("...aa->...a", Y)[..., None] * I


def centred_connection(U, h):
    """Ω_i(x) ≈ U†∂_iU (off-diagonal, gauge covariant at x) from the aligned logs to both neighbours,
    interior sites only: list of three (N−2)³ × 3 × 3 arrays."""
    out = []
    for i in range(3):
        Up = _sl(U, i, 2, None)
        Um = _sl(U, i, 0, -2)
        Uc = _sl(U, i, 1, -1)
        Om = (aligned_log(Uc, Up) - aligned_log(Uc, Um)) / (2.0 * h)
        idx = [slice(1, -1)] * 3
        idx[i] = slice(None)
        out.append(Om[tuple(idx)])
    return out


def spatial_links(U):
    Uh = _dag(U)
    return [_arc(jnp.matmul(Uh[:-1], U[1:])), _arc(jnp.matmul(Uh[:, :-1], U[:, 1:])),
            _arc(jnp.matmul(Uh[:, :, :-1], U[:, :, 1:]))]


class SpaceTime:
    def __init__(self, lat, dt, full=True, sponge=None, tol=1e-8, max_newton=10, local=True):
        self.lat, self.dt, self.full, self.local = lat, float(dt), bool(full), bool(local)
        self.tol, self.max_newton = float(tol), int(max_newton)
        self.M = 2.0 * lat.r[0] * lat.h ** 3
        from lattice3d import x6_to_X
        n = lat.N - 2

        def U_of(Ub, x):
            x6 = jnp.zeros((lat.N, lat.N, lat.N, 6)).at[1:-1, 1:-1, 1:-1].set(x.reshape(n, n, n, 6))
            return jnp.matmul(Ub, cayley3(x6_to_X(x6)))

        self.U_of = U_of
        zero = jnp.zeros(lat.nfree)
        dt = self.dt

        self.kin = jax.jit(self.kinetic)
        self.V = jax.jit(lat.energy_U)
        self.gradV = jax.jit(lambda U: jax.grad(lambda d: lat.energy_U(U_of(U, d)))(zero))
        self.D2K = jax.jit(lambda U0, U1: jax.grad(lambda d: self.kinetic(U0, U_of(U1, d)))(zero))

        def Phi(U0, y):
            return -dt * jax.grad(lambda d: self.kinetic(U_of(U0, d), U_of(U0, y)))(zero)

        self.Phi = jax.jit(Phi)

        def newton_update(U0, y, ptil, Ginv, m=12, rtol=1e-3):
            """One Newton step for −dt D₁K(q, q Cay(y)) = p̃, the linear system solved by right-
            preconditioned GMRES(m) written out here: Jacobian–vector products by a forward
            difference of Φ (one gradient each), preconditioner dt G_loc⁻¹ (the site-local continuum
            kinetic metric, 6 × 6 per site).  Returns the updated y and the residual before the update."""
            Py = np.asarray(Phi(U0, y))
            r = np.asarray(ptil) - Py
            beta = float(np.linalg.norm(r))
            if beta == 0.0:
                return y, 0.0
            Gi = np.asarray(Ginv)
            yn = np.asarray(y)
            ny = float(np.linalg.norm(yn))

            def prec(v):
                return dt * np.einsum("sij,sj->si", Gi, v.reshape(-1, 6)).ravel()

            def Jv(v):
                eps = 1e-7 * max(1.0, ny) / max(float(np.linalg.norm(v)), 1e-300)
                return (np.asarray(Phi(U0, jnp.asarray(yn + eps * v))) - Py) / eps

            V = [r / beta]
            H = np.zeros((m + 1, m))
            Z = []
            k_used = 0
            for k in range(m):
                z = prec(V[k])
                Z.append(z)
                w = Jv(z)
                for j in range(k + 1):
                    H[j, k] = float(np.dot(V[j], w))
                    w = w - H[j, k] * V[j]
                H[k + 1, k] = float(np.linalg.norm(w))
                k_used = k + 1
                e1 = np.zeros(k + 2)
                e1[0] = beta
                c, *_ = np.linalg.lstsq(H[:k + 2, :k + 1], e1, rcond=None)
                res = float(np.linalg.norm(H[:k + 2, :k + 1] @ c - e1))
                if res < rtol * beta or H[k + 1, k] < 1e-14 * beta:
                    break
                V.append(w / H[k + 1, k])
            dy = sum(ci * zi for ci, zi in zip(c, Z))
            self.last_gmres = k_used
            return jnp.asarray(yn + dy), beta

        def newton_block(U0, y, ptil, Ginv):
            """Newton step with the block-diagonal Jacobian dt⁻¹ G_loc (exact to leading order for
            the local kinetic term): one gradient per iteration."""
            r = ptil - Phi(U0, y)
            dy = dt * jnp.einsum("sij,sj->si", Ginv, r.reshape(-1, 6)).ravel()
            return y + dy, jnp.linalg.norm(r)

        self.newton_update = newton_update if not self.local else jax.jit(newton_block)
        self.last_gmres = 0
        self.pre_every = 2 if self.local else 10
        self._Ginv, self._pre_count = None, 0
        self.sig = None
        if sponge is not None:
            width, smax = sponge
            n = lat.N - 2
            x = lat.x1[1:-1]
            L = lat.x1[-1]
            d = np.maximum(np.maximum(np.abs(x)[:, None, None], np.abs(x)[None, :, None]), np.abs(x)[None, None, :])
            s = smax * np.clip((d - (L - width)) / width, 0.0, None) ** 2
            self.sig = jnp.asarray(np.repeat(s[..., None], 6, -1).ravel())

    def local_metric_inv(self, U):
        """Inverse of the site-local continuum kinetic metric G(x) (6 × 6, interior sites), from the
        centred gauge-covariant spatial connection Ω_i (centred_connection):
        T(x)/h³ = Σ_{a<b} r|ω₀^{ab}|² + Σ_a ½κ_a Σ_i (2 Im Σ_b ω̄₀^{ba}Ω_i^{ba})² + ½κ₃ Σ_i Σ_{a≠c}|ω₀^{ab}Ω_i^{bc} − Ω_i^{ab}ω₀^{bc}|²."""
        if not self.full:
            n = (self.lat.N - 2) ** 3
            return jnp.asarray(np.broadcast_to(np.eye(6) / self.M, (n, 6, 6)).copy())
        f = getattr(self, "_lmi", None)
        if f is None:
            from lattice3d import x6_to_X
            lat = self.lat
            r, kap, k3, h = lat.r, lat.kappa, lat.kappa3, lat.h

            def t_site(x6, Wi):
                X = x6_to_X(x6)
                T = 0.0
                for k, (a, b) in enumerate(PAIRS):
                    T = T + r[k] * _abs2(X[a, b])
                for i in range(3):
                    w = Wi[i]
                    for a in range(3):
                        F = 0.0
                        for b in range(3):
                            if b != a:
                                F = F + 2.0 * jnp.imag(jnp.conj(X[b, a]) * w[b, a])
                        T = T + 0.5 * kap[a] * F ** 2
                    for a in range(3):
                        for c in range(3):
                            if c == a:
                                continue
                            b = 3 - a - c
                            T = T + 0.5 * k3 * _abs2(X[a, b] * w[b, c] - w[a, b] * X[b, c])
                return T * h ** 3

            conn = jax.jit(lambda U: jnp.stack([o.reshape(-1, 3, 3) for o in centred_connection(U, h)], axis=1))
            hess = jax.jit(jax.vmap(jax.hessian(t_site), in_axes=(None, 0)))

            def lmi(U):
                # three separate steps: compiled as one function this hung in the XLA CPU runtime
                G = hess(jnp.zeros(6), conn(U))
                return jnp.asarray(np.linalg.inv(np.asarray(G)))

            f = self._lmi = lmi
        return f(jnp.asarray(U))

    def kinetic_local(self, U0, U1):
        """Site-local version: σ part from the temporal links as in kinetic(); the quartic and
        three-cycle time components from the continuum formulas at each interior site, with the
        velocity W = aligned_log(U0, U1)/dt and the centred spatial connection Ω_i of the midpoint
        field U0 Cay(W/2) (W is the same matrix in the frames of U0 and of the midpoint, so the
        products are consistent and the form is symmetric under U0 ↔ U1)."""
        lat, h, dt = self.lat, self.lat.h, self.dt
        Ot = _arc(jnp.matmul(_dag(U0), U1))
        Ks = 0.0
        for k, (a, b) in enumerate(PAIRS):
            Ks = Ks + lat.r[k] * 0.5 * jnp.sum(_abs2(Ot[..., a, b]) + _abs2(Ot[..., b, a]))
        Ks = Ks * h ** 3 / dt ** 2
        if not self.full:
            return Ks
        W = aligned_log(U0, U1)
        Um = jnp.matmul(U0, cayley3(0.5 * W))
        Om = centred_connection(Um, h)
        Wi = W[1:-1, 1:-1, 1:-1]
        KF, KC = 0.0, 0.0
        for i in range(3):
            w = Om[i]
            for a in range(3):
                F = 0.0
                for b in range(3):
                    if b != a:
                        F = F + 2.0 * jnp.imag(jnp.conj(Wi[..., b, a]) * w[..., b, a])
                KF = KF + 0.5 * lat.kappa[a] * jnp.sum(F ** 2)
            if lat.kappa3 != 0.0:
                for a in range(3):
                    for c in range(3):
                        if c == a:
                            continue
                        b = 3 - a - c
                        KC = KC + 0.5 * lat.kappa3 * jnp.sum(_abs2(Wi[..., a, b] * w[..., b, c] - w[..., a, b] * Wi[..., b, c]))
        return Ks + (KF + KC) * h ** 3 / dt ** 2

    def kinetic(self, U0, U1):
        if self.local:
            return self.kinetic_local(U0, U1)
        lat, h, dt = self.lat, self.lat.h, self.dt
        Ot = _arc(jnp.matmul(_dag(U0), U1))
        Ks = 0.0
        for k, (a, b) in enumerate(PAIRS):
            Ks = Ks + lat.r[k] * 0.5 * jnp.sum(_abs2(Ot[..., a, b]) + _abs2(Ot[..., b, a]))
        Ks = Ks * h ** 3 / dt ** 2
        if not self.full:
            return Ks
        O0, O1 = spatial_links(U0), spatial_links(U1)
        KF, KC = 0.0, 0.0
        for i in range(3):
            A, Cc = O0[i], O1[i]                                   # O^i at t_n and at t_{n+1}
            B, D = _sl(Ot, i, 1, None), _sl(Ot, i, 0, -1)          # O^t(x + e_i), O^t(x)
            for a in range(3):
                F = jnp.angle(A[..., a, a] * B[..., a, a] * jnp.conj(Cc[..., a, a]) * jnp.conj(D[..., a, a]))
                KF = KF + 0.5 * lat.kappa[a] * jnp.sum(F ** 2)
            if lat.kappa3 != 0.0:
                Ad, Bd, Cd, Dd = _dag(A), _dag(B), _dag(Cc), _dag(D)
                corners = ((A, B, D, Cc), (Ad, D, B, Cd), (Cc, Bd, Dd, A), (Cd, Dd, Bd, Ad))
                for a in range(3):
                    for c in range(3):
                        if c == a:
                            continue
                        b = 3 - a - c
                        for L1, L2, L3, L4 in corners:
                            Cv = L1[..., a, b] * L2[..., b, c] - L3[..., a, b] * L4[..., b, c]
                            KC = KC + 0.125 * lat.kappa3 * jnp.sum(_abs2(Cv))
        return Ks + (KF + KC) * h / dt ** 2

    def momentum_from_velocity(self, U0, v0):
        """p_0 such that the first step moves by exactly y = dt v0."""
        return self.Phi(U0, self.dt * v0) + 0.5 * self.dt * self.gradV(U0)

    def step(self, U, p, y_guess, gV=None):
        dt = self.dt
        if gV is None:
            gV = self.gradV(U)
        ptil = p - 0.5 * dt * gV
        if self._Ginv is None or self._pre_count % self.pre_every == 0:
            self._Ginv = self.local_metric_inv(U)
        self._pre_count += 1
        y = y_guess
        scale = max(1.0, float(jnp.linalg.norm(ptil)))
        for k in range(self.max_newton):
            y_new, res = self.newton_update(U, y, ptil, self._Ginv)
            if float(res) < self.tol * scale:
                break
            y = y_new
        U1 = self.U_of(U, y)
        gV1 = self.gradV(U1)
        p1 = dt * self.D2K(U, U1) - 0.5 * dt * gV1
        if self.sig is not None:
            p1 = p1 * (1.0 - dt * self.sig)
        return U1, p1, y, gV1, k, float(res)


def link_content(lat, U):
    """P_ab = h³ Σ_x ½(|U_ab|² + |U_ba|²) for the pairs 01, 12, 02: how much of each link the field
    carries (14.2 for one Q = 1 vortex, 21.0 for the Q = 2 ring, h = 0.3)."""
    Un = np.asarray(U)
    return {f"{a}{b}": float(lat.h ** 3 * np.sum(0.5 * (np.abs(Un[..., a, b]) ** 2 + np.abs(Un[..., b, a]) ** 2)))
            for a, b in ((0, 1), (1, 2), (0, 2))}


def radiated_energy_outside(lat, U, radius):
    Un = np.asarray(U)
    q = sum(1.0 - np.abs(Un[..., c, c]) ** 2 for c in range(3))
    r = np.sqrt(lat.X ** 2 + lat.Y ** 2 + lat.Z ** 2)
    return float(lat.h ** 3 * np.sum(q * (r > radius)))


def run(st, U, v0, steps, every, tag, extra=None, save_every=None, trace_every=None):
    """trace_every: also record cheap observables (V, line weights and rms radii) every so many steps,
    for spectra of the ringing."""
    lat = st.lat
    U = jnp.asarray(U)
    p = st.momentum_from_velocity(U, jnp.asarray(v0))
    y = st.dt * jnp.asarray(v0)
    gV = st.gradV(U)
    rows, trace, t0 = [], [], time.time()
    E0 = None
    y_prev = None
    for n in range(steps + 1):
        Vn = float(st.V(U))
        if trace_every and n % trace_every == 0:
            gm = lat.geometry(np.asarray(U))
            trace.append(dict(t=n * st.dt, V=Vn, w=[gm[c]["weight"] for c in range(3)],
                              R=[gm[c].get("radius") for c in range(3)], m=[gm[c].get("moments") for c in range(3)]))
        # guess: linear extrapolation of the chart displacement (the charts of consecutive steps
        # differ by O(dt|v|), so this is only a starting point for Newton)
        yg = y if y_prev is None else 2.0 * y - y_prev
        y_prev = y
        U1, p1, y, gV1, nk, res = st.step(U, p, yg, gV)
        if n % every == 0:
            K = float(st.kin(U, U1))
            V1 = float(st.V(U1))
            E = K + 0.5 * (Vn + V1)
            E0 = E if E0 is None else E0
            Q, Qa, _ = charge(lat, np.asarray(U), pad=1)
            gm = lat.geometry(np.asarray(U))
            row = dict(step=n, t=n * st.dt, T=K, V=Vn, E=E, Q=Q, lines=[gm[c]["weight"] for c in range(3)],
                       centroid0=gm[0].get("centroid"), centroid2=gm[2].get("centroid"),
                       outside4=radiated_energy_outside(lat, U, 4.0), links=link_content(lat, U), newton=nk, res=res,
                       wall=round(time.time() - t0, 1))
            rows.append(row)
            print(json.dumps({k: (round(x, 5) if isinstance(x, float) else x) for k, x in row.items()}), flush=True)
            if save_every and n % save_every == 0:
                np.savez_compressed(os.path.join(SCRATCH, f"dyn2_{tag}_t{n * st.dt:.2f}.npz"), U=np.asarray(U), p=np.asarray(p))
        U, p, gV = U1, p1, gV1
    out = dict(tag=tag, N=lat.N, h=lat.h, dt=st.dt, full=st.full, rows=rows)
    if trace:
        out["trace"] = trace
    if extra:
        out.update(extra)
    json.dump(out, open(os.path.join(DATA, f"flagpair_dyn2_{tag}.json"), "w"), indent=1)
    return U, p, rows


class _Coords:
    pass


def ring_boosted(Gf, u, lat, R, a, block, gamma, axis=0):
    """smith.ring evaluated at Lorentz-contracted coordinates about its centre a (motion along axis)."""
    from smith import ring
    c = _Coords()
    X = [lat.X, lat.Y, lat.Z]
    X[axis] = a[axis] + gamma * (X[axis] - a[axis])
    c.X, c.Y, c.Z = X
    return ring(Gf, u, c, R, a, block)


def cmd_test(N, h, dt):
    from lattice_dyn import chart_velocity_of_translation, log_map
    from lattice_string import X_to_x
    lat = Lattice(N, h)
    U = np.load(os.path.join(SCRATCH, f"lattice_ref_A01_N{N}_h{h:g}.npz"))["U"]
    E1 = lat.energy(U)
    st = SpaceTime(lat, dt, full=True)
    print("rest, full kinetic term", flush=True)
    run(st, U, np.zeros(lat.nfree), int(round(1.0 / dt)), int(round(0.2 / dt)), f"test_rest_N{N}_dt{dt:g}")
    print("sigma-only kinetic term, moving at 0.3 (compare lattice_dyn test_move)", flush=True)
    st0 = SpaceTime(lat, dt, full=False)
    v = chart_velocity_of_translation(lat, U, 0, 0.3)
    run(st0, U, v, int(round(3.0 / dt)), int(round(0.6 / dt)), f"test_move_sigma_N{N}_dt{dt:g}")
    print("full kinetic term, Lorentz-contracted vortex moving at 0.5: E should be close to gamma*E1 =",
          E1 / np.sqrt(1 - 0.25), flush=True)
    from run_flag_pair import setup
    from smith import ring
    Gf, Uf = setup(24, 32, 4)
    u = Uf[:, :3]
    Vb = 0.5
    g = 1.0 / np.sqrt(1.0 - Vb ** 2)
    Ua = ring_boosted(Gf, u, lat, np.eye(3), (0.0, 0.0, 0.0), (0, 1), g)
    Ub_ = ring_boosted(Gf, u, lat, np.eye(3), (1e-3 * Vb, 0.0, 0.0), (0, 1), g)
    for s in (Ua, Ub_):
        s[0], s[-1], s[:, 0], s[:, -1], s[:, :, 0], s[:, :, -1] = (np.eye(3),) * 6
    v = X_to_x(lat, log_map(Ua, Ub_)) / 1e-3
    ring0 = ring_boosted(Gf, u, lat, np.eye(3), (0.0, 0.0, 0.0), (0, 1), 1.0)
    ring0[0], ring0[-1], ring0[:, 0], ring0[:, -1], ring0[:, :, 0], ring0[:, :, -1] = (np.eye(3),) * 6
    run(st, Ua, v, int(round(4.0 / dt)), int(round(0.4 / dt)), f"test_boost_N{N}_dt{dt:g}",
        extra=dict(V=Vb, gamma=g, E_static_relaxed=E1, E_static_unrelaxed_ring=lat.energy(ring0)))


def cmd_collide(N, h, V, cfg, dt, T, sponge=None, s=1.2):
    from run_flag_pair import setup
    from smith import smooth_pair, configs
    from lattice_dyn import log_map
    from lattice_string import X_to_x
    Gf, Uf = setup(24, 32, 4)
    u = Uf[:, :3]
    lat = Lattice(N, h)
    (RA, aA), (RB, aB) = configs(0.93)[cfg]
    g = 1.0 / np.sqrt(1.0 - V ** 2)

    def field(shift):
        UA = ring_boosted(Gf, u, lat, RA, (aA[0] - s + shift, aA[1], aA[2]), (0, 1), g)
        UB = ring_boosted(Gf, u, lat, RB, (aB[0] + s - shift, aB[1], aB[2]), (1, 2), g)
        U = smooth_pair(UA, UB)[0]
        U[0], U[-1], U[:, 0], U[:, -1], U[:, :, 0], U[:, :, -1] = (np.eye(3),) * 6
        return U

    U = field(0.0)
    d = 1e-3
    v = X_to_x(lat, log_map(U, field(V * d))) / d
    st = SpaceTime(lat, dt, full=True, sponge=sponge)
    tag = f"collide_{cfg}_V{V:g}_N{N}" + ("_sponge" if sponge else "")
    steps = int(round(T / dt))
    run(st, U, v, steps, max(1, int(round(0.5 / dt))), tag, extra=dict(V=V, gamma=g, cfg=cfg, s=s, sponge=sponge),
        save_every=max(1, int(round(5.0 / dt))))


def cmd_ring(N, h, eps, dt, T, sponge=(1.5, 3.0), ref="fused02"):
    """Ringing of the relaxed Q = 2 ring (§4.1/§5.4, link 02): start at the static solution with the
    velocity of a slow dilation, U(x, t) = U(x/(1 + ε t)), i.e. ∂_tU = −ε (x·∇)U, and follow the
    free oscillation with the full kinetic term.  The rms radius of the ring oscillates at the
    frequencies of the axisymmetric in-block vibrations; §4.4 found one bound one, ω² = 0.465, with
    the full kinetic metric (the continuum starts at ω = 1)."""
    from lattice_dyn import log_map
    from lattice_string import X_to_x
    lat = Lattice(N, h)
    U = np.load(os.path.join(SCRATCH, f"lattice_ref_{ref}_N{N}_h{h:g}.npz"))["U"]
    P = [lat.X, lat.Y, lat.Z]
    Wt = 0.0
    for i in range(3):
        Wi = (log_map(U, np.roll(U, -1, axis=i)) - log_map(U, np.roll(U, 1, axis=i))) / (2.0 * h)
        Wt = Wt - eps * P[i][..., None, None] * Wi
    v = X_to_x(lat, Wt)
    st = SpaceTime(lat, dt, full=True, sponge=sponge)
    tag = f"ring_{ref}_eps{eps:g}_N{N}"
    run(st, U, v, int(round(T / dt)), max(1, int(round(1.0 / dt))), tag, extra=dict(eps=eps, ref=ref, sponge=sponge),
        trace_every=2)


if __name__ == "__main__":
    cmd = sys.argv[1]
    N, h = int(sys.argv[2]), float(sys.argv[3])
    if cmd == "test":
        cmd_test(N, h, float(sys.argv[4]))
    elif cmd == "ring":
        cmd_ring(N, h, float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))
    elif cmd == "collide":
        sp = (1.5, 3.0) if len(sys.argv) > 8 and sys.argv[8] == "sponge" else None
        cmd_collide(N, h, float(sys.argv[4]), sys.argv[5], float(sys.argv[6]), float(sys.argv[7]), sponge=sp)
