"""
A₂ flag-manifold Hopf solitons in axial symmetry.

Order parameter.  A Hermitian 3×3 X with non-degenerate spectrum defines three
orthogonal eigen-projectors P_a = Z_a Z_a†, i.e. a point of the flag manifold
F₂ = U(3)/U(1)³ (real dimension 6).  The Weyl group of A₂, S₃, permutes the P_a.
π₂(F₂) = ℤ², π₃(F₂) = π₃(U(3)) = ℤ, so Hopf-type solitons exist.

Energy (Faddeev–Skyrme type, no gauge field):
    E = ∫ d³x [ Σ_{a<b} r_ab |ω_i^{ab}|²  +  Σ_a (κ_a/2) Σ_{i<j} (F^{(a)}_{ij})²
                + (κ₃/2) Σ_{i<j} Σ_{a≠c} |C^{ac}_{ij}|²  +  Σ_a m_a² Σ_{c≠a} |Z_{ca}|² ],
    C^{ac}_{ij} = ω_i^{ab} ω_j^{bc} − ω_j^{ab} ω_i^{bc}   ({a, b, c} = {1, 2, 3}).
κ₃ = 0 is model M2 of PAPER1_REVIEW §4.2.  Its quartic term vanishes identically
on real flags (U ∈ SO(3), all F^{(a)} = 0), which carry Hopf degree 4k, so in
that sector the energy has infimum 0 (Derrick collapse).  κ₃ = κ_a = κ gives the
full commutator term (κ/2) Σ_{i<j} ‖[ω_i^off, ω_j^off]‖² (model M2').
    ω_i = U†∂_i U  (U = [Z_1 Z_2 Z_3]),   F^{(a)}_{ij} = 2 Im Σ_{b≠a} ω̄_i^{ba} ω_j^{ba}.
F^{(a)} is the pull-back of the Kähler form of the P_a factor
(F^{(a)}_{ij} = −i Tr P_a[∂_iP_a, ∂_jP_a]).  The vacuum is U = 1 (diagonal
phases are gauge).  With r = 2f², κ = 2/g², m² = f²μ² the configurations with
Z_3 = e_3 fixed are exactly the CP¹ Faddeev–Skyrme model with potential
f²μ²(1 − n_3) (the "embedded" Hopfions; three copies related by S₃).

Axial symmetry.  U(ρ,φ,z) = R_L(φ) Û(ρ,z), R_L = diag(e^{il_1φ}, e^{il_2φ}, e^{il_3φ});
then ω_φ = Û†(iL)Û/ρ.  The embedded m = 1 Hopfion has (l_1, l_2) = (0, 1);
l_3 labels the axially symmetric sector of the F₂ deformations.  On the axis
the columns must be eigenvectors of L: an off-diagonal tangent coordinate x_ab
is free there only if l_a = l_b.

Numerics as in hopfion_axisym (compactified Q2 grid, full meridional plane).
Tangent coordinates x_ab (a<b, complex) at each node, retraction
U = U₀ · Cay(X), Cay(X) = (1 − X/2)⁻¹(1 + X/2).  JAX gives exact gradients and
Hessian-vector products; Hessians are assembled by colouring.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

jax.config.update("jax_enable_x64", True)

from hopfion_axisym import sparse_solve   # noqa: E402
from newton import _share_1d              # noqa: E402

PAIRS = ((0, 1), (0, 2), (1, 2))
NT = 6          # (Re, Im) of x_01, x_02, x_12


def _bcoo(P):
    return jsparse.BCOO.from_scipy_sparse(sp.csr_matrix(P))


def _abs2(z):
    """|z|² written without abs(): jnp.abs has no second derivative at z = 0,
    and the normal components vanish identically on the embedded state."""
    return jnp.real(z) ** 2 + jnp.imag(z) ** 2


def cayley(X):
    I = jnp.eye(3, dtype=X.dtype)
    return jnp.linalg.solve(I - 0.5 * X, I + 0.5 * X)


def x_to_X(x6):
    """(n, 6) real -> (n, 3, 3) anti-Hermitian off-diagonal."""
    n = x6.shape[0]
    X = jnp.zeros((n, 3, 3), dtype=jnp.complex128)
    for k, (a, b) in enumerate(PAIRS):
        z = x6[:, 2 * k] + 1j * x6[:, 2 * k + 1]
        X = X.at[:, a, b].set(z)
        X = X.at[:, b, a].set(-jnp.conj(z))
    return X


class FlagModel:
    def __init__(self, grid, L=(0, 1, 2), r=(2.0, 2.0, 2.0), kappa=(2.0, 2.0, 2.0),
                 m2=(1.0, 1.0, 1.0), kappa3=0.0, disc="u"):
        G = grid
        assert not G.half
        assert disc in ("u", "proj", "align")
        self.G = G
        self.disc = disc
        self.L = tuple(int(l) for l in L)
        self.r = jnp.asarray(np.array(r, dtype=float))      # (r_01, r_02, r_12)
        self.kappa, self.m2 = kappa, m2
        self.kappa3 = float(kappa3)
        self.Pv, self.Pr, self.Pz = _bcoo(G.Pv), _bcoo(G.Pr), _bcoo(G.Pz)
        self.rho = jnp.asarray(G.rho)
        self.W = jnp.asarray(G.W)
        self.Lvec = jnp.asarray(np.array(self.L, dtype=float))
        if disc == "align":
            # element-local gauge: every node's column phases are aligned with the element's
            # centre node (odd I, odd J) before interpolation, so a change of nodal phases
            # multiplies each element by one overall phase per column and drops out.
            S = (abs(G.Pv) + abs(G.Pr) + abs(G.Pz)).tocoo()
            rows, cols = S.row, S.col
            centre = (G.I[cols] % 2 == 1) & (G.J[cols] % 2 == 1)
            nq = G.Pv.shape[0]
            ref = -np.ones(nq, dtype=np.int64)
            ref[rows[centre]] = cols[centre]
            assert np.all(ref >= 0), "every quadrature point needs an element centre node"
            val = lambda P: np.asarray(sp.csr_matrix(P)[rows, cols]).ravel()
            self.nq = nq
            self.a_rows, self.a_cols = jnp.asarray(rows), jnp.asarray(cols)
            self.a_ref = jnp.asarray(ref[rows])
            self.a_vv, self.a_vr, self.a_vz = (jnp.asarray(val(P)) for P in (G.Pv, G.Pr, G.Pz))

    def _interp_aligned(self, U):
        """Z, ∂_ρZ, ∂_zZ at the quadrature points in the element-local aligned gauge."""
        Ug = U[self.a_cols]                                    # (entries, 3, 3)
        Ur = U[self.a_ref]
        w = jnp.sum(jnp.conj(Ur) * Ug, axis=1)                  # <Z_a(centre), Z_a(n)>
        ph = jnp.conj(w) * jax.lax.rsqrt(_abs2(w) + 1e-300)
        Ua = Ug * ph[:, None, :]

        def ip(v):
            return jax.ops.segment_sum(v[:, None, None] * Ua, self.a_rows, num_segments=self.nq)
        return ip(self.a_vv), ip(self.a_vr), ip(self.a_vz)

    # ---------------------------------------------------------------------------
    def density_terms(self, U, r=None, k3=None):
        """Energy density pieces at the quadrature points for nodal U (nn,3,3).

        r (optional, may be traced) overrides the sigma couplings (r_01, r_02, r_12);
        k3 (optional, may be traced) overrides the three-cycle coefficient κ₃.
        """
        r = self.r if r is None else r
        with_c = (k3 is not None) or (self.kappa3 != 0.0)
        k3 = self.kappa3 if k3 is None else k3
        if self.disc == "proj":
            return self._density_proj(U, r, k3, with_c)
        if self.disc == "align":
            Z, Zr, Zz = self._interp_aligned(U)
        else:
            nn = U.shape[0]
            flat = jnp.concatenate([U.real.reshape(nn, 9), U.imag.reshape(nn, 9)], axis=1)

            def ip(P):
                o = P @ flat
                return (o[:, :9] + 1j * o[:, 9:]).reshape(-1, 3, 3)
            Z, Zr, Zz = ip(self.Pv), ip(self.Pr), ip(self.Pz)
        Zh = jnp.conj(jnp.swapaxes(Z, 1, 2))
        w_r = Zh @ Zr
        w_z = Zh @ Zz
        LZ = 1j * self.Lvec[None, :, None] * Z
        w_f = (Zh @ LZ) / self.rho[:, None, None]
        sig = 0.0
        for k, (a, b) in enumerate(PAIRS):
            s_ab = 0.0
            for w in (w_r, w_z, w_f):
                s_ab = s_ab + 0.5 * (_abs2(w[:, a, b]) + _abs2(w[:, b, a]))
            sig = sig + r[k] * s_ab
        sk = 0.0
        comps = ((w_r, w_z), (w_r, w_f), (w_z, w_f))
        for a in range(3):
            for (wi, wj) in comps:
                F = 0.0
                for b in range(3):
                    if b != a:
                        F = F + 2.0 * jnp.imag(jnp.conj(wi[:, b, a]) * wj[:, b, a])
                sk = sk + 0.5 * self.kappa[a] * F ** 2
        if with_c:
            # off-diagonal (three-cycle) part of [ω_i^off, ω_j^off]:
            #   C^{ac}_{ij} = ω_i^{ab} ω_j^{bc} − ω_j^{ab} ω_i^{bc},  {a, b, c} = {0, 1, 2}.
            # Its diagonal part is −i F^{(a)}_{ij}, so κ3 = κ gives the full commutator
            # (κ/2) Σ_{i<j} ‖[ω_i^off, ω_j^off]‖².  Zero on every embedded CP¹ configuration.
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
                    pot = pot + self.m2[a] * _abs2(Z[:, c, a])
        return sig, sk, pot

    def _density_proj(self, U, r, k3, with_c):
        """Gauge-invariant discretisation: interpolate the projectors P_a = Z_a Z_a†.

        The nodal column phases drop out exactly, so the discrete energy is a
        function of the nodal flags only.  With D_ρ, D_z the derivatives of the
        interpolated P̂_a and D_φ P̂ = (i/ρ)[L, P̂] (the R_L conjugation drops out of
        every trace):
            |ω_i^{ab}|²   = −½ Tr(D_iP_a D_iP_b)                        (a ≠ b)
            F^{(a)}_{ij}  = −i Tr(P_a [D_iP_a, D_jP_a])
            |C^{ac}_{ij}|² = ‖P_a (D_iP_b P_b D_jP_c − D_jP_b P_b D_iP_c) P_c‖²
            1 − |Z_aa|²   = Σ_{c≠a} Σ_d |(P_a)_cd|².
        For exact projectors these are the formulas of the U discretisation.
        """
        nn = U.shape[0]
        Pn = jnp.einsum("nia,nja->naij", U, jnp.conj(U))                 # (nn, a, i, j)
        flat = jnp.concatenate([Pn.real.reshape(nn, 27), Pn.imag.reshape(nn, 27)], axis=1)

        def ip(Pm):
            o = Pm @ flat
            return (o[:, :27] + 1j * o[:, 27:]).reshape(-1, 3, 3, 3)
        P, Pr, Pz = ip(self.Pv), ip(self.Pr), ip(self.Pz)
        Lm = jnp.diag(self.Lvec).astype(jnp.complex128)
        Pf = 1j * (jnp.einsum("ij,qajk->qaik", Lm, P) - jnp.einsum("qaij,jk->qaik", P, Lm))
        Pf = Pf / self.rho[:, None, None, None]
        D = (Pr, Pz, Pf)

        def tr(A, B):
            return jnp.real(jnp.einsum("qij,qji->q", A, B))
        sig = 0.0
        for k, (a, b) in enumerate(PAIRS):
            s_ab = 0.0
            for Di in D:
                s_ab = s_ab - 0.5 * tr(Di[:, a], Di[:, b])
            sig = sig + r[k] * s_ab
        comps = ((0, 1), (0, 2), (1, 2))
        sk = 0.0
        for a in range(3):
            for (i, j) in comps:
                Di, Dj = D[i][:, a], D[j][:, a]
                comm = Di @ Dj - Dj @ Di
                F = jnp.real(-1j * jnp.einsum("qij,qji->q", P[:, a], comm))
                sk = sk + 0.5 * self.kappa[a] * F ** 2
        if with_c:
            for (i, j) in comps:
                for a in range(3):
                    for c in range(3):
                        if c == a:
                            continue
                        b = 3 - a - c
                        M = P[:, a] @ (D[i][:, b] @ P[:, b] @ D[j][:, c]
                                       - D[j][:, b] @ P[:, b] @ D[i][:, c]) @ P[:, c]
                        sk = sk + 0.5 * k3 * jnp.real(jnp.einsum("qij,qij->q", M, jnp.conj(M)))
        # 1 − |Z_aa|² = Σ_{c≠a} Σ_d |(P_a)_cd|² for an exact projector.  This form is a sum of
        # squares of interpolated entries, whose leading part near the vacuum is the
        # consistent mass |x|²; the linear form 1 − (P_a)_aa would act like a lumped mass,
        # which is not positive on the stretched outer elements.
        pot = 0.0
        for a in range(3):
            sq = jnp.real(P[:, a]) ** 2 + jnp.imag(P[:, a]) ** 2          # (q, c, d)
            pot = pot + self.m2[a] * (jnp.sum(sq, axis=(1, 2)) - jnp.sum(sq[:, a, :], axis=1))
        return sig, sk, pot

    def energy_U(self, U, r=None, k3=None):
        sig, sk, pot = self.density_terms(U, r, k3)
        return jnp.sum(self.W * (sig + sk + pot))

    def parts(self, U, r=None, k3=None):
        sig, sk, pot = self.density_terms(jnp.asarray(U), r, k3)
        return {k: float(jnp.sum(self.W * v)) for k, v in (("sigma", sig), ("skyrme", sk), ("pot", pot))}

    # ---------------------------------------------------------------------------
    def free_mask(self, types=range(NT)):
        """(nn, 6) boolean: interior all free; axis only degenerate-l pairs."""
        G = self.G
        free = np.zeros((G.nn, NT), bool)
        interior = ~(G.inf_mask | G.axis_mask)
        for t in types:
            free[interior, t] = True
        ax = G.axis_mask & ~G.inf_mask
        for k, (a, b) in enumerate(PAIRS):
            if self.L[a] == self.L[b]:
                for t in (2 * k, 2 * k + 1):
                    if t in types:
                        free[ax, t] = True
        return free


class FlagProblem:
    """Energy as a function of tangent coordinates x at a base point U0.

    The jitted functions take U0, the sigma couplings r and κ₃ as arguments, so
    moving the base point (Newton re-anchoring) or changing r, κ₃ (continuation
    in the couplings) does not trigger recompilation.
    """

    def __init__(self, fm, U0, types=range(NT)):
        self.fm, self.G = fm, fm.G
        self.types = tuple(types)
        self.free = fm.free_mask(self.types)
        self.free_flat = np.nonzero(self.free.ravel())[0]
        self.nfree = int(self.free.sum())
        tbl = -np.ones(self.G.nn * NT, dtype=np.int64)
        tbl[self.free_flat] = np.arange(self.nfree)
        self.dof = tbl.reshape(self.G.nn, NT)
        ff = jnp.asarray(self.free_flat)
        nn = self.G.nn

        def U_of(U0, x):
            x6 = jnp.zeros(nn * NT).at[ff].set(x).reshape(nn, NT)
            return U0 @ cayley(x_to_X(x6))

        def energy(U0, x, r, k3):
            return fm.energy_U(U_of(U0, x), r, k3)

        self._U_of = jax.jit(U_of)
        self._energy = jax.jit(energy)
        self._grad = jax.jit(jax.grad(energy, argnums=1))
        self._hvp = jax.jit(lambda U0, x, d, r, k3: jax.jvp(
            lambda y: jax.grad(energy, argnums=1)(U0, y, r, k3), (x,), (d,))[1])
        self.r = fm.r
        self.k3 = jnp.asarray(fm.kappa3)
        self.set_base(U0)

    def set_base(self, U0):
        self.U0 = jnp.asarray(np.asarray(U0))

    def set_couplings(self, r=None, kappa3=None):
        if r is not None:
            self.r = jnp.asarray(np.array(r, dtype=float))
        if kappa3 is not None:
            self.k3 = jnp.asarray(float(kappa3))

    def U_of(self, x):
        return self._U_of(self.U0, x)

    def energy(self, x):
        return self._energy(self.U0, x, self.r, self.k3)

    def grad(self, x):
        return self._grad(self.U0, x, self.r, self.k3)

    def hessian(self, x=None):
        G, p = self.G, self.G.p
        x = jnp.zeros(self.nfree) if x is None else x
        period = 2 * p + 1
        rows_n = np.repeat(np.arange(G.nn), NT)
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
                for t in range(NT):
                    sel = colour & (self.dof[:, t] >= 0)
                    if not sel.any():
                        continue
                    d = np.zeros(self.nfree)
                    d[self.dof[sel, t]] = 1.0
                    col = np.asarray(self._hvp(self.U0, x, jnp.asarray(d), self.r, self.k3))
                    cd = self.dof[cnode, t]
                    good = ok & (cd >= 0) & colour[cnode]
                    R.append(rd[good]); C.append(cd[good]); V.append(col[rd[good]])
        H = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                          shape=(self.nfree, self.nfree))
        return (0.5 * (H + H.T)).tocsr()

    def mass(self):
        """L² metric of the perturbation: |δU|² = |X|² = 2 Σ_{a<b} |x_ab|²."""
        G = self.G
        Mc = (G.PvT @ sp.diags(G.W) @ G.Pv).tocsr()
        blocks = sp.block_diag([2.0 * Mc] * NT, format="csr")       # type-major
        perm = (np.arange(NT)[None, :] * G.nn + np.arange(G.nn)[:, None]).ravel()
        Mbig = blocks[perm][:, perm]
        return Mbig[self.free_flat][:, self.free_flat].tocsr()

    def lowest(self, H, nev=6, sigma=-1e-2):
        Mm = self.mass()
        A = (H - sigma * Mm).tocsr()
        try:
            import pypardiso
            s = pypardiso.PyPardisoSolver()
            s.set_matrix_type(11)
            s.factorize(A)
            solve = lambda b: s.solve(A, np.asarray(b, dtype=float))
        except Exception:
            lu = spla.splu(A.tocsc())
            solve = lu.solve
        op = spla.LinearOperator(H.shape, matvec=solve, dtype=float)
        vals, vecs = spla.eigsh(H, k=nev, M=Mm, sigma=sigma, which="LM", OPinv=op, tol=1e-9)
        o = np.argsort(vals)
        return vals[o], vecs[:, o]


def inertia(A):
    """(n_positive, n_negative, n_perturbed_pivots) of a sparse symmetric matrix
    from the Bunch–Kaufman LDLᵀ factorisation in PARDISO (Sylvester's law)."""
    import pypardiso
    s = pypardiso.PyPardisoSolver()
    s.set_matrix_type(-2)
    s.factorize(sp.triu(sp.csr_matrix(A), format="csr"))
    ip = s.get_iparms()
    out = (int(ip[22]), int(ip[23]), int(ip[14]))
    s.free_memory(everything=True)
    return out


def critical_ratio(H0, H1, lam_hi, nev=4):
    """Smallest θ of H0 v = θ H1 v (H1 ≻ 0), i.e. λ_c = −θ_min is the smallest λ with
    H0 + λH1 ⪰ 0.  Shift-invert about −lam_hi, which must lie below θ_min
    (checked by the caller with inertia(H0 + lam_hi H1))."""
    import pypardiso
    A = (H0 + lam_hi * H1).tocsr()
    s = pypardiso.PyPardisoSolver()
    s.set_matrix_type(11)
    s.factorize(A)
    op = spla.LinearOperator(H0.shape, matvec=lambda b: s.solve(A, np.asarray(b, dtype=float)),
                             dtype=float)
    vals, vecs = spla.eigsh(H0, k=nev, M=H1, sigma=-lam_hi, which="LM", OPinv=op, tol=1e-10)
    o = np.argsort(vals)
    return vals[o], vecs[:, o]


def newton_relax(fm, U0, types=range(NT), max_iter=60, tol=1e-9, lam0=1e-3, verbose=True,
                 lam_floor=1e-8, prob=None):
    """Damped Newton in tangent coordinates with re-anchoring after each step."""
    prob = FlagProblem(fm, U0, types) if prob is None else prob
    prob.set_base(U0)
    U = np.asarray(U0)
    hist = []
    lam = lam0
    E = None
    x0 = jnp.zeros(prob.nfree)
    for it in range(max_iter):
        prob.set_base(U)
        E = float(prob.energy(x0))
        g = np.asarray(prob.grad(x0))
        H = prob.hessian()
        dH = np.abs(H.diagonal()) + 1e-300
        gs = float(np.abs(g / np.sqrt(dH)).max())
        hist.append(dict(it=it, E=E, gscaled=gs, lam=lam))
        if verbose:
            print(f"    it {it:2d} E={E:.10f} |g/sqrtH|={gs:.2e} lam={lam:.0e}", flush=True)
        if gs < tol:
            return U, E, hist, True
        accepted = False
        for _ in range(14):
            dx = sparse_solve(H + max(lam, lam_floor) * sp.diags(dH), -g)
            if not np.all(np.isfinite(dx)):
                lam = max(lam * 10.0, 1e-6)
                continue
            En = float(prob.energy(jnp.asarray(dx)))
            pred = float(g @ dx + 0.5 * dx @ (H @ dx))
            if -pred < 1e-12 * abs(E) and En <= E + 1e-12 * abs(E):
                U = np.asarray(prob.U_of(jnp.asarray(dx)))
                return U, En, hist, True
            if En < E + 1e-4 * min(pred, 0.0):
                ratio = (En - E) / pred if pred < 0 else 1.0
                U = np.asarray(prob.U_of(jnp.asarray(dx)))
                lam = lam * 0.2 if ratio > 0.75 else lam
                accepted = True
                break
            lam = max(lam * 10.0, 1e-6)
        if not accepted:
            return U, E, hist, False
    return U, E, hist, False


# ---------------------------------------------------------------------------
# initial conditions
# ---------------------------------------------------------------------------

def embedded_hopfion(grid, h_nodes, block=(0, 1)):
    """Hopf projection of the SU(2) hedgehog embedded in the (a,b) block.

    z = (cos h + i sin h cosθ, i sin h sinθ) with the e^{iφ} of the second
    component carried by R_L (l_b − l_a = 1).  Z_a = z (in rows a, b), Z_b = z⊥.
    """
    r, ct, st = grid.node_r_theta()
    h = np.where(np.isfinite(r), h_nodes, 0.0)
    z1 = np.cos(h) + 1j * np.sin(h) * ct
    z2 = 1j * np.sin(h) * st
    a, b = block
    c = 3 - a - b
    U = np.zeros((grid.nn, 3, 3), complex)
    U[:, a, a] = z1
    U[:, b, a] = z2
    U[:, a, b] = -np.conj(z2)
    U[:, b, b] = np.conj(z1)
    U[:, c, c] = 1.0
    fin = ~np.isfinite(r)
    U[fin] = np.eye(3)
    return U
