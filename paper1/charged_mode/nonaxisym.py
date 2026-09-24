"""
Second variation of the charged, gauged Hopfion about its axisymmetric state,
one azimuthal sector at a time.

Co-rotating frame.  n = R₃(mφ) v(ρ,z,φ), A = A_ρ ρ̂ + A_z ẑ + A_φ φ̂.  With
T = e₃ × v, Φ = m/ρ + eA_φ and the physical derivatives
    p_ρ = ∂_ρ v,  p_z = ∂_z v,  q = ρ⁻¹ ∂_φ v,
the covariant derivatives are
    D_ρ v = p_ρ + eA_ρ T,  D_z v = p_z + eA_z T,  D_φ v = q + Φ T,
and (reduced with the unit-vector identities exactly as in hopfion_axisym)
    F_ρz = v·(p_ρ×p_z) + e(A_ρ p_z3 − A_z p_ρ3)
    F_ρφ = v·(p_ρ×q)  − Φ p_ρ3 + eA_ρ q_3
    F_zφ = v·(p_z×q)  − Φ p_z3 + eA_z q_3
    B_ρ = ρ⁻¹∂_φA_z − ∂_zA_φ,  B_φ = ∂_zA_ρ − ∂_ρA_z,
    B_z = ∂_ρA_φ + A_φ/ρ − ρ⁻¹∂_φA_ρ,
    ∇·A = ∂_ρA_ρ + A_ρ/ρ + ρ⁻¹∂_φA_φ + ∂_zA_z.
For φ-independent fields this is exactly the discrete energy of
hopfion_axisym, so the axisymmetric solution is a stationary point here too.

Perturbation in sector k (per node, 10 real coefficients):
    v = normalise(u₀ + Σ_i [c_i cos kφ + s_i sin kφ] e_i),   i = 1, 2 (tangent frame)
    A = A₀ + c_A cos kφ + s_A sin kφ.
The φ integral is done with M equally spaced samples (exact for the quadratic
form once M > 2k).  The Hessian is assembled by graph colouring with exact
Hessian-vector products from JAX.  The fixed-charge term enters through its
local part −(ω²/2)∫β²K with β frozen; the omitted non-local part is positive
semi-definite, so the computed Hessian is a lower bound of the true one.

Axis regularity (lab-frame smoothness at ρ = 0):
    k = 0 : δv = 0, δA_ρ = δA_φ = 0, δA_z free (cos part only)
    k = 1 : δv₁ + iδv₂ ∝ e^{-iφ}  (S₁ = −C₂, S₂ = C₁ in the axis frame
            e₁ = −x̂, e₂ = ŷ);  δA_ρ + iδA_φ ∝ e^{-iφ} (a_s = b_c, b_s = −a_c);
            δA_z = 0
    k ≥ 2 : every perturbation vanishes on the axis.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

jax.config.update("jax_enable_x64", True)

from newton import _frames, _share_1d  # noqa: E402

# per-node coefficient layout
CT1, CT2, ST1, ST2, CAR, CAZ, CAP, SAR, SAZ, SAP = range(10)
NT = 10


def _factorize(A):
    """Reusable sparse direct solver (PARDISO if available, else SuperLU)."""
    try:
        import pypardiso
        solver = pypardiso.PyPardisoSolver()
        solver.set_matrix_type(11)            # real nonsymmetric (robust for indefinite)
        Ac = sp.csr_matrix(A)
        solver.factorize(Ac)
        return lambda b: solver.solve(Ac, np.asarray(b, dtype=float))
    except Exception:
        lu = spla.splu(sp.csc_matrix(A))
        return lu.solve


def _bcoo(P):
    return jsparse.BCOO.from_scipy_sparse(sp.csr_matrix(P))


class SectorHessian:
    def __init__(self, model, U0, k, M=None):
        G = model.G
        assert not G.half, "use the full meridional plane"
        self.M_, self.G, self.k = model, G, k
        self.M = M if M is not None else (1 if k == 0 else 2 * k + 3)
        e, c2, c4, mu, m = model.e, model.c2, model.c4, model.mu, model.m
        self.U0 = U0.copy()
        u0 = U0[:, :3]
        e1, e2 = _frames(u0, G.sym_mask)
        self.e1, self.e2 = e1, e2
        # background charge sector (frozen)
        model.energy(U0)
        beta, I = model.last["beta"], model.last["I"]
        omega = model.N / I if model.charge else 0.0
        beta_q = G.Pv @ beta if model.charge else np.zeros(G.W.size)

        # ---- free-DOF mask and axis relations ---------------------------------
        free = np.zeros((G.nn, NT), bool)
        interior = ~(G.inf_mask | G.axis_mask)
        free[interior, :] = True
        if k == 0:
            free[:, [ST1, ST2, SAR, SAZ, SAP]] = False
        ax = G.axis_mask & ~G.inf_mask
        if k == 0:
            free[ax, CAZ] = True
        elif k == 1:
            assert np.allclose(e1[ax], [-1, 0, 0]) and np.allclose(e2[ax], [0, 1, 0])
            free[ax, CT1] = free[ax, CT2] = True
            free[ax, CAR] = free[ax, CAP] = True
        self.free = free
        self.axis_nodes = np.nonzero(ax)[0]
        self.nfree = int(free.sum())
        self.free_flat = np.nonzero(free.ravel())[0]
        # DOF index table (node, type) -> free index or -1
        tbl = -np.ones(G.nn * NT, dtype=np.int64)
        tbl[self.free_flat] = np.arange(self.nfree)
        self.dof = tbl.reshape(G.nn, NT)

        # ---- JAX constants ---------------------------------------------------
        phis = 2 * np.pi * np.arange(self.M) / self.M
        cs = np.cos(k * phis)
        sn = np.sin(k * phis)
        Pv, Pr, Pz = _bcoo(G.Pv), _bcoo(G.Pr), _bcoo(G.Pz)
        rho = jnp.asarray(G.rho)
        Wq = jnp.asarray(G.W) / self.M
        u0j, e1j, e2j = jnp.asarray(u0), jnp.asarray(e1), jnp.asarray(e2)
        A0 = jnp.asarray(U0[:, 3:6])
        csj, snj = jnp.asarray(cs)[:, None, None], jnp.asarray(sn)[:, None, None]
        free_flat = jnp.asarray(self.free_flat)
        axn = jnp.asarray(self.axis_nodes)
        nn, Mφ, kk = G.nn, self.M, float(k)
        betaq2 = jnp.asarray(beta_q ** 2)
        kap = model.kappa
        is_k1 = (k == 1)

        def interp(P, F):          # F: (M, nn, C) -> (M, nq, C)
            Mm, n, C = F.shape
            flat = jnp.transpose(F, (1, 0, 2)).reshape(n, Mm * C)
            out = P @ flat
            return jnp.transpose(out.reshape(-1, Mm, C), (1, 0, 2))

        def embed(x):
            X = jnp.zeros(nn * NT).at[free_flat].set(x).reshape(nn, NT)
            if is_k1 and axn.size:
                X = X.at[axn, ST1].set(-X[axn, CT2])
                X = X.at[axn, ST2].set(X[axn, CT1])
                X = X.at[axn, SAR].set(X[axn, CAP])
                X = X.at[axn, SAP].set(-X[axn, CAR])
            return X

        def energy(x):
            X = embed(x)
            c_t, s_t = X[None, :, 0:2], X[None, :, 2:4]
            c_A, s_A = X[None, :, 4:7], X[None, :, 7:10]
            t = c_t * csj + s_t * snj
            dt = kk * (-c_t * snj + s_t * csj)
            w = u0j[None] + t[..., 0:1] * e1j[None] + t[..., 1:2] * e2j[None]
            nw = jnp.sqrt(jnp.sum(w * w, axis=-1, keepdims=True))
            v = w / nw
            dw = dt[..., 0:1] * e1j[None] + dt[..., 1:2] * e2j[None]
            dv = (dw - v * jnp.sum(v * dw, axis=-1, keepdims=True)) / nw
            A = A0[None] + c_A * csj + s_A * snj
            dA = kk * (-c_A * snj + s_A * csj)
            F = jnp.concatenate([v, A], axis=-1)            # (M, nn, 6)
            Fd = jnp.concatenate([dv, dA], axis=-1)
            Qv, Qr, Qz = interp(Pv, F), interp(Pr, F), interp(Pz, F)
            Qf = interp(Pv, Fd) / rho[None, :, None]
            v1, v2, v3 = Qv[..., 0], Qv[..., 1], Qv[..., 2]
            Ar, Az, Ap = Qv[..., 3], Qv[..., 4], Qv[..., 5]
            pr, pz, q = Qr[..., 0:3], Qz[..., 0:3], Qf[..., 0:3]
            Ar_r, Az_r, Ap_r = Qr[..., 3], Qr[..., 4], Qr[..., 5]
            Ar_z, Az_z, Ap_z = Qz[..., 3], Qz[..., 4], Qz[..., 5]
            Ar_f, Az_f, Ap_f = Qf[..., 3], Qf[..., 4], Qf[..., 5]
            vv = Qv[..., 0:3]
            T = jnp.stack([-v2, v1, jnp.zeros_like(v1)], axis=-1)
            P = v1 * v1 + v2 * v2
            Phi = m / rho[None] + e * Ap
            Dr = pr + (e * Ar)[..., None] * T
            Dz = pz + (e * Az)[..., None] * T
            Df = q + Phi[..., None] * T
            cr = jnp.cross(pr, pz)
            F1 = jnp.sum(vv * cr, -1) + e * (Ar * pz[..., 2] - Az * pr[..., 2])
            F2 = jnp.sum(vv * jnp.cross(pr, q), -1) - Phi * pr[..., 2] + e * Ar * q[..., 2]
            F3 = jnp.sum(vv * jnp.cross(pz, q), -1) - Phi * pz[..., 2] + e * Az * q[..., 2]
            sig = 0.5 * c2 * (jnp.sum(Dr * Dr, -1) + jnp.sum(Dz * Dz, -1) + jnp.sum(Df * Df, -1))
            sk = 0.5 * c4 * (F1 ** 2 + F2 ** 2 + F3 ** 2)
            pot = 0.5 * c2 * mu * mu * (P + (v3 + 1.0) ** 2)
            Brho = Az_f - Ap_z
            Bphi = Ar_z - Az_r
            Bz = Ap_r + Ap / rho[None] - Ar_f
            div = Ar_r + Ar / rho[None] + Ap_f + Az_z
            mag = 0.5 * (Brho ** 2 + Bphi ** 2 + Bz ** 2) + 0.5 * kap * div ** 2
            K = c2 * P + c4 * (pr[..., 2] ** 2 + pz[..., 2] ** 2 + q[..., 2] ** 2)
            ch = -0.5 * omega ** 2 * betaq2[None] * K
            dens = sig + sk + pot + mag + ch
            return jnp.sum(Wq[None] * dens)

        self._energy = jax.jit(energy)
        self._grad = jax.jit(jax.grad(energy))
        x0 = jnp.zeros(self.nfree)
        self._hvp = jax.jit(lambda d: jax.jvp(jax.grad(energy), (x0,), (d,))[1])

    # -------------------------------------------------------------------------
    def check_stationary(self):
        g = np.asarray(self._grad(jnp.zeros(self.nfree)))
        return float(np.abs(g).max())

    def assemble(self):
        """Colouring with exact HVPs (distance-2p stencil of Q_p elements)."""
        G, p = self.G, self.G.p
        period = 2 * p + 1
        rows_n = np.repeat(np.arange(G.nn), NT)
        rows_t = np.tile(np.arange(NT), G.nn)
        rd = self.dof.ravel()
        ok_r = rd >= 0
        rows_n, rows_t, rd = rows_n[ok_r], rows_t[ok_r], rd[ok_r]
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
                    col = np.asarray(self._hvp(jnp.asarray(d)))
                    cd = self.dof[cnode, t]
                    good = ok & (cd >= 0) & colour[cnode]
                    R.append(rd[good])
                    C.append(cd[good])
                    V.append(col[rd[good]])
        H = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
                          shape=(self.nfree, self.nfree))
        asym = abs(H - H.T).max()
        H = 0.5 * (H + H.T)
        return H.tocsr(), float(asym)

    def mass(self, a_scale=1.0):
        """Consistent L² mass of the perturbation, ∫(|δv|² + a_scale|δA|²) d³x.

        With this metric the vacuum continuum of the matter sector starts at
        μ² (only the f² kinetic metric is used; the Skyrme part of the kinetic
        metric is left out, which can only lower the true frequencies).
        """
        G = self.G
        Mc = (G.PvT @ sp.diags(G.W) @ G.Pv).tocsr()
        fac = 1.0 if self.k == 0 else 0.5          # ⟨cos²⟩ = ⟨sin²⟩ = ½
        e = [self.e1, self.e2]
        zero = sp.csr_matrix((G.nn, G.nn))
        blocks = [[zero] * NT for _ in range(NT)]
        tan = ((CT1, ST1, 0), (CT2, ST2, 1))
        for (ci, si, a) in tan:
            for (cj, sj, b) in tan:
                Mab = sum(sp.diags(e[a][:, c]) @ Mc @ sp.diags(e[b][:, c]) for c in range(3))
                blocks[ci][cj] = fac * Mab
                blocks[si][sj] = fac * Mab
        for t in (CAR, CAZ, CAP, SAR, SAZ, SAP):
            blocks[t][t] = fac * a_scale * Mc
        Mbig = sp.bmat(blocks, format="csr")                    # type-major order
        perm = (np.arange(NT)[None, :] * G.nn + np.arange(G.nn)[:, None]).ravel()
        Mbig = Mbig[perm][:, perm]                               # node-major order
        Cm = self.constraint_matrix()
        return (Cm.T @ Mbig @ Cm).tocsr()

    def constraint_matrix(self):
        """Linear map from free DOFs to all node coefficients (axis relations)."""
        G = self.G
        rows = list(self.free_flat)
        cols = list(range(self.nfree))
        vals = [1.0] * self.nfree
        if self.k == 1:
            for n in self.axis_nodes:
                ct1, ct2 = self.dof[n, CT1], self.dof[n, CT2]
                car, cap = self.dof[n, CAR], self.dof[n, CAP]
                rows += [n * NT + ST1, n * NT + ST2, n * NT + SAR, n * NT + SAP]
                cols += [ct2, ct1, cap, car]
                vals += [-1.0, 1.0, 1.0, -1.0]
        return sp.csr_matrix((vals, (rows, cols)), shape=(G.nn * NT, self.nfree))

    def lowest(self, H, nev=10, sigma=-1e-3, a_scale=1.0):
        """Lowest generalised eigenpairs of H x = λ M x (shift-invert)."""
        Mm = self.mass(a_scale)
        lu = spla.splu((H - sigma * Mm).tocsc())
        op = spla.LinearOperator(H.shape, matvec=lu.solve, dtype=float)
        vals, vecs = spla.eigsh(H, k=nev, M=Mm, sigma=sigma, which="LM", OPinv=op)
        o = np.argsort(vals)
        return vals[o], vecs[:, o], Mm

    def split(self):
        """Free-DOF indices of the matter (tangent) and gauge-field coefficients."""
        types = np.tile(np.arange(NT), self.G.nn)[self.free_flat]
        return np.nonzero(types < 4)[0], np.nonzero(types >= 4)[0]

    def lowest_schur(self, H, nev=10, sigma=-1e-2):
        """Soliton spectrum with the gauge field relaxed for every matter mode.

        H ≥ 0  ⇔  H_AA > 0 and S = H_mm − H_mA H_AA⁻¹ H_Am ≥ 0.  H_AA is the
        gauge-fixed Maxwell operator plus e²M, positive definite with the
        fields vanishing at infinity.  S x = λ M_mm x is solved by shift-invert
        using (S − σM_mm)⁻¹ = [(H − σ diag(M_mm, 0))⁻¹]_mm.
        """
        im, ia = self.split()
        Mfull = self.mass(1.0)
        Mmm = Mfull[im][:, im].tocsr()
        Hmm = H[im][:, im].tocsr()
        HmA = H[im][:, ia].tocsr()
        HAm = H[ia][:, im].tocsr()
        solveA = _factorize(H[ia][:, ia])
        Mcoo = Mmm.tocoo()
        Mbig = sp.csr_matrix((Mcoo.data, (im[Mcoo.row], im[Mcoo.col])), shape=H.shape)
        solveK = _factorize(H - sigma * Mbig)
        nm = im.size

        def opinv(x):
            rhs = np.zeros(H.shape[0])
            rhs[im] = x
            return solveK(rhs)[im]

        def seff(x):
            return Hmm @ x - HmA @ solveA(HAm @ x)

        S = spla.LinearOperator((nm, nm), matvec=seff, dtype=float)
        OP = spla.LinearOperator((nm, nm), matvec=opinv, dtype=float)
        vals, vecs = spla.eigsh(S, k=nev, M=Mmm, sigma=sigma, which="LM", OPinv=OP, tol=1e-9)
        o = np.argsort(vals)
        full = np.zeros((H.shape[0], nev))
        full[im] = vecs[:, o]
        return vals[o], full, Mfull

    # generators of rigid motions (matter part) in free-DOF coordinates ----------
    def generators(self):
        """Tangent-coefficient vectors of x/y translation and x/y rotation (k = 1)."""
        G = self.G
        u = self.U0[:, :3]
        uu = u.reshape(G.nr, G.nz, 3)
        rr = G.rho_n.reshape(G.nr, G.nz)
        zz = G.z_n.reshape(G.nr, G.nz)
        with np.errstate(invalid="ignore", divide="ignore"):
            dr = np.zeros_like(uu)
            dz = np.zeros_like(uu)
            dr[1:-1] = (uu[2:] - uu[:-2]) / (rr[2:] - rr[:-2])[..., None]
            dz[:, 1:-1] = (uu[:, 2:] - uu[:, :-2]) / (zz[:, 2:] - zz[:, :-2])[..., None]
            T = np.stack([-uu[..., 1], uu[..., 0], np.zeros_like(rr)], -1)
            Tr = T / rr[..., None]
        dr, dz, Tr = (np.nan_to_num(a).reshape(-1, 3) for a in (dr, dz, Tr))
        rho = np.nan_to_num(G.rho_n, posinf=0.0)
        z = np.nan_to_num(G.z_n, posinf=0.0, neginf=0.0)
        gens = {}
        # translation along x: δv = cos φ ∂_ρ v − sin φ T/ρ
        gens["translation_x"] = (dr, -Tr)
        # rotation about x: δv = −(z/ρ) cos φ T + sin φ (ρ ∂_z v − z ∂_ρ v)
        gens["rotation_x"] = (-(z[:, None]) * Tr, rho[:, None] * dz - z[:, None] * dr)
        out = {}
        for name, (cvec, svec) in gens.items():
            X = np.zeros((G.nn, NT))
            X[:, CT1] = np.sum(cvec * self.e1, 1)
            X[:, CT2] = np.sum(cvec * self.e2, 1)
            X[:, ST1] = np.sum(svec * self.e1, 1)
            X[:, ST2] = np.sum(svec * self.e2, 1)
            out[name] = X.ravel()[self.free_flat]
        return out
