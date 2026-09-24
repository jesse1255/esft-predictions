"""
Axisymmetric charged, gauged Hopfion: finite-element core.

Model (units hbar = c = 1; same effective action as the 2026-09-08 note):

    E_N = ∫ d³x [ f²/2 |D n|² + 1/(4 g²) Σ_ij (n·D_i n × D_j n)²
                  + f² μ² (1 + n_3) + ½ |∇×A|² ]  +  N² / (2 I_eff),

    D_i n = ∂_i n + e A_i (e_3 × n),
    I_eff = min_β ∫ d³x [ β² K + |∇β|² / e² ],   β → 1 at infinity,
    K     = f² (1 - n_3²) + g⁻² |∇n_3|².

The fixed-charge term is exact for the isorotating ansatz with Gauss's law
solved: the electric potential is A_0 = ω (1 - β) / e with ω = N / I_eff.

Axial symmetry with azimuthal winding m:

    n = R_3(m φ) u(ρ, z),   u ∈ S²,   A = A_ρ ρ̂ + A_z ẑ + A_φ φ̂.

All quantities reduce to the meridional half plane.  With T = e_3 × u,
Φ = m/ρ + e A_φ:

    |D n|²      = |∂_ρ u + e A_ρ T|² + |∂_z u + e A_z T|² + Φ² |T|²
    Σ_ij F_ij²  = 2 [ F_ρz² + Φ² |∇u_3|² ]
    F_ρz        = u·(∂_ρ u × ∂_z u) + e (A_ρ ∂_z u_3 - A_z ∂_ρ u_3)

Discretisation: tensor-product Lagrange elements (order p = 1 or 2) on the
compactified grid ρ = a tan(πξ/2), z = a tan(πη/2), so the outer nodes sit at
spatial infinity (β = 1, A = 0, u = -e_3 imposed exactly there).  Gauss
quadrature with p+1 points per direction.  u is stored as a 3-vector at the
nodes; |T|² = u_1² + u_2² is used for (1 - u_3²), which is exact for unit
vectors and keeps the correct O(ρ²) behaviour on the axis.

With half=True only z ≥ 0 is stored and the reflection symmetry
(z → -z, u_2 → -u_2, A_ρ → -A_ρ, A_z → A_z, A_φ → A_φ) is imposed; every
integral is doubled so energies refer to all of space.

Gauge fixing.  When the matter phase is free, the gauge redundancy
(u phase, A_mer) is removed with the term κ/2 (∇·A)².  Because any
axisymmetric configuration can be gauge-transformed to ∇·A = 0 without
changing the energy, adding this term does not change the minimum.  When u is
held fixed (reproducing the h(r) calculations), κ is set to a tiny value
only to regularise the exterior; the gradient part of A then acts as a
phase relaxation of u, which is exactly the effect described in the note.
"""

import glob
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Optional MKL PARDISO (much faster sparse direct solves); SuperLU otherwise.
if "PYPARDISO_MKL_RT" not in os.environ:
    _cands = glob.glob("/usr/local/lib/libmkl_rt.so*") + glob.glob("/usr/lib/libmkl_rt.so*")
    if _cands:
        os.environ["PYPARDISO_MKL_RT"] = _cands[0]
try:
    import pypardiso as _pardiso
except Exception:  # pragma: no cover - fallback path
    _pardiso = None


def sparse_solve(A, b):
    """Solve A x = b for sparse A (PARDISO if available)."""
    if _pardiso is not None:
        return _pardiso.spsolve(sp.csr_matrix(A), np.asarray(b, dtype=float))
    return spla.spsolve(sp.csc_matrix(A), b)

TWO_PI = 2.0 * np.pi
FOUR_PI = 4.0 * np.pi

# Component order of the nodal array U[:, k]
U1, U2, U3, AR, AZ, AP = range(6)


# ---------------------------------------------------------------------------
# 1-D Lagrange elements
# ---------------------------------------------------------------------------

def _lagrange_reference(p):
    """Values/derivatives of order-p Lagrange basis at (p+1) Gauss points."""
    s_nodes = np.linspace(-1.0, 1.0, p + 1)
    gp, gw = np.polynomial.legendre.leggauss(p + 1)
    V = np.zeros((gp.size, p + 1))
    D = np.zeros((gp.size, p + 1))
    for k in range(p + 1):
        others = [s_nodes[m] for m in range(p + 1) if m != k]
        denom = np.prod([s_nodes[k] - o for o in others])
        V[:, k] = np.prod([gp - o for o in others], axis=0) / denom
        d = np.zeros_like(gp)
        for j in range(len(others)):
            rest = [gp - others[m] for m in range(len(others)) if m != j]
            d += np.prod(rest, axis=0) if rest else np.ones_like(gp)
        D[:, k] = d / denom
    return gp, gw, V, D


def _build_1d(nodes, p):
    """Sparse value/derivative interpolation from nodes to Gauss points."""
    ne = (nodes.size - 1) // p
    gp, gw, V, D = _lagrange_reference(p)
    nq = ne * (p + 1)
    rows, cols, v0, v1 = [], [], [], []
    xq = np.zeros(nq)
    wq = np.zeros(nq)
    for el in range(ne):
        idx = np.arange(el * p, el * p + p + 1)
        x0, x1 = nodes[el * p], nodes[el * p + p]
        hel = x1 - x0
        for g in range(p + 1):
            q = el * (p + 1) + g
            xq[q] = x0 + 0.5 * (gp[g] + 1.0) * hel
            wq[q] = 0.5 * gw[g] * hel
            rows.extend([q] * (p + 1))
            cols.extend(idx.tolist())
            v0.extend(V[g].tolist())
            v1.extend((D[g] * 2.0 / hel).tolist())
    shape = (nq, nodes.size)
    B0 = sp.csr_matrix((v0, (rows, cols)), shape=shape)
    B1 = sp.csr_matrix((v1, (rows, cols)), shape=shape)
    return B0, B1, xq, wq


def _map(s, a):
    """Compactifying map x = a tan(π s / 2) and its derivative."""
    c = np.cos(0.5 * np.pi * s)
    with np.errstate(divide="ignore"):
        x = a * np.tan(0.5 * np.pi * s)
        dx = 0.5 * np.pi * a / c ** 2
    return x, dx


class Grid:
    """Mapped tensor grid on the meridional half plane."""

    def __init__(self, ne_r, ne_z, p=2, a=3.0, half=True):
        self.p, self.a, self.half = p, a, half
        self.ne_r, self.ne_z = ne_r, ne_z
        self.xi = np.linspace(0.0, 1.0, p * ne_r + 1)
        if half:
            self.eta = np.linspace(0.0, 1.0, p * ne_z + 1)
        else:
            self.eta = np.linspace(-1.0, 1.0, 2 * p * ne_z + 1)
        self.nr, self.nz = self.xi.size, self.eta.size
        self.nn = self.nr * self.nz

        Br0, Br1, xq, wx = _build_1d(self.xi, p)
        Bz0, Bz1, eq, we = _build_1d(self.eta, p)
        self.nqr, self.nqz = xq.size, eq.size

        rho_q, drho_q = _map(xq, a)
        z_q, dz_q = _map(eq, a)
        self.rho = np.repeat(rho_q, self.nqz)
        self.zq = np.tile(z_q, self.nqr)
        self.jr = np.repeat(drho_q, self.nqz)      # dρ/dξ at quad points
        self.jz = np.tile(dz_q, self.nqr)          # dz/dη
        wxy = np.outer(wx, we).ravel()
        self.area = wxy * self.jr * self.jz        # dρ dz
        sym = 2.0 if half else 1.0
        self.W = sym * TWO_PI * self.rho * self.area   # d³x weight
        self.W2 = sym * self.area                      # dρ dz (full z)

        self.Pv = sp.kron(Br0, Bz0, format="csr")
        Pxi = sp.kron(Br1, Bz0, format="csr")
        Peta = sp.kron(Br0, Bz1, format="csr")
        self.Pr = sp.diags(1.0 / self.jr) @ Pxi
        self.Pz = sp.diags(1.0 / self.jz) @ Peta
        self.Pr = self.Pr.tocsr()
        self.Pz = self.Pz.tocsr()
        self.PvT = self.Pv.T.tocsr()
        self.PrT = self.Pr.T.tocsr()
        self.PzT = self.Pz.T.tocsr()

        # node coordinates (inf at the compactified boundary)
        rn, _ = _map(self.xi, a)
        zn, _ = _map(self.eta, a)
        rn[-1] = np.inf
        if half:
            zn[-1] = np.inf
        else:
            zn[0], zn[-1] = -np.inf, np.inf
        self.rho_n = np.repeat(rn, self.nz)
        self.z_n = np.tile(zn, self.nr)
        I, J = np.meshgrid(np.arange(self.nr), np.arange(self.nz), indexing="ij")
        self.I, self.J = I.ravel(), J.ravel()

        self.inf_mask = (self.I == self.nr - 1) | (self.J == self.nz - 1)
        if not half:
            self.inf_mask |= self.J == 0
        self.axis_mask = self.I == 0
        self.sym_mask = (self.J == 0) if half else np.zeros(self.nn, bool)

        # Free-DOF mask per component
        free = np.ones((self.nn, 6), bool)
        free[self.inf_mask, :] = False
        free[self.axis_mask, U1] = False
        free[self.axis_mask, U2] = False
        free[self.axis_mask, U3] = False
        free[self.axis_mask, AR] = False
        free[self.axis_mask, AP] = False
        free[self.sym_mask, U2] = False
        free[self.sym_mask, AR] = False
        self.free = free

        # Constant scalar Laplacian (for β): ∫ ∇a·∇b d³x
        self.S = (self.PrT @ sp.diags(self.W) @ self.Pr
                  + self.PzT @ sp.diags(self.W) @ self.Pz).tocsr()
        self.beta_free = ~self.inf_mask

    # -- helpers ---------------------------------------------------------
    def interp_points(self, rho, z):
        """Sparse matrix evaluating the nodal FE field at points (ρ, z)."""
        p = self.p
        s_nodes = np.linspace(-1.0, 1.0, p + 1)

        def basis_1d(x, nodes):
            ne = (nodes.size - 1) // p
            h = (nodes[-1] - nodes[0]) / ne
            el = np.clip(np.floor((x - nodes[0]) / h).astype(int), 0, ne - 1)
            loc = 2.0 * (x - (nodes[0] + el * h)) / h - 1.0
            vals = np.ones((x.size, p + 1))
            for k in range(p + 1):
                for m in range(p + 1):
                    if m != k:
                        vals[:, k] *= (loc - s_nodes[m]) / (s_nodes[k] - s_nodes[m])
            idx = el[:, None] * p + np.arange(p + 1)[None, :]
            return idx, vals

        with np.errstate(invalid="ignore"):
            xi = np.where(np.isfinite(rho), 2.0 / np.pi * np.arctan(np.abs(rho) / self.a), 1.0)
            et = np.where(np.isfinite(z), 2.0 / np.pi * np.arctan(z / self.a), np.sign(z))
        if self.half:
            et = np.abs(et)
        ii, vi = basis_1d(xi, self.xi)
        jj, vj = basis_1d(et, self.eta)
        npts = xi.size
        rows = np.repeat(np.arange(npts), (p + 1) ** 2)
        cols = (ii[:, :, None] * self.nz + jj[:, None, :]).reshape(npts, -1).ravel()
        vals = (vi[:, :, None] * vj[:, None, :]).reshape(npts, -1).ravel()
        return sp.csr_matrix((vals, (rows, cols)), shape=(npts, self.nn))

    def vacuum(self):
        U = np.zeros((self.nn, 6))
        U[:, U3] = -1.0
        return U

    def integrate(self, f):
        return float(np.dot(self.W, f))

    def node_r_theta(self):
        r = np.hypot(self.rho_n, self.z_n)
        with np.errstate(invalid="ignore", divide="ignore"):
            ct = np.where(r > 0, self.z_n / r, 1.0)
            st = np.where(r > 0, self.rho_n / r, 0.0)
        ct = np.where(np.isfinite(r), ct, 0.0)
        st = np.where(np.isfinite(r), st, 1.0)
        return r, ct, st


# ---------------------------------------------------------------------------
# Energy density and its partial derivatives at the quadrature points
# ---------------------------------------------------------------------------

class Model:
    def __init__(self, grid, e=0.3, N=1.0, m=1, f=1.0, g=1.0, mu=1.0,
                 kappa=1.0, charge=True):
        self.G = grid
        self.e, self.N, self.m = e, N, m
        self.c2, self.c4, self.mu = f * f, 1.0 / (g * g), mu
        self.kappa = kappa
        self.charge = charge and N != 0.0 and e != 0.0
        self._beta_lu = None
        self.last = {}

    # fields at quadrature points ---------------------------------------
    def interp(self, U):
        G = self.G
        return G.Pv @ U, G.Pr @ U, G.Pz @ U

    def local(self, Qv, Qr, Qz, want_grad=True, beta_q=None, omega=0.0):
        """Energy density (per unit d³x) and partials at quad points."""
        e, c2, c4, mu, m, kap = self.e, self.c2, self.c4, self.mu, self.m, self.kappa
        rho = self.G.rho
        u1, u2, u3 = Qv[:, U1], Qv[:, U2], Qv[:, U3]
        Ar, Az, Ap = Qv[:, AR], Qv[:, AZ], Qv[:, AP]
        pr1, pr2, pr3 = Qr[:, U1], Qr[:, U2], Qr[:, U3]
        pz1, pz2, pz3 = Qz[:, U1], Qz[:, U2], Qz[:, U3]
        Ar_r, Az_r, Ap_r = Qr[:, AR], Qr[:, AZ], Qr[:, AP]
        Ar_z, Az_z, Ap_z = Qz[:, AR], Qz[:, AZ], Qz[:, AP]

        Dr1 = pr1 - e * Ar * u2
        Dr2 = pr2 + e * Ar * u1
        Dz1 = pz1 - e * Az * u2
        Dz2 = pz2 + e * Az * u1
        P = u1 * u1 + u2 * u2
        Phi = m / rho + e * Ap
        c1 = pr2 * pz3 - pr3 * pz2
        c2v = pr3 * pz1 - pr1 * pz3
        c3 = pr1 * pz2 - pr2 * pz1
        F0 = u1 * c1 + u2 * c2v + u3 * c3
        F = F0 + e * (Ar * pz3 - Az * pr3)
        G3 = pr3 * pr3 + pz3 * pz3
        K = c2 * P + c4 * G3

        Brho = -Ap_z
        Bz = Ap_r + Ap / rho
        Bphi = Ar_z - Az_r
        Div = Ar_r + Ar / rho + Az_z

        dens_sig = 0.5 * c2 * (Dr1 ** 2 + Dr2 ** 2 + pr3 ** 2
                               + Dz1 ** 2 + Dz2 ** 2 + pz3 ** 2)
        dens_phi2 = 0.5 * c2 * Phi ** 2 * P
        dens_sk = 0.5 * c4 * (F ** 2 + Phi ** 2 * G3)
        # f²μ²(1 + u_3) written as f²μ²|u + e_3|²/2: identical for unit u,
        # and bounded below for the (non-unit) interpolant between nodes.
        dens_pot = 0.5 * c2 * mu * mu * (P + (u3 + 1.0) ** 2)
        dens_mag = 0.5 * (Brho ** 2 + Bz ** 2 + Bphi ** 2)
        dens_gf = 0.5 * kap * Div ** 2
        dens = dens_sig + dens_phi2 + dens_sk + dens_pot + dens_mag + dens_gf
        parts = dict(sigma=dens_sig + dens_phi2, skyrme=dens_sk, pot=dens_pot,
                     mag=dens_mag, gauge_fix=dens_gf, K=K, F0=F0, Phi=Phi,
                     Bphi=Bphi, Div=Div,
                     mag_tor=0.5 * Bphi ** 2,                  # B_φ, from meridional currents
                     mag_pol=0.5 * (Brho ** 2 + Bz ** 2))      # from A_φ
        if not want_grad:
            return dens, parts, None

        gv = np.zeros_like(Qv)
        gr = np.zeros_like(Qr)
        gz = np.zeros_like(Qz)
        cF = c4 * F
        PhiK = Phi * K
        # ∂/∂(∂_ρ u)
        gr[:, U1] = c2 * Dr1 + cF * (pz2 * u3 - pz3 * u2)
        gr[:, U2] = c2 * Dr2 + cF * (pz3 * u1 - pz1 * u3)
        gr[:, U3] = c2 * pr3 + cF * (pz1 * u2 - pz2 * u1 - e * Az) + c4 * Phi ** 2 * pr3
        # ∂/∂(∂_z u)
        gz[:, U1] = c2 * Dz1 + cF * (u2 * pr3 - u3 * pr2)
        gz[:, U2] = c2 * Dz2 + cF * (u3 * pr1 - u1 * pr3)
        gz[:, U3] = c2 * pz3 + cF * (u1 * pr2 - u2 * pr1 + e * Ar) + c4 * Phi ** 2 * pz3
        # ∂/∂u
        cpot = c2 * mu * mu
        gv[:, U1] = c2 * e * (Ar * Dr2 + Az * Dz2) + (c2 * Phi ** 2 + cpot) * u1 + cF * c1
        gv[:, U2] = -c2 * e * (Ar * Dr1 + Az * Dz1) + (c2 * Phi ** 2 + cpot) * u2 + cF * c2v
        gv[:, U3] = cF * c3 + cpot * (u3 + 1.0)
        # ∂/∂A
        gv[:, AR] = c2 * e * (-u2 * Dr1 + u1 * Dr2) + cF * e * pz3 + kap * Div / rho
        gv[:, AZ] = c2 * e * (-u2 * Dz1 + u1 * Dz2) - cF * e * pr3
        gv[:, AP] = e * PhiK + Bz / rho
        gr[:, AR] = kap * Div
        gz[:, AR] = Bphi
        gr[:, AZ] = -Bphi
        gz[:, AZ] = kap * Div
        gr[:, AP] = Bz
        gz[:, AP] = -Brho

        if beta_q is not None and omega != 0.0:
            # d/du of N²/(2 I):  -(ω²/2) β² ∂K
            s = -0.5 * omega * omega * beta_q * beta_q
            gv[:, U1] += s * 2.0 * c2 * u1
            gv[:, U2] += s * 2.0 * c2 * u2
            gr[:, U3] += s * 2.0 * c4 * pr3
            gz[:, U3] += s * 2.0 * c4 * pz3
        return dens, parts, (gv, gr, gz)

    # charge sector -------------------------------------------------------
    def beta_matrix(self, K):
        G = self.G
        return (G.S / (self.e ** 2) + G.PvT @ sp.diags(G.W * K) @ G.Pv).tocsr()

    def solve_beta(self, K):
        """Minimise ∫ β²K + |∇β|²/e² with β = 1 at infinity."""
        G = self.G
        Ab = self.beta_matrix(K)
        f = G.beta_free
        d = ~f
        Aff = Ab[f][:, f]
        rhs = -(Ab[f][:, d] @ np.ones(d.sum()))
        beta = np.ones(G.nn)
        beta[f] = sparse_solve(Aff, rhs)
        I = float(beta @ (Ab @ beta))
        return beta, I

    # total energy -------------------------------------------------------
    def energy(self, U, want_grad=True, frozen=None):
        """Total energy and nodal gradient.

        frozen = (beta, I) keeps the charge-sector solution fixed (used for
        finite-difference Hessians); otherwise β is re-solved exactly, so the
        gradient is the exact envelope derivative of E_N at fixed N.
        """
        G = self.G
        Qv, Qr, Qz = self.interp(U)
        beta_q, omega, I, beta = None, 0.0, None, None
        if self.charge and frozen is not None:
            beta, I = frozen
            omega = self.N / I
            beta_q = G.Pv @ beta
        elif self.charge:
            u1, u2 = Qv[:, U1], Qv[:, U2]
            K = self.c2 * (u1 * u1 + u2 * u2) + self.c4 * (Qr[:, U3] ** 2 + Qz[:, U3] ** 2)
            beta, I = self.solve_beta(K)
            omega = self.N / I
            beta_q = G.Pv @ beta
        dens, parts, grads = self.local(Qv, Qr, Qz, want_grad, beta_q, omega)
        E_static = G.integrate(dens)
        E_charge = 0.5 * self.N ** 2 / I if self.charge else 0.0
        self.last = dict(beta=beta, I=I, omega=omega, E_static=E_static,
                         E_charge=E_charge)
        E = E_static + E_charge
        if not want_grad:
            return E, None
        gv, gr, gz = grads
        W = G.W[:, None]
        grad = G.PvT @ (W * gv) + G.PrT @ (W * gr) + G.PzT @ (W * gz)
        return E, grad

    def breakdown(self, U):
        G = self.G
        Qv, Qr, Qz = self.interp(U)
        dens, parts, _ = self.local(Qv, Qr, Qz, want_grad=False)
        out = {k: G.integrate(v) for k, v in parts.items()
               if k in ("sigma", "skyrme", "pot", "mag", "gauge_fix", "mag_tor", "mag_pol")}
        if self.charge:
            beta, I = self.solve_beta(parts["K"])
            out["charge"] = 0.5 * self.N ** 2 / I
            out["I_eff"] = I
            out["omega"] = self.N / I
        out["total"] = sum(v for k, v in out.items()
                           if k in ("sigma", "skyrme", "pot", "mag", "gauge_fix", "charge"))
        return out

    # fixed-u solves for A ------------------------------------------------
    def solve_A_fixed_u(self, U, kappa_fix=None):
        """Exact minimisation over (A_ρ, A_z, A_φ) with u held fixed.

        Returns a copy of U with the optimal A.  With u fixed the problem is
        quadratic; A_φ decouples from the meridional pair (A_ρ, A_z).
        """
        G = self.G
        e, c2, c4, m = self.e, self.c2, self.c4, self.m
        kap = self.kappa if kappa_fix is None else kappa_fix
        Qv, Qr, Qz = self.interp(U)
        u1, u2, u3 = Qv[:, U1], Qv[:, U2], Qv[:, U3]
        pr1, pr2, pr3 = Qr[:, U1], Qr[:, U2], Qr[:, U3]
        pz1, pz2, pz3 = Qz[:, U1], Qz[:, U2], Qz[:, U3]
        rho = G.rho
        W = G.W
        P = u1 * u1 + u2 * u2
        G3 = pr3 * pr3 + pz3 * pz3
        K = c2 * P + c4 * G3
        F0 = (u1 * (pr2 * pz3 - pr3 * pz2) + u2 * (pr3 * pz1 - pr1 * pz3)
              + u3 * (pr1 * pz2 - pr2 * pz1))
        D = sp.diags
        Pv, Pr, Pz = G.Pv, G.Pr, G.Pz
        Prho = (Pr + D(1.0 / rho) @ Pv).tocsr()      # ∂_ρ + 1/ρ

        # A_φ block
        Hp = (Pz.T @ D(W) @ Pz + Prho.T @ D(W) @ Prho
              + Pv.T @ D(W * e * e * K) @ Pv).tocsr()
        bp = -(Pv.T @ (W * e * (m / rho) * K))
        fp = G.free[:, AP]
        Ap = np.zeros(G.nn)
        Ap[fp] = sparse_solve(Hp[fp][:, fp], bp[fp])

        # meridional block
        Mrr = c2 * P + c4 * pz3 * pz3
        Mzz = c2 * P + c4 * pr3 * pr3
        Mrz = -c4 * pr3 * pz3
        wr = c2 * (-u2 * pr1 + u1 * pr2) + c4 * F0 * pz3
        wz = c2 * (-u2 * pz1 + u1 * pz2) - c4 * F0 * pr3
        Hrr = (Pz.T @ D(W) @ Pz + kap * Prho.T @ D(W) @ Prho
               + Pv.T @ D(W * e * e * Mrr) @ Pv)
        Hzz = (Pr.T @ D(W) @ Pr + kap * Pz.T @ D(W) @ Pz
               + Pv.T @ D(W * e * e * Mzz) @ Pv)
        Hrz = (-Pz.T @ D(W) @ Pr + kap * Prho.T @ D(W) @ Pz
               + Pv.T @ D(W * e * e * Mrz) @ Pv)
        Hm = sp.bmat([[Hrr, Hrz], [Hrz.T, Hzz]], format="csr")
        bm = -np.concatenate([Pv.T @ (W * e * wr), Pv.T @ (W * e * wz)])
        fm = np.concatenate([G.free[:, AR], G.free[:, AZ]])
        x = np.zeros(2 * G.nn)
        x[fm] = sparse_solve(Hm[fm][:, fm], bm[fm])

        V = U.copy()
        V[:, AP] = Ap
        V[:, AR] = x[:G.nn]
        V[:, AZ] = x[G.nn:]
        return V

    # diagnostics ----------------------------------------------------------
    def diagnostics(self, U):
        G = self.G
        Qv, Qr, Qz = self.interp(U)
        dens, parts, _ = self.local(Qv, Qr, Qz, want_grad=False)
        K = parts["K"]
        out = {}
        # Hopf degree of the meridional map u : half-plane → S² (Q_H = m·deg)
        out["deg_u"] = float(np.dot(G.W2, parts["F0"]) / FOUR_PI)
        out["Q_H"] = self.m * out["deg_u"]
        r2 = G.rho ** 2 + G.zq ** 2
        if self.charge:
            beta, I = self.solve_beta(K)
            bq = G.Pv @ beta
            q = bq * K
            Qn = G.integrate(q)
            out["I_eff"] = I
            out["I_eff_check_int_betaK"] = Qn
            out["omega"] = self.N / I
            out["charge_Q"] = self.e * self.N
            out["r2_charge"] = G.integrate(r2 * q) / Qn
            out["rms_charge"] = float(np.sqrt(out["r2_charge"]))
            out["Dzz_charge"] = G.integrate((3 * G.zq ** 2 - r2) * q) / Qn
            out["beta_min"] = float(beta[np.isfinite(G.rho_n)].min())
            # form factors at |q| = 1: direction averaged, along z, transverse
            from scipy.special import j0
            rr = np.sqrt(r2)
            out["G0_q1"] = G.integrate(np.sinc(rr / np.pi) * q) / Qn
            out["Gz_q1"] = G.integrate(np.cos(G.zq) * q) / Qn
            out["Gx_q1"] = G.integrate(j0(G.rho) * q) / Qn
            # Derrick/virial identity for u(x/λ), A(x/λ)/λ, β(x/λ):
            #   E_2 - E_4 + 3 E_pot - E_mag - (ω²/2)(3 I_K2 + I_K4 + I_grad) = 0
            beta_sq = bq * bq
            u1, u2 = Qv[:, U1], Qv[:, U2]
            I_K2 = G.integrate(beta_sq * self.c2 * (u1 * u1 + u2 * u2))
            I_K4 = G.integrate(beta_sq * self.c4 * (Qr[:, U3] ** 2 + Qz[:, U3] ** 2))
            I_grad = float(beta @ (G.S @ beta)) / self.e ** 2
            w2 = (self.N / I) ** 2
            E2 = G.integrate(parts["sigma"])
            E4 = G.integrate(parts["skyrme"])
            Ep = G.integrate(parts["pot"])
            Em = G.integrate(parts["mag"] + parts["gauge_fix"])
            vir = E2 - E4 + 3 * Ep - Em - 0.5 * w2 * (3 * I_K2 + I_K4 + I_grad)
            out["virial_residual"] = vir
            out["virial_relative"] = vir / (E2 + E4 + 3 * Ep + Em)
        # magnetic moment (volume integral):  m_z = ½∫ ρ J_φ d³x, J_φ = -e Φ K
        out["mu_volume"] = -0.5 * self.e * G.integrate(G.rho * parts["Phi"] * K)
        out["mu_far"] = self.far_dipole(U)
        out["energy_density_rms_radius"] = float(np.sqrt(G.integrate(r2 * dens) / G.integrate(dens)))
        out["gauge_fix_energy"] = G.integrate(parts["gauge_fix"])
        return out

    def far_dipole(self, U, rmin=10.0, rmax=40.0):
        """Least-squares fit of A_φ to exterior multipoles ℓ = 1, 3, 5."""
        G = self.G
        r, ct, st = G.node_r_theta()
        sel = np.isfinite(r) & (r > rmin) & (r < rmax) & (st > 1e-6)
        if sel.sum() < 10:
            return float("nan")
        x, s, rr = ct[sel], st[sel], r[sel]
        # P_ℓ^1 without Condon–Shortley phase
        P11 = s
        P31 = 1.5 * s * (5 * x * x - 1)
        P51 = (15.0 / 8.0) * s * (21 * x ** 4 - 14 * x * x + 1)
        cols = [P11 / rr ** 2, P31 / rr ** 4, P51 / rr ** 6]
        if not G.half:
            P21 = 3 * x * s
            P41 = 2.5 * s * (7 * x ** 3 - 3 * x)
            cols += [P21 / rr ** 3, P41 / rr ** 5]
        Amat = np.stack(cols, axis=1)
        wts = rr ** 2  # emphasise the far region
        coef, *_ = np.linalg.lstsq(Amat * wts[:, None], U[sel, AP] * wts, rcond=None)
        return float(FOUR_PI * coef[0])


# ---------------------------------------------------------------------------
# Hopf projection of the hedgehog: the h(r) family of the earlier rounds
# ---------------------------------------------------------------------------

def hopf_ansatz(grid, h_nodes):
    """u(ρ,z) from the profile h at each node (vacuum u = -e_3).

    Z = (cos h + i sin h cosθ, i sin h sinθ e^{iφ});  n = -(Z†τZ) with the
    third component flipped so the vacuum is n_3 = -1:
      u_1 + i u_2 = 2 sin h sinθ (cos h - i sin h cosθ)
      u_3         = 2 sin²h sin²θ - 1
    Returns U (with A = 0) and du/dh at the nodes.
    """
    r, ct, st = grid.node_r_theta()
    h = np.where(np.isfinite(r), h_nodes, 0.0)
    sh, ch = np.sin(h), np.cos(h)
    U = grid.vacuum()
    U[:, U1] = 2 * sh * st * ch
    U[:, U2] = -2 * sh * sh * st * ct
    U[:, U3] = 2 * sh * sh * st * st - 1.0
    dU = np.zeros((grid.nn, 3))
    dU[:, 0] = 2 * np.cos(2 * h) * st
    dU[:, 1] = -2 * np.sin(2 * h) * st * ct
    dU[:, 2] = 2 * np.sin(2 * h) * st * st
    fin = np.isfinite(r)
    dU[~fin] = 0.0
    U[grid.axis_mask, :3] = (0.0, 0.0, -1.0)
    dU[grid.axis_mask] = 0.0
    return U, dU


def probe_field(grid, U, ell, R, L=12.0):
    """δu = ξ (e_3 - u_3 u),  ξ = (r/R)^ℓ e^{-r²/R²} (1 - r²/L²)^4 P_ℓ(cosθ)."""
    from scipy.special import eval_legendre
    r, ct, st = grid.node_r_theta()
    rr = np.where(np.isfinite(r), r, 0.0)
    xi = (rr / R) ** ell * np.exp(-(rr / R) ** 2) * np.clip(1 - (rr / L) ** 2, 0, None) ** 4
    xi *= eval_legendre(ell, ct)
    xi[~np.isfinite(r)] = 0.0
    u = U[:, :3]
    d = -u[:, 2:3] * u
    d[:, 2] += 1.0
    dU = np.zeros_like(U)
    dU[:, :3] = xi[:, None] * d
    dU[~grid.free] = 0.0
    return dU
