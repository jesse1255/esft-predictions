"""
Round 5: two Hopf solitons of the uncharged Faddeev–Skyrme model — binding,
repulsion, resonances, instability and tolerance.

Model (hopfion_axisym.Model with e = 0, so the gauge field decouples and is
held at A = 0):

    E = ∫ d³x [ f²/2 |∇n|² + 1/(4g²) Σ_ij (n·∂_i n × ∂_j n)² + f²μ² (1 + n_3) ],

f = g = μ = 1 unless stated.  n = R_3(mφ) u(ρ, z); a single soliton has m = 1
and Hopf charge Q = m·deg(u) = 1.  The field is a two-component object in
disguise: n = z†σz with z ∈ C², |z| = 1 (the "四象" level of the notes), and
the stereographic coordinate W = (u_1 + i u_2)/(1 − u_3) (W = 0 in the vacuum,
W = ∞ on the core ring) is the complex amplitude that superposes.

Subcommands
  single  ne_r ne_z a        relax one soliton on the half plane (reflection
                             symmetric, so it cannot drift), mirror it to the
                             full plane; energy, parts, virial, Q, ring
                             geometry, isorotational inertia
  spectrum ne_r ne_z a       lowest mass-normalised Hessian eigenvalues of the
                             single soliton (internal modes below the
                             continuum μ² = 1 are the "resonances")
  pairmap ne_r ne_z a        two solitons on the z axis at ±d/2 with relative
                             phase α, built by superposing W:
                               W = W_1(z − d/2) + e^{iα} W_1(z + d/2)
                             (homogeneous form, regular on the cores).
                             Interaction E(d, α) − E_1(+d/2) − E_1(−d/2).
  q2 ne_r ne_z a             charge-2 states: A_{2,1} (m = 2) and the merged
                             coaxial pair A_{1,2} (m = 1, deg 2); binding
                             energy 2E_1 − E_2.

Writes data/pair_<cmd>_ne{ne_r}x{ne_z}_a{a}.json.
"""

import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from hopfion_axisym import Grid, Model, hopf_ansatz, U1, U2, U3
from newton import NewtonRelaxer, _frames

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def freeze_A(G):
    G.free[:, 3:] = False
    return G


def model(G, m=1, mu=1.0, f=1.0, g=1.0):
    return Model(G, e=0.0, N=0.0, m=m, f=f, g=g, mu=mu, kappa=1.0)


def tag(ne_r, ne_z, a, extra=""):
    return f"ne{ne_r}x{ne_z}_a{a:g}{extra}"


def save(name, obj):
    fn = os.path.join(DATA, f"pair_{name}.json")
    with open(fn, "w") as fh:
        json.dump(obj, fh, indent=1)
    print("wrote", fn, flush=True)


# ---------------------------------------------------------------------------
# single soliton
# ---------------------------------------------------------------------------

def initial(G, R0=1.5):
    r, _, _ = G.node_r_theta()
    rr = np.where(np.isfinite(r), r, 1e9)
    h = np.pi / (1.0 + (rr / R0) ** 2)
    U, _ = hopf_ansatz(G, h)
    return U


def mirror_full(Gh, Uh):
    Gf = freeze_A(Grid(Gh.ne_r, Gh.ne_z, p=Gh.p, a=Gh.a, half=False))
    Uf = Gf.vacuum()
    U3h = Uh.reshape(Gh.nr, Gh.nz, 6)
    U3f = Uf.reshape(Gf.nr, Gf.nz, 6)
    mid = Gh.nz - 1
    U3f[:, mid:, :] = U3h
    lower = U3h[:, ::-1, :].copy()
    lower[:, :, U2] *= -1.0
    U3f[:, :mid + 1, :] = lower
    return Gf, U3f.reshape(-1, 6)


def parts_and_virial(M, U):
    b = M.breakdown(U)
    E2, E4, Ep = b["sigma"], b["skyrme"], b["pot"]
    vir = E2 - E4 + 3.0 * Ep
    return dict(E=E2 + E4 + Ep, E2=E2, E4=E4, E_pot=Ep, virial=vir,
                virial_relative=vir / (E2 + E4 + 3 * Ep))


def inertia_iso(M, U):
    """Isorotational inertia ∫ (f² |T|² + g⁻² |∇u_3|²) d³x (the 'K' density)."""
    Qv, Qr, Qz = M.interp(U)
    _, parts, _ = M.local(Qv, Qr, Qz, want_grad=False)
    return M.G.integrate(parts["K"])


def ring_geometry(G, U):
    rho = np.linspace(0.0, 6.0, 3001)
    P = G.interp_points(rho, np.zeros_like(rho))
    u3 = P @ U[:, U3]
    i = int(np.argmax(u3))
    return dict(core_radius=float(rho[i]), u3_max=float(u3[i]))


def relax_single(ne_r, ne_z, a, m=1, mu=1.0, verbose=True):
    fn = os.path.join(DATA, f"pair_single_state_{tag(ne_r, ne_z, a)}_m{m}_mu{mu:g}.npz")
    Gh = freeze_A(Grid(ne_r, ne_z, p=2, a=a, half=True))
    if os.path.exists(fn):
        Uh = np.load(fn)["U"]
    else:
        Mh = model(Gh, m=m, mu=mu)
        R = NewtonRelaxer(Mh, initial(Gh), verbose=verbose)
        Uh, _ = R.run(max_iter=80, tol=1e-9)
        np.savez_compressed(fn, U=Uh, converged=R.converged)
    Gf, Uf = mirror_full(Gh, Uh)
    return Gh, Uh, Gf, Uf


def cmd_single(ne_r, ne_z, a, mu=1.0):
    t0 = time.time()
    Gh, Uh, Gf, Uf = relax_single(ne_r, ne_z, a, mu=mu)
    Mh, Mf = model(Gh, mu=mu), model(Gf, mu=mu)
    out = dict(ne_r=ne_r, ne_z=ne_z, a=a, mu=mu, half=parts_and_virial(Mh, Uh),
               full=parts_and_virial(Mf, Uf), Q=Mf.diagnostics(Uf)["Q_H"],
               inertia_iso=inertia_iso(Mf, Uf), ring=ring_geometry(Gf, Uf), t=time.time() - t0)
    print(json.dumps(out, indent=1), flush=True)
    save(f"single_{tag(ne_r, ne_z, a)}_mu{mu:g}", out)
    return out


# ---------------------------------------------------------------------------
# spectrum ("resonances")
# ---------------------------------------------------------------------------

def lumped_mass(R):
    G = R.G
    vol = np.asarray((G.PvT @ sp.diags(G.W) @ G.Pv).diagonal()).ravel()
    mass = np.zeros(R.ndof)
    for k, msk in enumerate(R.masks):
        mass[R.offsets[k]:R.offsets[k + 1]] = vol[msk]
    return mass


def consistent_mass(R, e1, e2):
    """Galerkin mass matrix ∫ δu·δu d³x in the tangent coordinates (a1, a2).

    The lumped (diagonal) version puts spurious modes below the continuum in
    the huge outer elements of the compactified grid; with the consistent
    matrix the discrete eigenvalues are min–max upper bounds, so this cannot
    happen.
    """
    G = R.G
    Mv = (G.PvT @ sp.diags(G.W) @ G.Pv).tocoo()
    r, c, v = Mv.row, Mv.col, Mv.data
    frames = (e1, e2)
    blocks = [[None, None], [None, None]]
    for k in range(2):
        for kk in range(2):
            dot = np.sum(frames[k][r] * frames[kk][c], axis=1)
            full = sp.csr_matrix((v * dot, (r, c)), shape=(G.nn, G.nn))
            blocks[k][kk] = full[R.masks[k]][:, R.masks[kk]]
    return sp.bmat(blocks, format="csr")


def lowest(R, U, k=12, sigma=-0.05):
    M = R.M
    M.energy(U)
    e1, e2 = _frames(U[:, :3], R.G.sym_mask)
    H, g = R.hessian(U, e1, e2, None, central=True)
    Mm = consistent_mass(R, e1, e2)
    A = (H - sigma * Mm).tocsc()
    lu = spla.splu(A)
    op = spla.LinearOperator(H.shape, matvec=lu.solve, dtype=float)
    vals, vecs = spla.eigsh(H, k=k, M=Mm, sigma=sigma, which="LM", OPinv=op)
    order = np.argsort(vals)
    return vals[order], vecs[:, order], H, e1, e2, Mm


def generators(R, U, e1, e2):
    """Exact zero modes in tangent coordinates: isorotation, z translation."""
    G = R.G
    u = U[:, :3]
    T = np.stack([-u[:, 1], u[:, 0], np.zeros(G.nn)], axis=1)
    uu = u.reshape(G.nr, G.nz, 3)
    zz = G.z_n.reshape(G.nr, G.nz)
    d = np.zeros_like(uu)
    with np.errstate(invalid="ignore"):
        d[:, 1:-1] = (uu[:, 2:] - uu[:, :-2]) / (zz[:, 2:] - zz[:, :-2])[:, :, None]
    dz = np.nan_to_num(d.reshape(-1, 3))

    def to_x(vu):
        x = np.zeros(R.ndof)
        x[R.offsets[0]:R.offsets[1]] = np.sum(vu * e1, axis=1)[R.masks[0]]
        x[R.offsets[1]:R.offsets[2]] = np.sum(vu * e2, axis=1)[R.masks[1]]
        return x
    return dict(isorotation=to_x(T), z_translation=to_x(dz))


def cmd_spectrum(ne_r, ne_z, a, mu=1.0, k=14):
    Gh, Uh, Gf, Uf = relax_single(ne_r, ne_z, a, mu=mu)
    Mf = model(Gf, mu=mu)
    R = NewtonRelaxer(Mf, Uf, verbose=False)
    vals, vecs, H, e1, e2, Mm = lowest(R, Uf, k=k)
    gens = generators(R, Uf, e1, e2)
    nodes = np.concatenate([np.nonzero(msk)[0] for msk in R.masks[:2]])
    rr = np.hypot(Gf.rho_n, Gf.z_n)[nodes]
    rows = []
    for i, lam in enumerate(vals):
        v = vecs[:, i]
        vn = v / np.sqrt(v @ (Mm @ v))
        ov = {}
        for name, gvec in gens.items():
            gn = gvec / np.sqrt(gvec @ (Mm @ gvec))
            ov[name] = float(abs(vn @ (Mm @ gn)))
        w = vn * (Mm @ vn)
        inside = float(w[rr < 5.0].sum() / w.sum())          # weight within r < 5
        rows.append(dict(eigenvalue=float(lam), overlap=ov, weight_r_lt_5=inside))
        print(f"  λ = {lam:+.6f}   iso {ov['isorotation']:.3f}   z-trans {ov['z_translation']:.3f}   "
              f"weight(r<5) {inside:.3f}", flush=True)
    out = dict(ne_r=ne_r, ne_z=ne_z, a=a, mu=mu, continuum=mu ** 2, modes=rows,
               note="H v = λ M v with the consistent L² mass; λ < μ² = 1 and localised (weight(r<5) ≈ 1) "
                    "are bound modes of the soliton")
    save(f"spectrum_{tag(ne_r, ne_z, a)}_mu{mu:g}", out)
    return out


# ---------------------------------------------------------------------------
# two solitons by superposition of the stereographic amplitude
# ---------------------------------------------------------------------------

def shifted(Gf, Uf, z0):
    """Single soliton moved to z = z0 by interpolation; returns (a, b) with
    W = a / b, a = u_1 + i u_2, b = 1 − u_3 (normalised u)."""
    fin = np.isfinite(Gf.rho_n) & np.isfinite(Gf.z_n)
    rho = np.where(fin, Gf.rho_n, 0.0)
    z = np.where(fin, Gf.z_n - z0, 0.0)
    P = Gf.interp_points(rho, z)
    u = P @ Uf[:, :3]
    u /= np.linalg.norm(u, axis=1)[:, None]
    u[~fin] = (0.0, 0.0, -1.0)
    u[Gf.axis_mask] = (0.0, 0.0, -1.0)
    return u


def ab(u):
    return u[:, 0] + 1j * u[:, 1], 1.0 - u[:, 2]


def from_ab(num, den):
    n2, d2 = np.abs(num) ** 2, np.abs(den) ** 2
    s = n2 + d2
    w = num * np.conj(den)
    u = np.stack([2 * w.real / s, 2 * w.imag / s, (n2 - d2) / s], axis=1)
    return u


def pair_field(Gf, uA, uB, alpha):
    """W = W_A + e^{iα} W_B in homogeneous form (regular on both cores)."""
    aA, bA = ab(uA)
    aB, bB = ab(uB)
    num = aA * bB + np.exp(1j * alpha) * aB * bA
    den = bA * bB
    U = Gf.vacuum()
    U[:, :3] = from_ab(num, den)
    U[Gf.axis_mask, :3] = (0.0, 0.0, -1.0)
    U[Gf.inf_mask, :3] = (0.0, 0.0, -1.0)
    return U


def single_at(Gf, u):
    U = Gf.vacuum()
    U[:, :3] = u
    return U


def cmd_pairmap(ne_r, ne_z, a, mu=1.0, ds=None, alphas=None):
    Gh, Uh, Gf, Uf = relax_single(ne_r, ne_z, a, mu=mu)
    Mf = model(Gf, mu=mu)
    E1 = Mf.energy(Uf, want_grad=False)[0]
    if ds is None:
        ds = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]
    if alphas is None:
        alphas = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    rows = []
    for d in ds:
        uA, uB = shifted(Gf, Uf, +d / 2), shifted(Gf, Uf, -d / 2)
        EA = Mf.energy(single_at(Gf, uA), want_grad=False)[0]
        EB = Mf.energy(single_at(Gf, uB), want_grad=False)[0]
        Eint = []
        for al in alphas:
            U = pair_field(Gf, uA, uB, al)
            E = Mf.energy(U, want_grad=False)[0]
            Eint.append(E - EA - EB)
        Eint = np.array(Eint)
        # Fourier coefficients on the full circle (uniform α, so exact DFT):
        #   E_int(α) = c0 − B1 cos(α − α1) + B2 cos(2α − φ2) + …
        F = np.fft.rfft(Eint) / len(alphas)
        c0 = F[0].real
        B1, a1 = 2 * abs(F[1]), float(np.angle(-F[1].conj()))      # −B1 cos(α − α1)
        B2 = 2 * abs(F[2]) if len(F) > 2 else 0.0
        higher = float(2 * np.abs(F[3:]).sum()) if len(F) > 3 else 0.0
        imin, imax = int(np.argmin(Eint)), int(np.argmax(Eint))
        rows.append(dict(d=d, E_single_shift_error=[EA - E1, EB - E1], alpha=alphas.tolist(),
                         E_int=Eint.tolist(), c0=float(c0), B1=float(B1), alpha1=a1 % (2 * np.pi),
                         B2=float(B2), higher_harmonics=higher,
                         E_min=float(Eint[imin]), alpha_min=float(alphas[imin]),
                         E_max=float(Eint[imax]), alpha_max=float(alphas[imax])))
        print(f"d = {d:4.1f}:  min {Eint[imin]:+.5f} at {np.degrees(alphas[imin]):5.1f}°   max {Eint[imax]:+.5f} at "
              f"{np.degrees(alphas[imax]):5.1f}°   c0 {c0:+.5f}  B1 {B1:.5f} α1 {np.degrees(a1) % 360:5.1f}°  "
              f"B2 {B2:.5f}  rest {higher:.1e}  (shift err {EA - E1:+.1e})", flush=True)
    out = dict(ne_r=ne_r, ne_z=ne_z, a=a, mu=mu, E1=E1, rows=rows,
               description=__doc__.split("pairmap")[1].split("q2")[0].strip())
    save(f"pairmap_{tag(ne_r, ne_z, a)}_mu{mu:g}", out)
    return out


# ---------------------------------------------------------------------------
# relaxation at fixed separation and relative phase (Lagrange multipliers)
# ---------------------------------------------------------------------------
#
# The pair  W = e^{iα/2} W_1(z − d/2) + e^{−iα/2} W_1(z + d/2)  is invariant under
# the reflection z → −z, W → W̄ for every α (after the global rotation e^{−iα/2}), so it lives in the half plane
# z ≥ 0 with the existing symmetry (u_2 = 0 on z = 0); only soliton A is stored.
# Two collective coordinates of A are held fixed by Lagrange multipliers:
#   Z   = ∫ z q / ∫ q,  q = (1 + u_3)²       (core centroid, = d/2)
#   Φ   = arg ∫ w (u_1 + i u_2)               (isorotation angle, = α/2;
#          w = Gaussian window at the initial core, width 1)
# and everything else relaxes (KKT Newton with Levenberg damping).
# A single soliton at z_0 (full plane, same grid) relaxed with the same two
# constraints is the reference, so the discretisation error of a displaced
# soliton cancels.

def pair_field_sym(Gf, uA, uB, alpha):
    aA, bA = ab(uA)
    aB, bB = ab(uB)
    # e^{−iα/2}(W_A + e^{iα} W_B): same relative phase convention as pair_field
    num = np.exp(-0.5j * alpha) * aA * bB + np.exp(0.5j * alpha) * aB * bA
    U = Gf.vacuum()
    U[:, :3] = from_ab(num, bA * bB)
    U[Gf.axis_mask, :3] = (0.0, 0.0, -1.0)
    U[Gf.inf_mask, :3] = (0.0, 0.0, -1.0)
    return U


def upper_half(Gh, Gf, Uf):
    mid = Gh.nz - 1
    return Uf.reshape(Gf.nr, Gf.nz, 6)[:, mid:, :].reshape(-1, 6).copy()


class Constraints:
    def __init__(self, G, z0, phi, rc, sigma=1.0, z_half=False):
        self.G, self.z0, self.phi = G, z0, phi
        self.w = np.exp(-((G.rho - rc) ** 2 + (G.zq - z0) ** 2) / (2 * sigma ** 2))
        # centroid over the upper half only when both solitons are in the domain
        self.zmask = (G.zq > 0).astype(float) if z_half else np.ones_like(G.zq)

    def values(self, U):
        G = self.G
        Q = G.Pv @ U[:, :3]
        q = (1.0 + Q[:, 2]) ** 2 * self.zmask
        Den = float(G.W @ q)
        Z = float(G.W @ (G.zq * q)) / Den
        J = complex(G.W @ (self.w * (Q[:, 0] + 1j * Q[:, 1])))
        c2 = (np.exp(-1j * self.phi) * J).imag / abs(J)
        return np.array([Z - self.z0, c2]), Z, J, Den, Q

    def grads(self, U):
        G = self.G
        c, Z, J, Den, Q = self.values(U)
        g1 = np.zeros((G.nn, 6))
        g1[:, U3] = G.PvT @ (G.W * (G.zq - Z) * 2.0 * (1.0 + Q[:, 2]) * self.zmask) / Den
        g2 = np.zeros((G.nn, 6))
        g2[:, U1] = G.PvT @ (G.W * self.w * (-np.sin(self.phi))) / abs(J)
        g2[:, U2] = G.PvT @ (G.W * self.w * np.cos(self.phi)) / abs(J)
        return c, g1, g2, Z, np.angle(J)


def relax_constrained(G, U0, cons, mu=1.0, max_iter=60, tol=1e-8, verbose=False):
    M = model(G, mu=mu)
    R = NewtonRelaxer(M, U0, verbose=False)
    U = U0.copy()
    lam = 1e-3
    E, gU = M.energy(U)
    hist = []
    ones = np.ones(G.nn)
    for it in range(max_iter):
        e1, e2 = _frames(U[:, :3], G.sym_mask)
        H, _ = R.hessian(U, e1, e2, None)
        g = R._grad_x(U, gU, ones, e1, e2)
        c, g1, g2, Z, Phi = cons.grads(U)
        C = np.stack([R._grad_x(U, g1, ones, e1, e2), R._grad_x(U, g2, ones, e1, e2)], axis=1)
        dH = np.abs(H.diagonal()) + 1e-300
        # multipliers from the current gradient (least squares): g + C λ ⟂ C
        lam_mult, *_ = np.linalg.lstsq(C, -g, rcond=None)
        pg = g + C @ lam_mult
        pscaled = float(np.abs(pg / np.sqrt(dH)).max())
        hist.append(dict(it=it, E=E, proj_grad=pscaled, c=c.tolist()))
        if verbose:
            print(f"    it {it:2d} E = {E:.10f}  |P g|/√H = {pscaled:.2e}  c = {c[0]:+.1e} {c[1]:+.1e}  λ {lam:.0e}", flush=True)
        if pscaled < tol and np.abs(c).max() < 1e-9:
            return U, E, True, hist, Z, Phi
        accepted = False
        for _ in range(14):
            K = sp.bmat([[H + lam * sp.diags(dH), sp.csr_matrix(C)],
                         [sp.csr_matrix(C.T), None]], format="csc")
            rhs = np.concatenate([-g, -c])
            sol = spla.spsolve(K, rhs)
            dx = sol[:-2]
            Un, _ = R._assemble_U(dx, U, e1, e2)
            En, gUn = M.energy(Un)
            cn = cons.values(Un)[0]
            # merit: energy plus multiplier-weighted constraint violation
            m_old = E + float(lam_mult @ c) + 1e3 * float(c @ c)
            m_new = En + float(lam_mult @ cn) + 1e3 * float(cn @ cn)
            if m_new <= m_old + 1e-10 * abs(E):
                U, E, gU = Un, En, gUn
                lam = max(lam * 0.2, 1e-9)
                accepted = True
                break
            lam = max(lam * 10.0, 1e-6)
        if not accepted:
            break
    c = cons.values(U)[0]
    return U, E, False, hist, None, None


def cmd_constrained(ne_r, ne_z, a, mu=1.0, ds=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0), nalpha=12, suffix=""):
    Gh, Uh, Gf, Uf = relax_single(ne_r, ne_z, a, mu=mu, verbose=False)
    Mf = model(Gf, mu=mu)
    E1 = Mf.energy(Uf, want_grad=False)[0]
    rc = ring_geometry(Gf, Uf)["core_radius"]
    Ghp = freeze_A(Grid(ne_r, ne_z, p=2, a=a, half=True))
    rows = []
    for d in ds:
        uA, uB = shifted(Gf, Uf, +d / 2), shifted(Gf, Uf, -d / 2)
        # reference: soliton A alone, constrained to z = d/2 (full plane)
        UA = single_at(Gf, uA)
        consA = Constraints(Gf, d / 2, 0.0, rc)
        c0, _, _, _, phiA = consA.grads(UA)
        consA.phi = phiA
        Gfa = freeze_A(Grid(ne_r, ne_z, p=2, a=a, half=False))
        UAr, EAr, convA, hA, _, _ = relax_constrained(Gfa, UA, Constraints(Gfa, d / 2, phiA, rc), mu=mu)
        EA0 = Mf.energy(UA, want_grad=False)[0]
        for k in range(nalpha):
            al = 2 * np.pi * k / nalpha
            Ufull = pair_field_sym(Gf, uA, uB, al)
            E0 = Mf.energy(Ufull, want_grad=False)[0]
            U0 = upper_half(Ghp, Gf, Ufull)
            cons = Constraints(Ghp, d / 2, phiA - al / 2, rc, z_half=False)
            Ur, Er, conv, hist, Z, Phi = relax_constrained(Ghp, U0, cons, mu=mu)
            Q = model(Ghp, mu=mu).diagnostics(Ur)["Q_H"]
            row = dict(d=d, alpha=al, E_sup_int=E0 - 2 * EA0, E_relaxed_int=Er - 2 * EAr,
                       relaxation_gain=(E0 - Er) - 2 * (EA0 - EAr), converged=conv,
                       iterations=len(hist), Q=Q, ref_converged=convA)
            rows.append(row)
            print(f"d = {d:.2f} α = {np.degrees(al):5.1f}°:  superposition {row['E_sup_int']:+.5f}   "
                  f"relaxed {row['E_relaxed_int']:+.5f}   Q {Q:+.4f}  conv {conv} ({len(hist)} it)  ref {convA}",
                  flush=True)
    out = dict(ne_r=ne_r, ne_z=ne_z, a=a, mu=mu, E1=E1, rows=rows,
               note="pair in the reflection-symmetric half plane; core centroid Z = d/2 and isorotation "
                    "angle fixed by Lagrange multipliers; reference = single soliton at z = d/2 with the "
                    "same constraints on the full plane")
    save(f"constrained_{tag(ne_r, ne_z, a)}_mu{mu:g}{suffix}", out)
    return out


# ---------------------------------------------------------------------------
# resonances of the bound coaxial pair
# ---------------------------------------------------------------------------

def cmd_bound_spectrum(ne_r, ne_z, a, mu=1.0, k=10):
    """Relax the coaxial bound pair (m = 1, deg 2) in the symmetric half plane and
    list its lowest Hessian eigenvalues (consistent mass).  Modes below the
    continuum μ² that are localised are internal vibrations of the pair
    ("bond stretch", "relative twist"); their overlap with the two collective
    motions of soliton A (z translation, isorotation) identifies them."""
    Gh, Uh, Gf, Uf = relax_single(ne_r, ne_z, a, mu=mu, verbose=False)
    E1 = model(Gf, mu=mu).energy(Uf, want_grad=False)[0]
    fn = os.path.join(DATA, f"pair_bound_state_{tag(ne_r, ne_z, a)}_mu{mu:g}.npz")
    Ghp = freeze_A(Grid(ne_r, ne_z, p=2, a=a, half=True))
    M = model(Ghp, mu=mu)
    if os.path.exists(fn):
        Ub = np.load(fn)["U"]
    else:
        pm = json.load(open(os.path.join(DATA, f"pair_pairmap_{tag(ne_r, ne_z, a)}_mu{mu:g}.json")))
        row = min(pm["rows"], key=lambda r: abs(r["d"] - 2.0))
        uA, uB = shifted(Gf, Uf, 1.0), shifted(Gf, Uf, -1.0)
        U0 = upper_half(Ghp, Gf, pair_field_sym(Gf, uA, uB, row["alpha_min"]))
        R = NewtonRelaxer(M, U0, verbose=False)
        Ub, _ = R.run(max_iter=150, tol=1e-10)
        np.savez_compressed(fn, U=Ub)
    Eb = M.energy(Ub, want_grad=False)[0]
    R = NewtonRelaxer(M, Ub, verbose=False)
    vals, vecs, H, e1, e2, Mm = lowest(R, Ub, k=k, sigma=-0.02)
    gens = generators(R, Ub, e1, e2)
    nodes = np.concatenate([np.nonzero(msk)[0] for msk in R.masks[:2]])
    rr = np.hypot(Ghp.rho_n, Ghp.z_n)[nodes]
    rows = []
    for i, lam in enumerate(vals):
        v = vecs[:, i] / np.sqrt(vecs[:, i] @ (Mm @ vecs[:, i]))
        ov = {n: float(abs(v @ (Mm @ (gv / np.sqrt(gv @ (Mm @ gv)))))) for n, gv in gens.items()}
        w = v * (Mm @ v)
        rows.append(dict(eigenvalue=float(lam), omega=float(np.sqrt(max(lam, 0.0))), overlap=ov,
                         weight_r_lt_5=float(w[rr < 5.0].sum() / w.sum())))
        print(f"  λ = {lam:+.6f}  ω = {np.sqrt(max(lam, 0)):.4f}   twist {ov['isorotation']:.3f}   "
              f"stretch {ov['z_translation']:.3f}   weight(r<5) {rows[-1]['weight_r_lt_5']:.3f}", flush=True)
    out = dict(ne_r=ne_r, ne_z=ne_z, a=a, mu=mu, E_bound=Eb, E1=E1, binding=2 * E1 - Eb,
               continuum=mu ** 2, modes=rows)
    save(f"bound_spectrum_{tag(ne_r, ne_z, a)}_mu{mu:g}", out)
    return out


# ---------------------------------------------------------------------------
# charge-2 states
# ---------------------------------------------------------------------------

def cmd_q2(ne_r, ne_z, a, mu=1.0):
    out = dict(ne_r=ne_r, ne_z=ne_z, a=a, mu=mu)
    Gh, Uh, Gf, Uf = relax_single(ne_r, ne_z, a, mu=mu, verbose=False)
    E1 = model(Gh, mu=mu).energy(Uh, want_grad=False)[0]
    out["E1"] = E1
    # A_{2,1}: m = 2, meridional degree 1 (half plane, reflection symmetric)
    Gh2, Uh2, Gf2, Uf2 = relax_single(ne_r, ne_z, a, m=2, mu=mu)
    M2 = model(Gh2, m=2, mu=mu)
    E21 = M2.energy(Uh2, want_grad=False)[0]
    out["A21"] = dict(E=E21, parts=parts_and_virial(M2, Uh2), Q=model(Gf2, m=2, mu=mu).diagnostics(Uf2)["Q_H"],
                      ring=ring_geometry(Gf2, Uf2), binding=2 * E1 - E21, binding_fraction=(2 * E1 - E21) / (2 * E1))
    print("A21:", json.dumps(out["A21"], indent=1), flush=True)
    # A_{1,2}: the coaxial pair in its attractive channel, released without
    # constraints in the reflection-symmetric half plane (m = 1, deg 2)
    pm = json.load(open(os.path.join(DATA, f"pair_pairmap_{tag(ne_r, ne_z, a)}_mu{mu:g}.json")))
    merged = []
    for d0 in (2.0, 3.0):
        row = min(pm["rows"], key=lambda r: abs(r["d"] - d0))
        al = row["alpha_min"]
        uA, uB = shifted(Gf, Uf, +d0 / 2), shifted(Gf, Uf, -d0 / 2)
        Ghp = freeze_A(Grid(ne_r, ne_z, p=2, a=a, half=True))
        U0 = upper_half(Ghp, Gf, pair_field_sym(Gf, uA, uB, al))
        M1 = model(Ghp, mu=mu)
        E0 = M1.energy(U0, want_grad=False)[0]
        R = NewtonRelaxer(M1, U0, verbose=False)
        Ur, Er = R.run(max_iter=150, tol=1e-9)
        dg = M1.diagnostics(Ur)
        # where did the cores go: u_3 maxima along ρ = ring radius, z ≥ 0
        rho = np.full(801, ring_geometry(Gf, Uf)["core_radius"])
        zz = np.linspace(0.0, 4.0, 801)
        u3 = Ghp.interp_points(rho, zz) @ Ur[:, U3]
        merged.append(dict(d0=d0, alpha0=al, E_start=E0, E=Er, converged=R.converged,
                           iterations=len(R.history), Q=dg["Q_H"], binding=2 * E1 - Er,
                           u3_on_ring_line_max=float(u3.max()), z_of_max=float(zz[int(np.argmax(u3))]),
                           ring=ring_geometry(Ghp, Ur)))
        print("A12 from d0 =", d0, json.dumps(merged[-1], indent=1), flush=True)
    out["A12"] = merged
    save(f"q2_{tag(ne_r, ne_z, a)}_mu{mu:g}", out)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    ne_r, ne_z, a = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    mu = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    if cmd == "constrained_check":          # reduced fine-grid tolerance check
        cmd_constrained(ne_r, ne_z, a, mu=mu, ds=(2.0, 3.0), nalpha=8, suffix="_check")
        sys.exit(0)
    dict(single=cmd_single, spectrum=cmd_spectrum, pairmap=cmd_pairmap, q2=cmd_q2,
         constrained=cmd_constrained, bound_spectrum=cmd_bound_spectrum)[cmd](ne_r, ne_z, a, mu=mu)
