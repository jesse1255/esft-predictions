"""
Round 6: two Hopf solitons in the three-line (flag-manifold, "八卦") model.

Model: flag_axisym.FlagModel on F₂ = U(3)/U(1)³ with r = 2, κ = 2, m² = 1
(f = g = μ = 1), gauge-invariant "align" discretisation, three-cycle coupling
κ₃ (κ₃ = 2 is M2′).  On embedded configurations (one line untouched) the model
is exactly the two-component Faddeev–Skyrme model of round 5.  Round 3's
threshold κ₃* ≈ 0.919 was computed for L = (0, 1, 0); here L = (0, 1, 2), so
the stability of the embedded solitons is recomputed (subcommand q2).

Single soliton: the relaxed round-5 CP¹ soliton u(ρ, z) is mapped to the
block (a, b) of U with n = (u_1, u_2, −u_3) (a target reflection: it keeps the
Hopf degree, the energy and the sense of the azimuthal winding, and sends the
round-5 vacuum u_3 = −1 to the flag vacuum U = 1):
    z_1 = √((1 + n_3)/2),  z_2 = (n_1 + i n_2)/(2 z_1),
    Z_a = z_1 e_a + z_2 e_b,  Z_b = −z̄_2 e_a + z̄_1 e_b,  Z_c = e_c.
L = (0, 1, 2), so blocks (0,1) and (1,2) both carry winding +1.

Pairs on the z axis at ±d/2, product ansatz (axially consistent because
R_L(φ) Û R_L(φ)† ≡ R_L(φ) Û up to the diagonal gauge):
    Û = Û_A(z − d/2) · D(β, γ) · Û_B(z + d/2),   D = diag(1, e^{iβ}, e^{iγ}),
    same lines:      A, B in (0, 1)   (β plays the role of round 5's α)
    shared line:     A in (0, 1), B in (1, 2)   (two relative phases)
E_int = E(pair) − E(A alone, displaced) − E(B alone, displaced).

With L = (0, 1, 2) the azimuthal winding of block (a, b) is l_b − l_a, so the
embedded solitons of blocks (0, 1), (1, 2), (0, 2) have Hopf degree −1, −1, −2:
the degrees add like the heights of the SU(3) roots α₁, α₂, α₁ + α₂.

Subcommands
  check  ne_r ne_z a                       E_flag(embedded) vs E_CP¹, degree
  map    ne_r ne_z a  [k3] [kind] [order]  interaction on the (β, γ) torus
  relax  ne_r ne_z a  [k3] [kind] [d0]     release a pair from its most
                                           attractive (β, γ) at d0, relax freely
  q2     ne_r ne_z a  k3 [k3 ...]          single (0, 1) soliton and the Q = 2
                                           torus embedded in block (0, 2): relax,
                                           binding, lowest Hessian eigenvalues
  thresh ne_r ne_z a                       κ₃ below which the single soliton / the
                                           torus leak into the third line (L = (0,1,2))
  rot1   ne_r ne_z a  [k3]                 single soliton with stationary isorotation ω
  rotpair ne_r ne_z a k3 ωA ωB [d ...]     rotating different-link pair (resonance test)
  fission ne_r ne_z a [k3] [D ...]         path from the (0, 2) torus at fixed line-1
                                           weight D₁ (0 = torus, ≈ 28.6 = separated pair)
  fused  ne_r ne_z a  [k3] [d ...]         the same constraint, continued upward in d
                                           from the (0, 2) torus (fused branch)
  cons   ne_r ne_z a  [k3] [d ...]         shared-line pair relaxed at fixed
                                           separation (centroids of the line-0
                                           and line-2 defects fixed by Lagrange
                                           multipliers)
Writes data/flagpair_<cmd>_*.json.
"""

import json
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

from hopfion_axisym import Grid, sparse_solve
from flag_axisym import FlagModel, FlagProblem, newton_relax, inertia, NT
from flag_path import degree_parts
from flag_gauge import smooth_gauge
from run_hopf_pair import relax_single, shifted, model as cp1_model, tag

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SCRATCH = os.environ.get("FLAGPAIR_SCRATCH", os.path.join(HERE, "data"))   # large Hessian caches (.npz, gitignored)


def save(name, obj):
    fn = os.path.join(DATA, f"flagpair_{name}.json")
    with open(fn, "w") as fh:
        json.dump(obj, fh, indent=1)
    print("wrote", fn, flush=True)


def embed(G, u, block):
    """(nn, 3) unit vectors u (round-5 convention) → (nn, 3, 3) unitary."""
    n1, n2, n3 = u[:, 0], u[:, 1], -u[:, 2]
    z1 = np.sqrt(np.clip(0.5 * (1.0 + n3), 0.0, 1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        z2 = np.where(z1 > 1e-12, (n1 + 1j * n2) / (2.0 * np.maximum(z1, 1e-300)), 1.0 + 0j)
    nrm = np.sqrt(z1 ** 2 + np.abs(z2) ** 2)
    z1, z2 = z1 / nrm, z2 / nrm
    a, b = block
    c = 3 - a - b
    U = np.zeros((G.nn, 3, 3), complex)
    U[:, a, a] = z1
    U[:, b, a] = z2
    U[:, a, b] = -np.conj(z2)
    U[:, b, b] = z1
    U[:, c, c] = 1.0
    fin = np.isfinite(G.rho_n) & np.isfinite(G.z_n)
    U[~fin] = np.eye(3)
    return U


def line_defect(U, c):
    """Nodal 1 − |U_cc|²: gauge invariant, 0 in the vacuum, (1 − n₃)/2 on line a of an (a, b) soliton."""
    return 1.0 - np.abs(np.asarray(U)[:, c, c]) ** 2


def line_weights(G, U):
    return [float(G.W @ (G.Pv @ line_defect(U, c))) for c in range(3)]


def fmodel(G, k3, Om=None):
    if Om is not None:
        return RotFlagModel(G, Om, L=(0, 1, 2), kappa3=k3, disc="align")
    return FlagModel(G, L=(0, 1, 2), kappa3=k3, disc="align")


class RotFlagModel(FlagModel):
    """Stationary isorotation U(t) = e^{iΩt} U, Ω = diag(ω₀, ω₁, ω₂).

    The Lagrangian is T − V, where T has the form of the static energy with one
    spatial direction replaced by the time direction, w_t = U†(iΩ)U (no 1/ρ): the
    sigma term Σ r|w_t^{ab}|² and the quartic terms of the pairs (t, ρ), (t, z),
    (t, φ).  The potential is invariant under left diagonal phases.  Stationary
    rotating solitons are critical points of V − T at fixed Ω (energy_U below), and
    δ(V − T)|_Ω = δ(V + T)|_J, so interaction energies at fixed Ω equal those at
    fixed isospin charges J = ∂T/∂Ω to first order.  Link (a, b) rotates at
    ω_a − ω_b: a soliton in (0, 1) at ω_A = ω₀ − ω₁, one in (1, 2) at ω_B = ω₁ − ω₂,
    and the link (0, 2) that the cubic coupling x₀₁x₁₂x̄₀₂ feeds at ω_A + ω_B."""

    def __init__(self, grid, Om, **kw):
        super().__init__(grid, **kw)
        assert self.disc == "align"
        self.Om = jnp.asarray(np.array(Om, dtype=float))

    def kinetic_terms(self, U, r=None, k3=None):
        r = self.r if r is None else r
        k3 = self.kappa3 if k3 is None else k3
        Z, Zr, Zz = self._interp_aligned(U)
        Zh = jnp.conj(jnp.swapaxes(Z, 1, 2))
        w_r, w_z = Zh @ Zr, Zh @ Zz
        w_f = (Zh @ (1j * self.Lvec[None, :, None] * Z)) / self.rho[:, None, None]
        w_t = Zh @ (1j * self.Om[None, :, None] * Z)
        a2 = lambda z: jnp.real(z) ** 2 + jnp.imag(z) ** 2
        sig = 0.0
        for k, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            sig = sig + r[k] * 0.5 * (a2(w_t[:, a, b]) + a2(w_t[:, b, a]))
        sk = 0.0
        for a in range(3):
            for wi in (w_r, w_z, w_f):
                F = 0.0
                for b in range(3):
                    if b != a:
                        F = F + 2.0 * jnp.imag(jnp.conj(w_t[:, b, a]) * wi[:, b, a])
                sk = sk + 0.5 * self.kappa[a] * F ** 2
        for wi in (w_r, w_z, w_f):
            for a in range(3):
                for c in range(3):
                    if c == a:
                        continue
                    b = 3 - a - c
                    C = w_t[:, a, b] * wi[:, b, c] - wi[:, a, b] * w_t[:, b, c]
                    sk = sk + 0.5 * k3 * a2(C)
        return sig, sk

    def kinetic(self, U, r=None, k3=None):
        sig, sk = self.kinetic_terms(jnp.asarray(U), r, k3)
        return float(jnp.sum(self.W * (sig + sk)))

    def energy_U(self, U, r=None, k3=None):
        sig, sk = self.kinetic_terms(U, r, k3)
        return super().energy_U(U, r, k3) - jnp.sum(self.W * (sig + sk))


_JIT = {}


def energy(fm, U):
    import jax
    f = _JIT.get(id(fm))
    if f is None:
        f = _JIT[id(fm)] = jax.jit(fm.energy_U)
    return float(f(jnp.asarray(U)))


grid_of = {}


def setup(ne_r, ne_z, a):
    Gh, Uh, Gf, Uf = relax_single(ne_r, ne_z, a, verbose=False)
    grid_of[id(Gf)] = (ne_r, ne_z, a)
    return Gf, Uf


def cmd_check(ne_r, ne_z, a, k3=2.0):
    Gf, Uf = setup(ne_r, ne_z, a)
    E_cp1 = cp1_model(Gf).energy(Uf, want_grad=False)[0]
    out = dict(ne_r=ne_r, ne_z=ne_z, a=a, k3=k3, E_cp1=E_cp1, blocks={})
    fm = fmodel(Gf, k3)
    for blk in ((0, 1), (1, 2), (0, 2)):
        U = embed(Gf, Uf[:, :3], blk)
        E = energy(fm, U)
        Q, Qd, Q3 = degree_parts(fm, smooth_gauge(Gf, U)[0])
        out["blocks"][str(blk)] = dict(E=E, rel_diff=(E - E_cp1) / E_cp1, Q=Q, Q_3cycle=Q3,
                                       parts=fm.parts(U))
        print(f"block {blk}: E_flag = {E:.6f}  (CP¹ {E_cp1:.6f}, rel {(E - E_cp1) / E_cp1:+.2e})  Q = {Q:+.5f}",
              flush=True)
    save(f"check_{tag(ne_r, ne_z, a)}_k3{k3:g}", out)
    return out


def pair_U(Gf, uA, uB, blkA, blkB, beta, gamma, order="AB"):
    UA = embed(Gf, uA, blkA)
    UB = embed(Gf, uB, blkB)
    D = np.diag([1.0, np.exp(1j * beta), np.exp(1j * gamma)])
    if order == "AB":
        return UA @ D @ UB
    return D @ UB @ np.conj(D.T) @ UA


def cmd_map(ne_r, ne_z, a, k3=2.0, kind="shared", order="AB",
            ds=(2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0), nb=12):
    Gf, Uf = setup(ne_r, ne_z, a)
    fm = fmodel(Gf, k3)
    blkA, blkB = ((0, 1), (0, 1)) if kind == "same" else ((0, 1), (1, 2))
    phases = 2 * np.pi * np.arange(nb) / nb
    rows = []
    t0 = time.time()
    for d in ds:
        uA, uB = shifted(Gf, Uf, +d / 2), shifted(Gf, Uf, -d / 2)
        EA = energy(fm, embed(Gf, uA, blkA))
        EB = energy(fm, embed(Gf, uB, blkB))
        if kind == "same":
            grid = [(b, 0.0) for b in phases]
        else:
            grid = [(b, g) for b in phases for g in phases]
        E = np.array([energy(fm, pair_U(Gf, uA, uB, blkA, blkB, b, g, order)) - EA - EB for b, g in grid])
        i0, i1 = int(np.argmin(E)), int(np.argmax(E))
        row = dict(d=d, phases=[list(p) for p in grid], E_int=E.tolist(), E_min=float(E[i0]),
                   at_min=list(grid[i0]), E_max=float(E[i1]), at_max=list(grid[i1]), E_mean=float(E.mean()))
        if kind == "shared":
            M = E.reshape(nb, nb)
            F = np.fft.fft2(M) / M.size
            amps = {f"{kb},{kg}": float(abs(F[kb % nb, kg % nb]) * (1 if (kb, kg) == (0, 0) else 2))
                    for kb in range(-2, 3) for kg in range(-2, 3) if (kb, kg) >= (0, 0) or kb > 0}
            top = sorted(((v, k) for k, v in amps.items() if k != "0,0"), reverse=True)[:5]
            row["fourier_top"] = [(k, v) for v, k in top]
            row["fourier_mean"] = float(F[0, 0].real)
        else:
            F = np.fft.rfft(E) / len(E)
            row["B1"] = float(2 * abs(F[1]))
            row["alpha1_deg"] = float(np.degrees(np.angle(-F[1].conj())) % 360)
            row["c0"] = float(F[0].real)
        rows.append(row)
        extra = (f"  top harmonics {row['fourier_top'][:3]}" if kind == "shared"
                 else f"  B1 {row['B1']:.5f} α1 {row['alpha1_deg']:.1f}° c0 {row['c0']:+.5f}")
        print(f"d = {d:4.1f}: min {E[i0]:+.5f} at (β,γ) = ({np.degrees(grid[i0][0]):.0f}°, {np.degrees(grid[i0][1]):.0f}°)"
              f"   max {E[i1]:+.5f} at ({np.degrees(grid[i1][0]):.0f}°, {np.degrees(grid[i1][1]):.0f}°)"
              f"   mean {E.mean():+.5f}{extra}   ({time.time() - t0:.0f}s)", flush=True)
    out = dict(ne_r=ne_r, ne_z=ne_z, a=a, k3=k3, kind=kind, order=order, blocks=[blkA, blkB], rows=rows)
    save(f"map_{kind}_{order}_{tag(ne_r, ne_z, a)}_k3{k3:g}", out)
    return out


def cmd_relax(ne_r, ne_z, a, k3=2.0, kind="shared", d0=2.0):
    """Release a pair from the most attractive phases at d0 and relax fully."""
    Gf, Uf = setup(ne_r, ne_z, a)
    fm = fmodel(Gf, k3)
    blkA, blkB = ((0, 1), (0, 1)) if kind == "same" else ((0, 1), (1, 2))
    mp = json.load(open(os.path.join(DATA, f"flagpair_map_{kind}_AB_{tag(ne_r, ne_z, a)}_k3{k3:g}.json")))
    row = min(mp["rows"], key=lambda r: abs(r["d"] - d0))
    b, g = row["at_min"]
    uA, uB = shifted(Gf, Uf, +d0 / 2), shifted(Gf, Uf, -d0 / 2)
    U0 = pair_U(Gf, uA, uB, blkA, blkB, b, g)
    E0 = energy(fm, U0)
    E1 = energy(fm, embed(Gf, Uf[:, :3], (0, 1)))
    t0 = time.time()
    U, E, hist, conv = newton_relax(fm, U0, types=range(NT), max_iter=120, verbose=True)
    Q, Qd, Q3 = degree_parts(fm, smooth_gauge(Gf, U)[0])
    Dall = line_weights(Gf, U)
    out = dict(ne_r=ne_r, ne_z=ne_z, a=a, k3=k3, kind=kind, d0=d0, beta=b, gamma=g, E_start=E0, E=E,
               E1=E1, binding=2 * E1 - E, converged=conv, iterations=len(hist), Q=Q, Q_3cycle=Q3,
               D_lines=Dall, parts=fm.parts(U), t=time.time() - t0)
    print(json.dumps({k: v for k, v in out.items() if k != "parts"}, indent=1), flush=True)
    np.save(os.path.join(DATA, f"flagpair_state_{kind}_{tag(ne_r, ne_z, a)}_k3{k3:g}_d{d0:g}.npy"), U)
    save(f"relax_{kind}_{tag(ne_r, ne_z, a)}_k3{k3:g}_d{d0:g}", out)
    return out


# ---------------------------------------------------------------------------
# Q = 2 in block (0, 2) and stability of the embedded solitons (L = (0, 1, 2))
# ---------------------------------------------------------------------------

SECTORS = {"01": (0, 1), "02": (2, 3), "12": (4, 5)}      # types of x_01, x_02, x_12


def sector_fractions(prob, vecs):
    """Mass-norm fraction of each eigenvector in the x_01, x_02, x_12 sectors."""
    M = prob.mass()
    t = prob.free_flat % NT
    out = []
    for v in np.asarray(vecs).T:
        tot = float(v @ (M @ v))
        out.append({k: float((w := np.where(np.isin(t, ts), v, 0.0)) @ (M @ w)) / tot
                    for k, ts in SECTORS.items()})
    return out


def spectrum(fm, U, nev=8):
    prob = FlagProblem(fm, U)
    H = prob.hessian()
    vals, vecs = prob.lowest(H, nev=nev)
    fr = sector_fractions(prob, vecs)
    return [dict(ev=float(v), **{f"frac_{k}": round(f[k], 4) for k in SECTORS}) for v, f in zip(vals, fr)]


def relaxed_state(fm, U0, name, max_iter=60):
    """Newton-relax U0 in the flag model; cache the result by name."""
    fn = os.path.join(DATA, f"flagpair_state_{name}.npy")
    if os.path.exists(fn):
        U = np.load(fn)
        return U, energy(fm, U), True, 0
    U, E, hist, conv = newton_relax(fm, U0, types=range(NT), max_iter=max_iter, verbose=False)
    np.save(fn, U)
    return U, E, conv, len(hist)


def cmd_q2(ne_r, ne_z, a, k3s=(2.0,), nev=8):
    Gf, Uf = setup(ne_r, ne_z, a)
    _, _, Gf2, Uf2 = relax_single(ne_r, ne_z, a, m=2, verbose=False)
    E_cp1 = dict(single=cp1_model(Gf).energy(Uf, want_grad=False)[0],
                 A21=cp1_model(Gf2, m=2).energy(Uf2, want_grad=False)[0])
    T = tag(ne_r, ne_z, a)
    rows = []
    for k3 in k3s:
        fm = fmodel(Gf, k3)
        t0 = time.time()
        U1, E1, c1, n1 = relaxed_state(fm, embed(Gf, Uf[:, :3], (0, 1)), f"single01_{T}_k3{k3:g}")
        U2, E2, c2, n2 = relaxed_state(fm, embed(Gf, Uf2[:, :3], (0, 2)), f"A21in02_{T}_k3{k3:g}")
        Q1 = degree_parts(fm, smooth_gauge(Gf, U1)[0])
        Q2 = degree_parts(fm, smooth_gauge(Gf, U2)[0])
        row = dict(k3=k3, E1=E1, E_A21=E2, binding=2 * E1 - E2, binding_fraction=(2 * E1 - E2) / (2 * E1),
                   converged=[c1, c2], iterations=[n1, n2], Q1=Q1[0], Q1_3cycle=Q1[2], Q2=Q2[0], Q2_3cycle=Q2[2],
                   lines_single=line_weights(Gf, U1), lines_A21=line_weights(Gf, U2),
                   parts_single=fm.parts(U1), parts_A21=fm.parts(U2))
        if nev:
            row.update(spectrum_single=spectrum(fm, U1, nev), spectrum_A21=spectrum(fm, U2, nev))
        row["t"] = time.time() - t0
        rows.append(row)
        print(f"κ₃ = {k3:g}: E1 = {E1:.5f}  E(A21 in (0,2)) = {E2:.5f}  binding {2 * E1 - E2:.4f}"
              f"  Q = {Q1[0]:+.4f}, {Q2[0]:+.4f}  ({row['t']:.0f}s)", flush=True)
        for nm in ("single", "A21"):
            if f"spectrum_{nm}" in row:
                print(f"   {nm:6s} " + "  ".join(f"{s['ev']:+.4f}[{s['frac_01']:.2f},{s['frac_02']:.2f},"
                                              f"{s['frac_12']:.2f}]" for s in row[f"spectrum_{nm}"]), flush=True)
        save(f"q2_{T}", dict(ne_r=ne_r, ne_z=ne_z, a=a, E_cp1=E_cp1, rows=rows,
                             note="spectrum: lowest eigenvalues of the Hessian with the L² (consistent) mass, "
                                  "with the fraction of each eigenvector in the x_01, x_02, x_12 sectors"))
    return rows


# ---------------------------------------------------------------------------
# shared-line pair at fixed separation
# ---------------------------------------------------------------------------

class WeightModel(FlagModel):
    """'Energy' = D_c = ∫ (1 − |U_cc|²) d³x, so that FlagProblem assembles its Hessian
    (the curvature of the constraint, needed in the Lagrangian Hessian)."""

    def __init__(self, grid, line=1, **kw):
        super().__init__(grid, **kw)
        self.line = line

    def energy_U(self, U, r=None, k3=None):
        c = self.line
        return jnp.sum(self.W * (self.Pv @ (1.0 - jnp.abs(U[:, c, c]) ** 2)))


class LineWeight:
    """D_c = ∫ (1 − |U_cc|²) d³x of one line c (gauge invariant; 0 in the vacuum)."""

    def __init__(self, prob, line=1):
        fm, G = prob.fm, prob.G
        W, Pv = jnp.asarray(G.W), fm.Pv

        def w(U0, x):
            U = prob._U_of(U0, x)
            return jnp.stack([jnp.sum(W * (Pv @ (1.0 - jnp.abs(U[:, line, line]) ** 2)))])

        self.c = jax.jit(w)
        self.J = jax.jit(jax.jacrev(w, argnums=1))


class LineCentroids:
    """z-centroids of (1 − |U_00|²)² and (1 − |U_22|²)²: line 0 is excited only by
    soliton A (block (0, 1)), line 2 only by B (block (1, 2)).  Gauge invariant."""

    def __init__(self, prob, lines=(0, 2)):
        fm, G = prob.fm, prob.G
        W, zq, Pv = jnp.asarray(G.W), jnp.asarray(G.zq), fm.Pv

        def cents(U0, x):
            U = prob._U_of(U0, x)
            out = []
            for c in lines:
                q = Pv @ ((1.0 - jnp.abs(U[:, c, c]) ** 2) ** 2)
                out.append(jnp.sum(W * zq * q) / jnp.sum(W * q))
            return jnp.stack(out)

        self.c = jax.jit(cents)
        self.J = jax.jit(jax.jacrev(cents, argnums=1))


def relax_fixed(fm, U0, targets, max_iter=80, tol=1e-8, verbose=False, lines=(0, 2), kind="centroids"):
    """Newton–KKT: minimise E with the constraint values fixed at targets (Lagrange
    multipliers; Levenberg damping on the energy block; merit E + λ·c + 10³|c|²).
    kind = "centroids": z-centroids of the given lines; "weight": D of line lines[0]."""
    prob = FlagProblem(fm, U0)
    cons = LineCentroids(prob, lines) if kind == "centroids" else LineWeight(prob, lines[0])
    # the line weight is quadratic in the normal directions near an embedded solution, so its
    # curvature matters: use the Lagrangian Hessian H_E + λ H_D (λ = current multiplier estimate)
    pc = FlagProblem(WeightModel(fm.G, lines[0], L=fm.L, disc=fm.disc), U0) if kind == "weight" else None
    targets = np.asarray(targets, float)
    U = np.asarray(U0)
    x0 = jnp.zeros(prob.nfree)
    lam, hist = 1e-3, []
    for it in range(max_iter):
        prob.set_base(U)
        E = float(prob.energy(x0))
        g = np.asarray(prob.grad(x0))
        c = np.asarray(cons.c(prob.U0, x0)) - targets
        C = np.asarray(cons.J(prob.U0, x0)).T
        H = prob.hessian()
        dH = np.abs(H.diagonal()) + 1e-300
        lm, *_ = np.linalg.lstsq(C, -g, rcond=None)
        ps = float(np.abs((g + C @ lm) / np.sqrt(dH)).max())
        if pc is not None and ps < 1e-2 and np.abs(c).max() < 1e-3:
            # switch the constraint curvature on only near a KKT point, where the multiplier
            # estimate is reliable (far away it makes the step worse)
            pc.set_base(U)
            H = (H + lm[0] * pc.hessian()).tocsr()
        ps = float(np.abs((g + C @ lm) / np.sqrt(dH)).max())
        hist.append(dict(it=it, E=E, proj_grad=ps, c=c.tolist(), lam=lam))
        if verbose:
            print(f"    it {it:2d} E = {E:.10f}  |Pg|/√H = {ps:.2e}  c = {np.abs(c).max():.1e}  λ {lam:.0e}",
                  flush=True)
        if ps < tol and np.abs(c).max() < 1e-9:
            return U, E, True, hist, lm
        accepted = False
        for _ in range(14):
            A = H + lam * sp.diags(dH)
            K = sp.bmat([[A, sp.csr_matrix(C)], [sp.csr_matrix(C.T), None]], format="csr")
            sol = sparse_solve(K, np.concatenate([-g, -c]))
            dx = jnp.asarray(sol[:prob.nfree])
            En = float(prob.energy(dx))
            cn = np.asarray(cons.c(prob.U0, dx)) - targets
            if (np.all(np.isfinite(sol)) and
                    En + lm @ cn + 1e3 * cn @ cn <= E + lm @ c + 1e3 * c @ c + 1e-10 * abs(E)):
                U = np.asarray(prob.U_of(dx))
                lam = max(lam * 0.2, 1e-9)
                accepted = True
                break
            lam = max(lam * 10.0, 1e-6)
        if not accepted:
            break
    return U, E, False, hist, None


def _pardiso(A):
    import pypardiso
    s = pypardiso.PyPardisoSolver()
    s.set_matrix_type(11)
    A = sp.csr_matrix(A)
    s.factorize(A)
    return lambda b: s.solve(A, np.asarray(b, dtype=float))


def lowest_eig(H, M, lo=-4.0, hi=4.0, tol=1e-6, iters=6):
    """Lowest eigenvalue of H v = λ M v from the inertia of H − sM (Sylvester; bisection
    in s), eigenvector by shifted inverse iteration.  Robust where shift-invert Lanczos
    stalls on clustered spectra."""
    nneg = lambda s: inertia((H - s * M).tocsr())[1]
    while nneg(lo) > 0:
        lo = 2 * lo - 1.0
    while nneg(hi) == 0:
        hi = 2 * hi + 1.0
    while hi - lo > tol * max(1.0, abs(hi)):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if nneg(mid) == 0 else (lo, mid)
    shift = lo - 1e-5 * max(1.0, abs(lo))
    solve = _pardiso(H - shift * M)
    v = np.random.default_rng(0).standard_normal(H.shape[0])
    for _ in range(iters):
        v = solve(M @ v)
        v /= np.sqrt(v @ (M @ v))
    return float(v @ (H @ v)), v, (lo, hi)


def normal_hessians(fm, U, normal_types, cache=None):
    """H(κ₃ = 0) and H_C = ∂H/∂κ₃ in the normal block (cached as .npz in the scratch dir)."""
    pn = FlagProblem(fm, U, types=normal_types)
    if cache and os.path.exists(cache):
        z = np.load(cache)
        Ha = sp.csr_matrix((z["a_data"], z["a_ind"], z["a_ptr"]), shape=tuple(z["shape"]))
        HC = sp.csr_matrix((z["c_data"], z["c_ind"], z["c_ptr"]), shape=tuple(z["shape"]))
        return pn, Ha, HC
    pn.set_couplings(kappa3=0.0)
    Ha = pn.hessian()
    pn.set_couplings(kappa3=1.0)
    HC = (pn.hessian() - Ha).tocsr()
    if cache:
        np.savez(cache, a_data=Ha.data, a_ind=Ha.indices, a_ptr=Ha.indptr, c_data=HC.data,
                 c_ind=HC.indices, c_ptr=HC.indptr, shape=np.array(Ha.shape))
    return pn, Ha, HC


def kappa3_threshold(fm, U, normal_types, tol=1e-6, hi=4.0, cache=None, label=""):
    """Smallest κ₃ at which the Hessian of an embedded solution has no negative
    eigenvalue in the normal (third-line) block.  H(κ₃) = H(0) + κ₃ H_C with
    H_C ⪰ 0 (the three-cycle energy is ≥ 0 and vanishes on embedded
    configurations), so the negative index can only drop as κ₃ grows: bisection
    on the inertia (Sylvester).  The embedded solution itself does not depend on
    κ₃, and the in-block and normal blocks decouple (U(1) of the third line)."""
    pn, Ha, HC = normal_hessians(fm, U, normal_types, cache)
    M = pn.mass()
    n0 = inertia(Ha)[1]
    out = dict(normal_types=list(normal_types), n_negative_k3_0=n0)
    print(f"   {label}: negative normal eigenvalues at κ₃ = 0: {n0}", flush=True)
    if n0 == 0:
        k3s = 0.0
    elif inertia(Ha + hi * HC)[1] > 0:
        return dict(out, kappa3_star=None, note=f"still unstable at κ₃ = {hi}")
    else:
        lo = 0.0
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if inertia(Ha + mid * HC)[1] > 0 else (lo, mid)
        k3s = 0.5 * (lo + hi)
    out["kappa3_star"] = k3s
    print(f"   {label}: κ₃* = {k3s:.6f}", flush=True)
    for k3, key in ((0.0, "k3_0"), (2.0, "k3_2")):
        lam, v, br = lowest_eig((Ha + k3 * HC).tocsr(), M)
        out[f"lowest_normal_eig_{key}"] = lam
        out[f"fractions_{key}"] = sector_fractions(pn, v[:, None])[0]
        out[f"a1_{key}"] = float(v @ (HC @ v))            # dλ/dκ₃ of this mode
        print(f"   {label}: lowest normal eigenvalue at κ₃ = {k3:g}: {lam:+.6f}  "
              f"(fractions {out[f'fractions_{key}']}, dλ/dκ₃ = {out[f'a1_{key}']:.5f})", flush=True)
    return out


def cmd_thresh(ne_r, ne_z, a):
    """κ₃* of the single (0, 1) soliton and of the (0, 2) torus in the L = (0, 1, 2) sector."""
    Gf, _ = setup(ne_r, ne_z, a)
    T = tag(ne_r, ne_z, a)
    fm = fmodel(Gf, 2.0)
    out = dict(ne_r=ne_r, ne_z=ne_z, a=a, L=[0, 1, 2])
    for name, fn, types in (("A21in02", f"flagpair_state_A21in02_{T}_k32.npy", (0, 1, 4, 5)),
                            ("single01", f"flagpair_state_single01_{T}_k32.npy", (2, 3, 4, 5))):
        t0 = time.time()
        U = np.load(os.path.join(DATA, fn))
        out[name] = kappa3_threshold(fm, U, types, cache=os.path.join(SCRATCH, f"normal_hess_{name}_{T}.npz"),
                                     label=name)
        print(f"{name}: κ₃* = {out[name]['kappa3_star']}  (negative normal modes at κ₃ = 0: "
              f"{out[name]['n_negative_k3_0']}, {time.time() - t0:.0f}s)", flush=True)
        save(f"thresh_{T}", out)
    return out


def fission_seed(fm, U, normal_types=(0, 1, 4, 5), eps=0.1):
    """Displace the (0, 2) torus along its softest line-1 (normal) mode, max |x| = eps."""
    G = fm.G
    T = tag(*grid_of.get(id(G), (0, 0, 0)))
    pn, Ha, HC = normal_hessians(fm, U, normal_types, os.path.join(SCRATCH, f"normal_hess_A21in02_{T}.npz"))
    # At κ₃ = 2 the torus has no bound line-1 mode (the lowest normal eigenvalue is the
    # continuum edge 2m² = 2), so the seed direction is the softest *localised* line-1 mode:
    # the lowest mode of the κ₃ = 0 Hessian, where the torus splits most easily.
    lam, v, _ = lowest_eig(Ha, pn.mass())
    vals = [lam]
    fr = sector_fractions(pn, v[:, None])[0]
    x6 = np.zeros((G.nn, NT))
    for t in normal_types:
        sel = pn.dof[:, t] >= 0
        x6[sel, t] = v[pn.dof[sel, t]]
    pf = FlagProblem(fm, U)
    x = np.zeros(pf.nfree)
    for t in range(NT):
        sel = pf.dof[:, t] >= 0
        x[pf.dof[sel, t]] = x6[sel, t]
    x *= eps / np.abs(x).max()
    cons = LineCentroids(pf)
    best = None
    for s in (+1.0, -1.0):
        xs = jnp.asarray(s * x)
        z = np.asarray(cons.c(pf.U0, xs))
        if best is None or z[0] - z[1] > best[1]:
            best = (xs, float(z[0] - z[1]))
    Us = np.asarray(pf.U_of(best[0]))
    return Us, dict(eigenvalue=float(vals[0]), fractions={k: round(v, 4) for k, v in fr.items()},
                    dZ=best[1], E=energy(fm, Us), eps=eps)


def cmd_fused(ne_r, ne_z, a, k3=2.0, ds=(0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), seed_eps=0.1):
    """Fused branch: start from the relaxed Q = 2 torus in block (0, 2) (d = 0) and
    pull the line-0 and line-2 centroids apart step by step (continuation in d)."""
    Gf, Uf = setup(ne_r, ne_z, a)
    fm = fmodel(Gf, k3)
    T = tag(ne_r, ne_z, a)
    fn = os.path.join(DATA, f"flagpair_state_A21in02_{T}_k3{k3:g}.npy")
    U = np.load(fn)
    E0 = energy(fm, U)
    E1 = energy(fm, np.load(os.path.join(DATA, f"flagpair_state_single01_{T}_k3{k3:g}.npy")))
    rows = [dict(d=0.0, E=E0, E_int=E0 - 2 * E1, converged=True, Q=degree_parts(fm, smooth_gauge(Gf, U)[0])[0],
                 lines=line_weights(Gf, U), note="relaxed torus in block (0,2); reference 2 E1 (unconstrained)")]
    print(f"d = 0.00: E = {E0:.5f}  E_int = {E0 - 2 * E1:+.5f}", flush=True)
    # On the embedded torus |U_00| = |U_22| identically and the normal (line-1) directions move
    # them only at second order, so the constraint Jacobian of Z_A − Z_B vanishes there.  Seed
    # the continuation with the softest normal mode, signed so that Z_A − Z_B > 0.
    U, seed = fission_seed(fm, U, eps=seed_eps)
    rows[0]["seed"] = seed
    print(f"seed: softest normal eigenvalue {seed['eigenvalue']:+.5f}, fractions {seed['fractions']}, "
          f"Z_A − Z_B = {seed['dZ']:+.4f}, E = {seed['E']:.5f}", flush=True)
    for d in ds:
        t0 = time.time()
        UA0, UB0 = embed(Gf, shifted(Gf, Uf, +d / 2), (0, 1)), embed(Gf, shifted(Gf, Uf, -d / 2), (1, 2))
        UAr, EA, cA, _, _ = relax_fixed(fm, UA0, (d / 2,), lines=(0,))
        Ur, Er, conv, hist, lm = relax_fixed(fm, U, (d / 2, -d / 2))
        Q = degree_parts(fm, smooth_gauge(Gf, Ur)[0])
        row = dict(d=d, E=Er, EA=EA, E_int=Er - 2 * EA, converged=conv, iterations=len(hist), Q=Q[0],
                   Q_3cycle=Q[2], lines=line_weights(Gf, Ur), parts=fm.parts(Ur),
                   multipliers=None if lm is None else lm.tolist())
        rows.append(row)
        print(f"d = {d:.2f}: E = {Er:.5f}  E_int = {Er - 2 * EA:+.5f}  Q = {Q[0]:+.4f}  lines "
              f"{', '.join(f'{x:.2f}' for x in row['lines'])}  conv {conv} ({len(hist)} it, {time.time() - t0:.0f}s)",
              flush=True)
        if conv and abs(Q[0] + 2) < 0.02:
            U = Ur
            np.save(os.path.join(DATA, f"flagpair_state_fused_{T}_k3{k3:g}_d{d:g}.npy"), Ur)
        save(f"fused_{T}_k3{k3:g}", dict(ne_r=ne_r, ne_z=ne_z, a=a, k3=k3, rows=rows,
                                         note="continuation from the (0,2) torus; centroids of (1-|U_00|^2)^2 and "
                                              "(1-|U_22|^2)^2 fixed at +d/2, -d/2; reference 2 E_A(d/2)"))
    return rows


def cmd_thresh_A12(ne_r, ne_z, a, k3=2.0):
    """Round 5's coaxial same-line bound pair A₁,₂ (Q = 2, one link) in the flag model:
    embed in block (0, 1), relax, and find its κ₃* against leaking into line 2."""
    from run_hopf_pair import mirror_full, freeze_A
    Gf, Uf = setup(ne_r, ne_z, a)
    T = tag(ne_r, ne_z, a)
    Ghp = freeze_A(Grid(ne_r, ne_z, p=2, a=a, half=True))
    Ub = np.load(os.path.join(DATA, f"pair_bound_state_{T}_mu1.npz"))["U"]
    Gm, Ubf = mirror_full(Ghp, Ub)
    assert Gm.nn == Gf.nn
    fm = fmodel(Gf, k3)
    E_cp1 = cp1_model(Gm).energy(Ubf, want_grad=False)[0]
    U, E, conv, n = relaxed_state(fm, embed(Gf, Ubf[:, :3], (0, 1)), f"A12in01_{T}_k3{k3:g}")
    E1 = energy(fm, np.load(os.path.join(DATA, f"flagpair_state_single01_{T}_k3{k3:g}.npy")))
    Q = degree_parts(fm, smooth_gauge(Gf, U)[0])
    out = dict(ne_r=ne_r, ne_z=ne_z, a=a, E_cp1=E_cp1, E=E, E1=E1, binding=2 * E1 - E, converged=conv,
               iterations=n, Q=Q[0], lines=line_weights(Gf, U))
    print(f"A12 in (0,1): E = {E:.5f} (CP¹ {E_cp1:.5f})  binding {2 * E1 - E:.4f}  Q = {Q[0]:+.4f}  conv {conv}",
          flush=True)
    out["threshold"] = kappa3_threshold(fm, U, (2, 3, 4, 5), label="A12in01")
    save(f"thresh_A12_{T}", out)
    return out


def cmd_fission(ne_r, ne_z, a, k3=2.0, targets=(0.25, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0, 12.5, 15.0,
                                                   18.0, 21.0, 24.0, 27.0, 30.0)):
    """Path from the (0, 2) torus (line 1 untouched, D₁ = 0) towards the separated
    (0, 1) + (1, 2) pair (D₁ ≈ 2 × 14.3): minimise E at fixed line-1 weight D₁.
    Seeded with the softest localised line-1 mode (the κ₃ = 0 instability of the torus),
    then continued in D₁."""
    Gf, Uf = setup(ne_r, ne_z, a)
    fm = fmodel(Gf, k3)
    T = tag(ne_r, ne_z, a)
    U = np.load(os.path.join(DATA, f"flagpair_state_A21in02_{T}_k3{k3:g}.npy"))
    E1 = energy(fm, np.load(os.path.join(DATA, f"flagpair_state_single01_{T}_k3{k3:g}.npy")))
    E0 = energy(fm, U)
    rows = [dict(D1=0.0, E=E0, E_minus_2E1=E0 - 2 * E1, converged=True, lines=line_weights(Gf, U))]
    U, seed = fission_seed(fm, U, eps=0.3)
    print(f"torus E = {E0:.5f} (2E1 = {2 * E1:.5f});  seed D = {line_weights(Gf, U)}  E = {seed['E']:.5f}", flush=True)
    for s in targets:
        t0 = time.time()
        Ur, Er, conv, hist, lm = relax_fixed(fm, U, (s,), lines=(1,), kind="weight", tol=1e-6)
        Q = degree_parts(fm, smooth_gauge(Gf, Ur)[0])
        lw = line_weights(Gf, Ur)
        zc = [float(Gf.W @ (Gf.zq * (Gf.Pv @ line_defect(Ur, c) ** 2)) / (Gf.W @ (Gf.Pv @ line_defect(Ur, c) ** 2)))
              for c in (0, 2)]
        rc = [float(Gf.W @ (Gf.rho * (Gf.Pv @ line_defect(Ur, c) ** 2)) / (Gf.W @ (Gf.Pv @ line_defect(Ur, c) ** 2)))
              for c in (0, 1, 2)]
        row = dict(D1=s, E=Er, E_minus_2E1=Er - 2 * E1, converged=conv, iterations=len(hist), Q=Q[0],
                   Q_3cycle=Q[2], lines=lw, z_centroids_0_2=zc, rho_centroids=rc, parts=fm.parts(Ur),
                   multiplier=None if lm is None else float(lm[0]))
        rows.append(row)
        print(f"D₁ = {s:5.2f}: E = {Er:.5f}  E − 2E1 = {Er - 2 * E1:+9.4f}  dE/dD₁ = {row['multiplier'] if lm is not None else float('nan'):+.4f}"
              f"  Q = {Q[0]:+.4f}  lines {', '.join(f'{x:.2f}' for x in lw)}  z(0,2) {zc[0]:+.2f} {zc[1]:+.2f}"
              f"  ρ {', '.join(f'{x:.2f}' for x in rc)}  conv {conv} ({len(hist)} it, {time.time() - t0:.0f}s)", flush=True)
        if conv and abs(Q[0] + 2) < 0.02:
            U = Ur
            np.save(os.path.join(SCRATCH, f"flagpair_state_fission_{T}_k3{k3:g}_D{s:g}.npy"), Ur)
        save(f"fission_{T}_k3{k3:g}", dict(ne_r=ne_r, ne_z=ne_z, a=a, k3=k3, E1=E1, rows=rows, seed=seed,
                                           note="minimum of E at fixed line-1 weight D1 = int(1-|U_11|^2), "
                                                "continued from the (0,2) torus"))
        if not conv:
            break
    return rows


def cmd_rotpair(ne_r, ne_z, a, k3=2.0, omA=0.3, omB=0.3, ds=(2.0, 3.0)):
    """Different-link pair with stationary isorotation Ω = diag(ω_A, 0, −ω_B): soliton A
    (link 01) turns at ω_A, B (link 12) at ω_B, so the link 02 that the cubic coupling
    x₀₁x₁₂x̄₀₂ feeds turns at ω_A + ω_B.  Its quanta have mass μ = 1: for ω_A + ω_B → 1
    the field induced in link 02 becomes long ranged (decay √(1 − (ω_A + ω_B)²)), for
    ω_A = −ω_B it stays static.  E_int = (V − T) differences at fixed Ω, centroids of
    lines 0 and 2 fixed at ±d/2; references relaxed alone with the same Ω and constraint."""
    Gf, Uf = setup(ne_r, ne_z, a)
    T_ = tag(ne_r, ne_z, a)
    Om = (omA, 0.0, -omB)
    fm = fmodel(Gf, k3, Om)
    fs = fmodel(Gf, k3)
    key = f"rotpair_{T_}_k3{k3:g}_wA{omA:g}_wB{omB:g}"
    rows = []
    for d in ds:
        t0 = time.time()
        uA, uB = shifted(Gf, Uf, +d / 2), shifted(Gf, Uf, -d / 2)
        UAr, EA, cA, _, _ = relax_fixed(fm, embed(Gf, uA, (0, 1)), (d / 2,), lines=(0,))
        UBr, EB, cB, _, _ = relax_fixed(fm, embed(Gf, uB, (1, 2)), (-d / 2,), lines=(2,))
        fn_static = os.path.join(DATA, f"flagpair_state_cons_{T_}_k3{k3:g}_d{d:g}.npy")
        U0 = np.load(fn_static) if os.path.exists(fn_static) else pair_U(Gf, uA, uB, (0, 1), (1, 2), 0.0, 0.0)
        E_static_int = energy(fs, U0) - energy(fs, embed(Gf, uA, (0, 1))) - energy(fs, embed(Gf, uB, (1, 2)))
        Ur, Er, conv, hist, lm = relax_fixed(fm, U0, (d / 2, -d / 2))
        Q = degree_parts(fm, smooth_gauge(Gf, Ur)[0])
        TA, TB, Tp = fm.kinetic(UAr), fm.kinetic(UBr), fm.kinetic(Ur)
        row = dict(d=d, omA=omA, omB=omB, E_int=Er - EA - EB, V_int=(Er + Tp) - (EA + TA) - (EB + TB),
                   T_int=Tp - TA - TB, T_pair=Tp, T_A=TA, T_B=TB, converged=conv, ref_converged=[bool(cA), bool(cB)],
                   iterations=len(hist), Q=Q[0], lines=line_weights(Gf, Ur), start="static relaxed pair"
                   if os.path.exists(fn_static) else "superposition")
        rows.append(row)
        print(f"ω_A = {omA:+.3f} ω_B = {omB:+.3f}  d = {d:.2f}: E_int(Ω) = {row['E_int']:+.5f}   "
              f"[V part {row['V_int']:+.5f}, −T part {-row['T_int']:+.5f}]  T_pair {Tp:.4f}  Q {Q[0]:+.4f}"
              f"  lines {', '.join(f'{x:.2f}' for x in row['lines'])}  conv {conv}/{bool(cA)}/{bool(cB)}"
              f" ({len(hist)} it, {time.time() - t0:.0f}s)", flush=True)
        save(key, dict(ne_r=ne_r, ne_z=ne_z, a=a, k3=k3, Om=list(Om), rows=rows))
    return rows


def cmd_rot1(ne_r, ne_z, a, k3=2.0, oms=(0.1, 0.2, 0.3, 0.4, 0.45)):
    """Single soliton (link 01) with stationary isorotation ω: V, T, V − T, leakage."""
    Gf, Uf = setup(ne_r, ne_z, a)
    T_ = tag(ne_r, ne_z, a)
    U = np.load(os.path.join(DATA, f"flagpair_state_single01_{T_}_k3{k3:g}.npy"))
    rows = []
    for om in oms:
        fm = fmodel(Gf, k3, (om, 0.0, 0.0))
        t0 = time.time()
        U, E, hist, conv = newton_relax(fm, U, types=range(NT), max_iter=60, verbose=False)
        Tk = fm.kinetic(U)
        Q = degree_parts(fm, smooth_gauge(Gf, U)[0])
        row = dict(omega=om, V_minus_T=E, T=Tk, V=E + Tk, E_total=E + 2 * Tk, Lambda=2 * Tk / om ** 2,
                   J=2 * Tk / om, converged=conv, iterations=len(hist), Q=Q[0], lines=line_weights(Gf, U))
        rows.append(row)
        print(f"ω = {om:.3f}: V − T = {E:.5f}  V = {E + Tk:.5f}  T = {Tk:.5f}  Λ = {row['Lambda']:.3f}"
              f"  J = {row['J']:.3f}  Q {Q[0]:+.4f}  lines {', '.join(f'{x:.2f}' for x in row['lines'])}"
              f"  conv {conv} ({len(hist)} it, {time.time() - t0:.0f}s)", flush=True)
        save(f"rot1_{T_}_k3{k3:g}", dict(ne_r=ne_r, ne_z=ne_z, a=a, k3=k3, rows=rows))
    return rows


def cmd_cons(ne_r, ne_z, a, k3=2.0, ds=(3.0, 2.0, 1.5, 1.0, 0.5), check_B=False):
    """Shared-line pair relaxed at fixed separation d (A at +d/2, B at −d/2).
    References: A alone and B alone relaxed with their own centroid fixed at ±d/2."""
    Gf, Uf = setup(ne_r, ne_z, a)
    fm = fmodel(Gf, k3)
    T = tag(ne_r, ne_z, a)
    fn_out = os.path.join(DATA, f"flagpair_cons_{T}_k3{k3:g}.json")
    rows = json.load(open(fn_out))["rows"] if os.path.exists(fn_out) else []      # append to earlier scans
    rows = [r for r in rows if r["d"] not in ds]
    Uprev = None
    done = sorted({r["d"] for r in rows if r["d"] > max(ds)})
    if done:                                     # continue from the saved state of the nearest larger d
        fn_prev = os.path.join(DATA, f"flagpair_state_cons_{T}_k3{k3:g}_d{done[0]:g}.npy")
        Uprev = np.load(fn_prev) if os.path.exists(fn_prev) else None
    for d in ds:
        t0 = time.time()
        uA, uB = shifted(Gf, Uf, +d / 2), shifted(Gf, Uf, -d / 2)
        UA0, UB0 = embed(Gf, uA, (0, 1)), embed(Gf, uB, (1, 2))
        # single references: fix the one centroid that exists (line 0 for A, line 2 for B)
        UAr, EA, cA, hA, _ = relax_fixed(fm, UA0, (d / 2,), lines=(0,))
        if check_B or not any(r["d"] != d for r in rows):
            UBr, EB, cB, hB, _ = relax_fixed(fm, UB0, (-d / 2,), lines=(2,))
            print(f"   references at d = {d:.2f}: E_A = {EA:.6f}  E_B = {EB:.6f}  (B − A {EB - EA:+.1e})", flush=True)
        else:
            EB, cB = EA, cA          # z → −z with lines 0 ↔ 2 maps A onto B (checked at the first d)
        U0 = pair_U(Gf, uA, uB, (0, 1), (1, 2), 0.0, 0.0)
        E0 = energy(fm, U0)
        starts = [("superposition", U0)] + ([("previous", Uprev)] if Uprev is not None else [])
        best = None
        for nm, Us in starts:
            Ur, Er, conv, hist, lm = relax_fixed(fm, Us, (d / 2, -d / 2))
            Q = degree_parts(fm, smooth_gauge(Gf, Ur)[0])
            cand = dict(start=nm, E=Er, converged=conv, iterations=len(hist), Q=Q[0], Q_3cycle=Q[2],
                        multipliers=None if lm is None else lm.tolist(), lines=line_weights(Gf, Ur),
                        parts=fm.parts(Ur))
            print(f"   d = {d:.2f} start {nm:13s}: E = {Er:.5f}  E_int = {Er - EA - EB:+.5f}  Q = {Q[0]:+.4f}"
                  f"  conv {conv} ({len(hist)} it)", flush=True)
            if conv and abs(Q[0] + 2) < 0.02 and (best is None or Er < best[0]["E"]):
                best = (cand, Ur)
            cand["E_int"] = Er - EA - EB
            rows.append(dict(d=d, E_sup_int=E0 - energy(fm, UA0) - energy(fm, UB0), EA=EA, EB=EB,
                             ref_converged=[bool(cA), bool(cB)], **cand))
        if best is not None:
            Uprev = best[1]
            np.save(os.path.join(DATA, f"flagpair_state_cons_{T}_k3{k3:g}_d{d:g}.npy"), best[1])
        print(f"d = {d:.2f}: superposition E_int {rows[-1]['E_sup_int']:+.5f}   "
              f"relaxed E_int {min(r['E_int'] for r in rows if r['d'] == d):+.5f}   ({time.time() - t0:.0f}s)",
              flush=True)
        save(f"cons_{T}_k3{k3:g}", dict(ne_r=ne_r, ne_z=ne_z, a=a, k3=k3, rows=rows,
                                        note="A (block (0,1)) and B (block (1,2)) on the axis; centroids of "
                                             "(1-|U_00|^2)^2 and (1-|U_22|^2)^2 fixed at +d/2, -d/2; references "
                                             "relaxed alone with their own centroid fixed"))
    return rows


if __name__ == "__main__":
    cmd = sys.argv[1]
    ne_r, ne_z, a = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
    k3 = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    if cmd == "check":
        cmd_check(ne_r, ne_z, a, k3)
    elif cmd == "map":
        kind = sys.argv[6] if len(sys.argv) > 6 else "shared"
        order = sys.argv[7] if len(sys.argv) > 7 else "AB"
        cmd_map(ne_r, ne_z, a, k3, kind, order)
    elif cmd == "relax":
        kind = sys.argv[6] if len(sys.argv) > 6 else "shared"
        d0 = float(sys.argv[7]) if len(sys.argv) > 7 else 2.0
        cmd_relax(ne_r, ne_z, a, k3, kind, d0)
    elif cmd == "q2":
        # the embedded solutions do not depend on κ₃ (E_C = 0 and ∇E_C = 0 on them); stability
        # in κ₃ comes from the thresh subcommand, so by default no spectra here (nev = 0)
        cmd_q2(ne_r, ne_z, a, tuple(float(s) for s in sys.argv[5:]) or (2.0,), nev=int(os.environ.get("NEV", "0")))
    elif cmd == "thresh":
        cmd_thresh(ne_r, ne_z, a)
    elif cmd == "threshA12":
        cmd_thresh_A12(ne_r, ne_z, a)
    elif cmd == "rot1":
        cmd_rot1(ne_r, ne_z, a, k3)
    elif cmd == "rotpair":
        omA, omB = float(sys.argv[6]), float(sys.argv[7])
        ds = tuple(float(s) for s in sys.argv[8:]) or (2.0, 3.0)
        cmd_rotpair(ne_r, ne_z, a, k3, omA, omB, ds)
    elif cmd == "fission":
        ts = tuple(float(s) for s in sys.argv[6:])
        cmd_fission(ne_r, ne_z, a, k3, ts) if ts else cmd_fission(ne_r, ne_z, a, k3)
    elif cmd == "fused":
        ds = tuple(float(s) for s in sys.argv[6:])
        cmd_fused(ne_r, ne_z, a, k3, ds) if ds else cmd_fused(ne_r, ne_z, a, k3)
    elif cmd == "cons":
        ds = tuple(float(s) for s in sys.argv[6:])
        cmd_cons(ne_r, ne_z, a, k3, ds) if ds else cmd_cons(ne_r, ne_z, a, k3)
