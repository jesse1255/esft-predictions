"""
The flag model with n lines (U(n)/U(1)^n), on the lattice of lattice3d, and two paper results checked
on it (the user's "next step", 2026-10-03).

Energy (n lines; for n = 3 exactly lattice3d.Lattice):
    σ          Σ_links Σ_{a<b} r_ab ½(|Õ_ab|² + |Õ_ba|²) h
    Faddeev    Σ_plaq Σ_a ½κ_a (arg Π_a)² / h                    (Berry flux of each line)
    3-cycle    Σ_plaq ¼ Σ_corners Σ_{a≠c} ½κ₃ |Σ_{b∉{a,c}} (L1_ab L2_bc − L3_ab L4_bc)|² / h
               (the off-diagonal part of the commutator [ω_i, ω_j]: for n = 3 one middle line b,
                for n ≥ 4 a sum over all middle lines)
    potential  Σ_sites Σ_a m_a² (1 − |U_aa|²) h³
Linking rule (§5.5) for rings in links α = e_a − e_b and β = e_c − e_d:
    Q = q(α) + q(β) + (α·β) lk,  α·β = δ_ac − δ_ad − δ_bc + δ_bd ∈ {2, 1, 0, −1}.

1. Four lines: links that share no line (α·β = 0, e.g. 01 and 23).  A block-diagonal field
   U = U₀₁ ⊕ U₂₃ has ω = U†dU block diagonal, so every term splits: E(U₀₁ ⊕ U₂₃) = E(U₀₁) + E(U₂₃)
   for ANY positions, overlap or linking (no σ cross terms, each line's flux from its own block,
   no three-cycle path a → b → c across the blocks, potential additive).  The block-diagonal fields
   are the fixed set of the symmetry U → diag(1, 1, −1, −1) U (up to gauge), so the dynamics keeps
   them block diagonal: two such rings pass through each other without any interaction, and their
   charge is 1 + 1 whatever their linking.  (Whether a small off-block perturbation grows while they overlap is
   not computed: the σ and Faddeev terms have mixed second-order terms there.)
2. Charge: lattice3d.charge for n lines (sum over the n line bundles).

3. Can they really pass (`offblock`)?  The block-diagonal set is invariant, so the energy is even in the
   off-block perturbation X = [[0, Y], [−Y†, 0]] (Y couples lines 0, 1 with 2, 3) and its Hessian H_off
   decouples, even where the pair is not relaxed.  H_off is computed through U → U (1 + X + X²/2) (exact
   to second order) with a Hessian-vector product; its lowest eigenvalues by LOBPCG with a
   (α L + β)⁻¹ preconditioner (L the open-boundary lattice Laplacian, diagonal in the DCT;
   α = 2 r h, β = 4 m² h³ = the bottom of the vacuum continuum).  Configurations from the relaxed
   single ring (exact lattice symmetries only: rolls by whole sites, 90° rotations):
   single 01 ring; the 01 + 23 pair coaxial on top of each other; crossed at right angles; linked
   (crossed and shifted by ±2 sites).  λ_min < 0 would mean the passage is unstable (they would
   start to mix); 0 < λ_min < β is a bound leakage mode; ≥ β none.

Usage: python nlines.py check      → n = 3 equals lattice3d; four-line decoupling; charges
       python nlines.py offblock   → lowest eigenvalues of H_off for the single ring and the pairs
"""

import os
import sys
import json

import numpy as np
import jax
import jax.numpy as jnp

from lattice3d import Lattice, _abs2, _dag, _sl, _arc, SCRATCH, DATA

jax.config.update("jax_enable_x64", True)


def _arc_n(O):
    from lattice3d import _arc_scale
    s = _abs2(O)
    eye = jnp.eye(O.shape[-1], dtype=bool)
    return jnp.where(eye, O, O * _arc_scale(s))


class NLattice:
    def __init__(self, N, h, n, r=2.0, kappa=2.0, m2=1.0, kappa3=2.0):
        self.N, self.h, self.n = int(N), float(h), int(n)
        self.r = np.full((n, n), float(r)) if np.isscalar(r) else np.asarray(r, float)
        self.kappa = np.full(n, float(kappa)) if np.isscalar(kappa) else np.asarray(kappa, float)
        self.m2 = np.full(n, float(m2)) if np.isscalar(m2) else np.asarray(m2, float)
        self.kappa3 = float(kappa3)
        self._E = jax.jit(self.energy_U)

    def parts_U(self, U):
        h, n = self.h, self.n
        Uh = _dag(U)
        O = [jnp.matmul(Uh[:-1], U[1:]), jnp.matmul(Uh[:, :-1], U[:, 1:]), jnp.matmul(Uh[:, :, :-1], U[:, :, 1:])]
        O = [_arc_n(Oi) for Oi in O]
        Es = 0.0
        for Oi in O:
            for a in range(n):
                for b in range(a + 1, n):
                    Es = Es + self.r[a, b] * 0.5 * jnp.sum(_abs2(Oi[..., a, b]) + _abs2(Oi[..., b, a]))
        Es = Es * h
        EF, EC = 0.0, 0.0
        for d1, d2 in ((0, 1), (0, 2), (1, 2)):
            O1, O2 = O[d1], O[d2]
            A, Cc = _sl(O1, d2, 0, -1), _sl(O1, d2, 1, None)
            B, D = _sl(O2, d1, 1, None), _sl(O2, d1, 0, -1)
            for a in range(n):
                F = jnp.angle(A[..., a, a] * B[..., a, a] * jnp.conj(Cc[..., a, a]) * jnp.conj(D[..., a, a]))
                EF = EF + 0.5 * self.kappa[a] * jnp.sum(F ** 2)
            if self.kappa3 != 0.0:
                Ad, Bd, Cd, Dd = _dag(A), _dag(B), _dag(Cc), _dag(D)
                corners = ((A, B, D, Cc), (Ad, D, B, Cd), (Cc, Bd, Dd, A), (Cd, Dd, Bd, Ad))
                for a in range(n):
                    for c in range(n):
                        if c == a:
                            continue
                        for L1, L2, L3, L4 in corners:
                            Cv = 0.0
                            for b in range(n):
                                if b in (a, c):
                                    continue
                                Cv = Cv + L1[..., a, b] * L2[..., b, c] - L3[..., a, b] * L4[..., b, c]
                            EC = EC + 0.125 * self.kappa3 * jnp.sum(_abs2(Cv))
        EF, EC = EF / h, EC / h
        EV = 0.0
        for a in range(n):
            EV = EV + self.m2[a] * jnp.sum(1.0 - _abs2(U[..., a, a]))
        return Es, EF, EC, EV * h ** 3

    def energy_U(self, U):
        return sum(self.parts_U(U))

    def energy(self, U):
        return float(self._E(jnp.asarray(U)))


def charge_n(h, U, pad=1):
    """lattice3d.charge for n lines: Q = (1/8π²) Σ_a ∫ A_a·B_a over the n line bundles."""
    Qs = [_charge_line(h, U, a, pad) for a in range(U.shape[-1])]
    return sum(Qs), Qs


def _charge_line(h, U, a, pad):
    Uh = np.conj(np.swapaxes(U, -1, -2))
    O = [np.einsum("...ij,...jk->...ik", Uh[:-1], U[1:]), np.einsum("...ij,...jk->...ik", Uh[:, :-1], U[:, 1:]),
         np.einsum("...ij,...jk->...ik", Uh[:, :, :-1], U[:, :, 1:])]
    d = [Oi[..., a, a] for Oi in O]

    def flux(i, j):
        A, Bj = d[i], d[j]
        A0 = A[tuple(slice(0, -1) if k == j else slice(None) for k in range(3))]
        A1 = A[tuple(slice(1, None) if k == j else slice(None) for k in range(3))]
        B0 = Bj[tuple(slice(0, -1) if k == i else slice(None) for k in range(3))]
        B1 = Bj[tuple(slice(1, None) if k == i else slice(None) for k in range(3))]
        return np.angle(A0 * B1 * np.conj(A1) * np.conj(B0)) / h ** 2

    Fyz, Fzx, Fxy = flux(1, 2), -flux(0, 2), flux(0, 1)

    def centre(F, axis):
        sl0, sl1 = [slice(None)] * 3, [slice(None)] * 3
        sl0[axis], sl1[axis] = slice(0, -1), slice(1, None)
        return 0.5 * (F[tuple(sl0)] + F[tuple(sl1)])

    B = np.stack([centre(Fyz, 0), centre(Fzx, 1), centre(Fxy, 2)])
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
    A_ = np.real(np.fft.ifftn(1j * np.cross(K, Bk, axis=0) / K2, axes=(1, 2, 3)))
    return float(np.sum(A_ * Bp) * h ** 3 / (8 * np.pi ** 2))


def embed(U3, lines, n):
    """A field living in lines (a, b) of a 3 × 3 single-vortex field (its block (0, 1)) placed into
    lines `lines` of an n × n field (identity elsewhere)."""
    a, b = lines
    U = np.zeros(U3.shape[:3] + (n, n), complex)
    for i in range(n):
        U[..., i, i] = 1.0
    U[..., a, a], U[..., a, b], U[..., b, a], U[..., b, b] = U3[..., 0, 0], U3[..., 0, 1], U3[..., 1, 0], U3[..., 1, 1]
    return U


def cmd_check():
    from run_flag_pair import setup
    from smith import ring, smooth_pair, configs, rot_x
    N, h = 41, 0.3
    lat3 = Lattice(N, h)
    nl3 = NLattice(N, h, 3)
    UA = np.load(os.path.join(SCRATCH, "lattice_ref_A01_N41_h0.3.npz"))["U"]
    print("n = 3 vs lattice3d, relaxed single vortex:", repr(nl3.energy(UA)), repr(lat3.energy(UA)))
    Useed = np.load(os.path.join(SCRATCH, "lattice_seed_smith_touch_gs_N41_h0.3.npz"))["U"]
    print("n = 3 vs lattice3d, touching 01 + 12 seed:", repr(nl3.energy(Useed)), repr(lat3.energy(Useed)))
    Gf, Uf = setup(24, 32, 4)
    u = Uf[:, :3]
    cf = configs(0.93)
    nl4 = NLattice(N, h, 4)
    out = {}
    for name in ("touch", "linkp", "linkm", "cross2"):
        (RA, aA), (RB, aB) = cf[name]
        VA = ring(Gf, u, lat3, RA, aA, (0, 1))            # vortex A in block (0, 1) of a 3 × 3 field
        VB = ring(Gf, u, lat3, RB, aB, (0, 1))            # vortex B, same construction, placed in B's geometry
        A4 = embed(VA, (0, 1), 4)
        B4 = embed(VB, (2, 3), 4)
        both = A4 @ B4                                    # block diagonal: exact superposition
        EA, EB, EAB = nl4.energy(A4), nl4.energy(B4), nl4.energy(both)
        Q, Qa = charge_n(h, both, pad=1)
        # the same geometry with the second ring in link 12 (shares line 1 with 01), three lines
        V12 = ring(Gf, u, lat3, RB, aB, (1, 2))
        U3, _ = smooth_pair(VA, V12)
        E3 = lat3.energy(U3)
        out[name] = dict(E_A=EA, E_B=EB, E_AB=EAB, interaction=EAB - EA - EB, Q=Q, Q_lines=Qa,
                         three_lines_01_12=dict(E=E3, interaction=E3 - lat3.energy(VA) - lat3.energy(V12)))
        print(f"{name:7s} four lines, 01 + 23: E_A {EA:.4f} + E_B {EB:.4f} -> together {EAB:.4f} (interaction {EAB - EA - EB:+.2e}),"
              f" charge {Q:+.3f} {np.round(Qa, 3).tolist()};   three lines, 01 + 12 same geometry: interaction {out[name]['three_lines_01_12']['interaction']:+.2f}")
    json.dump(out, open(os.path.join(DATA, "nlines_four_decoupling.json"), "w"), indent=1)


def offblock_hvp(nl):
    """H_off(U) v as a jitted function of (U, v); U is an argument, so one compilation serves every U."""
    N = nl.N

    def build_X(th):
        Y = (th[..., 0:4] + 1j * th[..., 4:8]).reshape(th.shape[:3] + (2, 2))
        Z = jnp.zeros(th.shape[:3] + (2, 2), complex)
        top = jnp.concatenate([Z, Y], -1)
        bot = jnp.concatenate([-jnp.conj(jnp.swapaxes(Y, -1, -2)), Z], -1)
        return jnp.concatenate([top, bot], -2)

    def f(th, U):
        X = build_X(th)
        return nl.energy_U(U @ (jnp.eye(4) + X + 0.5 * X @ X))

    g = jax.grad(f)
    zero = jnp.zeros((N, N, N, 8))
    return jax.jit(lambda U, v: jax.jvp(lambda th: g(th, U), (zero,), (v,))[1])


def swap_blocks(v):
    """The off-block vector seen from the other block: lines (0, 1) <-> (2, 3) sends Y to −Y†."""
    Y = (v[..., 0:4] + 1j * v[..., 4:8]).reshape(v.shape[:3] + (2, 2))
    Y = -np.conj(np.swapaxes(Y, -1, -2)).reshape(v.shape[:3] + (4,))
    return np.concatenate([Y.real, Y.imag], -1)


def cmd_offblock(maxiter=60, tol=2e-3):
    import time
    from scipy.fft import dctn, idctn
    from scipy.sparse.linalg import LinearOperator, lobpcg
    N, h = 41, 0.3
    nl = NLattice(N, h, 4)
    hvp = offblock_hvp(nl)
    UA = np.load(os.path.join(SCRATCH, "lattice_ref_A01_N41_h0.3.npz"))["U"]
    rot = lambda F: np.rot90(F, 1, axes=(1, 2))                        # ring axis z -> y (exact on the lattice)
    shift = lambda F, k, ax: np.roll(F, k, axis=ax)
    configs = {
        "single": (embed(UA, (0, 1), 4), None, None),
        "coaxial_on_top": (embed(UA, (0, 1), 4) @ embed(UA, (2, 3), 4), lambda v: v, lambda v: v),
        "crossed": (embed(UA, (0, 1), 4) @ embed(rot(UA), (2, 3), 4), lambda v: v, rot),
        "linked": (embed(shift(UA, -2, 0), (0, 1), 4) @ embed(shift(rot(UA), 2, 0), (2, 3), 4),
                   lambda v: shift(v, -2, 0), lambda v: shift(rot(v), 2, 0)),
    }
    alpha, beta = 2 * 2.0 * h, 4 * 1.0 * h ** 3
    k = np.arange(N)
    l1 = 2 - 2 * np.cos(np.pi * k / N)
    Lk = l1[:, None, None] + l1[None, :, None] + l1[None, None, :]
    shape = (N, N, N, 8)

    def prec(V):
        V = np.asarray(V).reshape(shape + (-1,))
        out = np.empty_like(V)
        for j in range(V.shape[-1]):
            c = dctn(V[..., j], type=2, axes=(0, 1, 2), norm="ortho")
            out[..., j] = idctn(c / (alpha * Lk[..., None] + beta), type=2, axes=(0, 1, 2), norm="ortho")
        return out.reshape(-1, V.shape[-1])
    M = LinearOperator((np.prod(shape),) * 2, matmat=prec, matvec=lambda v: prec(v[:, None])[:, 0], dtype=float)
    results, vecs = {}, {}
    fn = os.path.join(DATA, "nlines_offblock.json")
    for name, (U, tA, tB) in configs.items():
        Uj = jnp.asarray(U)
        ncall = [0]

        def mv(V):
            V = np.asarray(V).reshape(shape + (-1,))
            out = np.empty_like(V)
            for j in range(V.shape[-1]):
                out[..., j] = np.asarray(hvp(Uj, jnp.asarray(V[..., j])))
                ncall[0] += 1
            return out.reshape(-1, V.shape[-1])
        A = LinearOperator((np.prod(shape),) * 2, matmat=mv, matvec=lambda v: mv(v[:, None])[:, 0], dtype=float)
        if name == "single":
            rng = np.random.default_rng(0)
            env = np.exp(-((np.indices((N, N, N)) - (N - 1) / 2) ** 2).sum(0) * h ** 2 / 4.0)[..., None]
            X0 = np.stack([(env * rng.normal(size=shape)).reshape(-1) for _ in range(3)], 1)
        else:
            vA = [vecs["single"][:, j].reshape(shape) for j in range(2)]
            vB = [tB(swap_blocks(v)) for v in vA]
            X0 = np.stack([tA(v).reshape(-1) for v in vA] + [v.reshape(-1) for v in vB], 1)
        t0 = time.time()
        lam, V, hist = lobpcg(A, X0, M=M, largest=False, tol=tol, maxiter=maxiter, retLambdaHistory=True)
        o = np.argsort(lam)
        lam, V = lam[o], V[:, o]
        res = [float(np.linalg.norm(mv(V[:, [j]])[:, 0] - lam[j] * V[:, j]) / np.linalg.norm(V[:, j])) for j in range(len(lam))]
        vecs[name] = V
        E = nl.energy(U)
        results[name] = dict(E=E, lam=[float(x) for x in lam], residual=res, hvp_calls=ncall[0],
                             iterations=len(hist), seconds=time.time() - t0, continuum_bottom=beta)
        print(f"{name:15s} E = {E:9.3f}   lowest off-block eigenvalues {np.round(lam, 5).tolist()}   residuals "
              f"{np.round(res, 5).tolist()}   ({len(hist)} iterations, {ncall[0]} products, {time.time() - t0:.0f} s)", flush=True)
        json.dump(results, open(fn, "w"), indent=1)
    np.save(os.path.join(SCRATCH, "nlines_offblock_vecs.npy"), {k: v for k, v in vecs.items()}, allow_pickle=True)
    print(f"continuum bottom beta = {beta:.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        cmd_check()
    elif sys.argv[1] == "offblock":
        cmd_offblock()
