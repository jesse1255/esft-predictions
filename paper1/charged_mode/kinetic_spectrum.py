"""
Vibrational spectra with the full kinetic metric (hypothesis sweep §3b, §4 item 3).

Rounds 5-6 used the L² metric ∫|δU|² for the small-oscillation problem.  The true
kinetic energy of M2′ also has the quartic terms with one time index, so
    G(δU, δU) = 2 T[δU̇ = δU],   T = ∫ [Σ r|w_t^{ab}|² + Σ_a κ_a/2 Σ_i (F^{(a)}_{ti})² + κ₃/2 Σ |C_{ti}|²],
w_t = Ẑ†·(aligned interpolation of δU) with the element phases of the base state.
Squared frequencies are the eigenvalues of H v = ω² G v.  Far away G → the L² form, so
the continuum still starts at ω² = μ² = 1.  With inertia counts (Sylvester) of H − sG:
    N(s) = number of ω² < s;  bound vibrations = N(1 − δ) − N(δ)  (zero modes excluded),
and the lowest ω² by bisection.  Blocks: in-block (the CP¹ spectrum of round 5) and
normal (leaking into the third line).
Usage: python kinetic_spectrum.py → data/flagpair_kinetic_spectrum.json
"""
import json, os, time
import numpy as np
import jax, jax.numpy as jnp
from flag_axisym import FlagModel, FlagProblem, inertia, NT, _abs2
from run_flag_pair import setup, fmodel, DATA, lowest_eig

class KinMetric(FlagModel):
    """energy_U(U) = T[δU = U − U₀] at the aligned base of U₀: its Hessian at U₀ is G."""

    def __init__(self, fm, U0):
        super().__init__(fm.G, L=fm.L, kappa3=fm.kappa3, disc="align")
        self.U0 = jnp.asarray(U0)
        Ug = self.U0[self.a_cols]; Ur = self.U0[self.a_ref]
        w = jnp.sum(jnp.conj(Ur) * Ug, axis=1)
        self.ph = jnp.conj(w) * jax.lax.rsqrt(_abs2(w) + 1e-300)
        self.Z0 = self._interp_aligned(self.U0)

    def _interp_with(self, V):
        Va = V[self.a_cols] * self.ph[:, None, :]
        return jax.ops.segment_sum(self.a_vv[:, None, None] * Va, self.a_rows, num_segments=self.nq)

    def energy_U(self, U, r=None, k3=None):
        r = self.r if r is None else r
        k3 = self.kappa3 if k3 is None else k3
        Z, Zr, Zz = self.Z0
        Zh = jnp.conj(jnp.swapaxes(Z, 1, 2))
        w_r, w_z = Zh @ Zr, Zh @ Zz
        w_f = (Zh @ (1j * self.Lvec[None, :, None] * Z)) / self.rho[:, None, None]
        w_t = Zh @ self._interp_with(U - self.U0)
        sig = 0.0
        for k, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            sig = sig + r[k] * 0.5 * (_abs2(w_t[:, a, b]) + _abs2(w_t[:, b, a]))
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
                    sk = sk + 0.5 * k3 * _abs2(C)
        return jnp.sum(self.W * (sig + sk))


def analyse(name, U, types, k3=2.0):
    Gf, _ = setup(24, 32, 4)
    fm = fmodel(Gf, k3)
    t0 = time.time()
    ph = FlagProblem(fm, U, types=types)
    H = ph.hessian()
    pg = FlagProblem(KinMetric(fm, U), U, types=types)
    G = pg.hessian()          # Hessian of T = G (T is exactly quadratic in δU)
    M = ph.mass()             # L² metric used before (continuum at 2 in that normalisation)
    cnt = lambda A, B, s: inertia((A - s * B).tocsr())[1]
    res = dict(name=name, types=list(types))
    for key, B, edge in (("full_metric", G, 1.0), ("L2_metric", M, 2.0)):
        n_lo = cnt(H, B, 1e-3 * edge)          # zero modes and unstable ones
        n_hi = cnt(H, B, (1 - 1e-3) * edge)    # everything below the continuum edge
        n_neg = cnt(H, B, -1e-3 * edge)
        res[key] = dict(continuum_edge=edge, n_negative=n_neg, n_below_small=n_lo, n_below_edge=n_hi,
                        bound_vibrations=n_hi - n_lo)
    lam, v, _ = lowest_eig(H.tocsr(), G, lo=-4.0, hi=4.0)
    res["lowest_omega2_full"] = lam
    # the lowest ω² above the zero modes: bisection on counts between small and the edge
    lo, hi, target = 1e-3, 1 - 1e-3, res["full_metric"]["n_below_small"] + 1
    if res["full_metric"]["bound_vibrations"] > 0:
        while hi - lo > 1e-5:
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if cnt(H, G, mid) < target else (lo, mid)
        res["lowest_bound_omega2_full"] = 0.5 * (lo + hi)
    res["t"] = time.time() - t0
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    T = "ne24x32_a4"
    load = lambda n: np.load(os.path.join(DATA, f"flagpair_state_{n}_{T}_k32.npy"))
    out = []
    for name, types in (("single01", (0, 1)), ("single01", (2, 3, 4, 5)),
                        ("A21in02", (2, 3)), ("A21in02", (0, 1, 4, 5)),
                        ("A12in01", (0, 1)), ("A12in01", (2, 3, 4, 5))):
        out.append(analyse(name, load(name), types))
        json.dump(dict(note=__doc__, rows=out), open(os.path.join(DATA, "flagpair_kinetic_spectrum.json"), "w"), indent=1)
