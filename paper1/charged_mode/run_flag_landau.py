"""
Order of the κ₃ transition at the S₃ point, from the local normal form.

At λ = 1 the embedded Hopfion loses stability at κ₃ = κ₃* through a complex
pair of normal modes ξ (the pair is the orbit of the phase of Z₃'s tilt).
Along the centre direction the reduced energy is

    E(t) = E₀ + ½ a₁ (κ₃ − κ₃*) t² + c₄ t⁴ + …,
    a₁ = ξᵀ H_C ξ,
    c₄ = E₄/24 − (1/8) gᵀ H⁻¹ g,
    E₄ = d⁴E(tξ)/dt⁴,   g = D³E[ξ, ξ, ·],

where the second term of c₄ comes from minimising over the transverse
(in-block) response η = −(t²/2) H⁻¹ g.  H⁻¹g is solved exactly in the R-even
in-block sector (flag_symmetry.py): g is R-even, and the only negative in-block
modes on the grid (lattice-lifted translation and isorotation) are R-odd.
The cubic term vanishes by the U(1) symmetry.  c₄ > 0: supercritical
(continuous; below κ₃* a small-amplitude F₂ branch appears with
ΔE = −a₁²(κ₃* − κ₃)²/(16 c₄)).  c₄ < 0: subcritical (first order; no
small-amplitude branch).  At κ₃ = κ₃* the value of c₄ does not depend on the
parametrisation of the centre manifold.

All derivatives are exact (JAX); only the Hessian solve is sparse-direct.

Usage: python run_flag_landau.py ne [--box 3.0] [--l3 0] [--disc u|proj]
Writes data/flag_landau_ne{ne}[_a{box}][_proj].json.  --disc proj uses the
gauge-invariant projector discretisation (flag_axisym.FlagModel).
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

from hopfion_axisym import Grid, sparse_solve
from hansatz import HProfile
from flag_axisym import FlagModel, FlagProblem, embedded_hopfion, newton_relax, inertia, NT
from flag_symmetry import reflection, restrict, sector_basis

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def file_tag(box=3.0, disc="u"):
    return ("" if box == 3.0 else f"_a{box:g}") + ("" if disc == "u" else f"_{disc}")


def embedded(ne, box, disc="u"):
    """Embedded m = 1 Hopfion, relaxed with the given discretisation (cached)."""
    fn = os.path.join(DATA, f"flag_embedded_ne{ne}{file_tag(box, disc)}.npy")
    G = Grid(ne, ne, p=2, a=box, half=False)
    if os.path.exists(fn):
        return G, np.load(fn)
    if disc == "u":
        cf = np.load(os.path.join(DATA, "state_ne32.npz"))["cf"]
        U0 = embedded_hopfion(G, HProfile(G).h_nodes(cf))
    else:
        U0 = embedded(ne, box, "u")[1]
    U, E, hist, conv = newton_relax(FlagModel(G, L=(0, 1, 2), disc=disc), U0, types=(0, 1),
                                    max_iter=60, verbose=False)
    np.save(fn, U)
    return G, U


def main(ne, box=3.0, l3=0, tol=1e-7, disc="u"):
    t_start = time.time()
    G, U = embedded(ne, box, disc)
    fm = FlagModel(G, L=(0, 1, l3), kappa3=1.0, disc=disc)
    E_emb = float(fm.energy_U(jnp.asarray(U), None, 0.0))
    # ---- threshold κ₃* in the normal block -----------------------------------------
    pn = FlagProblem(fm, U, types=(2, 3, 4, 5))
    pn.set_couplings(r=(2.0, 2.0, 2.0), kappa3=0.0)
    Ha = pn.hessian()
    pn.set_couplings(kappa3=1.0)
    HC = (pn.hessian() - Ha).tocsr()
    lo, hi = 0.0, 2.0
    assert inertia(Ha + hi * HC)[1] == 0 and inertia(Ha)[1] > 0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if inertia(Ha + mid * HC)[1] > 0:
            lo = mid
        else:
            hi = mid
    k3s = 0.5 * (lo + hi)
    vals, vecs = pn.lowest((Ha + k3s * HC).tocsr(), nev=4, sigma=-1e-3)
    xin = vecs[:, 0]
    Mn = pn.mass()
    xin = xin / np.sqrt(xin @ (Mn @ xin))
    a1 = float(xin @ (HC @ xin))
    print(f"ne={ne} a={box}: E_emb={E_emb:.6f}  κ3*={k3s:.7f}  lowest normal eigenvalues {vals}  "
          f"a1={a1:.5e}  ({time.time() - t_start:.0f}s)", flush=True)
    # ---- full problem at κ₃* ------------------------------------------------------
    pf = FlagProblem(fm, U, types=range(NT))
    pf.set_couplings(r=(2.0, 2.0, 2.0), kappa3=k3s)
    xi6 = np.zeros((G.nn, NT))
    for t in (2, 3, 4, 5):
        sel = pn.dof[:, t] >= 0
        xi6[sel, t] = xin[pn.dof[sel, t]]
    xi = np.zeros(pf.nfree)
    for t in range(NT):
        sel = pf.dof[:, t] >= 0
        xi[pf.dof[sel, t]] = xi6[sel, t]
    xi_j = jnp.asarray(xi)
    U0, r, k3 = pf.U0, pf.r, pf.k3
    energy = pf._energy                      # (U0, x, r, k3)

    def f(t):
        return energy(U0, t * xi_j, r, k3)

    d1 = jax.jacfwd(f)                       # forward mode: scalar argument
    d2 = jax.jacfwd(d1)
    d3 = jax.jacfwd(d2)
    d4 = jax.jit(jax.jacfwd(d3))
    t0 = jnp.asarray(0.0)
    E2, E3, E4 = float(d2(t0)), float(d3(t0)), float(d4(t0))
    # finite-difference check of E4
    h = 0.05
    Ef = [float(f(jnp.asarray(s * h))) for s in (-2, -1, 0, 1, 2)]
    E4_fd = (Ef[0] - 4 * Ef[1] + 6 * Ef[2] - 4 * Ef[3] + Ef[4]) / h ** 4
    # cubic coupling g = D³E[ξ, ξ, ·]
    x0 = jnp.zeros(pf.nfree)
    q = lambda x: jnp.dot(xi_j, pf._hvp(U0, x, xi_j, r, k3))
    g = np.asarray(jax.grad(q)(x0))
    types = np.tile(np.arange(NT), G.nn)[pf.free_flat]
    ib = np.nonzero(types < 2)[0]
    nb = np.nonzero(types >= 2)[0]
    g_normal = float(np.abs(g[nb]).max())
    H = pf.hessian()
    Hb = H[ib][:, ib].tocsr()
    gb = g[ib]
    # exact zero mode of the in-block Hessian: the isorotation diag(1, e^{iα}, 1)
    Uj = np.asarray(U0)
    D = np.diag([0.0, 1.0, 0.0]) * 1j
    X = np.conj(np.swapaxes(Uj, 1, 2)) @ (D @ Uj)          # U†(iD)U, anti-Hermitian
    z6 = np.zeros((G.nn, NT))
    z6[:, 0], z6[:, 1] = X[:, 0, 1].real, X[:, 0, 1].imag
    z = np.zeros(pf.nfree)
    for t in range(NT):
        sel = pf.dof[:, t] >= 0
        z[pf.dof[sel, t]] = z6[sel, t]
    zb = z[ib]
    zres = float(np.abs(Hb @ zb).max() / (abs(Hb).max() * np.abs(zb).max()))   # relative
    Mf = pf.mass()
    Mb = Mf[ib][:, ib].tocsr()
    g_dot_z = float(gb @ zb / np.sqrt((zb @ (Mb @ zb)) * (gb @ gb)))
    # first version (kept for comparison): zero mode projected out, shifted solve.
    # The shift eps = 1e-9·max|H_b| is not small against the in-block spectrum, so
    # this underestimates gᵀH⁻¹g (and overestimates c₄, by 1–13 % on the grids used).
    gb_p = gb - (Mb @ zb) * ((zb @ gb) / (zb @ (Mb @ zb)))
    eps = 1e-9 * float(abs(Hb).max())
    eta_shift = sparse_solve((Hb + eps * Mb).tocsr(), gb_p)
    eta_shift = eta_shift - zb * ((zb @ (Mb @ eta_shift)) / (zb @ (Mb @ zb)))
    gHg_shift = float(gb_p @ eta_shift)
    # exact solve in the R-even in-block sector.  g is R-even (it does not depend on
    # which member of the critical pair ξ is), and on the grid the only negative
    # in-block modes (lattice-lifted translation and isorotation) are R-odd, so
    # H_b restricted to the even sector is positive definite: no shift, no projection.
    perm, sign = reflection(G, pf)
    pb, sb = restrict(perm, sign, ib)
    Bb = sector_basis(pb, sb, +1)
    g_even = Bb @ (Bb.T @ gb)
    g_odd_rel = float(np.linalg.norm(gb - g_even) / np.linalg.norm(gb))
    Hbe = (Bb.T @ Hb @ Bb).tocsr()
    inertia_even = inertia(Hbe)
    eta = Bb @ sparse_solve(Hbe, Bb.T @ gb)
    gHg = float(gb @ eta)
    c4 = E4 / 24.0 - gHg / 8.0
    c4_shift = E4 / 24.0 - gHg_shift / 8.0
    # direct check of the reduced quartic: E(tξ + η(t)) with η = −(t²/2)H⁻¹g
    etaf = np.zeros(pf.nfree)
    etaf[ib] = eta
    red = []
    for tt in (0.05, 0.1, 0.2):
        x = tt * xi - 0.5 * tt * tt * etaf
        red.append(dict(t=tt, dE=float(energy(U0, jnp.asarray(x), r, k3)) - E_emb,
                        quartic=c4 * tt ** 4))
    out = dict(ne=ne, box=box, l3=l3, disc=disc, E_embedded=E_emb, kappa3_star=k3s,
               lowest_normal_at_star=vals.tolist(), a1=a1, E2=E2, E3=E3, E4=E4, E4_fd=E4_fd,
               g_normal_max=g_normal, g_inblock_max=float(np.abs(gb).max()),
               zero_mode_residual=zres, g_dot_zero_mode=g_dot_z, g_odd_relative=g_odd_rel,
               inblock_even_inertia=list(inertia_even), gHg=gHg, c4=c4,
               gHg_shifted_solve=gHg_shift, c4_shifted_solve=c4_shift, shift_eps=eps,
               order="continuous (supercritical)" if c4 > 0 else "first order (subcritical)",
               reduced_energy_check=red, t_total=time.time() - t_start)
    print(json.dumps({k: out[k] for k in ("kappa3_star", "a1", "E2", "E3", "E4", "E4_fd",
                                          "g_normal_max", "g_dot_zero_mode", "zero_mode_residual",
                                          "g_odd_relative", "inblock_even_inertia", "gHg", "c4",
                                          "gHg_shifted_solve", "c4_shifted_solve", "shift_eps",
                                          "order")}, indent=1), flush=True)
    print("reduced energy check:", red, flush=True)
    tag = file_tag(box, disc)
    with open(os.path.join(DATA, f"flag_landau_ne{ne}{tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez_compressed(os.path.join(DATA, f"flag_landau_vec_ne{ne}{tag}.npz"), xi=xi, eta=etaf,
                        kappa3_star=k3s, a1=a1, c4=c4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ne", type=int)
    ap.add_argument("--box", type=float, default=3.0)
    ap.add_argument("--l3", type=int, default=0)
    ap.add_argument("--disc", default="u", choices=("u", "proj", "align"))
    a = ap.parse_args()
    main(a.ne, a.box, a.l3, disc=a.disc)
