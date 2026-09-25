"""
Both sides of the κ₃ threshold at the S₃ point, with the Hopf charge tracked.

Uses the normal form from run_flag_landau.py (κ₃*, a₁, c₄, critical mode ξ and
transverse response η).  For c₄ > 0 the normal form predicts, at κ₃ = κ₃* − δ,
a small-amplitude F₂ minimum at t* = √(a₁δ/(4c₄)) with

    ΔE = E_F − E_emb = −a₁² δ² / (16 c₄),

and at κ₃ = κ₃* + δ no such state (the embedded Hopfion is the local minimum).

Lattice zero modes.  The discrete embedded Hopfion is a saddle: its two
continuum zero modes in the axisymmetric sector (z-translation and the
isorotation diag(1, e^{iα}, 1)) acquire small negative eigenvalues on the grid
(−0.141, −0.006 at ne = 16; −0.027, −0.0012 at ne = 24; −0.0083, −0.0004 at
ne = 32, i.e. ∝ ne⁻⁴).  An unconstrained Newton run slides along them (at
ne = 16 the Hopfion drifts to the coarse outer grid and unwinds: E → 0, Q → 0).
Both modes are odd under the exact symmetry of the discrete problem

    R:  U(ρ, z) ↦ S · conj U(ρ, −z) · S,   S = diag(1, −1, 1),

which fixes the embedded state (checked to 1e-12).  On tangent coordinates R is
the signed mirror X_ab(ρ, z) ↦ s_a s_b · conj X_ab(ρ, −z).  The runs below are
therefore done in the R-even sector (the slice that fixes the centre at z = 0
and the isorotation phase).  The critical pair {ξ, Jξ} contains exactly one
R-even direction, ξ_e; a₁ and c₄ are the same for every member of the pair.
Every end state is reported with the inertia of the full Hessian split into its
R-even and R-odd blocks, so what the slice removes is visible.

Newton (all six tangent coordinates, R-even) is started from
U_emb·Cay(t*ξ_e − t*²η/2) below and from U_emb·Cay(t*ξ_e) above.  Reported for
each end state: ΔE, the normal amplitude D = ∫(1 − |Z₃₃|²), the Hopf degree Q
(and its three-cycle part), the symmetry defect and the two sector spectra.

Usage: python run_flag_twosided.py ne [--box 3.0] [--deltas 0.01,0.02,0.04]
Writes data/flag_twosided_ne{ne}[_a{box}].json.
"""

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax.numpy as jnp

from hopfion_axisym import sparse_solve
from flag_axisym import FlagModel, FlagProblem, inertia, NT
from flag_path import degree_parts
from run_flag_landau import embedded
from flag_symmetry import symmetry_defect, reflection, sector_basis

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

def normal_amplitude(G, U):
    Uq = G.Pv @ U[:, 2, 2]
    return float(G.W @ (1.0 - np.abs(Uq) ** 2))


def newton_sector(pf, U0, B, max_iter=60, tol=1e-9, lam0=1e-3, lam_floor=1e-8):
    """flag_axisym.newton_relax restricted to the column space of B (x = B y)."""
    U = np.asarray(U0)
    x0 = jnp.zeros(pf.nfree)
    lam, hist = lam0, []
    BT = B.T.tocsr()
    for it in range(max_iter):
        pf.set_base(U)
        E = float(pf.energy(x0))
        g = BT @ np.asarray(pf.grad(x0))
        H = (BT @ pf.hessian() @ B).tocsr()
        dH = np.abs(H.diagonal()) + 1e-300
        gs = float(np.abs(g / np.sqrt(dH)).max())
        hist.append(dict(it=it, E=E, gscaled=gs, lam=lam))
        if gs < tol:
            return U, E, hist, True
        accepted = False
        for _ in range(14):
            dy = sparse_solve(H + max(lam, lam_floor) * sp.diags(dH), -g)
            if not np.all(np.isfinite(dy)):
                lam = max(lam * 10.0, 1e-6)
                continue
            dx = B @ dy
            En = float(pf.energy(jnp.asarray(dx)))
            pred = float(g @ dy + 0.5 * dy @ (H @ dy))
            if -pred < 1e-12 * abs(E) and En <= E + 1e-12 * abs(E):
                return np.asarray(pf.U_of(jnp.asarray(dx))), En, hist, True
            if En < E + 1e-4 * min(pred, 0.0):
                ratio = (En - E) / pred if pred < 0 else 1.0
                U = np.asarray(pf.U_of(jnp.asarray(dx)))
                lam = lam * 0.2 if ratio > 0.75 else lam
                accepted = True
                break
            lam = max(lam * 10.0, 1e-6)
        if not accepted:
            return U, E, hist, False
    return U, E, hist, False


def sector_spectrum(H, M, B, nev=5, sigma=-1.0):
    BT = B.T.tocsr()
    Hs, Ms = (BT @ H @ B).tocsr(), (BT @ M @ B).tocsr()
    A = (Hs - sigma * Ms).tocsc()
    lu = spla.splu(A)
    op = spla.LinearOperator(Hs.shape, matvec=lu.solve, dtype=float)
    vals = spla.eigsh(Hs, k=nev, M=Ms, sigma=sigma, which="LM", OPinv=op, tol=1e-9,
                      return_eigenvectors=False)
    n_pos, n_neg, n_pert = inertia(Hs)
    return dict(inertia=[n_pos, n_neg, n_pert], lowest=np.sort(vals).tolist())


def main(ne, box=3.0, deltas=(0.01, 0.02, 0.04), l3=0):
    tag = "" if box == 3.0 else f"_a{box:g}"
    G, Uemb = embedded(ne, box)
    vec = np.load(os.path.join(DATA, f"flag_landau_vec_ne{ne}{tag}.npz"))
    xi, eta = vec["xi"], vec["eta"]
    k3s, a1, c4 = float(vec["kappa3_star"]), float(vec["a1"]), float(vec["c4"])
    fm = FlagModel(G, L=(0, 1, l3), kappa3=1.0)
    pf = FlagProblem(fm, Uemb, types=range(NT))
    perm, sign = reflection(G, pf)
    R = lambda x: sign * x[perm]
    Be, Bo = sector_basis(perm, sign, +1), sector_basis(perm, sign, -1)
    assert Be.shape[1] + Bo.shape[1] == pf.nfree
    Mf = pf.mass()
    mn = lambda x: float(np.sqrt(x @ (Mf @ x)))
    # the R-even member of the critical pair (same a₁, c₄ by the U(1) symmetry)
    xe = 0.5 * (xi + R(xi))
    even_fraction = mn(xe) / mn(xi)
    xe = xe * (mn(xi) / mn(xe))
    eta_parity = float(eta @ (Mf @ R(eta)) / (eta @ (Mf @ eta)))
    E_emb = float(fm.energy_U(jnp.asarray(Uemb), None, 0.0))
    Q_emb = degree_parts(fm, Uemb)
    # check that R is a symmetry of the discrete energy (random perturbation)
    pf.set_base(Uemb)
    pf.set_couplings(r=(2.0, 2.0, 2.0), kappa3=k3s)
    rng = np.random.default_rng(1)
    xr = 1e-2 * rng.standard_normal(pf.nfree)
    e1, e2 = float(pf.energy(jnp.asarray(xr))), float(pf.energy(jnp.asarray(R(xr))))
    out = dict(ne=ne, box=box, l3=l3, kappa3_star=k3s, a1=a1, c4=c4, E_embedded=E_emb,
               Q_embedded=Q_emb, symmetry_defect_embedded=symmetry_defect(G, Uemb),
               energy_R_check=abs(e1 - e2) / abs(e1), xi_even_fraction=even_fraction,
               eta_R_parity=eta_parity, n_even=int(Be.shape[1]), n_odd=int(Bo.shape[1]),
               description=__doc__, embedded=[], runs=[])
    print(f"ne={ne} a={box}: κ3*={k3s:.6f} a1={a1:.4f} c4={c4:.5f} E_emb={E_emb:.6f} "
          f"Q_emb={Q_emb[0]:+.5f} sym.defect={out['symmetry_defect_embedded']:.1e} "
          f"|E(Rx)−E(x)|/E={out['energy_R_check']:.1e} ξ even fraction={even_fraction:.3f} "
          f"η parity={eta_parity:+.4f}", flush=True)
    fname = os.path.join(DATA, f"flag_twosided_ne{ne}{tag}.json")
    states = {}
    for delta in deltas:
        for side, k3 in (("below", k3s - delta), ("above", k3s + delta)):
            # reference: the embedded state at this κ₃
            pf.set_base(Uemb)
            pf.set_couplings(r=(2.0, 2.0, 2.0), kappa3=k3)
            H = pf.hessian()
            out["embedded"].append(dict(kappa3=k3, even=sector_spectrum(H, Mf, Be),
                                        odd=sector_spectrum(H, Mf, Bo)))
            ref = out["embedded"][-1]
            print(f"  embedded at κ3={k3:.5f}: even inertia {ref['even']['inertia']} lowest "
                  f"{np.round(ref['even']['lowest'], 5).tolist()} | odd inertia "
                  f"{ref['odd']['inertia']} lowest {np.round(ref['odd']['lowest'], 5).tolist()}",
                  flush=True)
            t0 = time.time()
            tstar = np.sqrt(a1 * delta / (4.0 * c4)) if c4 > 0 else 0.5
            x = tstar * xe - (0.5 * tstar ** 2 * eta if side == "below" else 0.0)
            Ustart = np.asarray(pf.U_of(jnp.asarray(x)))
            U, E, hist, conv = newton_sector(pf, Ustart, Be)
            pf.set_base(U)
            H = pf.hessian()
            Q, Qd, Q3 = degree_parts(fm, U)
            row = dict(side=side, delta=delta, kappa3=k3, t_start=float(tstar), E=E,
                       dE=E - E_emb,
                       dE_predicted=(-(a1 * delta) ** 2 / (16 * c4) if side == "below" else 0.0),
                       D=normal_amplitude(G, U), Q=Q, Q_diag=Qd, Q_3cycle=Q3,
                       symmetry_defect=symmetry_defect(G, U), converged=conv,
                       newton_iterations=len(hist), gscaled=hist[-1]["gscaled"],
                       even=sector_spectrum(H, Mf, Be), odd=sector_spectrum(H, Mf, Bo),
                       t=time.time() - t0)
            out["runs"].append(row)
            # end state relative to the embedded one: U_emb† U is block-diagonal with
            # unit-modulus diagonal iff the flag is unchanged (only nodal phases moved)
            W = np.conj(np.swapaxes(Uemb, 1, 2)) @ U
            row["flag_change_offdiag_max"] = float(np.abs(W - W * np.eye(3)).max())
            states[f"{side}_{delta:g}"] = U
            np.savez_compressed(os.path.join(DATA, f"flag_twosided_states_ne{ne}{tag}.npz"), **states)
            print(f"  {side:5s} δ={delta}: κ3={k3:.5f} ΔE={row['dE']:+.5e} "
                  f"(pred {row['dE_predicted']:+.5e}) D={row['D']:.4e} Q={Q:+.5f} "
                  f"(3-cycle {Q3:+.2e}) conv={conv} it={row['newton_iterations']} "
                  f"defect={row['symmetry_defect']:.1e} flag change={row['flag_change_offdiag_max']:.1e}\n"
                  f"        even inertia {row['even']['inertia']} lowest "
                  f"{np.round(row['even']['lowest'], 5).tolist()}\n"
                  f"        odd  inertia {row['odd']['inertia']} lowest "
                  f"{np.round(row['odd']['lowest'], 5).tolist()}  ({row['t']:.0f}s)", flush=True)
            with open(fname, "w") as fh:
                json.dump(out, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ne", type=int)
    ap.add_argument("--box", type=float, default=3.0)
    ap.add_argument("--deltas", default="0.01,0.02,0.04")
    a = ap.parse_args()
    main(a.ne, a.box, tuple(float(v) for v in a.deltas.split(",")))
