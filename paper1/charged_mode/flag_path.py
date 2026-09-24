"""
Where does the embedded Hopfion go when it is pushed along its unstable normal
mode?  Descent path with a capped step, recording at every step

    E and its parts (sigma, Skyrme, potential),
    R_E  = energy-weighted rms radius,
    Q    = (1/24π²) ∫ Tr(ω̃ ∧ ω̃ ∧ ω̃)   (degree of the lift, ω̃ = ω − iL dφ),
    Q_CS = the part of Q carried by the diagonal (Kähler) connections,
    D    = ∫ (1 − |Z_33|²) d³x.

In the continuum Q cannot change along a continuous path.  On the lattice it
can only change once the texture reaches the grid scale.  If E falls while R_E
shrinks steadily and the Skyrme energy stays small, the texture is collapsing:
the fourth-order term Σ_a (F^{(a)})² does not control the part of Q carried by
the three-cycle term Tr(ω^{12} ω^{23} ω^{31}).

Usage: python flag_path.py ne l3 lam [--eps 0.3 --max_step 0.1 --n_steps 400 --k3 0 --mode file]
Writes data/flag_path_ne{ne}_l{l3}_lam{lam}[_k3X].json.
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
from flag_axisym import FlagModel, FlagProblem, NT

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def degree_parts(fm, U):
    """Q (degree of the lift) and its diagonal-connection part, at the quadrature points."""
    G = fm.G
    nn = U.shape[0]
    Uj = jnp.asarray(U)
    flat = jnp.concatenate([Uj.real.reshape(nn, 9), Uj.imag.reshape(nn, 9)], axis=1)

    def ip(P):
        o = P @ flat
        return (o[:, :9] + 1j * o[:, 9:]).reshape(-1, 3, 3)

    Z, Zr, Zz = ip(fm.Pv), ip(fm.Pr), ip(fm.Pz)
    Zh = jnp.conj(jnp.swapaxes(Z, 1, 2))
    wr, wz = Zh @ Zr, Zh @ Zz
    L = fm.Lvec
    wf = (Zh @ (1j * L[None, :, None] * Z)) / fm.rho[:, None, None]
    wf = wf - 1j * jnp.diag(L)[None] / fm.rho[:, None, None]          # ω̃_φ = ω_φ − iL/ρ
    # (ρ̂, φ̂, ẑ) is right-handed:  Tr ω̃³ = 3 Tr(ω_ρ [ω_φ, ω_z]) d³x
    tr = jnp.trace(wr @ (wf @ wz - wz @ wf), axis1=1, axis2=2)
    Q = float(jnp.sum(fm.W * jnp.real(3.0 * tr)) / (24.0 * np.pi ** 2))
    # Tr(ω_ρ[ω_φ,ω_z]) = Σ_{abc} ω_ρ^{ab}(ω_φ^{bc}ω_z^{ca} − ω_z^{bc}ω_φ^{ca}); the
    # three-cycle part is the sum over a, b, c all distinct.
    cyc = 0.0
    for (a, b, c) in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        cyc = cyc + wr[:, a, b] * (wf[:, b, c] * wz[:, c, a] - wz[:, b, c] * wf[:, c, a])
    Q3 = float(jnp.sum(fm.W * jnp.real(3.0 * cyc)) / (24.0 * np.pi ** 2))
    return Q, Q - Q3, Q3


def rms_radius(fm, U, r):
    G = fm.G
    sig, sk, pot = fm.density_terms(jnp.asarray(U), r)
    dens = np.asarray(sig + sk + pot)
    r2 = G.rho ** 2 + G.zq ** 2
    return float(np.sqrt((G.W @ (r2 * dens)) / (G.W @ dens)))


def main(ne, l3, lam, eps=0.3, max_step=0.1, n_steps=400, k3=0.0, mode=None):
    G = Grid(ne, ne, p=2, a=3.0, half=False)
    Uemb = np.load(os.path.join(DATA, f"flag_embedded_ne{ne}.npy"))
    x6 = np.load(mode or os.path.join(DATA, f"flag_mode_ne{ne}_l{l3}.npy"))
    r = (2.0, 2.0 * lam, 2.0 * lam)
    fm = FlagModel(G, L=(0, 1, l3), r=r, kappa3=k3)
    prob = FlagProblem(fm, Uemb, types=range(NT))
    x = np.zeros(prob.nfree)
    for t in range(NT):
        sel = prob.dof[:, t] >= 0
        x[prob.dof[sel, t]] = x6[sel, t]
    U = np.asarray(prob.U_of(jnp.asarray(eps * x)))
    rows = []
    fname = os.path.join(DATA, f"flag_path_ne{ne}_l{l3}_lam{lam:g}"
                         + ("" if k3 == 0.0 else f"_k3{k3:g}") + ".json")
    E0 = float(fm.energy_U(jnp.asarray(Uemb)))
    Q0 = degree_parts(fm, Uemb)
    print(f"ne={ne} l3={l3} λ={lam} κ3={k3}: embedded E={E0:.6f}  Q={Q0}", flush=True)
    lam_lm = 1e-2
    x0 = jnp.zeros(prob.nfree)
    t0 = time.time()
    for it in range(n_steps):
        prob.set_base(U)
        E = float(prob.energy(x0))
        g = np.asarray(prob.grad(x0))
        if it % 5 == 0 or it < 5:
            parts = fm.parts(U, prob.r)
            Q, Qcs, Q3 = degree_parts(fm, U)
            Uq = G.Pv @ U[:, 2, 2]
            row = dict(it=it, E=E, parts=parts, R_E=rms_radius(fm, U, prob.r), Q=Q, Q_diag=Qcs,
                       Q_3cycle=Q3, D=float(G.W @ (1.0 - np.abs(Uq) ** 2)),
                       gmax=float(np.abs(g).max()), t=time.time() - t0)
            rows.append(row)
            print(f"  it {it:3d} E={E:.6f} (σ {parts['sigma']:.3f}, sk {parts['skyrme']:.3f}, "
                  f"pot {parts['pot']:.3f}) R_E={row['R_E']:.4f} Q={Q:+.4f} "
                  f"[diag {Qcs:+.4f}, 3-cycle {Q3:+.4f}] D={row['D']:.4f} |g|={row['gmax']:.2e}",
                  flush=True)
            with open(fname, "w") as fh:
                json.dump(dict(ne=ne, l3=l3, lam=lam, kappa3=k3, eps=eps, max_step=max_step,
                               E_embedded=E0, Q_embedded=Q0, path=rows), fh, indent=1)
        if E < 1e-6 or np.abs(g).max() < 1e-7:
            break
        H = prob.hessian()
        dH = np.abs(H.diagonal()) + 1e-300
        for _ in range(30):
            dx = sparse_solve(H + lam_lm * sp.diags(dH), -g)
            if not np.all(np.isfinite(dx)) or np.abs(dx).max() > max_step:
                lam_lm *= 4.0
                continue
            En = float(prob.energy(jnp.asarray(dx)))
            if En < E:
                U = np.asarray(prob.U_of(jnp.asarray(dx)))
                lam_lm = max(lam_lm / 2.0, 1e-8)
                break
            lam_lm *= 4.0
        else:
            print("  no descent step found", flush=True)
            break


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ne", type=int)
    ap.add_argument("l3", type=int)
    ap.add_argument("lam", type=float)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--max_step", type=float, default=0.1)
    ap.add_argument("--n_steps", type=int, default=400)
    ap.add_argument("--k3", type=float, default=0.0)
    ap.add_argument("--mode", default=None)
    a = ap.parse_args()
    main(a.ne, a.l3, a.lam, a.eps, a.max_step, a.n_steps, a.k3, a.mode)
