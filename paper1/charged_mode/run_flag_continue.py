"""
Follow the F₂ branch below κ₃* in the R-even slice, with the Hopf charge tracked.

Starts from the δ = 0.04 end state of run_flag_twosided.py and lowers κ₃ in
steps, re-converging with Newton in the R-even coordinates at every step.
Answers whether the κ₃ = 0.6 charge loss of the first round (ne = 16, no
symmetry restriction) happens on the branch itself or only off the slice.

Recorded per step: ΔE = E − E_emb (E_emb does not depend on κ₃), D, Q and its
three-cycle part, the energy-weighted rms radius R_E, the LDLᵀ inertia of the
R-even and R-odd Hessian blocks.

Usage: python run_flag_continue.py ne [--box 3.0] [--stop 0.3] [--step 0.04]
Writes data/flag_continue_ne{ne}[_a{box}].json.
"""

import argparse
import json
import os
import time

import numpy as np
import jax.numpy as jnp

from flag_axisym import FlagModel, FlagProblem, inertia, NT
from flag_path import degree_parts
from flag_symmetry import symmetry_defect, reflection, sector_basis
from run_flag_landau import embedded
from run_flag_twosided import newton_sector, normal_amplitude

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def rms_radius(fm, U, r, k3):
    G = fm.G
    sig, sk, pot = fm.density_terms(jnp.asarray(U), r, k3)
    dens = np.asarray(sig + sk + pot)
    return float(np.sqrt((G.W @ ((G.rho ** 2 + G.zq ** 2) * dens)) / (G.W @ dens)))


def main(ne, box=3.0, stop=0.3, step=0.04):
    tag = "" if box == 3.0 else f"_a{box:g}"
    G, Uemb = embedded(ne, box)
    ts = json.load(open(os.path.join(DATA, f"flag_twosided_ne{ne}{tag}.json")))
    U = np.load(os.path.join(DATA, f"flag_twosided_states_ne{ne}{tag}.npz"))["below_0.04"]
    k3s = ts["kappa3_star"]
    fm = FlagModel(G, L=(0, 1, 0), kappa3=1.0)
    pf = FlagProblem(fm, U, types=range(NT))
    perm, sign = reflection(G, pf)
    Be, Bo = sector_basis(perm, sign, +1), sector_basis(perm, sign, -1)
    E_emb = float(fm.energy_U(jnp.asarray(Uemb), None, 0.0))
    out = dict(ne=ne, box=box, kappa3_star=k3s, E_embedded=E_emb, Q_embedded=ts["Q_embedded"],
               description=__doc__, rows=[])
    fname = os.path.join(DATA, f"flag_continue_ne{ne}{tag}.json")
    k3 = k3s - 0.04
    while k3 >= stop - 1e-12:
        t0 = time.time()
        pf.set_couplings(r=(2.0, 2.0, 2.0), kappa3=k3)
        U, E, hist, conv = newton_sector(pf, U, Be, max_iter=80)
        pf.set_base(U)
        H = pf.hessian()
        Q, Qd, Q3 = degree_parts(fm, U)
        row = dict(kappa3=k3, E=E, dE=E - E_emb, D=normal_amplitude(G, U), Q=Q, Q_3cycle=Q3,
                   R_E=rms_radius(fm, U, pf.r, k3), converged=conv, newton_iterations=len(hist),
                   symmetry_defect=symmetry_defect(G, U),
                   inertia_even=list(inertia((Be.T @ H @ Be).tocsr())),
                   inertia_odd=list(inertia((Bo.T @ H @ Bo).tocsr())), t=time.time() - t0)
        out["rows"].append(row)
        print(f"κ3={k3:.4f}: ΔE={row['dE']:+.5e} D={row['D']:.4f} Q={Q:+.5f} (3-cycle {Q3:+.4f}) "
              f"R_E={row['R_E']:.4f} conv={conv} it={len(hist)} even neg={row['inertia_even'][1]} "
              f"odd neg={row['inertia_odd'][1]} ({row['t']:.0f}s)", flush=True)
        with open(fname, "w") as fh:
            json.dump(out, fh, indent=1)
        if not conv:
            break
        k3 = round(k3 - step, 10)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ne", type=int)
    ap.add_argument("--box", type=float, default=3.0)
    ap.add_argument("--stop", type=float, default=0.3)
    ap.add_argument("--step", type=float, default=0.04)
    a = ap.parse_args()
    main(a.ne, a.box, a.stop, a.step)
