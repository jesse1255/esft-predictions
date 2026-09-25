"""
How much does the discrete flag energy depend on the nodal column phases?

In the continuum U ↦ U·diag(e^{iθ_a(x)}) is a gauge transformation and the
energy does not change.  On the grid U is interpolated node by node, so the
discrete energy does depend on the nodal phases.  Newton in the Cayley chart
(off-diagonal coordinates only) never optimises them; they change only as a
side effect of re-anchoring, i.e. a closed path in the flag manifold can come
back with different phases (a holonomy).

1. Scaling: apply a smooth phase field θ(ρ, z) = A·exp(−((ρ − 0.9)² + z²)/0.25)
   (columns 0 and 1, opposite signs) to the embedded Hopfion and record
   ΔE(ne) for ne = 16, 24, 32.
2. If the ne = 16 continuation states exist (run_flag_continue.py 16 and the
   _up_large scan), take the end state T at κ₃ = 0.9863, which as a flag lies
   within 1 % of g(U_emb) = P₀₂ S U_emb S P₀₂ (an exact symmetry image of the
   embedded Hopfion), and swap the nodal phases between T and g(U_emb).

Usage: python check_gauge_phase.py   → data/check_gauge_phase.json
"""

import json
import os

import numpy as np
import jax.numpy as jnp

from flag_axisym import FlagModel
from run_flag_landau import embedded

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    out = dict(description=__doc__, scaling={}, phase_swap=None)
    for ne in (16, 24, 32):
        G, U = embedded(ne, 3.0)
        fm = FlagModel(G, L=(0, 1, 0), kappa3=1.0)
        E0 = float(fm.energy_U(jnp.asarray(U), None, 0.0))
        rho = np.nan_to_num(G.rho_n, posinf=1e3)
        z = np.nan_to_num(G.z_n, posinf=1e3, neginf=-1e3)
        row = dict(E_embedded=E0)
        for A in (0.5, 1.5):
            th = A * np.exp(-((rho - 0.9) ** 2 + z ** 2) / 0.25)
            D = np.exp(1j * np.stack([th, -th, 0 * th], axis=1))
            row[f"dE_A{A:g}"] = float(fm.energy_U(jnp.asarray(U * D[:, None, :]), None, 0.0)) - E0
        out["scaling"][str(ne)] = row
        print(f"ne={ne}: ΔE(A=0.5) = {row['dE_A0.5']:+.4e}   ΔE(A=1.5) = {row['dE_A1.5']:+.4e}",
              flush=True)
    sfile = os.path.join(DATA, "flag_continue_states_ne16.npz")
    if os.path.exists(sfile) and "0.9863" in np.load(sfile):
        G, U = embedded(16, 3.0)
        T = np.load(sfile)["0.9863"]
        P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], float)
        S = np.diag([1.0, -1.0, 1.0])
        Ut = P @ S @ U @ S @ P
        fm = FlagModel(G, L=(0, 1, 0), kappa3=1.0)
        E = lambda V: float(fm.energy_U(jnp.asarray(V), None, 0.9863))
        proj = lambda V: np.einsum("nia,nja->naij", V, np.conj(V))
        ov = np.einsum("nia,nia->na", np.conj(T), Ut)
        ph = ov / np.maximum(np.abs(ov), 1e-300)
        dphi = np.angle(ph)
        out["phase_swap"] = dict(
            kappa3=0.9863, E_T=E(T), E_image=E(Ut),
            E_T_with_image_phases=E(T * ph[:, None, :]),
            E_image_with_T_phases=E(Ut * np.conj(ph)[:, None, :]),
            flag_difference_max=float(np.abs(proj(T) - proj(Ut)).max()),
            phase_difference_max=np.abs(dphi).max(axis=0).tolist(),
            phase_difference_rms=np.sqrt((dphi ** 2).mean(axis=0)).tolist())
        print(json.dumps(out["phase_swap"], indent=1), flush=True)
    with open(os.path.join(DATA, "check_gauge_phase.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
