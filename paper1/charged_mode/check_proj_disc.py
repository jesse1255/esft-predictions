"""
Validation of the gauge-invariant discretisations of the flag energy
(FlagModel(disc="proj") and FlagModel(disc="align")).

1. Gauge invariance: random nodal column phases U → U·diag(e^{iθ_a(n)}) change
   the projector and the element-aligned ("align") energies only at round-off;
   the U discretisation changes.
2. Consistency: for the embedded Hopfion, and for the same Hopfion tilted along
   the critical normal mode (so that the three-cycle term is non-zero), both
   discretisations approach the same energies as ne grows.
3. In-block spectra of the embedded Hopfion in both discretisations (needs the
   relaxed states from run_flag_landau.py [--disc proj]).

Usage: python check_proj_disc.py   → data/check_proj_disc.json
"""

import json
import os

import numpy as np
import jax.numpy as jnp

from flag_axisym import FlagModel, FlagProblem, NT
from run_flag_landau import embedded

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def main():
    out = dict(description=__doc__, rows=[])
    rng = np.random.default_rng(3)
    for ne in (16, 24, 32):
        G, U = embedded(ne, 3.0)
        fu = FlagModel(G, L=(0, 1, 0), kappa3=1.0, disc="u")
        fp = FlagModel(G, L=(0, 1, 0), kappa3=1.0, disc="proj")
        fa = FlagModel(G, L=(0, 1, 0), kappa3=1.0, disc="align")
        th = rng.uniform(-np.pi, np.pi, size=(G.nn, 3))
        Ug = U * np.exp(1j * th)[:, None, :]
        row = dict(ne=ne)
        row["gauge_change_u"] = float(fu.energy_U(jnp.asarray(Ug), None, 1.0) - fu.energy_U(jnp.asarray(U), None, 1.0))
        row["gauge_change_proj"] = float(fp.energy_U(jnp.asarray(Ug), None, 1.0) - fp.energy_U(jnp.asarray(U), None, 1.0))
        row["gauge_change_align"] = float(fa.energy_U(jnp.asarray(Ug), None, 1.0) - fa.energy_U(jnp.asarray(U), None, 1.0))
        row["embedded_align"] = fa.parts(U, None, 1.0)
        row["embedded_u"] = fu.parts(U, None, 1.0)
        row["embedded_proj"] = fp.parts(U, None, 1.0)
        vfile = os.path.join(DATA, f"flag_landau_vec_ne{ne}.npz")
        if os.path.exists(vfile):
            vec = np.load(vfile)
            pf = FlagProblem(fu, U, types=range(NT))
            Ut = np.asarray(pf.U_of(jnp.asarray(0.5 * vec["xi"])))
            row["tilted_u"] = fu.parts(Ut, None, 1.0)
            row["tilted_proj"] = fp.parts(Ut, None, 1.0)
            row["tilted_align"] = fa.parts(Ut, None, 1.0)
        out["rows"].append(row)
        print(f"ne={ne}: gauge change  U-disc {row['gauge_change_u']:+.3e}   proj {row['gauge_change_proj']:+.3e}"
              f"   align {row['gauge_change_align']:+.3e}")
        for key in ("embedded", "tilted"):
            if f"{key}_u" in row:
                for d in ("u", "align", "proj"):
                    v = row[f"{key}_{d}"]
                    print(f"   {key:8s} {d:5s} " + "  ".join(f"{k}={x:.5f}" for k, x in v.items())
                          + f"  total={sum(v.values()):.5f}")
    out["inblock"] = inblock_spectra()
    with open(os.path.join(DATA, "check_proj_disc.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def inblock_spectra(nes=(16, 24, 32)):
    """In-block (CP¹-direction) Hessian of the embedded Hopfion, both discretisations.

    The Rayleigh quotients of the isorotation generator U†(i diag(0,1,0))U and of
    the z-translation generator U†∂_zU are listed with the two lowest eigenvalues.
    With the projector energy the isorotation is an exact symmetry of the chart
    (the nodal phases no longer matter), so its Rayleigh quotient is zero up to
    round-off; the translation is still broken by the non-uniform grid.
    """
    rows = []
    for ne in nes:
        for disc in ("u", "proj", "align"):
            fn = os.path.join(DATA, f"flag_embedded_ne{ne}" + ("" if disc == "u" else f"_{disc}") + ".npy")
            if not os.path.exists(fn):
                continue
            G, U = embedded(ne, 3.0, disc)
            fm = FlagModel(G, L=(0, 1, 0), kappa3=0.0, disc=disc)
            p = FlagProblem(fm, U, types=(0, 1))
            H, M = p.hessian(), p.mass()
            vals, vecs = p.lowest(H, nev=3, sigma=-0.3)

            def to_x(X):
                x6 = np.zeros((G.nn, NT))
                x6[:, 0], x6[:, 1] = X[:, 0, 1].real, X[:, 0, 1].imag
                x = np.zeros(p.nfree)
                for t in range(NT):
                    sel = p.dof[:, t] >= 0
                    x[p.dof[sel, t]] = x6[sel, t]
                return x
            Uh = np.conj(np.swapaxes(U, 1, 2))
            z_iso = to_x(Uh @ (np.diag([0.0, 1.0, 0.0]) * 1j @ U))
            zc = G.z_n.reshape(G.nr, G.nz)
            Ug = U.reshape(G.nr, G.nz, 3, 3)
            Uz = np.zeros_like(U)
            for i in range(G.nr):
                idx = np.nonzero(np.isfinite(zc[i]))[0]
                Uz.reshape(G.nr, G.nz, 3, 3)[i, idx] = np.gradient(Ug[i, idx], zc[i][idx], axis=0)
            Uz[~np.isfinite(G.z_n) | ~np.isfinite(G.rho_n)] = 0
            t_z = to_x(Uh @ Uz)
            rq = lambda x: float(x @ (H @ x) / (x @ (M @ x)))
            row = dict(ne=ne, disc=disc, lowest=vals.tolist(), rayleigh_isorotation=rq(z_iso),
                       rayleigh_translation=rq(t_z))
            rows.append(row)
            print(f"in-block ne={ne} {disc:4s}: lowest {np.round(vals, 6).tolist()}  "
                  f"RQ(isorotation) {row['rayleigh_isorotation']:+.2e}  RQ(z-translation) "
                  f"{row['rayleigh_translation']:+.2e}", flush=True)
    return rows


if __name__ == "__main__":
    main()
