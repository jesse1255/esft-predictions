"""
Relax the charged Hopfion at other couplings (e, N, μ), starting from the
e = 0.3, N = 1, μ = 1 state, and store states + diagnostics.

Usage: python run_params.py 48 64
Writes data/params_ne{ne}.json and (not committed) data/state_{tag}_ne{ne}.npz.
"""

import json
import os
import sys

import numpy as np

from hopfion_axisym import Grid, Model
from newton import NewtonRelaxer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# (e, N, mu).  N = 1/2 is the charge of the fermionic quantisation (see
# quantize.py); e = 0.6 then gives Q = eN = 0.3 ≈ √(4πα).
POINTS = [(0.3, 1.0, 1.0), (0.3, 0.5, 1.0), (0.6, 0.5, 1.0), (0.6, 1.0, 1.0),
          (0.6, 0.5, 0.5), (0.6, 0.5, 2.0)]


def tag(e, N, mu):
    return f"e{e:g}_N{N:g}_mu{mu:g}"


def main(ne_list):
    for ne in ne_list:
        base = np.load(os.path.join(DATA, f"state_ne{ne}.npz"))
        G = Grid(ne, ne, p=2, a=float(base["a"]), half=True)
        rows = []
        prev = base["V_full"]
        for (e, N, mu) in POINTS:
            M = Model(G, e=e, N=N, mu=mu, kappa=1.0)
            R = NewtonRelaxer(M, prev, verbose=False)
            V, E = R.run(max_iter=80, tol=1e-9)
            d = M.diagnostics(V)
            b = M.breakdown(V)
            rows.append(dict(e=e, N=N, mu=mu, E=E, converged=R.converged,
                             newton_iterations=len(R.history), breakdown=b,
                             rms_charge=d["rms_charge"], Dzz=d["Dzz_charge"],
                             mu_volume=d["mu_volume"], mu_far=d["mu_far"],
                             I_eff=d["I_eff"], omega=d["omega"], Q_H=d["Q_H"],
                             virial_relative=d["virial_relative"],
                             M_times_rms=E * d["rms_charge"]))
            np.savez_compressed(os.path.join(DATA, f"state_{tag(e, N, mu)}_ne{ne}.npz"),
                                V=V, e=e, N=N, mu=mu, ne=ne, a=float(base["a"]), p=2)
            print(f"ne={ne} {tag(e, N, mu)}: E={E:.8f} rms={d['rms_charge']:.6f} "
                  f"mu={d['mu_volume']:.5f} I={d['I_eff']:.4f} it={len(R.history)} "
                  f"conv={R.converged}", flush=True)
            prev = V
        with open(os.path.join(DATA, f"params_ne{ne}.json"), "w") as fh:
            json.dump(dict(ne=ne, points=rows), fh, indent=1)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [48])
