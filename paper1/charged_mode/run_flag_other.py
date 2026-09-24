"""
Stability of the *other* embedded Hopfions (blocks 13 and 23) along the same
coupling line r = (r12, r13, r23) = (2, 2λ, 2λ), in model M2' (κ₃ = κ = 2).

Relabelling 2 ↔ 3 maps the block-13 embedding to the block-12 embedding of the
model (r12, r13, r23) = (2λ, 2, 2λ): its own stiffness is 2λ, and it is rotated
out of its plane against the stiffnesses 2 and 2λ.  For each λ the embedded
state is relaxed with its own stiffness, then the normal Hessian is checked.

Usage: python run_flag_other.py ne [kappa3]
Writes data/flag_other[_k3X]_ne{ne}.json.
"""

import json
import os
import sys
import time

import numpy as np

from hopfion_axisym import Grid
from hansatz import HProfile
from flag_axisym import FlagModel, FlagProblem, embedded_hopfion, newton_relax, inertia

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LAMBDAS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def main(ne, k3=2.0, l3_list=(0, -1)):
    G = Grid(ne, ne, p=2, a=3.0, half=False)
    cf = np.load(os.path.join(DATA, "state_ne32.npz"))["cf"]
    U = embedded_hopfion(G, HProfile(G).h_nodes(cf))
    out = dict(ne=ne, kappa3=k3, note="block-13 embedding, relabelled to block 12 with "
               "r = (2λ, 2, 2λ)", rows=[])
    tag = "" if k3 == 2.0 else f"_k3{k3:g}"
    for lam in LAMBDAS:
        t0 = time.time()
        r = (2.0 * lam, 2.0, 2.0 * lam)
        fm = FlagModel(G, L=(0, 1, 2), r=r, kappa3=k3)
        U, E, hist, conv = newton_relax(fm, U, types=(0, 1), max_iter=60, verbose=False)
        row = dict(lam=lam, E_embedded=E, converged=conv, sectors=[])
        for l3 in l3_list:
            prob = FlagProblem(FlagModel(G, L=(0, 1, l3), r=r, kappa3=k3), U, types=(2, 3, 4, 5))
            H = prob.hessian()
            nneg = inertia(H)[1]
            sec = dict(l3=l3, n_negative=nneg)
            if nneg == 0:
                vals, _ = prob.lowest(H, nev=2, sigma=-5e-2)
                sec["lowest"] = vals.tolist()
            row["sectors"].append(sec)
        out["rows"].append(row)
        print(f"λ={lam}: E_13 = {E:.6f} conv={conv}  "
              + "  ".join(f"l3={s['l3']}: neg={s['n_negative']} low={s.get('lowest')}"
                          for s in row["sectors"]) + f"  ({time.time() - t0:.0f}s)", flush=True)
        with open(os.path.join(DATA, f"flag_other{tag}_ne{ne}.json"), "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), float(a[1]) if len(a) > 1 else 2.0)
