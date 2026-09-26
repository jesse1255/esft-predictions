"""
Small-step descent from the d = 1 fusion saddle (round 6).

Damped Newton (newton_relax) is not a gradient flow: from both sides of the saddle it
ended in the (0,2) ring.  Here the path is a preconditioned gradient flow
    x ← x − η (H₀ + cM)⁻¹ g(x),   H₀ = Hessian at the saddle,  c > |λ_min|,
re-anchored every step, with η small enough that the energy decreases monotonically.
A fixed positive-definite metric cannot jump between basins, so the side the flow ends
on is the side of the separatrix the initial displacement ±ε v lies on (in that metric).
Records E, the separation Z_A − Z_B of the line-0/line-2 defect centroids and the line
weights along the path.

Usage: python saddle_flow.py [eps] [eta] [steps]  → data/flagpair_saddle_flow_*.json
"""
import json, os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax.numpy as jnp
from run_flag_pair import (setup, fmodel, FlagProblem, lowest_eig, LineCentroids, consistent_line_weight,
                           energy, DATA)

eps = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
eta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 300
Gf, _ = setup(24, 32, 4)
fm = fmodel(Gf, 2.0)
U0 = np.load(os.path.join(DATA, "flagpair_state_cons_ne24x32_a4_k32_d1.npy"))
E1 = energy(fm, np.load(os.path.join(DATA, "flagpair_state_single01_ne24x32_a4_k32.npy")))
prob = FlagProblem(fm, U0)
H0 = prob.hessian(); M = prob.mass()
lam, v, _ = lowest_eig(H0, M)
v = v / np.abs(v).max()
c = 2.0 * abs(lam) + 1.0
P = spla.splu(sp.csc_matrix(H0 + c * M))
cons = LineCentroids(prob)
out = dict(eps=eps, eta=eta, c=c, lambda_min=lam, E1=E1, paths={})
for sgn in (+1, -1):
    prob.set_base(U0)
    U = np.asarray(prob.U_of(jnp.asarray(sgn * eps * v)))
    path, t0 = [], time.time()
    x0 = jnp.zeros(prob.nfree)
    for it in range(steps):
        prob.set_base(U)
        E = float(prob.energy(x0))
        g = np.asarray(prob.grad(x0))
        Z = np.asarray(cons.c(prob.U0, x0))
        if it % 10 == 0 or it == steps - 1:
            lw = [float(consistent_line_weight(fm, jnp.asarray(U), k)) for k in range(3)]
            path.append(dict(it=it, E=E, E_minus_2E1=E - 2 * E1, sep=float(Z[0] - Z[1]), lines=lw))
            print(f"side {sgn:+d} it {it:3d}: E − 2E1 = {E - 2 * E1:+9.4f}  Z_A − Z_B = {Z[0] - Z[1]:+.4f}  "
                  f"lines {', '.join(f'{w:.2f}' for w in lw)}  ({time.time() - t0:.0f}s)", flush=True)
        dx = -eta * P.solve(g)
        h = eta
        while float(prob.energy(jnp.asarray(dx))) > E and h > 1e-4:     # keep the flow monotone
            h *= 0.5; dx *= 0.5
        U = np.asarray(prob.U_of(jnp.asarray(dx)))
    out["paths"][f"{sgn:+d}"] = path
    json.dump(out, open(os.path.join(DATA, f"flagpair_saddle_flow_ne24x32_a4_k32_d1.json"), "w"), indent=1)
