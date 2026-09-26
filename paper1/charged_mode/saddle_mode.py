# What is the unstable mode of the d = 1 fixed-separation state?  Projection on the centroid
# constraints (separation, sum) and its effect on the ring radii of lines 0 and 2.
import os, json, numpy as np, jax.numpy as jnp
from run_flag_pair import (setup, fmodel, FlagProblem, lowest_eig, LineCentroids, line_defect, DATA, SCRATCH)
Gf, _ = setup(24, 32, 4)
fm = fmodel(Gf, 2.0)
U0 = np.load(os.path.join(DATA, "flagpair_state_cons_ne24x32_a4_k32_d1.npy"))
prob = FlagProblem(fm, U0)
H = prob.hessian(); M = prob.mass()
lam, v, _ = lowest_eig(H, M)
v = v / np.sqrt(v @ (M @ v))
cons = LineCentroids(prob)
C = np.asarray(cons.J(prob.U0, jnp.zeros(prob.nfree))).T      # (nfree, 2): d Z_A, d Z_B
dZ = C.T @ v
def radii(U):
    out = []
    for c in (0, 2):
        q = Gf.Pv @ line_defect(U, c) ** 2
        out.append(float(Gf.W @ (Gf.rho * q) / (Gf.W @ q)))
    return out
eps = 1e-3
Up = np.asarray(prob.U_of(jnp.asarray(eps * v))); Um = np.asarray(prob.U_of(jnp.asarray(-eps * v)))
r0 = radii(U0); rp = radii(Up); rm = radii(Um)
dr = [(a - b) / (2 * eps) for a, b in zip(rp, rm)]
# unit-norm separation direction for comparison: the constrained-Newton 'force' direction
res = dict(lambda_=lam, dZA_dv=float(dZ[0]), dZB_dv=float(dZ[1]), d_sep=float(dZ[0] - dZ[1]), d_sum=float(dZ[0] + dZ[1]),
           rho_centroid_lines02=r0, d_rho_dv=dr)
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(DATA, "flagpair_saddle_mode_ne24x32_a4_k32_d1.json"), "w"), indent=1)
