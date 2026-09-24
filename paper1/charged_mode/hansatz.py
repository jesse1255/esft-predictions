"""
The restricted h(r) family of the earlier rounds, with complete EM relaxation.

u(ρ,z) is the Hopf projection of the hedgehog with profile h(r) (h(0) = π,
h(∞) = 0), see hopfion_axisym.hopf_ansatz.  h is a clamped cubic B-spline in
s = r/L_h on [0, 1] with h(L_h) = h'(L_h) = 0 and knots clustered near the
core.  For every h the three electromagnetic sub-problems are solved exactly
on the grid (A_φ, meridional A with only a 1e-8 exterior regulariser so that
its gradient part acts as a phase relaxation of u, and β), so this is the
complete-basis limit of the note's E_N[h].
"""

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

from hopfion_axisym import Model, hopf_ansatz


class HProfile:
    def __init__(self, grid, ncoef=60, L_h=12.0, k=3, cluster=1.2):
        self.G, self.L_h, self.k = grid, L_h, k
        x = np.linspace(0.0, 1.0, ncoef - k + 1)
        inner = x ** cluster
        self.knots = np.concatenate([[0.0] * k, inner, [1.0] * k])
        self.nb = self.knots.size - k - 1
        r, _, _ = grid.node_r_theta()
        s = np.where(np.isfinite(r), np.minimum(r / L_h, 1.0), 1.0)
        self.B = BSpline.design_matrix(s, self.knots, k).tocsr()
        self.free = np.arange(1, self.nb - 2)   # c_0 = π; last two = 0

    def coefs(self, cf):
        c = np.zeros(self.nb)
        c[0] = np.pi
        c[self.free] = cf
        return c

    def h_nodes(self, cf):
        return self.B @ self.coefs(cf)

    def h_of_r(self, cf, r):
        s = np.clip(np.asarray(r) / self.L_h, 0.0, 1.0)
        return BSpline(self.knots, self.coefs(cf), self.k)(s)

    def fit(self, h_func, npts=2000):
        """Least-squares coefficients reproducing a profile h(r) (callable).

        Sampled densely on [0, L_h] so that every coefficient is determined.
        """
        r = np.linspace(0.0, self.L_h, npts)
        s = r / self.L_h
        Bg = BSpline.design_matrix(s, self.knots, self.k).toarray()
        rhs = h_func(r) - np.pi * Bg[:, 0]
        cf, *_ = np.linalg.lstsq(Bg[:, self.free], rhs, rcond=None)
        return cf


def optimize_h(grid, prof, cf0, e=0.3, N=1.0, kappa_fix=1e-8, maxiter=3000,
               verbose=False):
    """Minimise E_N[h] with all EM fields re-solved for every trial profile."""
    M = Model(grid, e=e, N=N, kappa=kappa_fix)
    state = {"n": 0}

    def fun(cf):
        U, dU = hopf_ansatz(grid, prof.h_nodes(cf))
        V = M.solve_A_fixed_u(U)
        E, g = M.energy(V)
        gh = np.sum(g[:, :3] * dU, axis=1)
        gh[~grid.free[:, 2]] = 0.0
        gc = prof.B.T @ gh
        state.update(V=V, E=E)
        state["n"] += 1
        if verbose and state["n"] % 25 == 0:
            print(f"    h-opt eval {state['n']}: E = {E:.10f}", flush=True)
        return E, gc[prof.free]

    res = minimize(fun, cf0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxfun=2 * maxiter, gtol=1e-10,
                                ftol=1e-16, maxcor=40))
    # final consistent evaluation at the returned coefficients
    E, gc = fun(res.x)
    info = dict(message=str(res.message), nit=int(res.nit), nfev=state["n"],
                E=float(E), grad_max=float(np.abs(gc).max()))
    return res.x, state["V"], M, info
