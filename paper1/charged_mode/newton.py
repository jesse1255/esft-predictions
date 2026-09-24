"""
Damped Newton relaxation for the axisymmetric charged Hopfion.

Unknowns per node: two tangent coordinates (a1, a2) of u ∈ S² and the three
components of A.  The Hessian is assembled by coloured forward differences of
the exact gradient (β frozen, so every column is local); the tiny non-local
part coming from the fixed-charge term (∝ ω² ≈ 2×10⁻⁴) is left out, which
only makes the otherwise quadratic convergence linear with a very small ratio.
Every trial step is accepted or rejected on the exact E_N (β re-solved).
"""

import time

import numpy as np
import scipy.sparse as sp

from hopfion_axisym import AR, AZ, AP, sparse_solve


def _frames(u, sym_mask):
    """Orthonormal tangent frame (e1, e2) at each node.

    On the symmetry plane u_2 = 0 is imposed, and e1 lies in the (1,3) plane.
    """
    c = np.zeros_like(u)
    use_e1 = np.abs(u[:, 1]) > 0.9
    c[~use_e1, 1] = 1.0
    c[use_e1, 0] = 1.0
    e1 = np.cross(c, u)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(u, e1)
    return e1, e2


def _share_1d(i, j, p):
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    if p == 1:
        return (hi - lo) <= 1
    return np.ceil((hi - p) / p) <= np.floor(lo / p)


class NewtonRelaxer:
    def __init__(self, model, U, fix_u=False, verbose=True):
        self.M, self.G = model, model.G
        self.U = U.copy()
        self.fix_u = fix_u
        self.verbose = verbose
        G = self.G
        self.p = G.p
        ufree = G.free[:, 0] & G.free[:, 2]          # u free at node
        self.fa1 = ufree & (not fix_u)
        self.fa2 = ufree & ~G.sym_mask & (not fix_u)
        self.fA = G.free[:, 3:].copy()
        # DOF numbering: [a1 | a2 | Ar | Az | Ap] over free nodes
        masks = [self.fa1, self.fa2, self.fA[:, 0], self.fA[:, 1], self.fA[:, 2]]
        self.masks = masks
        self.offsets = np.cumsum([0] + [m.sum() for m in masks])
        self.ndof = int(self.offsets[-1])
        self.dof_of = []
        for k, m in enumerate(masks):
            idx = -np.ones(G.nn, dtype=np.int64)
            idx[m] = self.offsets[k] + np.arange(m.sum())
            self.dof_of.append(idx)
        self.history = []
        self.converged = False

    # ------------------------------------------------------------------
    def _assemble_U(self, x, U0, e1, e2):
        G = self.G
        U = U0.copy()
        a1 = np.zeros(G.nn)
        a2 = np.zeros(G.nn)
        a1[self.masks[0]] = x[self.offsets[0]:self.offsets[1]]
        a2[self.masks[1]] = x[self.offsets[1]:self.offsets[2]]
        w = U0[:, :3] + a1[:, None] * e1 + a2[:, None] * e2
        nw = np.linalg.norm(w, axis=1)
        U[:, :3] = w / nw[:, None]
        for k, comp in enumerate((AR, AZ, AP)):
            m = self.masks[2 + k]
            U[m, comp] = U0[m, comp] + x[self.offsets[2 + k]:self.offsets[3 + k]]
        return U, nw

    def _grad_x(self, U, gU, nw, e1, e2):
        u = U[:, :3]
        gu = gU[:, :3]
        ug = np.sum(u * gu, axis=1)
        pg = (gu - ug[:, None] * u) / nw[:, None]
        g1 = np.sum(pg * e1, axis=1)
        g2 = np.sum(pg * e2, axis=1)
        parts = [g1[self.masks[0]], g2[self.masks[1]]]
        for k, comp in enumerate((AR, AZ, AP)):
            parts.append(gU[self.masks[2 + k], comp])
        return np.concatenate(parts)

    def hessian(self, U0, e1, e2, frozen, eps=None, central=False):
        """Coloured finite-difference Hessian in (a1, a2, A) coordinates.

        Forward differences (eps = 1e-6) are ample for Newton steps; central
        differences (eps = 1e-4, error O(eps²)) are used for eigenvalue checks.
        """
        if eps is None:
            eps = 1e-4 if central else 1e-6
        G = self.G
        p = self.p
        period = 2 * p + 1
        x0 = np.zeros(self.ndof)
        Ub, nwb = self._assemble_U(x0, U0, e1, e2)
        _, gUb = self.M.energy(Ub, frozen=frozen)
        gb = self._grad_x(Ub, gUb, nwb, e1, e2)
        rows, cols, vals = [], [], []
        # row DOFs: all free (node, var) pairs
        row_nodes, row_dofs = [], []
        for k in range(5):
            nodes = np.nonzero(self.masks[k])[0]
            row_nodes.append(nodes)
            row_dofs.append(self.dof_of[k][nodes])
        rn_all = np.concatenate(row_nodes)
        rd_all = np.concatenate(row_dofs)
        ri, rj = G.I[rn_all], G.J[rn_all]
        for ci in range(period):
            for cj in range(period):
                colour = ((G.I % period) == ci) & ((G.J % period) == cj)
                # column node coupled to each row node for this colour
                di = (ci - ri + p) % period - p
                dj = (cj - rj + p) % period - p
                ci_n, cj_n = ri + di, rj + dj
                ok = (ci_n >= 0) & (ci_n < G.nr) & (cj_n >= 0) & (cj_n < G.nz)
                ok &= _share_1d(ri, ci_n, p) & _share_1d(rj, cj_n, p)
                col_node = np.where(ok, ci_n * G.nz + np.where(ok, cj_n, 0), 0)
                for k in range(5):
                    sel = colour & self.masks[k]
                    if not sel.any():
                        continue
                    d = np.zeros(self.ndof)
                    d[self.dof_of[k][sel]] = 1.0
                    Up, nwp = self._assemble_U(eps * d, U0, e1, e2)
                    _, gUp = self.M.energy(Up, frozen=frozen)
                    gp = self._grad_x(Up, gUp, nwp, e1, e2)
                    if central:
                        Um, nwm = self._assemble_U(-eps * d, U0, e1, e2)
                        _, gUm = self.M.energy(Um, frozen=frozen)
                        gm = self._grad_x(Um, gUm, nwm, e1, e2)
                        col = (gp - gm) / (2.0 * eps)
                    else:
                        col = (gp - gb) / eps
                    cdof = self.dof_of[k][col_node]
                    good = ok & (cdof >= 0) & self.masks[k][col_node]
                    rows.append(rd_all[good])
                    cols.append(cdof[good])
                    vals.append(col[rd_all[good]])
        H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.ndof, self.ndof))
        H = 0.5 * (H + H.T)
        return H.tocsr(), gb

    # ------------------------------------------------------------------
    def run(self, max_iter=40, tol=1e-9, lam0=1e-3, max_time=None):
        """Damped Newton iteration.

        Convergence is declared when the Newton decrement (the predicted
        energy decrease of the undamped step) falls below the round-off level
        of E, or when the Hessian-scaled gradient max |g_i|/sqrt(H_ii) < tol.
        The raw max |g_i| is also recorded; it can stay at ~1e-7 on the
        outermost nodes of the compactified grid, whose huge volume weights
        make it dynamically irrelevant.
        """
        G, M = self.G, self.M
        U = self.U
        E, gU = M.energy(U)
        lam = lam0
        t_start = time.time()
        noise = 1e-12 * max(abs(E), 1.0)
        for it in range(max_iter):
            u = U[:, :3]
            e1, e2 = _frames(u, G.sym_mask)
            frozen = (M.last["beta"], M.last["I"]) if M.charge else None
            t0 = time.time()
            H, _ = self.hessian(U, e1, e2, frozen)
            g = self._grad_x(U, gU, np.ones(G.nn), e1, e2)
            dH = np.abs(H.diagonal()) + 1e-300
            gmax = float(np.abs(g).max())
            gscaled = float(np.abs(g / np.sqrt(dH)).max())
            dx0 = sparse_solve(H, -g)
            decrement = float(-g @ dx0)
            th = time.time() - t0
            self.history.append(dict(it=it, E=E, gmax=gmax, gscaled=gscaled,
                                     newton_decrement=decrement, lam=lam, t_hess=th))
            if self.verbose:
                print(f"  it {it:3d}  E = {E:.12f}  |g|max = {gmax:.2e}  |g/sqrtH|max = {gscaled:.2e}"
                      f"  dec = {decrement:.2e}  lam = {lam:.0e}  ({th:.1f}s, ndof {self.ndof})", flush=True)
            if gscaled < tol or 0 <= decrement < noise:
                self.converged = True
                break
            if max_time is not None and time.time() - t_start > max_time:
                break
            accepted = False
            for _ in range(12):
                dx = dx0 if lam == 0 else sparse_solve(H + lam * sp.diags(dH), -g)
                Un, _ = self._assemble_U(dx, U, e1, e2)
                En, gUn = M.energy(Un)
                pred = float(g @ dx + 0.5 * dx @ (H @ dx))
                if En < E + 1e-4 * min(pred, 0.0) or (abs(pred) < noise and En < E + noise):
                    accepted = True
                    ratio = (En - E) / pred if pred < -noise else 1.0
                    U, E, gU = Un, En, gUn
                    lam = lam * 0.2 if ratio > 0.75 else lam
                    if lam < 1e-9:
                        lam = 0.0
                    break
                lam = max(lam * 10.0, 1e-6)
            if not accepted:
                if self.verbose:
                    print("  step rejected repeatedly; stopping", flush=True)
                break
        else:
            self.converged = False
        self.U = U
        self.E = E
        self.hessian_last = H
        return U, E
