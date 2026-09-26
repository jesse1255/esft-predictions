"""
Shape of the lowest modes of one azimuthal sector k (flag_sector.FlagSector) at an
axisymmetric state: which "curved" motion lowers the energy?

Perturbation (body frame): Ŵ = Û Cay(X), X = X_c cos kφ + X_s sin kφ, lab field
U = R_L(φ) Ŵ.  Every eigenvalue of a sector k ≥ 1 is doubly degenerate (rotating
the pattern about the axis mixes cos and sin), so the modes come in (x, y) pairs.

Observables (first order in the mode, per line c = 0, 1, 2), from the defect density
q_c = 1 − |U_cc|² (gauge invariant; = (1 − n₃)/2 on the lines of an embedded soliton):
    δq_c = −2 Re( conj(Û_cc) (Û X)_cc ),  split into cos kφ and sin kφ parts.
For k = 1, with x = ρ cos φ (dV = 2πρ dρ dz = G.W at the quadrature points):
    lateral shift   s_c = ∫ δq_c x dV / ∫ q_c dV           = ½ ΣW ρ δq_c^cos / ΣW q_c
    tilt angle      θ_c = −∫ δq_c x (z − z̄_c) dV / ∫ q_c x² dV
                        = −ΣW ρ (z − z̄_c) δq_c^cos / ΣW ρ² q_c           (about the y axis)
(a ring of radius R tilted by θ about the y axis through its centre has z − z̄ = −θx).
For a thin ring the only k = 1 motions are these two (in-plane shift, tilt), so a mode
of two rings is a combination of: common shift (translation, zero mode), common tilt +
opposite shift (rigid rotation, zero mode), relative shift ("side slip") and relative
tilt.  The degenerate pair is rotated so that the line-0 response points along x.

Also printed: the share of the mode in each link (01, 02, 12) in the L² metric, where
along z the mode lives, and the term-by-term split of its Rayleigh quotient
(σ, Faddeev-type quartic, three-cycle κ₃, potential).

Usage: python sector_mode.py <state.npy> k [nneg]   → data/flagpair_sector_mode_<label>_k<k>.json
H and M are cached in $FLAGPAIR_SCRATCH (default /tmp/flagpair).
"""

import json
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import jax
import jax.numpy as jnp

from flag_axisym import PAIRS
from flag_sector import FlagSector, count_below, NS, _x6_to_X
from run_flag_pair import setup, fmodel, _pardiso, DATA

SCRATCH = os.environ.get("FLAGPAIR_SCRATCH", "/tmp/flagpair")


def cached(S, fn):
    if os.path.exists(fn):
        z = np.load(fn)
        mk = lambda p: sp.csr_matrix((z[p + "d"], z[p + "i"], z[p + "p"]), shape=tuple(z["shape"]))
        return mk("H"), mk("M")
    H, M = S.hessian(), S.mass()
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez(fn, Hd=H.data, Hi=H.indices, Hp=H.indptr, Md=M.data, Mi=M.indices, Mp=M.indptr,
             shape=np.array(H.shape))
    return H, M


def block_inverse_iteration(H, M, shift, nvec, iters=40, seed=0):
    """Eigenpairs of H v = λ M v nearest to `shift` (Rayleigh–Ritz on a block)."""
    solve = _pardiso(H - shift * M)
    V = np.random.default_rng(seed).standard_normal((H.shape[0], nvec))
    lam_old = None
    for it in range(iters):
        W = np.asarray(solve(M @ V))
        if W.ndim == 1:
            W = W[:, None]
        Hs, Ms = W.T @ (H @ W), W.T @ (M @ W)
        Hs, Ms = 0.5 * (Hs + Hs.T), 0.5 * (Ms + Ms.T)
        lam, C = sla.eigh(Hs, Ms)
        V = W @ C
        V /= np.sqrt(np.einsum("ij,ij->j", V, M @ V))[None, :]
        if lam_old is not None and np.max(np.abs(lam - lam_old)) < 1e-10 * max(1.0, np.abs(lam).max()):
            break
        lam_old = lam
    res = np.linalg.norm(H @ V - (M @ V) * lam[None, :], axis=0)
    return lam, V, res, it + 1


def split_parts(S, v):
    """Second derivative of each energy term along x = t v at t = 0."""
    fm = S.fm
    out = {}
    base = dict(r=fm.r, kappa=fm.kappa, kappa3=fm.kappa3, m2=fm.m2)
    z3, r0 = (0.0, 0.0, 0.0), jnp.zeros(3)
    variants = {
        "sigma": dict(kappa=z3, kappa3=0.0, m2=z3),
        "faddeev_quartic": dict(r=r0, kappa3=0.0, m2=z3),
        "three_cycle": dict(r=r0, kappa=z3, m2=z3),
        "potential": dict(r=r0, kappa=z3, kappa3=0.0),
    }
    vj = jnp.asarray(v)
    for name, change in variants.items():
        for key, val in change.items():
            setattr(fm, key, val)
        S_ = FlagSector(fm, S.U0, S.k, S.M)
        f = lambda t: S_._energy(t * vj)
        out[name] = float(jax.jacfwd(jax.jacfwd(f))(0.0))
        for key, val in base.items():
            setattr(fm, key, val)
    return out


def observables(S, v):
    """Line shifts, tilts, link shares and z-profile of a sector-1 mode x."""
    G, fm = S.G, S.fm
    U0 = np.asarray(S.U0)
    xc, xs = (np.asarray(a) for a in S._fields(jnp.asarray(v)))
    Xc, Xs = np.asarray(_x6_to_X(jnp.asarray(xc))), np.asarray(_x6_to_X(jnp.asarray(xs)))
    Pv = G.Pv
    W, rho, zq = G.W, G.rho, G.zq
    out = {"lines": {}}
    for c in range(3):
        q = Pv @ (1.0 - np.abs(U0[:, c, c]) ** 2)
        dq = {}
        for nm, X in (("cos", Xc), ("sin", Xs)):
            UX = np.einsum("nij,nj->ni", U0, X[:, :, c])     # (Û X)_{·c}
            dq[nm] = Pv @ (-2.0 * np.real(np.conj(U0[:, c, c]) * UX[:, c]))
        N = W @ q
        zbar = (W @ (zq * q)) / N
        R2 = (W @ (rho ** 2 * q)) / N
        s_x = 0.5 * (W @ (rho * dq["cos"])) / N
        s_y = 0.5 * (W @ (rho * dq["sin"])) / N
        th_y = -(W @ (rho * (zq - zbar) * dq["cos"])) / (W @ (rho ** 2 * q))
        th_x = (W @ (rho * (zq - zbar) * dq["sin"])) / (W @ (rho ** 2 * q))
        out["lines"][c] = dict(weight=float(N), z_bar=float(zbar), rho_rms=float(np.sqrt(R2)),
                               shift_x=float(s_x), shift_y=float(s_y),
                               tilt_about_y=float(th_y), tilt_about_x=float(th_x))
    # link shares (L² metric) and where the mode lives along z
    dens_p = []
    for p in range(3):
        d = 0.0
        for x6 in (xc, xs):
            d = d + (Pv @ x6[:, 2 * p]) ** 2 + (Pv @ x6[:, 2 * p + 1]) ** 2
        dens_p.append(0.5 * d)                 # ⟨cos²⟩ = ⟨sin²⟩ = ½
    tot = sum(W @ d for d in dens_p)
    out["link_share"] = {f"{a}{b}": float((W @ dens_p[p]) / tot) for p, (a, b) in enumerate(PAIRS)}
    dens = sum(dens_p)
    out["z_profile"] = dict(mean_z=float((W @ (zq * dens)) / tot),
                            share_z_pos=float((W @ (dens * (zq > 0))) / tot),
                            share_between=float((W @ (dens * (np.abs(zq) < 0.5))) / tot),
                            rho_mean=float((W @ (rho * dens)) / tot))
    return out


def rotate_pair(S, V):
    """Rotate a degenerate (x, y) pair so that line 0 moves along x in the first vector."""
    o = [observables(S, V[:, j])["lines"][0] for j in range(2)]
    a = np.array([o[0]["shift_x"], o[1]["shift_x"]])
    b = np.array([o[0]["tilt_about_y"], o[1]["tilt_about_y"]])
    w = a if np.linalg.norm(a) > 1e-3 * np.linalg.norm(b) else b
    w = w / np.linalg.norm(w)
    return np.column_stack([V @ w, V @ np.array([-w[1], w[0]])])


def main():
    path = sys.argv[1]
    k = int(sys.argv[2])
    nneg = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    label = os.path.basename(path)[:-4]
    t0 = time.time()
    Gf, _ = setup(24, 32, 4)
    fm = fmodel(Gf, 2.0)
    U = np.load(path)
    S = FlagSector(fm, U, k)
    S._fields = jax.jit(lambda x: _fields_of(S, x))
    H, M = cached(S, os.path.join(SCRATCH, f"sector_HM_{label}_k{k}.npz"))
    print(f"H, M ready ({time.time() - t0:.0f} s), nfree {S.nfree}", flush=True)
    res = dict(label=label, k=k, nfree=S.nfree, modes=[])
    groups = [("negative", None, nneg), ("near_zero", -0.02, 6)]
    for name, shift, nv in groups:
        if shift is None:
            lo = -10.0
            while count_below(H, M, lo) > 0:
                lo *= 2
            hi = 0.0
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if count_below(H, M, mid) == 0 else (lo, mid)
            shift = lo - 0.02 * abs(lo)
        lam, V, r, its = block_inverse_iteration(H, M, shift, nv + 2)
        print(f"{name}: shift {shift:+.4f}, {its} iterations, λ = {np.round(lam, 5).tolist()}, "
              f"residual {np.round(r, 7).tolist()}", flush=True)
        take = [j for j in range(len(lam)) if (name == "negative" and lam[j] < -0.05)
                or (name == "near_zero" and abs(lam[j]) < 0.05)]
        Vt = V[:, take]
        lt = lam[take]
        j = 0
        while j < len(take):
            if j + 1 < len(take) and abs(lt[j + 1] - lt[j]) < 1e-4 * max(1.0, abs(lt[j])) and k > 0:
                pair = rotate_pair(S, Vt[:, j:j + 2])
                vecs, lams = [pair[:, 0], pair[:, 1]], [lt[j], lt[j + 1]]
                j += 2
            else:
                vecs, lams = [Vt[:, j]], [lt[j]]
                j += 1
            for v, l in zip(vecs, lams):
                row = dict(group=name, lam=float(l), rayleigh=float(v @ (H @ v) / (v @ (M @ v))))
                row.update(observables(S, v))
                if name == "negative":
                    row["parts"] = split_parts(S, v / np.sqrt(v @ (M @ v)))
                res["modes"].append(row)
                print(json.dumps(row), flush=True)
                np.save(os.path.join(SCRATCH, f"sector_mode_{label}_k{k}_{name}_{len(res['modes'])}.npy"), v)
    res["t"] = time.time() - t0
    fn = os.path.join(DATA, f"flagpair_sector_mode_{label}_k{k}.json")
    json.dump(res, open(fn, "w"), indent=1)
    print("wrote", fn, flush=True)


def _fields_of(S, x):
    """Same split as FlagSector.fields (kept outside the class so the cached jit is reusable)."""
    G = S.G
    nn = G.nn
    axis = G.axis_mask & ~G.inf_mask
    x12 = jnp.zeros(nn * NS).at[jnp.asarray(S.free_flat)].set(x).reshape(nn, NS)
    xc, xs = x12[:, :6], x12[:, 6:]
    sgn = jnp.asarray(S.axis_sign)
    is_axis = jnp.asarray(axis.astype(float))
    der = jnp.stack([-xc[:, 1::2], xc[:, 0::2]], axis=2).reshape(nn, 6) * jnp.repeat(sgn, 2, axis=1)
    xs = xs * (1.0 - is_axis[:, None]) + der * is_axis[:, None]
    return xc, xs


if __name__ == "__main__":
    main()
