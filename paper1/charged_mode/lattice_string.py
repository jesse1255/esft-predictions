"""
Minimum-energy path of 01 + 12 → 02 on the 3D lattice by the (simplified, climbing)
string method, E & Vanden-Eijnden, J. Chem. Phys. 126, 164103 (2007) (依記憶，未核對頁碼).

A chain of images U⁽⁰⁾ … U⁽ᴹ⁻¹⁾ between two fixed end states is evolved by
preconditioned steepest descent (lattice3d.Precond: the vacuum Hessian is 1 in the
variables y, x = S y, U ← U Cay(X(x))) and redistributed after every step to equal
arc length in the gauge-invariant metric
    d(U, V)² = h³ Σ_x Σ_a ‖P_a(U) − P_a(V)‖²,   P_a = u_a u_a†.
Interpolation between neighbouring images: align the column phases of V to U site by
site, Y = off-diagonal part of Cay⁻¹(U†V), U Cay(tY).  After `climb_after` steps the
highest interior image climbs (its gradient component along the tangent is reversed),
so it converges to the saddle point (the transition state) of the path.

The path is not assumed to be axisymmetric: the images come from lattice_path.py
(coaxial and zipped states), so the string can bend.

Usage:
    python lattice_string.py N h M steps tau out_tag anchor1.npz anchor2.npz ... anchorK.npz
      (anchors in path order; the first and last are the fixed ends)
Rows: data/flagpair_lattice_string_<out_tag>.json; images: $FLAGPAIR_SCRATCH/string_<out_tag>.npz
"""

import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from lattice3d import Lattice, Precond, cayley, SCRATCH, DATA, PAIRS


def projectors(U):
    return np.einsum("...ra,...sa->...ars", U, np.conj(U))


def dist2(lat, U, V):
    return lat.h ** 3 * float(np.sum(np.abs(projectors(U) - projectors(V)) ** 2))


def align(U, V):
    """V with column phases rotated so that diag(U†V) is real positive."""
    w = np.einsum("...ra,...ra->...a", np.conj(U), V)
    ph = np.where(np.abs(w) > 1e-14, np.conj(w) / np.maximum(np.abs(w), 1e-300), 1.0)
    return V * ph[..., None, :]


def cay_inv(W):
    I = np.eye(3)
    return 2.0 * np.linalg.solve((W + I).swapaxes(-1, -2), (W - I).swapaxes(-1, -2)).swapaxes(-1, -2)


def offdiag(Y):
    return Y - np.einsum("...aa->...a", Y)[..., None] * np.eye(3)


def log_map(U, V):
    """Off-diagonal Y with U Cay(Y) ≈ V (after aligning V's column phases to U)."""
    Va = align(U, V)
    W = np.einsum("...ba,...bc->...ac", np.conj(U), Va)
    return offdiag(cay_inv(W))


def exp_map(U, Y):
    I = np.eye(3)
    return U @ np.linalg.solve(I - 0.5 * Y, I + 0.5 * Y)


def interp(U, V, t):
    return exp_map(U, t * log_map(U, V))


def X_to_x(lat, Y):
    """(N,N,N,3,3) off-diagonal anti-Hermitian → interior 6-vector (the parametrisation of U_of)."""
    Yi = Y[1:-1, 1:-1, 1:-1]
    out = np.zeros(Yi.shape[:-2] + (6,))
    for k, (a, b) in enumerate(PAIRS):
        out[..., 2 * k] = Yi[..., a, b].real
        out[..., 2 * k + 1] = Yi[..., a, b].imag
    return out.ravel()


def build_initial(lat, anchors, M):
    """Piecewise interpolation between the anchors, then equal arc length."""
    d = [np.sqrt(dist2(lat, anchors[i], anchors[i + 1])) for i in range(len(anchors) - 1)]
    L = sum(d)
    imgs = []
    for i in range(len(anchors) - 1):
        n = max(1, int(round((M - 1) * d[i] / L)))
        for j in range(n):
            imgs.append(interp(anchors[i], anchors[i + 1], j / n))
    imgs.append(anchors[-1])
    for _ in range(3):
        imgs, ell = redistribute_to(lat, imgs, M)
    return imgs


def redistribute_to(lat, imgs, M):
    d = np.array([np.sqrt(dist2(lat, imgs[i], imgs[i + 1])) for i in range(len(imgs) - 1)])
    ell = np.concatenate([[0.0], np.cumsum(d)])
    target = np.linspace(0.0, ell[-1], M)
    new = [imgs[0]]
    for j in range(1, M - 1):
        i = int(np.clip(np.searchsorted(ell, target[j]) - 1, 0, len(imgs) - 2))
        t = (target[j] - ell[i]) / max(d[i], 1e-300)
        new.append(interp(imgs[i], imgs[i + 1], float(np.clip(t, 0.0, 1.0))))
    new.append(imgs[-1])
    return new, ell


def main():
    N, h, M, steps, tau, tag = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), \
        float(sys.argv[5]), sys.argv[6]
    anchors_f = sys.argv[7:]
    lat = Lattice(N, h)
    S = Precond(lat)
    n = N - 2
    k = np.arange(1, n + 1)
    lam1 = (2.0 - 2.0 * np.cos(np.pi * k / (n + 1))) / h ** 2
    Lk = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]
    finv = (4.0 * h ** 3 * (Lk + 1.0)) ** 0.5            # S⁻¹ in the sine basis
    from scipy.fft import dstn

    def S_inv(v):
        a = dstn(np.asarray(v).reshape(n, n, n, 6), type=1, axes=(0, 1, 2), norm="ortho")
        a *= finv[..., None]
        return dstn(a, type=1, axes=(0, 1, 2), norm="ortho").ravel()

    Ej = jax.jit(lat.energy_U)
    gj = jax.jit(jax.grad(lambda Ub, x: lat.energy_U(lat.U_of(Ub, x)), argnums=1))

    anchors = [np.load(f)["U"] for f in anchors_f]
    imgs = build_initial(lat, anchors, M)
    rows_fn = os.path.join(DATA, f"flagpair_lattice_string_{tag}.json")
    hist = []
    climb_after = int(0.4 * steps)
    t0 = time.time()
    for it in range(steps + 1):
        E = np.array([float(Ej(jnp.asarray(U))) for U in imgs])
        if it % 10 == 0 or it == steps:
            _, ell = redistribute_to(lat, imgs, M)
            imax = int(np.argmax(E[1:-1])) + 1
            row = dict(it=it, E=E.tolist(), ell=ell.tolist(), imax=imax, Emax=float(E[imax]), t=time.time() - t0)
            hist.append(row)
            print(json.dumps(dict(it=it, imax=imax, Emax=round(float(E[imax]), 4),
                                  E=[round(float(e), 2) for e in E], t=round(time.time() - t0))), flush=True)
            with open(rows_fn, "w") as fh:
                json.dump(dict(N=N, h=h, M=M, tau=tau, anchors=[os.path.basename(f) for f in anchors_f], hist=hist), fh)
            np.savez_compressed(os.path.join(SCRATCH, f"string_{tag}.npz"), imgs=np.stack(imgs), E=E)
        if it == steps:
            break
        imax = int(np.argmax(E[1:-1])) + 1
        new = [imgs[0]]
        for i in range(1, M - 1):
            U = imgs[i]
            g = np.asarray(gj(jnp.asarray(U), jnp.zeros(lat.nfree)))
            gy = S(g)
            if it >= climb_after and i == imax:
                ty = S_inv(X_to_x(lat, log_map(U, imgs[i + 1]) - log_map(U, imgs[i - 1])))
                ty /= np.linalg.norm(ty) + 1e-300
                gy = gy - 2.0 * (ty @ gy) * ty
            x = S(-tau * gy)
            new.append(np.asarray(lat.U_of(jnp.asarray(U), jnp.asarray(x))))
        new.append(imgs[-1])
        imgs, _ = redistribute_to(lat, new, M)


if __name__ == "__main__":
    main()
