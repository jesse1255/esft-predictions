"""
Time evolution of the lattice flag model (lattice3d), to see collisions, resonance and emitted waves.

Lagrangian L = T − V, V = the static lattice energy of lattice3d.Lattice, and the kinetic term of
the σ part only:  T = ∫ Σ_{a<b} r |ω_t^{ab}|²,  ω_t = U†∂_tU  (the quartic and three-cycle terms
also have time components; they are left out in this first version, which changes the inertia of
the cores but not the linear waves, whose speed is 1).

In the chart U = U_b Cay(X(x)) about the current field, ω_t = Ẋ at X = 0, so T = ½ M Σ ẋ² with
M = 2 r h³ per real component.  Integrator: kick–drift–kick leapfrog, the base is moved to the new
field after every drift and the velocity components are carried over to the new chart (an O(dt|v|)
error in the frame of the tangent space; monitored through the total energy).

Diagnostics per output step: T, V, total, the lattice charge, line weights, and the energy that has
left a ball around the origin (the radiation).

Usage: python lattice_dyn.py test N h            → a vortex at rest and a moving vortex (energy conservation)
       python lattice_dyn.py collide N h V cfg   → two rings (smith.configs cfg) sent at each other with speed V
"""

import json
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

from lattice3d import Lattice, charge, SCRATCH, DATA, PAIRS

jax.config.update("jax_enable_x64", True)


def log_map(U, V):
    """Off-diagonal Y with U Cay(Y) ≈ V: the column phases of V are first aligned to U site by site
    (diag(U†V) real positive), so the result lives in U's chart whatever the gauge phases."""
    w = np.einsum("...ra,...ra->...a", np.conj(U), V)
    ph = np.where(np.abs(w) > 1e-14, np.conj(w) / np.maximum(np.abs(w), 1e-300), 1.0)
    Va = V * ph[..., None, :]
    W = np.einsum("...ba,...bc->...ac", np.conj(U), Va)
    I = np.eye(3)
    Y = 2.0 * np.linalg.solve((W + I).swapaxes(-1, -2), (W - I).swapaxes(-1, -2)).swapaxes(-1, -2)
    return Y - np.einsum("...aa->...a", Y)[..., None] * I


def chart_velocity_of_translation(lat, U, axis, speed):
    """ẋ for U(x − speed·t·ê): ω_t = −speed·ω_e, with ω_e from the aligned Cayley logarithm of the
    links to both neighbours (central difference), off-diagonal part, interior sites."""
    from lattice_string import X_to_x
    Up = np.roll(U, -1, axis=axis)
    Um = np.roll(U, 1, axis=axis)
    W = (log_map(U, Up) - log_map(U, Um)) / (2.0 * lat.h)
    return X_to_x(lat, -speed * W)


class Dyn:
    def __init__(self, lat, dt):
        self.lat, self.dt = lat, float(dt)
        self.M = 2.0 * lat.r[0] * lat.h ** 3
        self.grad = jax.jit(jax.grad(lambda Ub, x: lat.energy_U(lat.U_of(Ub, x)), argnums=1))
        self.Uof = jax.jit(lat.U_of)
        self.E = jax.jit(lat.energy_U)
        self.x0 = jnp.zeros(lat.nfree)

    def kinetic(self, v):
        return 0.5 * self.M * float(jnp.sum(v ** 2))

    def step(self, U, v, g=None):
        if g is None:
            g = self.grad(U, self.x0)
        vh = v - 0.5 * self.dt * g / self.M
        U = self.Uof(U, self.dt * vh)
        g = self.grad(U, self.x0)
        v = vh - 0.5 * self.dt * g / self.M
        return U, v, g


def radiated_energy_outside(lat, U, radius):
    """Static energy density outside a ball (a crude radiation meter): σ + potential per site."""
    Un = np.asarray(U)
    q = sum(1.0 - np.abs(Un[..., c, c]) ** 2 for c in range(3))
    r = np.sqrt(lat.X ** 2 + lat.Y ** 2 + lat.Z ** 2)
    return float(lat.h ** 3 * np.sum(q * (r > radius)))


def run(lat, U, v, dt, steps, every, tag, extra=None):
    dyn = Dyn(lat, dt)
    U = jnp.asarray(U)
    v = jnp.asarray(v)
    rows = []
    g = None
    t0 = time.time()
    for n in range(steps + 1):
        if n % every == 0:
            T = dyn.kinetic(v)
            V = float(dyn.E(U))
            Q, Qa, _ = charge(lat, np.asarray(U), pad=1)
            gm = lat.geometry(np.asarray(U))
            row = dict(step=n, t=n * dt, T=T, V=V, E=T + V, Q=Q, lines=[gm[c]["weight"] for c in range(3)],
                       outside4=radiated_energy_outside(lat, U, 4.0), wall=round(time.time() - t0, 1))
            rows.append(row)
            print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}), flush=True)
            if n % (every * 5) == 0:
                np.savez_compressed(os.path.join(SCRATCH, f"dyn_{tag}_t{n * dt:.2f}.npz"), U=np.asarray(U), v=np.asarray(v))
        if n < steps:
            U, v, g = dyn.step(U, v, g)
    out = dict(tag=tag, N=lat.N, h=lat.h, dt=dt, rows=rows)
    if extra:
        out.update(extra)
    fn = os.path.join(DATA, f"flagpair_dyn_{tag}.json")
    json.dump(out, open(fn, "w"), indent=1)
    return U, v, rows


def cmd_test(N, h, dt=0.02):
    lat = Lattice(N, h)
    U = np.load(os.path.join(SCRATCH, f"lattice_ref_A01_N{N}_h{h:g}.npz"))["U"]
    v = np.zeros(lat.nfree)
    run(lat, U, v, dt, 100, 20, f"test_rest_N{N}")
    v = chart_velocity_of_translation(lat, U, 0, 0.3)
    run(lat, U, v, dt, 300, 30, f"test_move_N{N}")


def cmd_collide(N, h, V, cfg, dt=0.02, steps=1500, every=50, s=1.2):
    """Two rings of links 01 and 12 in the geometry smith.configs[cfg], first pulled apart by ±s along
    x, then sent at each other with speeds ±V.  The initial velocity is the time derivative of the
    combined field itself: both rings moved by ±V·δt, combined again, and the aligned Cayley
    logarithm taken (so it lives in the chart of the combined field, whatever its gauge phases)."""
    from run_flag_pair import setup
    from smith import ring, smooth_pair, configs
    from lattice_string import X_to_x
    Gf, Uf = setup(24, 32, 4)
    u = Uf[:, :3]
    lat = Lattice(N, h)
    (RA, aA), (RB, aB) = configs(0.93)[cfg]

    def field(shift):
        UA = ring(Gf, u, lat, RA, (aA[0] - s + shift, aA[1], aA[2]), (0, 1))
        UB = ring(Gf, u, lat, RB, (aB[0] + s - shift, aB[1], aB[2]), (1, 2))
        return smooth_pair(UA, UB)[0]

    U = field(0.0)
    d = 1e-3
    v = X_to_x(lat, log_map(U, field(V * d))) / d
    run(lat, U, v, dt, steps, every, f"collide_{cfg}_V{V:g}_N{N}", extra=dict(V=V, cfg=cfg, s=s))


if __name__ == "__main__":
    cmd = sys.argv[1]
    N, h = int(sys.argv[2]), float(sys.argv[3])
    if cmd == "test":
        cmd_test(N, h)
    elif cmd == "collide":
        cmd_collide(N, h, float(sys.argv[4]), sys.argv[5])
