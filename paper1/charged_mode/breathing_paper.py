"""
Paper calculation for the breathing of the Q = 2 ring (the user's objection, 2026-10-03: nothing is
static, the vibration keeps flowing out, the frequency need not be one number).  No time evolution:
one static energy evaluation and one kinetic-energy evaluation of the dilation velocity.

Collective coordinate: the size λ of the ring, U_λ(x) = U(x/λ).  Derrick scaling in 3D:
    V(λ) = E₂ λ + E₄ / λ + E₀ λ³               (σ, quartic + three-cycle, potential)
    T    = ½ I(λ) λ̇²,  I(λ) = I_σ λ + I₄ / λ    (σ and quartic time components)
Small oscillations: ω₀² = V''(1) / I(1).  Anharmonic frequency shift (Lindstedt–Poincaré, after
s = ∫ √(I/I₀) dλ makes the mass constant): s̈ + ω₀² s = −α s² − β s³ gives
    ω = ω₀ + κ A²,   κ = 3β/(8ω₀) − 5α²/(12ω₀³)        (Landau–Lifshitz, Mechanics §28).
Radiation: the continuum starts at the mass gap ω = 1.  A bound vibration with ω_b < 1 can only
radiate through a harmonic n ω_b > 1; the n-th harmonic is generated with amplitude ∝ Aⁿ, so the
radiated power ∝ A²ⁿ and dE/dt ∝ −Eⁿ: E(t) ∝ (1 + t/τ)^(−1/(n−1)), a power law.

Usage: python breathing_paper.py
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SCRATCH = os.environ.get("FLAGPAIR_SCRATCH", "/tmp/flagpair")


def inputs():
    """E₂, E₄, E₀ of the relaxed Q = 2 ring and I_σ, I = I_σ + I₄ of its dilation (h = 0.3)."""
    import jax.numpy as jnp
    from lattice3d import Lattice
    from lattice_dyn2 import SpaceTime
    from lattice_dyn import log_map
    from lattice_string import X_to_x
    lat = Lattice(41, 0.3)
    U = np.load(os.path.join(SCRATCH, "lattice_ref_fused02_N41_h0.3.npz"))["U"]
    p = lat.parts(U)
    P = [lat.X, lat.Y, lat.Z]
    Wt = 0.0
    for i in range(3):
        Wt = Wt - P[i][..., None, None] * (log_map(U, np.roll(U, -1, axis=i)) - log_map(U, np.roll(U, 1, axis=i))) / (2.0 * lat.h)
    v = X_to_x(lat, Wt)
    dt, eps = 1e-3, 1e-3
    I = {}
    for full in (False, True):
        st = SpaceTime(lat, dt, full=full)
        K = float(st.kin(jnp.asarray(U), st.U_of(jnp.asarray(U), jnp.asarray(dt * eps * v))))
        I["full" if full else "sigma"] = 2.0 * K / eps ** 2
    return dict(E2=p["sigma"], E4=p["faddeev"] + p["three_cycle"], E0=p["potential"], I_sigma=I["sigma"], I_full=I["full"])


def breathing(E2, E4, E0, I_sigma, I_full):
    I4 = I_full - I_sigma
    k = 2 * E4 + 6 * E0                      # V''(1)
    v3 = (-6 * E4 + 6 * E0) / 6.0            # V'''(1)/6
    v4 = 24 * E4 / 24.0                      # V''''(1)/24
    I0 = I_sigma + I4
    a1 = (I_sigma - I4) / I0                 # I(1+x) = I0 (1 + a1 x + a2 x² + …)
    a2 = (2 * I4) / (2 * I0)
    c2, c3 = a1 / 4, a2 / 6 - a1 ** 2 / 24   # s = x + c2 x² + c3 x³
    b2, b3 = -c2, 2 * c2 ** 2 - c3           # x = s + b2 s² + b3 s³
    w3 = k * b2 + v3
    w4 = 0.5 * k * (b2 ** 2 + 2 * b3) + 3 * v3 * b2 + v4
    om0 = np.sqrt(k / I0)
    alpha, beta = 3 * w3 / I0, 4 * w4 / I0
    kappa = 3 * beta / (8 * om0) - 5 * alpha ** 2 / (12 * om0 ** 3)
    return dict(virial=E2 - E4 + 3 * E0, k=k, I0=I0, I4=I4, omega0=om0, alpha=alpha, beta=beta, kappa=kappa)


def main():
    inp = inputs()
    b = breathing(**inp)
    print("inputs:", {k: round(v, 3) for k, v in inp.items()})
    print("Derrick breathing: omega0 = %.4f (the bound mode of §4.4 is 0.682; a pure dilation is a trial shape, so"
          " this is an upper bound)" % b["omega0"])
    print("anharmonic shift kappa = %+.4f  (alpha = %.4f, beta = %.4f): omega(A) = omega0 + kappa A^2" % (b["kappa"], b["alpha"], b["beta"]))
    for A in (0.07, 0.15, 0.3):
        print("   A = %.2f: shift %+.5f (%.2f %%)" % (A, b["kappa"] * A ** 2, 100 * b["kappa"] * A ** 2 / b["omega0"]))
    modes = {"single vortex (in block)": 0.781, "Q = 2 ring (in block)": 0.465, "Q = 2 ring (toward line 1)": 0.893,
             "A12 (in block, softest)": 0.0285}
    rad = {}
    for name, w2 in modes.items():
        w = np.sqrt(w2)
        n = int(np.ceil(1.0 / w - 1e-12))
        rad[name] = dict(omega=w, n=n, power_law=f"P ∝ A^{2 * n}", energy_decay=f"E ∝ (1 + t/τ)^(-1/{n - 1})")
        print("   %-28s omega_b = %.3f: first harmonic above the gap n = %d -> P ∝ A^%d, E(t) ∝ (1+t/τ)^(-1/%d)" % (name, w, n, 2 * n, n - 1))
    # leak rate from the small-kick run (upper limit: the early transient of the kick is included)
    d = json.load(open(os.path.join(DATA, "flagpair_dyn2_ring_fused02_eps0.05_N41.json")))
    rows = d["rows"]
    t = np.array([r["t"] for r in rows]); E = np.array([r["E"] for r in rows])
    sel = t >= 8
    P = -np.polyfit(t[sel], E[sel], 1)[0]
    Em = rows[0]["T"]
    Gam = P / Em ** 2
    print("small kick (eps = 0.05): mode energy %.3f, radiated power %.4f per unit time -> Gamma = P/E^2 = %.2e,"
          " tau = 1/(Gamma E) = %.0f (lower limit)" % (Em, P, Gam, 1 / (Gam * Em)))
    for eps in (0.15, 0.3):
        Ek = Em * (eps / 0.05) ** 2
        print("   kick eps = %.2f: mode energy %.1f -> tau about %.0f" % (eps, Ek, 1 / (Gam * Ek)))
    json.dump(dict(inputs=inp, breathing=b, radiation=rad, leak=dict(P=P, E_mode=Em, Gamma=Gam, tau=1 / (Gam * Em))),
              open(os.path.join(DATA, "breathing_paper.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
