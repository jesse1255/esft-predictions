"""
Magnetic moment of the quantised charged Hopfion: exact coupling scaling and
the size of the rotational contribution.

Scaling (exact in the model, f fixed).  With y = f g x and Â = A/f the energy is
E = (f/g) Ê(ẽ, μ̃), ẽ = e/g, μ̃ = μ/(fg), so
    M = (f/g) m̂,   r = r̂/(fg),   μ_s = μ̂/(f g²),   V₁₁ = v̂/(f g³),
and for the J = ½ state (Q = eN = ẽ g N)
    g_static ≡ 2 M μ_lab/(Q J) = κ_g(ẽ, μ̃) · (M r_rms)²,
    κ_g = 2 μ̂_lab / (ẽ N J m̂ r̂²)            (g-independent),
    ΔE_rot / M = [J(J+1) − K²] / (2 V₁₁ M) ∝ g⁴.
So the static-current g factor and (M r)² fall together as 1/g⁴.

Rotational part (estimate only).  Rotating the body about a perpendicular
axis with Ω = J_⊥/V₁₁ carries the charge density along; if that current were
purely convective, j = ρ_Q Ω × x, the body moment would gain
(Q/2)⟨y² + z²⟩_Q Ω_⊥ and
    g_rot ≈ (2/3) M ⟨y² + z²⟩_Q / V₁₁          (J = ½, K = −½),
which is also g-independent.  The divergence-free part of the O(ωΩ) current
(Skyrme term, induced A) is not computed, so g_rot is an estimate, not a result.

Usage: python run_gscaling.py 48
Writes data/gscaling_ne{ne}.json and data/state_e{e}_N0.5_mu1_ne{ne}.npz.
"""

import json
import os
import sys

import numpy as np

from hopfion_axisym import Grid, Model
from newton import NewtonRelaxer
from quantize import inertia_x, full_plane

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
E_PHYS = 2.0 * np.sqrt(4.0 * np.pi / 137.035999)     # Q = e/2 = √(4πα)
SMALL_E = (0.2, 0.15, 0.1, 0.05)


def charge_moments(G, M, U):
    Qv, Qr, Qz = M.interp(U)
    dens, parts, _ = M.local(Qv, Qr, Qz, want_grad=False)
    beta, _ = M.solve_beta(parts["K"])
    q = (G.Pv @ beta) * parts["K"]
    Qn = G.integrate(q)
    return dict(rho2=G.integrate(G.rho ** 2 * q) / Qn, z2=G.integrate(G.zq ** 2 * q) / Qn)


def point(ne, e, N, mu, V):
    Gh = Grid(ne, ne, p=2, a=3.0, half=True)
    Mh = Model(Gh, e=e, N=N, mu=mu, kappa=1.0)
    E, _ = Mh.energy(V, want_grad=False)
    d = Mh.diagnostics(V)
    mom = charge_moments(Gh, Mh, V)
    Gf, Uf = full_plane(ne, V)
    V11 = inertia_x(Model(Gf, e=e, N=N, mu=mu, kappa=1.0), Uf)["total"]
    J, K = abs(N), -N
    Mq = E + (J * (J + 1) - K * K) / (2.0 * V11)
    r = d["rms_charge"]
    mu_lab = d["mu_volume"] * (J * K) / (J * (J + 1))
    Q = e * N
    g_static = 2.0 * Mq * abs(mu_lab) / (abs(Q) * J)
    yz2 = 0.5 * mom["rho2"] + mom["z2"]
    g_rot = 2.0 * Mq * (Q / 2.0) * yz2 / V11 * (J * (1.0 - K * K / (J * (J + 1)))) / (Q * J)
    return dict(e_tilde=e, N=N, mu_tilde=mu, E=E, M=Mq, rms_charge=r, mu_static=d["mu_volume"],
                mu_far=d["mu_far"], mu_lab=mu_lab, V11=V11, rho2_Q=mom["rho2"], z2_Q=mom["z2"],
                g_static=g_static, kappa_g=g_static / (Mq * r) ** 2, g_rot_estimate=g_rot,
                rot_fraction=(Mq - E) / Mq, M_times_r=Mq * r,
                v11_times_M_over_r2=V11 / (Mq * r * r))


def main(ne):
    rows = []
    for (e, mu) in ((0.3, 1.0), (0.6, 1.0), (0.6, 0.5), (0.6, 2.0)):
        V = np.load(os.path.join(DATA, f"state_e{e:g}_N0.5_mu{mu:g}_ne{ne}.npz"))["V"]
        rows.append(point(ne, e, 0.5, mu, V))
        print({k: rows[-1][k] for k in ("e_tilde", "mu_tilde", "M", "kappa_g", "g_rot_estimate",
                                          "rot_fraction")}, flush=True)
    G = Grid(ne, ne, p=2, a=3.0, half=True)
    prev = np.load(os.path.join(DATA, f"state_e0.3_N0.5_mu1_ne{ne}.npz"))["V"]
    for e in SMALL_E:
        M = Model(G, e=e, N=0.5, mu=1.0, kappa=1.0)
        R = NewtonRelaxer(M, prev, verbose=False)
        V, E = R.run(max_iter=80, tol=1e-9)
        np.savez_compressed(os.path.join(DATA, f"state_e{e:g}_N0.5_mu1_ne{ne}.npz"),
                            V=V, e=e, N=0.5, mu=1.0, ne=ne, a=3.0, p=2)
        row = point(ne, e, 0.5, 1.0, V)
        row["converged"] = R.converged
        rows.append(row)
        print({k: row[k] for k in ("e_tilde", "mu_tilde", "M", "kappa_g", "g_rot_estimate",
                                   "rot_fraction", "converged")}, flush=True)
        prev = V
    # self-consistent physical-charge line: ẽ = E_PHYS / g
    mu1 = sorted([r for r in rows if r["mu_tilde"] == 1.0], key=lambda r: r["e_tilde"])
    et = np.array([r["e_tilde"] for r in mu1])
    line = []
    for g in np.linspace(1.0, 16.0, 61):
        e_t = E_PHYS / g
        if e_t < et.min() or e_t > et.max():
            continue
        itp = lambda key: float(np.interp(e_t, et, [r[key] for r in mu1]))
        Mr = itp("M_times_r") / g ** 2
        kg, gr = itp("kappa_g"), itp("g_rot_estimate")
        line.append(dict(g=float(g), e_tilde=e_t, M_times_r=Mr, g_static=kg * Mr ** 2,
                         g_rot_estimate=gr, rot_fraction=itp("rot_fraction") * g ** 4))
    out = dict(ne=ne, description=__doc__, e_phys=E_PHYS, rows=rows, physical_charge_line=line)
    with open(os.path.join(DATA, f"gscaling_ne{ne}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for L in line[::4]:
        print(f"g={L['g']:.2f} ẽ={L['e_tilde']:.3f} M·r={L['M_times_r']:.3f} "
              f"g_static={L['g_static']:.4g} g_rot≈{L['g_rot_estimate']:.3f} "
              f"ΔE_rot/M={L['rot_fraction']:.3g}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 48)
