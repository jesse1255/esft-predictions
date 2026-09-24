"""
Rigid-rotor (collective-coordinate) quantisation of the axisymmetric charged
Hopfion and the fate of its static-current magnetic moment.

Collective coordinates.  For the m = 1 axial Hopfion a spatial rotation about
the symmetry axis by β equals an isorotation by −β, so the orbit of rigid
rotations and isorotations is SO(3) itself.  The body-axis projection of the
spin is fixed by the U(1) charge, K₃ = −mN, and the rotor Hamiltonian is

    H = E_N + [J(J+1) − K₃²] / (2 V₁₁),

where E_N already contains the isorotation energy N²/(2 I_eff) (I_eff = V₃₃ with
electric screening).  A 2π spatial rotation and a 2π isorotation are the same
loop, so:
  * bosonic quantisation (loop contractible):  J integer,  N integer;
  * fermionic quantisation (loop non-contractible; stated in the literature
    for odd Hopf charge — Krusch & Speight 2006, to be checked): J and N both
    half-integer, so the smallest charge is Q = e/2.

Moments of inertia (per unit angular velocity Ω about x):
    V₁₁ = ∫ [ f²|δv|² + g⁻² Σ_i (v·(δv × D_i v))² + |δA|² ] d³x,
    δv = −[ρ sinφ ∂_z v − z sinφ ∂_ρ v − (z cosφ/ρ) T],   δA = −𝒟A + x̂ × A,
with 𝒟 = y∂_z − z∂_y.  Cross terms with the isorotation vanish after the φ
average; the Coulomb part of the electric field only rotates rigidly.

Magnetic moment.  The spatial current J = −e w − e² M A contains no ω, so the
classical moment is purely static and lies along the body symmetry axis:
μ_body = μ_s ẑ_body.  In the rotor state |J, M=J, K⟩,
    ⟨μ_z⟩ = μ_s ⟨cos θ⟩ = μ_s · M K / (J(J+1)),
i.e. −μ_s/3 for J = ½ and −μ_s/2 for J = 1 (K = −J).  The quadrupole of a
J = ½ state vanishes (⟨P₂(cos θ)⟩ = 0).  Currents induced by rotation about
x, y are of relative order Ω·R ~ R/V₁₁ and are estimated, not included.
"""

import json
import os
import sys

import numpy as np

from hopfion_axisym import Grid, Model, U1, U2, U3, AR, AZ, AP

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def inertia_x(model, U, nphi=8):
    """V₁₁ = matter (sigma + Skyrme) + electromagnetic part, full plane."""
    G = model.G
    e, c2, c4, m = model.e, model.c2, model.c4, model.m
    Qv, Qr, Qz = model.interp(U)
    rho, z = G.rho, G.zq
    v = Qv[:, :3]
    pr, pz = Qr[:, :3], Qz[:, :3]
    Ar, Az, Ap = Qv[:, AR], Qv[:, AZ], Qv[:, AP]
    T = np.stack([-v[:, 1], v[:, 0], np.zeros_like(rho)], 1)
    Dr = pr + (e * Ar)[:, None] * T
    Dz = pz + (e * Az)[:, None] * T
    Df = (m / rho + e * Ap)[:, None] * T
    Ar_r, Az_r, Ap_r = Qr[:, AR], Qr[:, AZ], Qr[:, AP]
    Ar_z, Az_z, Ap_z = Qz[:, AR], Qz[:, AZ], Qz[:, AP]
    sig = sk = em = 0.0
    for j in range(nphi):
        ph = 2 * np.pi * (j + 0.5) / nphi
        c, s = np.cos(ph), np.sin(ph)
        dv = -(rho[:, None] * s * pz - z[:, None] * s * pr - (z * c / rho)[:, None] * T)
        sig += np.dot(G.W, c2 * np.sum(dv * dv, 1))
        sk += np.dot(G.W, c4 * sum(np.sum(v * np.cross(dv, D), 1) ** 2 for D in (Dr, Dz, Df)))
        # Cartesian vector potential and its rotation generator
        Ax, Ay = Ar * c - Ap * s, Ar * s + Ap * c
        dAx_r, dAx_z = Ar_r * c - Ap_r * s, Ar_z * c - Ap_z * s
        dAy_r, dAy_z = Ar_r * s + Ap_r * c, Ar_z * s + Ap_z * c
        dAx_f, dAy_f = -Ay, Ax                       # ∂_φ of Cartesian components
        Dop = lambda fr, fz, ff: rho * s * fz - z * s * fr - (z * c / rho) * ff
        dAX = -Dop(dAx_r, dAx_z, dAx_f)
        dAY = -Dop(dAy_r, dAy_z, dAy_f) - Az
        dAZ = -Dop(Az_r, Az_z, 0.0) + Ay
        em += np.dot(G.W, dAX ** 2 + dAY ** 2 + dAZ ** 2)
    return dict(matter_sigma=sig / nphi, matter_skyrme=sk / nphi, em=em / nphi,
                total=(sig + sk + em) / nphi)


def rotor_levels(E_N, V11, N, J):
    K = -N
    return E_N + (J * (J + 1) - K * K) / (2.0 * V11)


def analyse(G, U, e, N, mu):
    M = Model(G, e=e, N=N, mu=mu, kappa=1.0)
    E, _ = M.energy(U, want_grad=False)
    d = M.diagnostics(U)
    Qv, Qr, Qz = M.interp(U)
    K = M.c2 * (Qv[:, U1] ** 2 + Qv[:, U2] ** 2) + M.c4 * (Qr[:, U3] ** 2 + Qz[:, U3] ** 2)
    V11 = inertia_x(M, U)
    J = abs(N)                      # lowest allowed spin: J = |K₃| = |N|
    Mq = rotor_levels(E, V11["total"], N, J)
    proj = (J * (-N)) / (J * (J + 1))      # ⟨cos θ⟩ for M = J, K = −N
    mu_s = d["mu_volume"]
    mu_lab = mu_s * proj
    Q = e * N
    g_like = 2 * Mq * abs(mu_lab) / (abs(Q) * J)
    return dict(e=e, N=N, mu_coupling=mu, E_N=E, I_eff=d["I_eff"], V33_bare=G.integrate(K),
                V11=V11, J=J, K3=-N, statistics="fermionic" if (2 * J) % 2 == 1 else "bosonic",
                rotational_energy=Mq - E, rotational_fraction=(Mq - E) / E,
                M_quantum=Mq, charge_Q=Q, rms_charge=d["rms_charge"],
                M_times_rms=Mq * d["rms_charge"], mu_static=mu_s, projection=proj,
                mu_lab=mu_lab, g_like=g_like,
                rot_current_estimate=d["rms_charge"] / V11["total"],
                quadrupole_lab_J_half="0 (⟨P₂⟩ = 0 for J = 1/2)" if J == 0.5 else None)


def full_plane(ne, V):
    from run_checks import mirror_to_full
    Gh = Grid(ne, ne, p=2, a=3.0, half=True)
    return mirror_to_full(Gh, V)


def main(ne_list):
    out = []
    for ne in ne_list:
        for (e, N, mu) in ((0.3, 1.0, 1.0), (0.3, 0.5, 1.0), (0.6, 0.5, 1.0), (0.6, 1.0, 1.0),
                           (0.6, 0.5, 0.5), (0.6, 0.5, 2.0)):
            fn = os.path.join(DATA, f"state_e{e:g}_N{N:g}_mu{mu:g}_ne{ne}.npz")
            if not os.path.exists(fn):
                continue
            V = np.load(fn)["V"]
            Gf, Uf = full_plane(ne, V)
            r = analyse(Gf, Uf, e, N, mu)
            r["ne"] = ne
            out.append(r)
            print(f"ne={ne} e={e} N={N} mu={mu}: E={r['E_N']:.6f} V11={r['V11']['total']:.3f} "
                  f"(sig {r['V11']['matter_sigma']:.2f}, sk {r['V11']['matter_skyrme']:.2f}, "
                  f"em {r['V11']['em']:.3f}) I_eff={r['I_eff']:.3f} dE_rot={r['rotational_energy']:.3e} "
                  f"mu_s={r['mu_static']:.4f} mu_lab={r['mu_lab']:.4f} g={r['g_like']:.4g} "
                  f"M*r={r['M_times_rms']:.2f}", flush=True)
    with open(os.path.join(DATA, "quantization.json"), "w") as fh:
        json.dump(dict(description=__doc__, rows=out), fh, indent=1)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [48])
