"""
Round 4 (2026-09-25): calculations behind STEELMAN_R4.md.

The question of this round is the opposite of the audit: for each statement of
Paper One, is there a setting in which it is exactly true, and what does that
setting cost?  Every number quoted in STEELMAN_R4.md is produced here and
written to steelman_results.json.  Only numpy/scipy are needed.

Sections
  S1  the Two-Range profile is Fetter's vortex ansatz; it is the exact vortex of
      a Ginzburg-Landau model with a sextic potential; exact GP vortex for
      comparison; the logarithmic UV divergence that really does force a core
  S2  the paper's E(R) = 4πn²[R − a·arctan(R/a)] is the angular energy of a
      global O(3) hedgehog; exact hedgehog of an O(3) model with a sextic
      potential; Mexican-hat hedgehog for comparison
  S3  "odd n → spin 1/2 from the phase bundle over S²": monopole harmonics.
      A charged scalar on the sphere of directions around a core that carries
      Chern number n has J ∈ n/2 + ℤ≥0
  S4  the electron's sizes inside the Standard Model (Darwin/zitterbewegung,
      Welton fluctuation smearing, electroweak range) and why the 1S-2S bound
      lands on the electroweak scale
  S5  sine-Gordon dictionary: the paper's own L₀ in 1+1 dimensions; classical
      kink size versus mass, M·r_rms = 4π/β²
  S6  Lorentz violation: the paper's bound at collider/cosmic-ray energies and
      the scale a single emergent medium would need
  S7  vortex-ring kinematics (no rest frame)

Literature numbers are marked LIT and listed in STEELMAN_R4.md §0 with their
source; they only enter S4 (particle masses, G_F) and S6 (GRB bounds).
"""

import json
import math
import os

import numpy as np
from scipy.integrate import quad, solve_bvp
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}

# ---------------------------------------------------------------------------
# constants (CODATA 2018, PDG 2024 masses: LIT)
# ---------------------------------------------------------------------------
alpha = 1 / 137.035999084
me_GeV = 0.51099895000e-3
hbarc_GeVfm = 0.1973269804
GeV_to_Hz = 2.417989242e23
G_F = 1.1663788e-5                 # GeV⁻²
m_W, m_Z, m_H, m_t = 80.369, 91.1876, 125.20, 172.57
v_higgs = 246.21965
sin2W = 0.23122                    # MS-bar at m_Z (LIT)
lamC_fm = hbarc_GeVfm / me_GeV     # reduced Compton wavelength, 386.16 fm


def rel(x, y):
    return float(abs(x - y) / abs(y))


# ---------------------------------------------------------------------------
# S1  vortex line
# ---------------------------------------------------------------------------
def s1():
    out = {}
    a = 1.0
    # (a) Fetter profile f = r/√(r²+a²) solves f'' + f'/r − f/r² = V'(f) with
    #     V(f) = (1−f²)²(3−2f²)/(4a²)
    r = np.linspace(0.01, 50, 20001)
    f = r / np.sqrt(r * r + a * a)
    fp = a * a / (r * r + a * a) ** 1.5
    fpp = -3 * a * a * r / (r * r + a * a) ** 2.5
    Vp = lambda f: (-4 * f * (1 - f * f) * (3 - 2 * f * f) - 4 * f * (1 - f * f) ** 2) / (4 * a * a)
    res = fpp + fp / r - f / r ** 2 - Vp(f)
    # near f = 1: V ≈ (1 − f)²/a², so the amplitude mode has m² = V''(1) = 2/a²
    out["fetter_exact_in_sextic_GL"] = dict(potential="V(f) = (1-f^2)^2 (3-2f^2)/(4 a^2)",
                                             max_residual=float(np.abs(res).max()),
                                             radial_mass_squared=float(2.0 / a ** 2))
    # (b) GP/GL vortex, units ξ = 1:  f'' + f'/r − n²f/r² + f(1−f²) = 0
    rows = []
    for n in (1, 2):
        R = 40.0
        x = np.linspace(1e-4, R, 4000)

        def rhs(x, y):
            return np.vstack([y[1], -y[1] / x + n * n * y[0] / x ** 2 - y[0] * (1 - y[0] ** 2)])

        def bc(ya, yb):
            return np.array([ya[0], yb[0] - (1 - n * n / (2 * R * R))])
        y0 = np.vstack([np.tanh(x / 1.5) ** n, n * (1 - np.tanh(x / 1.5) ** 2) / 1.5])
        sol = solve_bvp(rhs, bc, x, y0, tol=1e-9, max_nodes=200000)
        fe = sol.sol(x)[0]
        fpe = sol.sol(x)[1]

        def energy(fv, fpv, xs, n):
            dens = 0.5 * fpv ** 2 + 0.5 * n * n * fv ** 2 / xs ** 2 + 0.25 * (1 - fv ** 2) ** 2
            return float(np.trapezoid(2 * np.pi * xs * dens, xs))

        E_ex = energy(fe, fpe, x, n)

        def fetter_E(av):
            ff = x ** n / (x ** (2 * n) + av ** (2 * n)) ** 0.5
            ffp = np.gradient(ff, x)
            return energy(ff, ffp, x, n)
        opt = minimize_scalar(fetter_E, bounds=(0.3, 5.0), method="bounded", options=dict(xatol=1e-7))
        a_opt = float(opt.x)
        ff = x ** n / (x ** (2 * n) + a_opt ** (2 * n)) ** 0.5
        # effective core of the exact solution: fit f² n²/r² ≈ n²/(r²+a²) (n = 1 form)
        mask = x < 10
        afit = minimize_scalar(lambda av: np.sum((fe[mask] - x[mask] ** n / (x[mask] ** (2 * n) + av ** (2 * n)) ** 0.5) ** 2),
                               bounds=(0.3, 5.0), method="bounded").x
        rows.append(dict(n=n, bvp_status=int(sol.status), E_exact_R40=E_ex, E_fetter_R40=float(opt.fun),
                         fetter_a_opt=a_opt, fetter_a_opt_over_sqrt2=a_opt / math.sqrt(2),
                         energy_excess_rel=(float(opt.fun) - E_ex) / E_ex,
                         max_abs_profile_diff=float(np.abs(fe - ff).max()),
                         least_squares_core_a=float(afit)))
    out["gp_vortex"] = rows
    # (c) pure phase winding (no amplitude): energy per length π n² ln(R/r_min) — UV divergent
    out["pure_phase_log_divergence"] = [dict(r_min=rm, E_per_length_over_pi=float(math.log(40.0 / rm)))
                                        for rm in (1.0, 1e-2, 1e-4, 1e-8)]
    return out


# ---------------------------------------------------------------------------
# S2  global O(3) hedgehog
# ---------------------------------------------------------------------------
def s2():
    out = {}
    a = 1.0
    # (a) the paper's energy formula is the angular energy of f = r/√(r²+a²):
    #     ½|∇φ|²_angular = f²/r² = 1/(r²+a²)
    rows = []
    for R in (10.0, 100.0, 1000.0):
        num = quad(lambda r: 4 * np.pi * r * r * (r / math.sqrt(r * r + a * a)) ** 2 / r ** 2, 0, R, limit=400)[0]
        paper = 4 * np.pi * (R - a * math.atan(R / a))
        rows.append(dict(R=R, hedgehog_angular=num, paper_formula=paper, rel_diff=rel(num, paper)))
    out["paper_E_equals_hedgehog_angular_energy"] = rows
    # (b) f = r/√(r²+a²) solves f'' + 2f'/r − 2f/r² = V'(f) with V = (1−f²)²(2−f²)/(2a²)
    r = np.linspace(0.01, 50, 20001)
    f = r / np.sqrt(r * r + a * a)
    fp = a * a / (r * r + a * a) ** 1.5
    fpp = -3 * a * a * r / (r * r + a * a) ** 2.5
    Vp = lambda f: (-4 * f * (1 - f * f) * (2 - f * f) - 2 * f * (1 - f * f) ** 2) / (2 * a * a)
    res = fpp + 2 * fp / r - 2 * f / r ** 2 - Vp(f)
    # energy: E(R) = 4π∫[½f'² + f²/r² + V] r² dr = 4πR + E_core + O(1/R)
    V = lambda f: (1 - f * f) ** 2 * (2 - f * f) / (2 * a * a)
    dens = lambda r: 0.5 * (a * a / (r * r + a * a) ** 1.5) ** 2 + 1 / (r * r + a * a) + V(r / math.sqrt(r * r + a * a))
    Ecore = []
    for R in (50.0, 200.0, 1000.0):
        E = quad(lambda r: 4 * np.pi * r * r * dens(r), 0, R, limit=800)[0]
        Ecore.append(dict(R=R, E=E, E_minus_4piR=E - 4 * np.pi * R))
    # radial stability of this hedgehog (hedgehog-symmetric perturbations δf)
    N, L = 6000, 60.0
    h = L / N
    rr = h * np.arange(1, N)
    ff = rr / np.sqrt(rr * rr + a * a)
    Vpp = -(5 - 24 * ff ** 2 + 15 * ff ** 4) / a ** 2        # V' = −(5f − 8f³ + 3f⁵)/a²
    # u = r δf:  −u'' + (2/r² + V'') u = ω² u
    d = 2 / h ** 2 + 2 / rr ** 2 + Vpp
    e = -np.ones(N - 2) / h ** 2
    w = eigh_tridiagonal(d, e, select="i", select_range=(0, 2))[0]
    out["hedgehog_exact_in_sextic_O3"] = dict(potential="V(f) = (1-f^2)^2 (2-f^2)/(2 a^2)",
                                              max_residual=float(np.abs(res).max()),
                                              radial_mass_squared=float(4.0 / a ** 2),
                                              energy=Ecore, lowest_radial_omega2=w.tolist(),
                                              note="lowest radial ω² in a box of radius 60: positive, "
                                                   "continuum starts at m² = 4/a²; angular (non-hedgehog) "
                                                   "perturbations not checked here")
    # (c) Mexican-hat hedgehog for comparison: f'' + 2f'/r − 2f/r² + λ f(1−f²) = 0, λ = 1
    Rb = 40.0
    x = np.linspace(1e-4, Rb, 4000)

    def rhs(x, y):
        return np.vstack([y[1], -2 * y[1] / x + 2 * y[0] / x ** 2 - y[0] * (1 - y[0] ** 2)])

    def bc(ya, yb):
        return np.array([ya[0], yb[0] - (1 - 1 / Rb ** 2)])
    y0 = np.vstack([np.tanh(x / 2), (1 - np.tanh(x / 2) ** 2) / 2])
    sol = solve_bvp(rhs, bc, x, y0, tol=1e-9, max_nodes=200000)
    fe = sol.sol(x)[0]
    mask = x < 10
    afit = minimize_scalar(lambda av: np.sum((fe[mask] - x[mask] / np.sqrt(x[mask] ** 2 + av * av)) ** 2),
                           bounds=(0.3, 6), method="bounded").x
    fr = x / np.sqrt(x ** 2 + afit ** 2)
    out["mexican_hat_hedgehog"] = dict(bvp_status=int(sol.status), least_squares_core_a=float(afit),
                                       max_abs_profile_diff=float(np.abs(fe - fr).max()))
    return out


# ---------------------------------------------------------------------------
# S3  monopole harmonics
# ---------------------------------------------------------------------------
def s3(N=20000):
    """Spectrum of −(1/sinθ)∂θ(sinθ∂θ) + (m − q(1−cosθ))²/sin²θ on θ ∈ (0, π).

    This is the kinetic angular operator L_kin² of a charged scalar around a
    Dirac monopole of strength q = n/2 (Dirac-string gauge A_φ = q(1−cosθ)/sinθ,
    string on the negative z axis, ψ ∝ e^{imφ} with integer m).  The physical
    J_z is m − q.  Theory (Wu–Yang 1976): eigenvalues j(j+1) − q² with
    j = |q|, |q|+1, …, each j with J_z = −j … j.  Here the operator is
    discretised as a cell-centred finite-volume Sturm–Liouville problem (zero
    flux through the poles, where sinθ = 0), so no boundary behaviour is put in
    by hand; j is read off from the eigenvalue.
    """
    rows = []
    h = math.pi / N
    th = (np.arange(N) + 0.5) * h
    sf = np.sin(np.arange(1, N) * h)                   # interior faces
    W = h * np.sin(th)
    for n in (0, 1, 2, 3):
        q = n / 2
        levels = []
        for m in range(-4, 4 + n + 1):
            mu = m - q * (1 - np.cos(th))
            Ad = np.zeros(N)
            Ad[:-1] += sf / h
            Ad[1:] += sf / h
            Ad += mu ** 2 * h / np.sin(th)
            d = Ad / W
            e = -(sf / h) / np.sqrt(W[:-1] * W[1:])
            w = eigh_tridiagonal(d, e, select="i", select_range=(0, 3))[0]
            for lam in w:
                jj = -0.5 + math.sqrt(0.25 + lam + q * q)
                levels.append(dict(m=m, Jz=m - q, eigenvalue=float(lam), j=jj))
        byj = {}
        for L in levels:
            jr = round(L["j"] * 2) / 2
            byj.setdefault(jr, []).append(L)
        table = []
        for jr in sorted(byj)[:3]:
            Jz = sorted(L["Jz"] for L in byj[jr])
            table.append(dict(j=jr, Jz_values=Jz, multiplicity=len(Jz), expected=int(round(2 * jr + 1)),
                              eigenvalue_theory=jr * (jr + 1) - q * q,
                              max_eigenvalue_error=max(abs(L["eigenvalue"] - (jr * (jr + 1) - q * q)) for L in byj[jr])))
        rows.append(dict(n=n, q=q, lowest_j=table))
    return rows


# ---------------------------------------------------------------------------
# S4  electron sizes in the Standard Model; the 73 GeV coincidence
# ---------------------------------------------------------------------------
def s4():
    out = {}
    psi2_1S = (me_GeV * alpha) ** 3 / math.pi          # |ψ_1S(0)|², GeV³
    d_psi2 = (1 - 1 / 8) * psi2_1S                     # 1S − 2S
    # finite-size shift of an S level: ΔE = (2π/3) α ⟨r²⟩ |ψ(0)|²
    shift = lambda r2_GeV2, p2: (2 * math.pi / 3) * alpha * r2_GeV2 * p2
    # Darwin term = finite size with ⟨r²⟩ = (3/4)(ħ/mc)²
    r2_D = 0.75 / me_GeV ** 2
    E_D_1S = shift(r2_D, psi2_1S)
    out["darwin"] = dict(r_equiv_fm=math.sqrt(0.75) * lamC_fm, E1S_GeV=E_D_1S,
                         E1S_over_alpha4_m_over_2=E_D_1S / (alpha ** 4 * me_GeV / 2))
    # Welton: ⟨δr²⟩ = (2α/π)(ħ/mc)² ln(k_max/k_min); with k_max ~ m and k_min at
    # the binding energy the log is ln[1/(Zα)²], which reproduces Bethe's leading
    # term (4α/3π)(Zα)⁴m/n³·ln[1/(Zα)²]; 2S Lamb shift estimate
    r2_W = (2 * alpha / math.pi) / me_GeV ** 2 * math.log(1 / alpha ** 2)
    psi2_2S = psi2_1S / 8
    out["welton"] = dict(rms_fm=math.sqrt(r2_W) * hbarc_GeVfm,
                         lamb_2S_estimate_MHz=shift(r2_W, psi2_2S) * GeV_to_Hz / 1e6,
                         note="leading-log estimate (Welton 1948 / Bethe 1947); the measured 2S-2P "
                              "Lamb shift is 1057.8 MHz (LIT); the full Bethe logarithm lowers the estimate")
    # weak-interaction scale of hydrogen levels: G_F |ψ(0)|²
    nu_F = G_F * d_psi2 * GeV_to_Hz
    out["fermi_scale_1S2S_Hz"] = nu_F
    out["weak_range_fm"] = dict(hbar_over_mW=hbarc_GeVfm / m_W, hbar_over_mZ=hbarc_GeVfm / m_Z)

    # the paper's bound: shift(a²) = δν  →  Λ = 1/a
    def Lam_of(dnu):
        r2 = dnu / GeV_to_Hz / ((2 * math.pi / 3) * alpha * d_psi2)
        return 1 / math.sqrt(r2)
    L10 = Lam_of(10.0)
    out["paper_bound"] = dict(dnu_Hz=10.0, Lambda_GeV=L10, a_fm=hbarc_GeVfm / L10,
                              a_times_mW=m_W / L10, dnu_over_nuF=10.0 / nu_F)
    # closed form: Λ² = (2π/3)(α/G_F)(ν_F/δν)
    out["closed_form"] = dict(sqrt_alpha_over_GF_GeV=math.sqrt(alpha / G_F),
                              tree_level_sqrt_alpha_over_GF=m_W * math.sqrt(sin2W) * (math.sqrt(2) / math.pi) ** 0.5,
                              Lambda_check=math.sqrt((2 * math.pi / 3) * alpha / G_F * nu_F / 10.0))
    landmarks = dict(mW=m_W, mZ=m_Z, mH=m_H, mt=m_t, v=v_higgs, v_over_sqrt2=v_higgs / math.sqrt(2),
                     v_over_2=v_higgs / 2)
    scan = []
    for dnu in (1.0, 3.0, 10.0, 30.0, 100.0):
        L = Lam_of(dnu)
        near = min(landmarks, key=lambda k: abs(math.log(L / landmarks[k])))
        scan.append(dict(dnu_Hz=dnu, Lambda_GeV=L, nearest=near, ratio=L / landmarks[near]))
    out["lambda_vs_precision"] = scan
    # look-elsewhere: fraction of log-uniform Λ in [Λ(100 Hz), Λ(1 Hz)] within ±10 % / ±15 % of a landmark
    lo, hi = Lam_of(100.0), Lam_of(1.0)
    g = np.exp(np.linspace(math.log(lo), math.log(hi), 200001))
    lm = np.array(list(landmarks.values()))
    dist = np.min(np.abs(np.log(g[:, None] / lm[None, :])), axis=1)
    out["look_elsewhere"] = dict(range_GeV=[lo, hi], frac_within_10pct=float(np.mean(dist < math.log(1.10))),
                                 frac_within_15pct=float(np.mean(dist < math.log(1.15))))
    return out


# ---------------------------------------------------------------------------
# S5  sine-Gordon dictionary (1+1 dimensions)
# ---------------------------------------------------------------------------
def s5():
    """L₀ = (f²/2)(∂θ)² − Λ⁴(1 − cos θ) in 1+1D.  φ = fθ, β = 1/f, m = Λ²/f.
    Kink θ = 4 arctan e^{mx}: M_cl = 8fΛ² = 8m/β², topological charge
    (1/2π)∫∂ₓθ = 1, charge density (m/π) sech(mx), r_rms = π/(2m).
    Coleman: 4π/β² = 1 + g/π (massive Thirring coupling g); β² = 4π free fermion,
    β² = 8π Kosterlitz–Thouless point (cos term marginal)."""
    f, Lam = 1.0, 1.0
    m = Lam ** 2 / f
    x = np.linspace(-40, 40, 400001)
    th = 4 * np.arctan(np.exp(m * x))
    thx = np.gradient(th, x)
    M = np.trapezoid(0.5 * f * f * thx ** 2 + Lam ** 4 * (1 - np.cos(th)), x)
    Q = np.trapezoid(thx, x) / (2 * math.pi)
    rho = thx / (2 * math.pi)
    rrms = math.sqrt(np.trapezoid(x * x * rho, x))
    rows = []
    for b2 in (0.1, 1.0, 4 * math.pi, 6 * math.pi, 8 * math.pi):
        rows.append(dict(beta2=b2, M_cl_times_r_rms=4 * math.pi / b2, thirring_g_over_pi=4 * math.pi / b2 - 1))
    return dict(numeric=dict(M=M, M_expected=8 * f * Lam ** 2, Q=Q, r_rms=rrms, r_rms_expected=math.pi / (2 * m),
                             M_r=M * rrms, M_r_expected=4 * math.pi * f * f),
                table=rows)


# ---------------------------------------------------------------------------
# S6  Lorentz violation
# ---------------------------------------------------------------------------
def s6(E_QG2_GeV=None):
    Lc = 73.0
    rows = [dict(label=lab, k_GeV=k, delta_conservative=(k / Lc) ** 2)
            for lab, k in (("atomic (27 eV)", 27.2e-9), ("LEP beam (104.5 GeV)", 104.5),
                           ("cosmic-ray electrons (10 TeV)", 1e4), ("GRB photons (10 TeV)", 1e4))]
    out = dict(paper_conservative=rows)
    # Bogoliubov: ω² = c²k² + (ħk²/2m)² = c²k²[1 + (kξ)²/2], ξ = ħ/(√2 m c)
    # written as ω² = k²(1 + k²/E_QG,2²):  E_QG,2 = √2/ξ  →  ξ = √2 ħc / E_QG,2
    # the Fetter core of the same medium is a = √2 ξ  →  a = 2ħc / E_QG,2
    if E_QG2_GeV:
        out["emergent_single_scale"] = [dict(source=src, E_QG2_GeV=E, xi_max_fm=math.sqrt(2) * hbarc_GeVfm / E,
                                             core_a_max_fm=2 * hbarc_GeVfm / E,
                                             ratio_to_paper_a=(2 * hbarc_GeVfm / E) / 2.7e-3)
                                        for src, E in E_QG2_GeV]
    return out


# ---------------------------------------------------------------------------
# S7  vortex ring kinematics (thin ring, solid-body core, Kelvin/Saffman)
# ---------------------------------------------------------------------------
def s7():
    """E = ½ρκ²R[ln(8R/a) − 7/4], P = πρκR², U = dE/dP = κ/(4πR)[ln(8R/a) − 1/4].
    Units ρ = κ = a = 1.  U > 0 for every ring with R ≫ a: there is no rest frame,
    and E ∝ √P ln P, not √(P² + M²)."""
    rows = []
    for R in (3.0, 10.0, 100.0, 1e4):
        E = 0.5 * R * (math.log(8 * R) - 1.75)
        P = math.pi * R * R
        U = (math.log(8 * R) - 0.25) / (4 * math.pi * R)
        rows.append(dict(R_over_a=R, E=E, P=P, U=U, E_over_sqrtP_lnP=E / (math.sqrt(P) * math.log(P))))
    return rows


if __name__ == "__main__":
    OUT["S1_vortex"] = s1()
    OUT["S2_hedgehog"] = s2()
    OUT["S3_monopole_harmonics"] = s3()
    OUT["S4_electron_sizes"] = s4()
    OUT["S5_sine_gordon"] = s5()
    # LIT: GRB 090510, subluminal, 95 % CL (Vasileiou et al. 2013); GRB 221009A,
    # 6×10⁻⁸ E_Pl with E_Pl = 1.22×10¹⁹ GeV (LHAASO, Cao et al. 2024)
    OUT["S6_lorentz"] = s6([("Vasileiou 2013, GRB 090510", 1.3e11), ("LHAASO 2024, GRB 221009A", 6e-8 * 1.22089e19)])
    OUT["S7_vortex_ring"] = s7()
    with open(os.path.join(HERE, "steelman_results.json"), "w") as fh:
        json.dump(OUT, fh, indent=1)
    print(json.dumps(OUT, indent=1)[:20000])
